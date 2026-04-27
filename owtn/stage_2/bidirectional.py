"""Bidirectional MCTS orchestration: forward → backward → Phase 3.

Per `docs/stage-2/mcts.md` §Bidirectional Phases:

- **Phase 1 (forward)**: MCTS rooted at the 1-node anchor seed. Permitted
  edge types are `causal` and `implication`. New beats are added downstream
  of the anchor (anchor → new). Search direction: anchor → resolution.
- **Phase 2 (backward)**: MCTS reset; the forward winner becomes the root.
  Permitted edge types are `causal`, `constraint`, `disclosure`, `motivates`,
  with `implication` allowed but de-emphasized. New beats are added upstream
  of the anchor (new → anchor). Search direction: anchor → opening.
- **Phase 3 (refinement)**: short `add_edge`-only pass. Lives in
  `owtn.stage_2.refinement`.

What's in this module:
- `PhaseContext` — bundle of phase metadata MCTS expansion needs.
- `make_phase_filtered_expand_fn` — wraps a base expand_fn with phase-rule
  filtering. Drops actions whose edge type / direction / endpoint scope
  violates the active phase.
- `run_bidirectional` — runs forward then backward, returns the backward
  winner's DAG.

Tests inject mock expand_factory callables; production wires the real LLM
expansion call (`owtn.stage_2.operators.propose_actions_via_llm`).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from owtn.models.stage_2.actions import (
    Action,
    AddBeatAction,
    AddEdgeAction,
    RewriteBeatAction,
)
from owtn.models.stage_2.dag import DAG, EdgeType
from owtn.models.stage_2.mcts_node import Phase
from owtn.stage_2.mcts import MCTS, MCTSConfig
from owtn.stage_2.mcts import ExpandFn, RolloutFn


logger = logging.getLogger(__name__)


# ----- Phase rules (from docs/stage-2/mcts.md §Bidirectional Phases) -----


_FORWARD_EDGE_TYPES: frozenset[EdgeType] = frozenset({"causal", "implication"})
_BACKWARD_EDGE_TYPES: frozenset[EdgeType] = frozenset(
    {"causal", "constraint", "disclosure", "motivates", "implication"}
)
_REFINEMENT_EDGE_TYPES: frozenset[EdgeType] = frozenset(
    {"causal", "disclosure", "implication", "constraint", "motivates"}
)


@dataclass(frozen=True)
class PhaseContext:
    """Metadata threaded through expansion calls during one phase.

    Fields:
        phase: which phase ("forward", "backward", "refinement").
        anchor_id: the role-bearing node's id. Phase rules constrain which
            DAG nodes are valid expansion endpoints relative to this anchor.
        permitted_edge_types: edge types this phase allows (others rejected
            at filter time).
        permitted_directions: which `add_beat.direction` values are allowed.
            Forward only allows "downstream"; backward only "upstream";
            refinement allows neither (no add_beat).
        allow_add_beat / allow_add_edge / allow_rewrite_beat: action-type
            gating. Refinement only allows add_edge.
        pacing_hint: short string substituted into the expansion prompt's
            `{PACING_HINT}` placeholder. Empty for refinement.
    """
    phase: Phase
    anchor_id: str
    permitted_edge_types: frozenset[EdgeType]
    permitted_directions: frozenset[Literal["downstream", "upstream"]]
    allow_add_beat: bool
    allow_add_edge: bool
    allow_rewrite_beat: bool
    pacing_hint: str = ""


def forward_phase_context(*, anchor_id: str, pacing_hint: str = "") -> PhaseContext:
    return PhaseContext(
        phase="forward",
        anchor_id=anchor_id,
        permitted_edge_types=_FORWARD_EDGE_TYPES,
        permitted_directions=frozenset({"downstream"}),
        allow_add_beat=True,
        allow_add_edge=True,
        allow_rewrite_beat=True,
        pacing_hint=pacing_hint,
    )


def backward_phase_context(*, anchor_id: str, pacing_hint: str = "") -> PhaseContext:
    return PhaseContext(
        phase="backward",
        anchor_id=anchor_id,
        permitted_edge_types=_BACKWARD_EDGE_TYPES,
        permitted_directions=frozenset({"upstream"}),
        allow_add_beat=True,
        allow_add_edge=True,
        allow_rewrite_beat=True,
        pacing_hint=pacing_hint,
    )


def refinement_phase_context(*, anchor_id: str) -> PhaseContext:
    return PhaseContext(
        phase="refinement",
        anchor_id=anchor_id,
        permitted_edge_types=_REFINEMENT_EDGE_TYPES,
        permitted_directions=frozenset(),  # no add_beat
        allow_add_beat=False,
        allow_add_edge=True,
        allow_rewrite_beat=False,
        pacing_hint="",
    )


# ----- Phase-rule action filtering -----


def action_valid_for_phase(action: Action, ctx: PhaseContext, dag: DAG) -> bool:
    """Returns True iff the action satisfies the phase's edge-type, direction,
    and endpoint-scope rules.

    Used to drop LLM-proposed actions that violate phase rules before they
    reach the DAG validator. Most invalid actions would be rejected by the
    DAG validator anyway, but filtering here keeps logs clean (no noisy
    ValidationErrors for actions that are obviously phase-mismatched).
    """
    # Action-type gating
    if isinstance(action, AddBeatAction):
        if not ctx.allow_add_beat:
            return False
        if action.edge_type not in ctx.permitted_edge_types:
            return False
        if action.direction not in ctx.permitted_directions:
            return False
        # Endpoint-scope: forward requires anchor_id be anchor or descendant
        # of anchor. Backward requires anchor_id be anchor or ancestor.
        # We approximate "anchor or descendant/ancestor" by checking the
        # node exists; full topological reachability is the DAG validator's
        # job. Phase 6 ships the looser check; tighten if pilot data shows
        # cross-phase scope violations.
        node_ids = {n.id for n in dag.nodes}
        if action.anchor_id not in node_ids:
            return False
        return True
    if isinstance(action, AddEdgeAction):
        if not ctx.allow_add_edge:
            return False
        if action.edge_type not in ctx.permitted_edge_types:
            return False
        node_ids = {n.id for n in dag.nodes}
        return action.src_id in node_ids and action.dst_id in node_ids
    if isinstance(action, RewriteBeatAction):
        if not ctx.allow_rewrite_beat:
            return False
        node_ids = {n.id for n in dag.nodes}
        return action.node_id in node_ids
    return False  # pragma: no cover — discriminated union covers all branches


def make_phase_filtered_expand_fn(
    base_expand_fn: ExpandFn,
    ctx: PhaseContext,
) -> ExpandFn:
    """Wrap an expand_fn with phase-rule filtering.

    The wrapped fn calls `base_expand_fn`, then drops any actions whose
    edge type, direction, or endpoint scope violates the phase. Returned
    list preserves the LLM's ranked order (filtering is a per-action
    boolean, not a re-sort).
    """
    async def filtered(dag: DAG) -> list[Action]:
        proposed = await base_expand_fn(dag)
        valid = [a for a in proposed if action_valid_for_phase(a, ctx, dag)]
        dropped = len(proposed) - len(valid)
        if dropped:
            logger.info(
                "%s phase: dropped %d/%d actions for phase-rule violations",
                ctx.phase, dropped, len(proposed),
            )
        return valid

    return filtered


# ----- Orchestration -----


@dataclass
class BidirectionalResult:
    """Output of one bidirectional run.

    Fields:
        winner_dag: best DAG from the backward phase (Phase 3 not yet applied).
        forward_winner_dag: best DAG from the forward phase (root of Phase 2).
        forward_iterations / backward_iterations: actual iteration counts run.
        forward_mcts / backward_mcts: the MCTS objects, kept for metric
            extraction (phase improvement rate, visit counts, etc.).
    """
    winner_dag: DAG
    forward_winner_dag: DAG
    forward_iterations: int
    backward_iterations: int
    forward_mcts: MCTS = field(repr=False)
    backward_mcts: MCTS = field(repr=False)


# Type alias: a function that, given a phase context, returns an expand_fn
# for that phase. The factory pattern lets the bidirectional orchestrator
# build phase-specific expand_fns without knowing the prompt details.
ExpandFactory = Callable[[PhaseContext], ExpandFn]


def _anchor_id(dag: DAG) -> str:
    """Look up the role-bearing node's id (the anchor). DAG validation
    enforces exactly one role-bearing node, so this is unambiguous on
    valid DAGs."""
    for n in dag.nodes:
        if n.role is not None:
            return n.id
    raise ValueError("DAG has no role-bearing node (anchor)")


async def run_bidirectional(
    seed_dag: DAG,
    *,
    expand_factory: ExpandFactory,
    rollout_fn: RolloutFn,
    config: MCTSConfig | None = None,
    forward_iterations: int = 50,
    backward_iterations: int = 50,
    forward_pacing_hint: str = "",
    backward_pacing_hint: str = "",
) -> BidirectionalResult:
    """Run forward then backward MCTS phases.

    Forward phase: MCTS rooted at `seed_dag` (typically the 1-node anchor
    seed from `seed_root`). Permitted edges = causal + implication. New
    beats added downstream.

    Backward phase: fresh MCTS rooted at the forward winner. Permitted edges
    = causal + constraint + disclosure + motivates + implication. New beats
    added upstream.

    The backward MCTS is constructed fresh — UCB statistics from the forward
    phase do NOT carry over (per `mcts.md` §Why reset the MCTS tree between
    phases). Only the DAG carries forward.
    """
    cfg = config or MCTSConfig()
    anchor = _anchor_id(seed_dag)

    forward_ctx = forward_phase_context(anchor_id=anchor, pacing_hint=forward_pacing_hint)
    forward_base = expand_factory(forward_ctx)
    forward_expand = make_phase_filtered_expand_fn(forward_base, forward_ctx)
    forward_mcts = MCTS(
        seed_dag,
        expand_fn=forward_expand,
        rollout_fn=rollout_fn,
        phase="forward",
        config=cfg,
    )
    await forward_mcts.run(forward_iterations)
    forward_winner = forward_mcts.best_terminal().dag

    backward_anchor = _anchor_id(forward_winner)  # same anchor; sanity-rebind in case of any node-id renames
    backward_ctx = backward_phase_context(anchor_id=backward_anchor, pacing_hint=backward_pacing_hint)
    backward_base = expand_factory(backward_ctx)
    backward_expand = make_phase_filtered_expand_fn(backward_base, backward_ctx)
    backward_mcts = MCTS(
        forward_winner,
        expand_fn=backward_expand,
        rollout_fn=rollout_fn,
        phase="backward",
        config=cfg,
    )
    await backward_mcts.run(backward_iterations)
    backward_winner = backward_mcts.best_terminal().dag

    return BidirectionalResult(
        winner_dag=backward_winner,
        forward_winner_dag=forward_winner,
        forward_iterations=forward_mcts.iterations,
        backward_iterations=backward_mcts.iterations,
        forward_mcts=forward_mcts,
        backward_mcts=backward_mcts,
    )
