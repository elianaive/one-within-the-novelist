"""Bidirectional orchestration tests. Mocked expansion + rollout.

Phase 6 exit criteria covered here:
- Forward phase: edge-type filtering rejects non-{causal, implication}
  proposals; produces a DAG with only those types from the anchor forward.
- Backward phase: grows ancestors correctly; tree state is fresh (UCB
  visit counts not carried over from forward).
- Phase 3 refinement runs (covered in `test_refinement.py`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from owtn.models.stage_2.actions import (
    Action,
    AddBeatAction,
    AddEdgeAction,
    RewriteBeatAction,
)
from owtn.models.stage_2.dag import DAG
from owtn.stage_2.bidirectional import (
    PhaseContext,
    action_valid_for_phase,
    backward_phase_context,
    forward_phase_context,
    make_phase_filtered_expand_fn,
    refinement_phase_context,
    run_bidirectional,
)
from owtn.stage_2.mcts import MCTSConfig


# ----- Helpers -----


def _seed_dag(concept_id: str = "seed") -> DAG:
    """1-node anchor seed (post-seed_root state)."""
    return DAG(
        concept_id=concept_id,
        preset="cassandra_ish",
        motif_threads=["motif a", "motif b"],
        concept_demands=[],
        nodes=[{
            "id": "anchor",
            "sketch": "A complete anchor sketch with enough words to validate.",
            "role": ["climax"],
            "motifs": [],
        }],
        edges=[],
        character_arcs=[],
        story_constraints=[],
        target_node_count=5,
    )


def _make_add_beat(
    new_node_id: str,
    *,
    anchor_id: str = "anchor",
    direction: str = "downstream",
    edge_type: str = "causal",
    realizes: str = "this is a substantive realizes payload of four tokens",
) -> AddBeatAction:
    return AddBeatAction(
        anchor_id=anchor_id,
        direction=direction,
        new_node_id=new_node_id,
        sketch=f"A new beat sketch for {new_node_id} with enough words to validate.",
        edge_type=edge_type,
        edge_payload={"realizes": realizes} if edge_type == "causal" else
                     {"entails": realizes} if edge_type == "implication" else
                     {"reframes": realizes, "withheld": realizes} if edge_type == "disclosure" else
                     {"prohibits": realizes} if edge_type == "constraint" else
                     {"agent": "x", "goal": realizes},
    )


def _constant_rollout(value: float) -> Callable[[DAG], Awaitable[float]]:
    async def rollout(dag: DAG) -> float:
        return value
    return rollout


# ----- Phase context unit tests -----


class TestPhaseContexts:
    def test_forward_permitted_edge_types(self) -> None:
        ctx = forward_phase_context(anchor_id="anchor")
        assert ctx.permitted_edge_types == frozenset({"causal", "implication"})
        assert ctx.permitted_directions == frozenset({"downstream"})
        assert ctx.allow_add_beat is True
        assert ctx.allow_add_edge is True

    def test_backward_permitted_edge_types(self) -> None:
        ctx = backward_phase_context(anchor_id="anchor")
        assert ctx.permitted_edge_types == frozenset(
            {"causal", "constraint", "disclosure", "motivates", "implication"}
        )
        assert ctx.permitted_directions == frozenset({"upstream"})

    def test_refinement_disallows_add_beat(self) -> None:
        ctx = refinement_phase_context(anchor_id="anchor")
        assert ctx.allow_add_beat is False
        assert ctx.allow_add_edge is True
        assert ctx.allow_rewrite_beat is False
        assert ctx.permitted_directions == frozenset()


# ----- Action validation per phase -----


class TestActionPhaseValidation:
    def test_forward_rejects_disclosure_add_beat(self) -> None:
        seed = _seed_dag()
        ctx = forward_phase_context(anchor_id="anchor")
        action = _make_add_beat("x", edge_type="disclosure")
        assert action_valid_for_phase(action, ctx, seed) is False

    def test_forward_accepts_causal_downstream_add_beat(self) -> None:
        seed = _seed_dag()
        ctx = forward_phase_context(anchor_id="anchor")
        action = _make_add_beat("x", direction="downstream", edge_type="causal")
        assert action_valid_for_phase(action, ctx, seed) is True

    def test_forward_rejects_upstream_add_beat(self) -> None:
        seed = _seed_dag()
        ctx = forward_phase_context(anchor_id="anchor")
        action = _make_add_beat("x", direction="upstream", edge_type="causal")
        assert action_valid_for_phase(action, ctx, seed) is False

    def test_backward_accepts_disclosure_upstream(self) -> None:
        seed = _seed_dag()
        ctx = backward_phase_context(anchor_id="anchor")
        action = _make_add_beat("x", direction="upstream", edge_type="disclosure")
        assert action_valid_for_phase(action, ctx, seed) is True

    def test_backward_rejects_downstream(self) -> None:
        seed = _seed_dag()
        ctx = backward_phase_context(anchor_id="anchor")
        action = _make_add_beat("x", direction="downstream", edge_type="disclosure")
        assert action_valid_for_phase(action, ctx, seed) is False

    def test_refinement_rejects_add_beat(self) -> None:
        seed = _seed_dag()
        ctx = refinement_phase_context(anchor_id="anchor")
        action = _make_add_beat("x", edge_type="causal")
        assert action_valid_for_phase(action, ctx, seed) is False

    def test_refinement_accepts_add_edge_for_known_nodes(self) -> None:
        """Build a real 2-node DAG (via DAG(...) so validators run) and
        confirm refinement accepts a properly-typed add_edge between them."""
        from owtn.models.stage_2.dag import DAG, Node
        two_node = DAG(
            concept_id="t",
            preset="cassandra_ish",
            motif_threads=["m1", "m2"],
            concept_demands=[],
            nodes=[
                Node(id="anchor",
                     sketch="A complete anchor sketch with enough words to validate.",
                     role=["climax"], motifs=[]),
                Node(id="epilogue",
                     sketch="An epilogue beat with enough words to satisfy validation.",
                     role=None, motifs=[]),
            ],
            edges=[],
            character_arcs=[],
            story_constraints=[],
            target_node_count=5,
        )
        ctx = refinement_phase_context(anchor_id="anchor")
        action = AddEdgeAction(
            src_id="anchor",
            dst_id="epilogue",
            edge_type="disclosure",
            edge_payload={
                "reframes": "the anchor's framing is reframed by the epilogue retroactively",
                "withheld": "from the reader, that the anchor was already pointing at the epilogue",
            },
        )
        assert action_valid_for_phase(action, ctx, two_node) is True


# ----- Phase-filtered expand_fn -----


class TestPhaseFilter:
    def test_filter_drops_disallowed_edge_types(self) -> None:
        """Forward phase: an LLM proposing disclosure actions has them
        dropped before MCTS sees them. Causal actions pass through."""
        seed = _seed_dag()
        ctx = forward_phase_context(anchor_id="anchor")

        async def base_expand(dag: DAG) -> list[Action]:
            return [
                _make_add_beat("good", edge_type="causal"),
                _make_add_beat("bad", edge_type="disclosure"),  # forward rejects disclosure
                _make_add_beat("also_good", edge_type="implication"),
            ]

        wrapped = make_phase_filtered_expand_fn(base_expand, ctx)
        result = asyncio.run(wrapped(seed))

        ids = [a.new_node_id for a in result if isinstance(a, AddBeatAction)]
        assert "good" in ids
        assert "also_good" in ids
        assert "bad" not in ids


# ----- Bidirectional orchestration -----


class TestBidirectionalOrchestration:
    """Phase 6 exit: forward produces only causal/implication; backward
    grows ancestors with fresh tree state."""

    def test_forward_then_backward_runs(self) -> None:
        seed = _seed_dag()

        per_phase_actions = {
            "forward": [
                _make_add_beat("forward_beat", direction="downstream", edge_type="causal"),
            ],
            "backward": [
                _make_add_beat(
                    "backward_beat",
                    direction="upstream",
                    edge_type="disclosure",
                    realizes="this is a substantive payload field with several words",
                ),
            ],
        }

        def factory(ctx: PhaseContext):
            phase = ctx.phase
            actions = per_phase_actions[phase]

            async def expand(dag: DAG) -> list[Action]:
                if not getattr(expand, "_called", False):
                    expand._called = True  # type: ignore[attr-defined]
                    return list(actions)
                return []

            return expand

        # Higher-reward rollouts when the canned beat made it into the DAG —
        # makes best_terminal pick the deeper branch reliably.
        async def rollout(dag: DAG) -> float:
            if any(n.id in {"forward_beat", "backward_beat"} for n in dag.nodes):
                return 0.9
            return 0.4

        result = asyncio.run(run_bidirectional(
            seed,
            expand_factory=factory,
            rollout_fn=rollout,
            forward_iterations=3,
            backward_iterations=3,
            config=MCTSConfig(seed=42),
        ))

        # Forward winner should contain "forward_beat" (downstream of anchor).
        forward_node_ids = {n.id for n in result.forward_winner_dag.nodes}
        assert "forward_beat" in forward_node_ids

        # Backward winner should contain "backward_beat" (upstream of anchor).
        winner_node_ids = {n.id for n in result.winner_dag.nodes}
        assert "backward_beat" in winner_node_ids
        # Plus the forward-phase beat (carried into Phase 2 root).
        assert "forward_beat" in winner_node_ids

    def test_forward_phase_rejects_disclosure(self) -> None:
        """If the forward-phase expand_fn proposes a disclosure action, the
        phase filter drops it — the resulting DAG has no disclosure edges
        from the forward phase."""
        seed = _seed_dag()

        def factory(ctx: PhaseContext):
            async def expand(dag: DAG) -> list[Action]:
                if not getattr(expand, "_called", False):
                    expand._called = True  # type: ignore[attr-defined]
                    return [
                        _make_add_beat(
                            "naughty",
                            direction="downstream",
                            edge_type="disclosure",  # forward forbids
                            realizes="this is a substantive payload field of several words",
                        ),
                    ]
                return []
            return expand

        result = asyncio.run(run_bidirectional(
            seed,
            expand_factory=factory,
            rollout_fn=_constant_rollout(0.5),
            forward_iterations=3,
            backward_iterations=0,  # skip backward to isolate forward
            config=MCTSConfig(seed=42),
        ))

        # No disclosure edges in the forward winner — phase filter dropped them.
        assert all(e.type != "disclosure" for e in result.forward_winner_dag.edges)
        # And no "naughty" beat was added.
        assert all(n.id != "naughty" for n in result.forward_winner_dag.nodes)

    def test_backward_starts_with_fresh_tree_state(self) -> None:
        """Backward phase's MCTS instance has its own visit counts —
        forward-phase visits do not bleed into the backward UCB stats."""
        seed = _seed_dag()

        def factory(ctx: PhaseContext):
            async def expand(dag: DAG) -> list[Action]:
                return []  # no expansion; just checking tree freshness
            return expand

        result = asyncio.run(run_bidirectional(
            seed,
            expand_factory=factory,
            rollout_fn=_constant_rollout(0.5),
            forward_iterations=5,
            backward_iterations=3,
            config=MCTSConfig(seed=42),
        ))

        # Backward MCTS ran 3 iterations on its own tree.
        assert result.backward_iterations == 3
        # Forward and backward MCTS are distinct objects with distinct trees.
        assert result.forward_mcts is not result.backward_mcts
        # Visit counts are independent: backward root's count is bounded by
        # backward_iterations regardless of how many forward iterations ran.
        # (When expand_fn returns [], the first iteration produces no rollout
        # because the leaf is marked fully_expanded inline; subsequent
        # iterations rollout from the now-terminal root.)
        assert 0 <= result.backward_mcts.root.visits <= 3
        assert 0 <= result.forward_mcts.root.visits <= 5
        # Critically, backward visits do NOT include forward visits — if
        # they did, backward.root.visits could be 8 (5+3). Bound is 3.
        assert result.backward_mcts.root.visits != (
            result.forward_mcts.root.visits + result.backward_mcts.root.visits - result.forward_mcts.root.visits
        ) or True  # tautology; the real check is the bound above.
