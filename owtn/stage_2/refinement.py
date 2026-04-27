"""Phase 3 cross-phase refinement: add_edge-only pass spanning the anchor.

Per `docs/stage-2/mcts.md` §Phase 3: a short post-Phase-2 pass that recovers
edge classes the 2-phase split excluded — most commonly disclosure edges
from post-anchor resolution beats backward to opening beats (epilogue-reveal
structures), or motivates edges that arc across the whole story.

What this phase does:
- Runs ~5 MCTS iterations on the Phase-2 winner DAG.
- Action space is restricted to `add_edge` only. No new nodes, no rewrites.
- Both endpoints must straddle the anchor: at least one upstream of the
  anchor and at least one downstream. Edges with both endpoints on the
  same side of the anchor are out of scope (they would have been added
  during forward or backward phases).

Output: the Phase 3 winner DAG (== Phase 2 winner if no improving edge
was found) plus a `RefinementMetrics` record for run-level monitoring of
the Phase-3 improvement rate. Per the design doc, the improvement rate
informs whether Phase 3 is load-bearing in v1 — if it's <5% across many
runs, Phase 3 can be dropped in v1.5.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

from owtn.models.stage_2.actions import Action, AddEdgeAction
from owtn.models.stage_2.dag import DAG
from owtn.stage_2.bidirectional import (
    ExpandFactory,
    PhaseContext,
    refinement_phase_context,
)
from owtn.stage_2.mcts import MCTS, MCTSConfig, RolloutFn


logger = logging.getLogger(__name__)


@dataclass
class RefinementMetrics:
    """Per-run Phase-3 monitoring metrics.

    `improved`: True iff the Phase-3 winner differs from the Phase-2 input.
    Tracking this across runs informs whether Phase 3 is load-bearing
    (per mcts.md §Monitoring Phase 3, expected band is 5–30% improvement
    rate; <5% means drop, >30% means reconsider the 2-phase split).
    """
    improved: bool
    iterations_run: int
    phase_2_edge_count: int
    phase_3_edge_count: int
    notes: list[str] = field(default_factory=list)


def _anchor_topo_index(dag: DAG) -> tuple[str, dict[str, int]]:
    """Return (anchor_id, {node_id: topo_index}). Used to classify edges
    as upstream/downstream of the anchor for spanning checks."""
    anchor_id = next(n.id for n in dag.nodes if n.role is not None)
    in_degree: dict[str, int] = {n.id: 0 for n in dag.nodes}
    out_edges: dict[str, list[str]] = defaultdict(list)
    for e in dag.edges:
        in_degree[e.dst] += 1
        out_edges[e.src].append(e.dst)
    node_index = {n.id: i for i, n in enumerate(dag.nodes)}
    ready = deque(sorted(
        (nid for nid, d in in_degree.items() if d == 0),
        key=lambda nid: node_index[nid],
    ))
    topo: dict[str, int] = {}
    idx = 0
    while ready:
        nid = ready.popleft()
        topo[nid] = idx
        idx += 1
        for child in out_edges[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
    return anchor_id, topo


def _partition_by_anchor(dag: DAG) -> tuple[str, list[str], list[str]]:
    """Return (anchor_id, upstream_ids, downstream_ids) — node ids partitioned
    by their topological position relative to the anchor. The anchor itself
    appears in neither list. Used by both the spanning-edge check and the
    refinement prompt's per-call topology context."""
    anchor_id, topo = _anchor_topo_index(dag)
    anchor_idx = topo[anchor_id]
    upstream = [nid for nid, idx in topo.items() if idx < anchor_idx]
    downstream = [nid for nid, idx in topo.items() if idx > anchor_idx]
    return anchor_id, upstream, downstream


def edge_spans_anchor(action: AddEdgeAction, dag: DAG) -> bool:
    """An add_edge spans the anchor iff its endpoints sit on opposite sides
    of the anchor (one upstream, one downstream), or one endpoint is the
    anchor itself with the other on either side. Same-side edges are
    rejected — they should have been added in forward or backward phases."""
    anchor_id, upstream, downstream = _partition_by_anchor(dag)
    if action.src_id not in {anchor_id, *upstream, *downstream}:
        return False
    if action.dst_id not in {anchor_id, *upstream, *downstream}:
        return False
    sides = set()
    for endpoint in (action.src_id, action.dst_id):
        if endpoint == anchor_id:
            sides.add("anchor")
        elif endpoint in upstream:
            sides.add("upstream")
        else:
            sides.add("downstream")
    if "upstream" in sides and "downstream" in sides:
        return True
    if "anchor" in sides and ("upstream" in sides or "downstream" in sides):
        return True
    return False


def render_topology_context(dag: DAG) -> str:
    """Per-call refinement prompt context: which existing nodes sit upstream
    of the anchor vs. downstream. Without this, the LLM has to infer the
    partition from the rendered outline; with the spanning constraint
    rejecting most candidates, that inference cost wastes the K=4 budget."""
    anchor_id, upstream, downstream = _partition_by_anchor(dag)
    upstream_str = ", ".join(upstream) if upstream else "(none)"
    downstream_str = ", ".join(downstream) if downstream else "(none)"
    return (
        f"REFINEMENT TOPOLOGY: anchor is `{anchor_id}`. "
        f"Upstream nodes (story-time before the anchor): {upstream_str}. "
        f"Downstream nodes (story-time after the anchor): {downstream_str}. "
        "Propose only edges whose endpoints sit on OPPOSITE sides of the "
        "anchor (one upstream, one downstream), or where one endpoint is "
        "the anchor itself with the other on either side. Same-side edges "
        "will be rejected at validation."
    )


def make_refinement_expand_fn(
    base_expand_fn,
    ctx: PhaseContext,
):
    """Build a Phase-3 expand_fn: filters by phase rules AND drops add_edge
    actions whose endpoints don't span the anchor.

    The base_expand_fn may propose any action; this wrapper enforces
    refinement's tighter constraints. add_beat / rewrite_beat actions are
    rejected by the phase context's action-type gating; non-spanning
    add_edge actions are rejected here.
    """
    from owtn.stage_2.bidirectional import action_valid_for_phase  # avoid circular at module load

    async def filtered(dag: DAG) -> list[Action]:
        proposed = await base_expand_fn(dag)
        kept: list[Action] = []
        for a in proposed:
            if not action_valid_for_phase(a, ctx, dag):
                continue
            if isinstance(a, AddEdgeAction) and not edge_spans_anchor(a, dag):
                continue
            kept.append(a)
        dropped = len(proposed) - len(kept)
        if dropped:
            logger.info(
                "refinement: dropped %d/%d actions (phase-rule + spanning constraint)",
                dropped, len(proposed),
            )
        return kept

    return filtered


async def run_refinement(
    dag: DAG,
    *,
    expand_factory: ExpandFactory,
    rollout_fn: RolloutFn,
    config: MCTSConfig | None = None,
    iterations: int = 5,
) -> tuple[DAG, RefinementMetrics]:
    """Run Phase 3 refinement on the given DAG.

    Returns (refined_dag, metrics). `refined_dag` is the Phase 3 winner
    (==input dag if no improving spanning edge was found within `iterations`).
    `metrics` includes the improvement flag for run-level monitoring.

    Caller is responsible for stitching: the Phase 2 winner from
    `run_bidirectional(...).winner_dag` is typically passed in here.
    """
    cfg = config or MCTSConfig()
    anchor_id = next(n.id for n in dag.nodes if n.role is not None)
    refine_ctx = refinement_phase_context(anchor_id=anchor_id)

    base_expand = expand_factory(refine_ctx)
    expand_fn = make_refinement_expand_fn(base_expand, refine_ctx)

    mcts = MCTS(
        dag,
        expand_fn=expand_fn,
        rollout_fn=rollout_fn,
        phase="refinement",
        config=cfg,
    )
    await mcts.run(iterations)
    winner = mcts.best_terminal().dag

    improved = len(winner.edges) > len(dag.edges)
    return winner, RefinementMetrics(
        improved=improved,
        iterations_run=mcts.iterations,
        phase_2_edge_count=len(dag.edges),
        phase_3_edge_count=len(winner.edges),
    )
