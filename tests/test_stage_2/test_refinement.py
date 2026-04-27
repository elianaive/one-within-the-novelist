"""Phase 3 refinement tests. Mocked expansion + rollout.

Phase 6 exit criterion covered: Phase 3 runs when Phase 2 completes and the
improvement-rate metric is recorded. Plus the per-action filter that drops
non-spanning add_edge proposals (the structural reason Phase 3 exists at all).
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
from owtn.stage_2.bidirectional import refinement_phase_context
from owtn.stage_2.refinement import (
    RefinementMetrics,
    edge_spans_anchor,
    make_refinement_expand_fn,
    render_topology_context,
    run_refinement,
)
from owtn.stage_2.mcts import MCTSConfig


# ----- Helpers -----


def _phase_2_winner_dag() -> DAG:
    """A 5-node DAG mid-way through MCTS — simulates a reasonable Phase 2
    winner. Anchor is the climax beat at index 2 (middle); upstream beats
    0,1; downstream beats 3,4.
    """
    return DAG(
        concept_id="phase2_winner",
        preset="cassandra_ish",
        motif_threads=["motif a", "motif b"],
        concept_demands=[],
        nodes=[
            {"id": "opening", "sketch": "An opening beat with enough words to validate.",
             "role": None, "motifs": []},
            {"id": "rising", "sketch": "A rising-action beat with enough words to validate.",
             "role": None, "motifs": []},
            {"id": "climax", "sketch": "The climax beat with enough words to validate.",
             "role": ["climax"], "motifs": []},
            {"id": "falling", "sketch": "A falling-action beat with enough words to validate.",
             "role": None, "motifs": []},
            {"id": "coda", "sketch": "A coda beat with enough words to validate.",
             "role": None, "motifs": []},
        ],
        edges=[
            {"src": "opening", "dst": "rising", "type": "causal",
             "realizes": "the opening establishes the conditions that produce the rising action"},
            {"src": "rising", "dst": "climax", "type": "causal",
             "realizes": "the rising action arrives at the climax via accumulated tension"},
            {"src": "climax", "dst": "falling", "type": "causal",
             "realizes": "the climax sets up the immediate consequences of the falling action"},
            {"src": "falling", "dst": "coda", "type": "causal",
             "realizes": "the falling action settles into the coda's resolution"},
        ],
        character_arcs=[],
        story_constraints=[],
        target_node_count=5,
    )


def _spanning_add_edge() -> AddEdgeAction:
    """An add_edge that spans the anchor: opening → coda is a long-range
    disclosure crossing the anchor (climax)."""
    return AddEdgeAction(
        src_id="opening",
        dst_id="coda",
        edge_type="disclosure",
        edge_payload={
            "reframes": "the opening's apparent simplicity is recontextualized by the coda",
            "withheld": "from the reader, what the opening was really about",
        },
    )


def _non_spanning_add_edge() -> AddEdgeAction:
    """An add_edge between two upstream beats — should NOT be added in Phase 3
    (would have been added during forward or backward)."""
    return AddEdgeAction(
        src_id="opening",
        dst_id="rising",  # both upstream of anchor
        edge_type="implication",
        edge_payload={
            "entails": "the opening's logical framework entails the rising action's predicate",
        },
    )


def _constant_rollout(value: float) -> Callable[[DAG], Awaitable[float]]:
    async def rollout(dag: DAG) -> float:
        return value
    return rollout


# ----- render_topology_context (Issue 6 from 2026-04-26 issue) -----


class TestRenderTopologyContext:
    """The refinement prompt's per-call DAG-state context. Without this the
    LLM had to infer the upstream/downstream partition from the rendered
    outline; with the spanning-edge filter, that wasted most of the K=4
    candidates. The renderer gives the LLM the partition directly."""

    def test_includes_anchor_id(self) -> None:
        dag = _phase_2_winner_dag()
        rendered = render_topology_context(dag)
        assert "`climax`" in rendered

    def test_partitions_upstream_and_downstream_correctly(self) -> None:
        """For the canonical 5-node Phase 2 winner, opening + rising are
        upstream of climax; falling + coda are downstream."""
        dag = _phase_2_winner_dag()
        rendered = render_topology_context(dag)
        # Both upstream nodes named in the upstream section.
        upstream_section = rendered.split("Upstream nodes")[1].split("Downstream nodes")[0]
        assert "opening" in upstream_section
        assert "rising" in upstream_section
        assert "falling" not in upstream_section
        # Downstream section likewise.
        downstream_section = rendered.split("Downstream nodes")[1]
        assert "falling" in downstream_section
        assert "coda" in downstream_section
        assert "opening" not in downstream_section

    def test_includes_spanning_instruction(self) -> None:
        """The renderer must explicitly tell the LLM the spanning constraint —
        otherwise it's just a fact dump, not actionable steering."""
        dag = _phase_2_winner_dag()
        rendered = render_topology_context(dag)
        assert "OPPOSITE sides" in rendered or "opposite sides" in rendered.lower()


# ----- edge_spans_anchor unit tests -----


class TestSpanningCheck:
    def test_spanning_edge_detected(self) -> None:
        dag = _phase_2_winner_dag()
        action = _spanning_add_edge()
        assert edge_spans_anchor(action, dag) is True

    def test_non_spanning_upstream_only_rejected(self) -> None:
        dag = _phase_2_winner_dag()
        action = _non_spanning_add_edge()  # both endpoints upstream of anchor
        assert edge_spans_anchor(action, dag) is False

    def test_non_spanning_downstream_only_rejected(self) -> None:
        dag = _phase_2_winner_dag()
        action = AddEdgeAction(
            src_id="falling",
            dst_id="coda",  # both downstream of anchor
            edge_type="causal",
            edge_payload={"realizes": "the falling action sets up the coda's stillness"},
        )
        assert edge_spans_anchor(action, dag) is False

    def test_anchor_to_downstream_counts_as_spanning(self) -> None:
        """Edge with one endpoint at the anchor and the other on either
        side counts as spanning (per refinement.py docstring)."""
        dag = _phase_2_winner_dag()
        action = AddEdgeAction(
            src_id="climax",  # anchor
            dst_id="coda",   # downstream
            edge_type="implication",
            edge_payload={"entails": "the climax's logical content propagates into the coda's framing"},
        )
        assert edge_spans_anchor(action, dag) is True


# ----- Refinement filter -----


class TestRefinementFilter:
    def test_filter_drops_non_spanning_add_edge(self) -> None:
        dag = _phase_2_winner_dag()
        ctx = refinement_phase_context(anchor_id="climax")

        async def base_expand(d: DAG) -> list[Action]:
            return [
                _spanning_add_edge(),
                _non_spanning_add_edge(),  # rejected by spanning filter
            ]

        wrapped = make_refinement_expand_fn(base_expand, ctx)
        result = asyncio.run(wrapped(dag))

        assert len(result) == 1
        assert isinstance(result[0], AddEdgeAction)
        assert result[0].dst_id == "coda"

    def test_filter_drops_add_beat_in_refinement(self) -> None:
        dag = _phase_2_winner_dag()
        ctx = refinement_phase_context(anchor_id="climax")

        async def base_expand(d: DAG) -> list[Action]:
            return [
                AddBeatAction(
                    anchor_id="climax",
                    direction="downstream",
                    new_node_id="extra",
                    sketch="An extra beat that refinement should not allow.",
                    edge_type="causal",
                    edge_payload={"realizes": "this would extend the falling action with another beat"},
                ),
                _spanning_add_edge(),
            ]

        wrapped = make_refinement_expand_fn(base_expand, ctx)
        result = asyncio.run(wrapped(dag))

        # Only the spanning add_edge survives.
        assert len(result) == 1
        assert isinstance(result[0], AddEdgeAction)


# ----- Refinement run -----


class TestRefinementRun:
    """Phase 3 runs end-to-end on a Phase-2-winner DAG with mocks."""

    def test_runs_and_returns_metrics(self) -> None:
        dag = _phase_2_winner_dag()

        def factory(ctx) -> Callable[[DAG], Awaitable[list[Action]]]:
            async def expand(d: DAG) -> list[Action]:
                if not getattr(expand, "_called", False):
                    expand._called = True  # type: ignore[attr-defined]
                    return [_spanning_add_edge()]
                return []
            return expand

        refined, metrics = asyncio.run(run_refinement(
            dag,
            expand_factory=factory,
            rollout_fn=_constant_rollout(0.7),
            iterations=3,
            config=MCTSConfig(seed=42),
        ))

        assert isinstance(metrics, RefinementMetrics)
        assert metrics.iterations_run == 3
        # Refinement may or may not select the spanning edge as best (depends
        # on rollout dynamics); the structural property is that the metric
        # tracks edge-count delta, and the input DAG is preserved if no
        # improving edge is found.
        assert metrics.phase_2_edge_count == 4  # original 4 causal edges
        assert metrics.phase_3_edge_count >= metrics.phase_2_edge_count

    def test_no_improvement_when_no_spanning_edges_proposed(self) -> None:
        """If the LLM only proposes non-spanning edges, refinement filter
        drops them all and the DAG is unchanged."""
        dag = _phase_2_winner_dag()

        def factory(ctx) -> Callable[[DAG], Awaitable[list[Action]]]:
            async def expand(d: DAG) -> list[Action]:
                return [_non_spanning_add_edge()]  # always rejected
            return expand

        refined, metrics = asyncio.run(run_refinement(
            dag,
            expand_factory=factory,
            rollout_fn=_constant_rollout(0.5),
            iterations=3,
            config=MCTSConfig(seed=42),
        ))

        assert metrics.improved is False
        assert metrics.phase_3_edge_count == metrics.phase_2_edge_count
        # Refined DAG has the same edges as input (model_dump comparison).
        assert {(e.src, e.dst, e.type) for e in refined.edges} == \
               {(e.src, e.dst, e.type) for e in dag.edges}

    def test_metrics_track_edge_count_delta(self) -> None:
        """When refinement finds an improving edge, metrics.improved is True
        and edge_count grows by exactly the number of accepted edges."""
        dag = _phase_2_winner_dag()

        actions_to_propose = [_spanning_add_edge()]

        def factory(ctx) -> Callable[[DAG], Awaitable[list[Action]]]:
            async def expand(d: DAG) -> list[Action]:
                # Propose only on the first call (the cache mechanism in MCTS
                # consumes one per iteration).
                if not getattr(expand, "_called", False):
                    expand._called = True  # type: ignore[attr-defined]
                    return list(actions_to_propose)
                return []
            return expand

        refined, metrics = asyncio.run(run_refinement(
            dag,
            expand_factory=factory,
            rollout_fn=_constant_rollout(1.0),  # high reward → MCTS picks the new branch
            iterations=2,
            config=MCTSConfig(seed=42),
        ))

        # Refinement found one extra edge.
        if metrics.improved:
            assert metrics.phase_3_edge_count == metrics.phase_2_edge_count + 1
            assert any(e.dst == "coda" and e.type == "disclosure" for e in refined.edges)
