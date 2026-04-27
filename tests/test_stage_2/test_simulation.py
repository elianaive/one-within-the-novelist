"""Tests for `owtn.stage_2.simulation` — bounded rollout walk per BiT-MCTS.

The simulator extends a partial DAG up to s_max times via the search's
expansion machinery, evaluates each accepted step against the running
champion, and halts on reward non-improvement. Walk extensions are
ephemeral; they don't enter the search tree.

Tests cover:
- Reward-improving walks accept up to s_max extensions
- Non-improving extensions trigger early stop
- Walk halts when expand_fn raises or returns no valid actions
- min_partial_size gates skip the walk and just cheap-judge
- The simulator closure plumbs cheap_judge + expand_fn correctly
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from owtn.evaluation.models import STAGE_2_DIMENSION_NAMES
from owtn.evaluation.stage_2 import CheapJudgeOutcome, CompareInputs
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.actions import AddBeatAction
from owtn.models.stage_2.dag import DAG, Node
from owtn.stage_2.simulation import (
    SimulationResult,
    make_simulator,
    simulate_bounded,
)
from tests.conftest import HILLS_GENOME


@pytest.fixture
def hills_concept() -> ConceptGenome:
    return ConceptGenome.model_validate(HILLS_GENOME)


@pytest.fixture
def seed_dag(hills_concept: ConceptGenome) -> DAG:
    return DAG(
        concept_id="t", preset="phoebe_ish",
        motif_threads=["a", "b"], concept_demands=[],
        nodes=[Node(
            id="anchor", sketch=hills_concept.anchor_scene.sketch,
            role=[hills_concept.anchor_scene.role], motifs=[],
        )],
        edges=[], character_arcs=[], story_constraints=[],
        target_node_count=8,
    )


def _judge() -> JudgePersona:
    return JudgePersona(
        id="t", name="t", identity="t", values=["v"], exemplars=["e"],
        lean_in_signals=["s"], harshness="standard", priority="primary",
        model=["claude-sonnet-4-6"], temperature=0.0,
    )


def _outcome(reward: float) -> CheapJudgeOutcome:
    return CheapJudgeOutcome(
        challenger_wins=reward > 0.5,
        reward=reward,
        resolved_votes={d: "tie" for d in STAGE_2_DIMENSION_NAMES},
        cost=0.001,
    )


def _add_beat(node_id: str, anchor: str = "anchor") -> AddBeatAction:
    return AddBeatAction(
        action_type="add_beat",
        anchor_id=anchor,
        direction="downstream",
        new_node_id=node_id,
        sketch="A specific subsequent beat with concrete imagery.",
        edge_type="causal",
        edge_payload={"realizes": "the next thing happens because the prior thing did"},
    )


# ----- simulate_bounded -----


class TestSimulateBounded:
    def test_walks_to_smax_when_reward_keeps_improving(
        self, seed_dag: DAG, hills_concept: ConceptGenome,
    ) -> None:
        """Reward improves at every step — walk takes the full s_max=3 steps."""
        rewards = iter([0.4, 0.5, 0.7, 0.9])  # initial + 3 improving steps

        async def fake_judge(inputs, *, cheap_judge):
            return _outcome(next(rewards))

        async def fake_expand(dag):
            new_id = f"b{len(dag.nodes)}"
            return [_add_beat(new_id, anchor="anchor")]

        with patch("owtn.stage_2.simulation.cheap_judge_compare", side_effect=fake_judge):
            result = asyncio.run(simulate_bounded(
                seed_dag,
                concept=hills_concept, champion=seed_dag,
                cheap_judge=_judge(), expand_fn=fake_expand, s_max=3,
            ))

        assert result.steps_accepted == 3
        assert result.outcome.reward == pytest.approx(0.9)
        assert len(result.walked_partial.nodes) == 4  # seed + 3 added

    def test_early_stops_when_reward_decreases(
        self, seed_dag: DAG, hills_concept: ConceptGenome,
    ) -> None:
        """First step improves; second decreases — walk halts after step 1."""
        rewards = iter([0.4, 0.6, 0.5])

        async def fake_judge(inputs, *, cheap_judge):
            return _outcome(next(rewards))

        async def fake_expand(dag):
            return [_add_beat(f"b{len(dag.nodes)}")]

        with patch("owtn.stage_2.simulation.cheap_judge_compare", side_effect=fake_judge):
            result = asyncio.run(simulate_bounded(
                seed_dag,
                concept=hills_concept, champion=seed_dag,
                cheap_judge=_judge(), expand_fn=fake_expand, s_max=3,
            ))

        assert result.steps_accepted == 1
        assert result.outcome.reward == pytest.approx(0.6)
        assert len(result.walked_partial.nodes) == 2

    def test_halts_when_expand_raises(
        self, seed_dag: DAG, hills_concept: ConceptGenome,
    ) -> None:
        rewards = iter([0.4])

        async def fake_judge(inputs, *, cheap_judge):
            return _outcome(next(rewards))

        async def failing_expand(dag):
            raise RuntimeError("expansion failed")

        with patch("owtn.stage_2.simulation.cheap_judge_compare", side_effect=fake_judge):
            result = asyncio.run(simulate_bounded(
                seed_dag,
                concept=hills_concept, champion=seed_dag,
                cheap_judge=_judge(), expand_fn=failing_expand, s_max=3,
            ))

        assert result.steps_accepted == 0
        assert result.walked_partial is seed_dag
        assert result.outcome.reward == pytest.approx(0.4)

    def test_halts_when_no_valid_action(
        self, seed_dag: DAG, hills_concept: ConceptGenome,
    ) -> None:
        """expand_fn returns actions but none apply cleanly (e.g., duplicate
        node_id) — walk halts with steps_accepted=0 after the initial eval."""
        rewards = iter([0.4])

        async def fake_judge(inputs, *, cheap_judge):
            return _outcome(next(rewards))

        async def empty_expand(dag):
            # Action references the existing anchor as new_node_id → duplicate-id
            # → apply_action raises → simulator skips → no valid action remains.
            return [_add_beat("anchor")]

        with patch("owtn.stage_2.simulation.cheap_judge_compare", side_effect=fake_judge):
            result = asyncio.run(simulate_bounded(
                seed_dag,
                concept=hills_concept, champion=seed_dag,
                cheap_judge=_judge(), expand_fn=empty_expand, s_max=3,
            ))

        assert result.steps_accepted == 0
        assert result.walked_partial is seed_dag

    def test_skips_first_invalid_takes_second(
        self, seed_dag: DAG, hills_concept: ConceptGenome,
    ) -> None:
        """Multiple candidates per expansion: simulator takes first valid."""
        rewards = iter([0.4, 0.6, 0.5])

        async def fake_judge(inputs, *, cheap_judge):
            return _outcome(next(rewards))

        async def expand_with_invalid_first(dag):
            return [
                _add_beat("anchor"),  # duplicate id - invalid
                _add_beat(f"valid_{len(dag.nodes)}"),  # valid second choice
            ]

        with patch("owtn.stage_2.simulation.cheap_judge_compare", side_effect=fake_judge):
            result = asyncio.run(simulate_bounded(
                seed_dag,
                concept=hills_concept, champion=seed_dag,
                cheap_judge=_judge(), expand_fn=expand_with_invalid_first, s_max=3,
            ))

        # First step accepts (reward 0.4 → 0.6); second step rejected (0.6 → 0.5)
        assert result.steps_accepted == 1


# ----- make_simulator (the closure builder) -----


class TestMakeSimulator:
    def test_skips_walk_when_partial_below_min_size(
        self, seed_dag: DAG, hills_concept: ConceptGenome,
    ) -> None:
        """Partial has 1 node, min_partial_size=3 → walk skipped, just cheap-judge."""
        async def fake_judge(inputs, *, cheap_judge):
            return _outcome(0.4)

        expand_calls = []

        async def fake_expand(dag):
            expand_calls.append(dag)
            return [_add_beat("never")]

        sim = make_simulator(
            cheap_judge=_judge(), expand_fn=fake_expand,
            s_max=3, min_partial_size=3,
        )
        with patch("owtn.stage_2.simulation.cheap_judge_compare", side_effect=fake_judge):
            result = asyncio.run(sim(CompareInputs(
                challenger=seed_dag, champion=seed_dag, concept=hills_concept,
            )))

        # No expansion call — gate skipped the walk.
        assert expand_calls == []
        assert result.steps_accepted == 0
        assert result.walked_partial is seed_dag

    def test_walks_when_partial_meets_min_size(
        self, seed_dag: DAG, hills_concept: ConceptGenome,
    ) -> None:
        rewards = iter([0.4, 0.6, 0.5])

        async def fake_judge(inputs, *, cheap_judge):
            return _outcome(next(rewards))

        async def fake_expand(dag):
            return [_add_beat(f"b{len(dag.nodes)}")]

        sim = make_simulator(
            cheap_judge=_judge(), expand_fn=fake_expand,
            s_max=3, min_partial_size=1,  # gate satisfied
        )
        with patch("owtn.stage_2.simulation.cheap_judge_compare", side_effect=fake_judge):
            result = asyncio.run(sim(CompareInputs(
                challenger=seed_dag, champion=seed_dag, concept=hills_concept,
            )))

        assert result.steps_accepted == 1
        assert result.outcome.reward == pytest.approx(0.6)
