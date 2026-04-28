"""Unit tests for `owtn.stage_2.scalar_handoff`.

Confirms scalar-mode handoff rescore correctly ranks entries by aggregate
score and sets fields the downstream `build_handoff_for_concept` expects.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from owtn.evaluation.scalar.types import ScoreCard
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG
from owtn.stage_2.scalar_handoff import rescore_entries_scalar
from owtn.stage_2.tournament import TournamentEntry
from tests.conftest import HILLS_GENOME


def _entry(preset: str) -> TournamentEntry:
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    dag = DAG(
        concept_id="c1", preset=preset,
        motif_threads=["m1"], concept_demands=[],
        nodes=[{"id": "anchor", "sketch": concept.anchor_scene.sketch,
                "role": [concept.anchor_scene.role], "motifs": []}],
        edges=[], character_arcs=[], story_constraints=[],
        target_node_count=5,
    )
    return TournamentEntry(preset=preset, dag=dag, mcts_reward=0.5)


class FakeScorer:
    """Returns scores keyed by preset for predictable ranking tests."""
    def __init__(self, scores_by_preset: dict[str, float]):
        self.scores_by_preset = scores_by_preset

    async def score(self, dag):
        agg = self.scores_by_preset[dag.preset]
        return ScoreCard(
            dim_scores={"x": agg * 20},
            aggregate=agg,
            n_calls=1,
            judge_label="fake",
        )


def test_rescore_ranks_by_aggregate_descending():
    entries = [_entry("a"), _entry("b"), _entry("c")]
    scorer = FakeScorer({"a": 0.6, "b": 0.9, "c": 0.7})

    with patch("owtn.stage_2.scalar_handoff.build_scorer_from_config", return_value=scorer):
        ranked = asyncio.run(rescore_entries_scalar(
            entries, composition_name="handoff_rescore",
        ))

    # Best-first: b (0.9), c (0.7), a (0.6)
    assert [e.preset for e in ranked] == ["b", "c", "a"]
    # mcts_reward updated to scalar aggregate
    assert ranked[0].mcts_reward == 0.9
    assert ranked[2].mcts_reward == 0.6
    # wins assigned by rank position (top = most wins)
    assert ranked[0].wins == 2
    assert ranked[1].wins == 1
    assert ranked[2].wins == 0


def test_rescore_single_entry():
    entries = [_entry("solo")]
    scorer = FakeScorer({"solo": 0.5})

    with patch("owtn.stage_2.scalar_handoff.build_scorer_from_config", return_value=scorer):
        ranked = asyncio.run(rescore_entries_scalar(
            entries, composition_name="handoff_rescore",
        ))

    assert len(ranked) == 1
    assert ranked[0].mcts_reward == 0.5
    assert ranked[0].wins == 0
