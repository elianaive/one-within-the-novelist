"""Tier 3 (concept-demand fidelity) verdict tests.

Spec: `docs/stage-2/evaluation.md` §Tier 3.

Coverage:
- Empty `concept_demands` short-circuits without LLM call.
- Missing classifier model logs a warning and skips.
- LLM verdict count mismatch → skipped, `failed=False`.
- LLM call raising → skipped, `failed=False`.
- Successful all-satisfied response → `failed=False`, verdicts populated.
- One `failed` verdict in the list → `failed=True`.
- Tier 3 priority gate demotes failed entries below all-satisfied entries
  in `TournamentEntry.sort_key` and in `rescore_entries_scalar`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from owtn.evaluation.scalar.types import ScoreCard
from owtn.evaluation.stage_2_tier3 import (
    Tier3Result,
    _Tier3LLMResponse,
    evaluate_concept_demands,
)
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG
from owtn.models.stage_2.handoff import ConceptDemandVerdict
from owtn.stage_2.scalar_handoff import rescore_entries_scalar
from owtn.stage_2.tournament import TournamentEntry
from tests.conftest import HILLS_GENOME


# ----- Helpers -----


@dataclass
class _FakeQueryResult:
    content: object
    cost: float = 0.005
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    thought: str = ""
    model_name: str = "deepseek-v4-flash"


def _seed_dag(*, concept_demands: list[str], preset: str = "cassandra_ish") -> DAG:
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    return DAG(
        concept_id="c1", preset=preset,
        motif_threads=["the hills"], concept_demands=list(concept_demands),
        nodes=[{"id": "anchor", "sketch": concept.anchor_scene.sketch,
                "role": [concept.anchor_scene.role], "motifs": []}],
        edges=[], character_arcs=[], story_constraints=[],
        target_node_count=5,
    )


def _concept() -> ConceptGenome:
    return ConceptGenome.model_validate(HILLS_GENOME)


# ----- evaluate_concept_demands -----


class TestEmptyDemands:
    def test_no_demands_short_circuits_without_llm_call(self) -> None:
        dag = _seed_dag(concept_demands=[])
        with patch("owtn.evaluation.stage_2_tier3.query_async") as q:
            result = asyncio.run(evaluate_concept_demands(
                dag, concept=_concept(), classifier_model="dummy",
            ))
        assert q.call_count == 0
        assert result.skipped_reason == "no_concept_demands"
        assert result.verdicts == []
        assert result.failed is False


class TestNoClassifierModel:
    def test_missing_classifier_model_skips_with_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        dag = _seed_dag(concept_demands=["the closing beat must address the reader"])
        with caplog.at_level(logging.WARNING, logger="owtn.evaluation.stage_2_tier3"):
            with patch("owtn.evaluation.stage_2_tier3.query_async") as q:
                result = asyncio.run(evaluate_concept_demands(
                    dag, concept=_concept(), classifier_model=None,
                ))
        assert q.call_count == 0
        assert result.skipped_reason == "no_classifier_model"
        assert result.failed is False
        assert any("Tier 3 skipped" in rec.message for rec in caplog.records)


class TestLLMSuccess:
    def test_all_satisfied_returns_failed_false(self) -> None:
        dag = _seed_dag(concept_demands=["d1", "d2"])
        verdicts = [
            ConceptDemandVerdict(demand="d1", verdict="satisfied", rationale="anchor handles it"),
            ConceptDemandVerdict(demand="d2", verdict="partial", rationale="motif gestures at it"),
        ]
        response = _Tier3LLMResponse(verdicts=verdicts)

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=response)

        with patch("owtn.evaluation.stage_2_tier3.query_async", side_effect=fake_query):
            result = asyncio.run(evaluate_concept_demands(
                dag, concept=_concept(), classifier_model="dummy",
            ))

        assert result.skipped_reason is None
        assert result.failed is False
        assert len(result.verdicts) == 2
        assert result.verdicts[0].verdict == "satisfied"
        assert result.verdicts[1].verdict == "partial"

    def test_one_failed_makes_overall_failed(self) -> None:
        dag = _seed_dag(concept_demands=["d1", "d2"])
        response = _Tier3LLMResponse(verdicts=[
            ConceptDemandVerdict(demand="d1", verdict="satisfied", rationale="ok"),
            ConceptDemandVerdict(demand="d2", verdict="failed", rationale="absent"),
        ])

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=response)

        with patch("owtn.evaluation.stage_2_tier3.query_async", side_effect=fake_query):
            result = asyncio.run(evaluate_concept_demands(
                dag, concept=_concept(), classifier_model="dummy",
            ))

        assert result.failed is True


class TestLLMFailure:
    def test_provider_error_returns_skipped_not_failed(self) -> None:
        dag = _seed_dag(concept_demands=["d1"])

        async def boom(**kwargs):
            raise RuntimeError("provider down")

        with patch("owtn.evaluation.stage_2_tier3.query_async", side_effect=boom):
            result = asyncio.run(evaluate_concept_demands(
                dag, concept=_concept(), classifier_model="dummy",
            ))

        # Extraction failure must NOT be treated as DAG failure.
        assert result.failed is False
        assert result.skipped_reason == "llm_error:RuntimeError"

    def test_verdict_count_mismatch_skipped(self) -> None:
        dag = _seed_dag(concept_demands=["d1", "d2"])
        # LLM returned only 1 verdict for 2 demands.
        response = _Tier3LLMResponse(verdicts=[
            ConceptDemandVerdict(demand="d1", verdict="satisfied", rationale="ok"),
        ])

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=response)

        with patch("owtn.evaluation.stage_2_tier3.query_async", side_effect=fake_query):
            result = asyncio.run(evaluate_concept_demands(
                dag, concept=_concept(), classifier_model="dummy",
            ))

        assert result.failed is False
        assert result.skipped_reason == "verdict_count_mismatch"


# ----- Priority gate: TournamentEntry.sort_key -----


class TestSortKeyPriority:
    def test_failed_demand_demotes_below_all_satisfied(self) -> None:
        """A demand-failed entry with more wins still ranks below an
        all-satisfied entry with fewer wins."""
        dag = _seed_dag(concept_demands=[])
        winner_with_failed_demand = TournamentEntry(
            preset="a", dag=dag, wins=10, dim_wins_total=80,
            concept_demand_failed=True,
        )
        loser_all_satisfied = TournamentEntry(
            preset="b", dag=dag, wins=0, dim_wins_total=0,
            concept_demand_failed=False,
        )
        ranked = sorted([winner_with_failed_demand, loser_all_satisfied],
                        key=lambda e: e.sort_key())
        assert [e.preset for e in ranked] == ["b", "a"]

    def test_within_tier_falls_back_to_pairwise(self) -> None:
        """Two failed-demand entries fall back to wins/dim_wins."""
        dag = _seed_dag(concept_demands=[])
        e1 = TournamentEntry(preset="a", dag=dag, wins=2, dim_wins_total=10,
                             concept_demand_failed=True)
        e2 = TournamentEntry(preset="b", dag=dag, wins=5, dim_wins_total=20,
                             concept_demand_failed=True)
        ranked = sorted([e1, e2], key=lambda e: e.sort_key())
        assert [e.preset for e in ranked] == ["b", "a"]


# ----- Priority gate: scalar handoff -----


class _FakeScorer:
    def __init__(self, scores_by_preset: dict[str, float]):
        self.scores_by_preset = scores_by_preset

    async def score(self, dag):
        agg = self.scores_by_preset[dag.preset]
        return ScoreCard(
            dim_scores={"x": agg * 20}, aggregate=agg,
            n_calls=1, judge_label="fake",
        )


class TestScalarHandoffPriority:
    def test_demand_failed_sinks_below_satisfied_regardless_of_aggregate(self) -> None:
        """Scalar rescore must demote a high-aggregate demand-failed entry
        below a lower-aggregate all-satisfied entry."""
        dag_a = _seed_dag(concept_demands=[], preset="a")
        dag_b = _seed_dag(concept_demands=[], preset="b")
        # 'a' would win on aggregate alone (0.95 vs 0.6) but has a failed
        # demand; 'b' should rank first.
        entries = [
            TournamentEntry(preset="a", dag=dag_a, mcts_reward=0.0,
                            concept_demand_failed=True),
            TournamentEntry(preset="b", dag=dag_b, mcts_reward=0.0,
                            concept_demand_failed=False),
        ]
        scorer = _FakeScorer({"a": 0.95, "b": 0.6})

        with patch("owtn.stage_2.scalar_handoff.build_scorer_from_config",
                   return_value=scorer):
            ranked = asyncio.run(rescore_entries_scalar(
                entries, composition_name="handoff_rescore",
            ))

        assert [e.preset for e in ranked] == ["b", "a"]
