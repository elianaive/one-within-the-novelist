"""Unit tests for owtn.evaluation.scalar — types, rubric loading, factory.

LLM-calling tests are marked live_api; offline tests cover protocol shape,
config loading, and aggregation math.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from owtn.evaluation.scalar import (
    AggregatedScoreCard,
    AnchorCalibrator,
    AtomicPerDimScorer,
    Dim,
    EnsembleScorer,
    Rubric,
    ScoreCard,
    SingleCallScorer,
    bootstrap_ci,
    load_rubric,
)
from owtn.evaluation.scalar.scorer import _aggregate, _polarity_correct
from owtn.evaluation.scalar.factory import build_scorer_from_config


def _renderer(x):
    return str(x)


class TestRubricLoading:
    def test_load_concept_rubric(self):
        rubric = load_rubric("concept")
        assert isinstance(rubric, Rubric)
        assert len(rubric.dims) == 9
        assert "indelibility" in rubric.dim_names
        assert rubric.scale_max == 20

    def test_load_dag_rubric(self):
        rubric = load_rubric("dag")
        assert len(rubric.dims) == 8
        assert "concept_fidelity_thematic" in rubric.dim_names
        for d in rubric.dims:
            assert d.weight > 0

    def test_missing_rubric(self):
        with pytest.raises(FileNotFoundError):
            load_rubric("does_not_exist")


class TestAggregation:
    def setup_method(self):
        self.rubric = Rubric(dims=(
            Dim("a", "alpha", weight=2.0),
            Dim("b", "beta", weight=1.0),
        ), scale_max=20)

    def test_weighted_mean(self):
        # (2*20 + 1*10) / 3 = 50/3 = 16.67 -> /20 = 0.833
        agg = _aggregate({"a": 20.0, "b": 10.0}, self.rubric)
        assert abs(agg - (50/3) / 20) < 1e-6

    def test_polarity_inversion(self):
        rubric = Rubric(dims=(Dim("x", "neg", polarity="negative"),), scale_max=20)
        corrected = _polarity_correct({"x": 5}, rubric)
        assert corrected["x"] == 15.0   # 20 - 5

    def test_polarity_passthrough(self):
        corrected = _polarity_correct({"a": 18, "b": 3}, self.rubric)
        assert corrected == {"a": 18.0, "b": 3.0}

    def test_empty_dim_scores_zero_aggregate(self):
        assert _aggregate({}, self.rubric) == 0.0


class TestAnchorCalibrator:
    def test_monotonic_interp(self):
        cal = AnchorCalibrator(
            raw_anchor_aggregates=[0.3, 0.5, 0.8],
            expected_normalized=[0.2, 0.5, 0.9],
        )
        assert cal.calibrate(0.3) == pytest.approx(0.2)
        assert cal.calibrate(0.5) == pytest.approx(0.5)
        assert cal.calibrate(0.8) == pytest.approx(0.9)
        # Linear interp midpoint
        assert cal.calibrate(0.4) == pytest.approx(0.35)

    def test_clamped_outside_anchors(self):
        cal = AnchorCalibrator(
            raw_anchor_aggregates=[0.3, 0.8],
            expected_normalized=[0.2, 0.9],
        )
        # numpy.interp clamps at the endpoints
        assert cal.calibrate(0.1) == pytest.approx(0.2)
        assert cal.calibrate(1.0) == pytest.approx(0.9)

    def test_empty_calibrator_passthrough(self):
        cal = AnchorCalibrator(raw_anchor_aggregates=[], expected_normalized=[])
        assert cal.calibrate(0.7) == 0.7


class TestBootstrapCI:
    def test_ci_brackets_aggregate(self):
        rubric = Rubric(dims=(Dim("a", "_"),Dim("b", "_"),Dim("c", "_")), scale_max=20)
        card = ScoreCard(
            dim_scores={"a": 18.0, "b": 12.0, "c": 6.0},
            aggregate=12.0/20.0,
            n_calls=1,
            judge_label="t",
        )
        lo, hi = bootstrap_ci(card, rubric, n=200)
        assert 0.0 <= lo <= hi <= 1.0
        # With three observations of [18, 12, 6], CI should bracket 0.6 (=12/20)
        assert lo < 0.65 < hi


class TestFactory:
    def test_build_neutral_scorer(self):
        scorer = build_scorer_from_config("rollout_reward", _renderer)
        assert isinstance(scorer, AtomicPerDimScorer)
        assert scorer.persona_system_msg is None
        assert scorer.judge_model == "deepseek-v4-flash"
        # DeepSeek vendor toggle plumbed via reasoning_disabled flag
        assert scorer.sampling_kwargs.get("extra_body") == {"thinking": {"type": "disabled"}}

    def test_build_ensemble_scorer(self):
        scorer = build_scorer_from_config("handoff_rescore", _renderer)
        assert isinstance(scorer, EnsembleScorer)
        assert len(scorer.base_scorers) == 4
        for member in scorer.base_scorers:
            assert isinstance(member, AtomicPerDimScorer)
            assert member.persona_system_msg is not None

    def test_unknown_composition(self):
        with pytest.raises(KeyError):
            build_scorer_from_config("does_not_exist", _renderer)
