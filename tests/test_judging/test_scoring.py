"""Tests for owtn.judging.scoring — Hölder mean and selection score math."""

import math

import numpy as np
import pytest

from owtn.judging.scoring import (
    AggregateResult,
    aggregate_judge_scores,
    holder_mean,
    selection_score,
)


class TestHolderMean:
    def test_uniform_scores(self):
        """Uniform scores should return that score regardless of p."""
        assert holder_mean([3, 3, 3, 3, 3]) == pytest.approx(3.0, abs=1e-4)
        assert holder_mean([5, 5, 5, 5, 5]) == pytest.approx(5.0, abs=1e-4)

    def test_p1_is_arithmetic_mean(self):
        """At p=1.0, Hölder mean equals arithmetic mean."""
        scores = [1, 2, 3, 4, 5]
        expected = np.mean(scores)
        assert holder_mean(scores, p=1.0) == pytest.approx(expected, abs=1e-4)

    def test_skewed_scores_penalized(self):
        """Skewed scores should be pulled down from arithmetic mean."""
        skewed = holder_mean([5, 5, 5, 5, 1], p=0.4)
        arithmetic = np.mean([5, 5, 5, 5, 1])
        # Hölder mean < arithmetic mean when scores are uneven
        assert skewed < arithmetic
        # But still above the minimum
        assert skewed > 1.0

    def test_lower_p_harsher(self):
        """Lower p should penalize unevenness more."""
        scores = [5, 5, 5, 5, 1]
        harsh = holder_mean(scores, p=0.2)
        mild = holder_mean(scores, p=0.6)
        assert harsh < mild

    def test_empty_scores(self):
        assert holder_mean([]) == 0.0

    def test_single_score(self):
        assert holder_mean([4.0]) == pytest.approx(4.0, abs=1e-4)

    def test_zeros_handled(self):
        """Zeros should not cause math errors."""
        result = holder_mean([0, 0, 0, 5])
        assert result >= 0
        assert math.isfinite(result)

    def test_all_zeros(self):
        result = holder_mean([0, 0, 0])
        assert result >= 0
        assert result < 0.01


class TestSelectionScore:
    def test_no_bonus_below_threshold(self):
        """No diversity bonus when std < threshold."""
        score = selection_score(judge_mean=3.5, judge_std=0.5, std_threshold=0.8)
        assert score == pytest.approx(3.5)

    def test_bonus_above_threshold(self):
        """Diversity bonus activates above threshold."""
        score = selection_score(judge_mean=3.5, judge_std=1.8, std_threshold=0.8)
        expected_bonus = (1.8 - 0.8) * 0.15
        assert score == pytest.approx(3.5 + expected_bonus)

    def test_exactly_at_threshold(self):
        """At exactly the threshold, bonus is zero."""
        score = selection_score(judge_mean=4.0, judge_std=0.8, std_threshold=0.8)
        assert score == pytest.approx(4.0)

    def test_custom_weights(self):
        score = selection_score(
            judge_mean=3.0, judge_std=2.0,
            diversity_weight=0.5, std_threshold=1.0,
        )
        expected = 3.0 + (2.0 - 1.0) * 0.5
        assert score == pytest.approx(expected)


class TestAggregateJudgeScores:
    def test_basic_aggregation(self):
        result = aggregate_judge_scores([3.0, 3.5, 4.0])
        assert isinstance(result, AggregateResult)
        assert result.judge_mean == pytest.approx(3.5, abs=1e-4)
        assert result.holder_score == pytest.approx(3.5, abs=1e-4)
        assert result.judge_std > 0
        assert result.diversity_bonus == 0.0  # std too low for bonus

    def test_high_variance_bonus(self):
        """Polarizing judges should trigger diversity bonus."""
        result = aggregate_judge_scores([1.0, 5.0])
        assert result.judge_std > 0.8
        assert result.diversity_bonus > 0
        assert result.combined_score > result.judge_mean
        # holder_score is raw mean — no bonus
        assert result.holder_score == pytest.approx(result.judge_mean)

    def test_holder_score_vs_combined(self):
        """holder_score (cell replacement) has no bonus; combined_score does."""
        result = aggregate_judge_scores([1.0, 5.0])
        assert result.holder_score < result.combined_score
        assert result.holder_score == pytest.approx(result.judge_mean)

    def test_empty_input(self):
        result = aggregate_judge_scores([])
        assert result.combined_score == 0.0
        assert result.holder_score == 0.0
        assert result.judge_mean == 0.0

    def test_single_judge(self):
        result = aggregate_judge_scores([4.0])
        assert result.judge_mean == pytest.approx(4.0)
        assert result.holder_score == pytest.approx(4.0)
        assert result.judge_std == pytest.approx(0.0)
        assert result.diversity_bonus == 0.0

    def test_unanimous_high(self):
        result = aggregate_judge_scores([4.5, 4.5, 4.5])
        assert result.judge_mean == pytest.approx(4.5)
        assert result.holder_score == pytest.approx(4.5)
        assert result.diversity_bonus == 0.0
        assert result.combined_score == pytest.approx(4.5)
