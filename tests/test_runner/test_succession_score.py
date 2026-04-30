"""Tests for the challenger-succession score rule.

See `lab/issues/2026-04-18-champion-succession-score-bug.md`.
"""

from __future__ import annotations

from owtn.stage_1.runner import _SUCCESSION_EPS, _challenger_succession_score


class TestChallengerSuccessionScore:
    def test_low_match_score_still_succeeds_high_incumbent(self):
        """Core bug: a challenger that wins 6-1-2 (0.78) against an
        incumbent that had won 8-1-0 (0.83) against a weak seed must still
        outrank the incumbent after succession."""
        out = _challenger_succession_score(
            match_score=0.78, incumbent_score=0.83,
        )
        assert out > 0.83
        assert out == 0.83 + _SUCCESSION_EPS

    def test_high_match_score_preserved_verbatim(self):
        """If the match score exceeds the incumbent, use the raw match score."""
        out = _challenger_succession_score(
            match_score=0.89, incumbent_score=0.50,
        )
        assert out == 0.89

    def test_equal_scores_get_incumbent_plus_eps(self):
        out = _challenger_succession_score(
            match_score=0.5, incumbent_score=0.5,
        )
        assert out == 0.5 + _SUCCESSION_EPS
        assert out > 0.5

    def test_monotonic_across_succession_chain(self):
        """A chain of successive upsets each with decreasing match scores
        still produces monotonically increasing champion scores."""
        score = 0.83
        for match_score in (0.78, 0.78, 0.67, 0.56, 0.56):
            new_score = _challenger_succession_score(
                match_score=match_score, incumbent_score=score,
            )
            assert new_score > score
            score = new_score
        # After 5 successions, the score has drifted by 5 * eps.
        assert abs(score - (0.83 + 5 * _SUCCESSION_EPS)) < 1e-12
