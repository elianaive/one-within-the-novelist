"""Tests for pairwise vote resolution and classification under the tie vote.

Covers the full (fwd, rev) case table from
`lab/issues/2026-04-18-reintroduce-harshness-pairwise.md`.
"""

from __future__ import annotations

import pytest

from owtn.evaluation.models import DIMENSION_NAMES, PairwiseJudgment
from owtn.evaluation.pairwise import (
    _aggregate,
    _classify_resolution,
    _classify_votes,
    _flip_votes,
    _resolve_votes,
)


def _judgment(**votes) -> PairwiseJudgment:
    """Build a PairwiseJudgment with per-dim votes; unspecified dims default to 'a'."""
    full = {dim: votes.get(dim, "a") for dim in DIMENSION_NAMES}
    return PairwiseJudgment(reasoning="stub", **full)


class TestFlipVotes:
    def test_flip_a_and_b(self):
        j = _judgment(novelty="a", grip="b")
        flipped = _flip_votes(j)
        assert flipped["novelty"] == "b"
        assert flipped["grip"] == "a"

    def test_tie_is_identity(self):
        j = _judgment(novelty="tie")
        flipped = _flip_votes(j)
        assert flipped["novelty"] == "tie"


class TestResolveVotes:
    """The 9 combinations from the case table."""

    @pytest.mark.parametrize("fwd,rev,expected", [
        # Same-side agreement → that side
        ("a", "a", "a"),
        ("b", "b", "b"),
        # Same tie → tie
        ("tie", "tie", "tie"),
        # Position disagreement → tie (resolution-tie)
        ("a", "b", "tie"),
        ("b", "a", "tie"),
        # One-side uncertainty → tie (soft-tie; conservative)
        ("a", "tie", "tie"),
        ("tie", "a", "tie"),
        ("b", "tie", "tie"),
        ("tie", "b", "tie"),
    ])
    def test_case_table(self, fwd, rev, expected):
        fwd_votes = {d: fwd for d in DIMENSION_NAMES}
        rev_votes = {d: rev for d in DIMENSION_NAMES}
        resolved = _resolve_votes(fwd_votes, rev_votes)
        for d in DIMENSION_NAMES:
            assert resolved[d] == expected, f"({fwd}, {rev}) → {resolved[d]}, expected {expected}"


class TestClassifyResolution:
    @pytest.mark.parametrize("fwd,rev,expected", [
        ("a", "a", "confident-a"),
        ("b", "b", "confident-b"),
        ("tie", "tie", "confident-tie"),
        ("a", "b", "resolution-tie"),
        ("b", "a", "resolution-tie"),
        ("a", "tie", "soft-tie"),
        ("tie", "a", "soft-tie"),
        ("b", "tie", "soft-tie"),
        ("tie", "b", "soft-tie"),
    ])
    def test_classifications(self, fwd, rev, expected):
        assert _classify_resolution(fwd, rev) == expected

    def test_classify_votes_covers_all_dims(self):
        fwd = {d: "a" for d in DIMENSION_NAMES}
        rev = {d: "a" for d in DIMENSION_NAMES}
        fwd["novelty"] = "tie"
        rev["novelty"] = "tie"
        fwd["grip"] = "a"
        rev["grip"] = "b"
        classes = _classify_votes(fwd, rev)
        assert classes["novelty"] == "confident-tie"
        assert classes["grip"] == "resolution-tie"
        assert classes["indelibility"] == "confident-a"


class TestAggregate:
    """Tie = abstention, not veto. Majority of non-tie votes per dim wins."""

    def _mk(self, *votes_per_judge) -> list[dict[str, str]]:
        """votes_per_judge: list of per-judge vote lists, one value per dim."""
        return [
            {dim: votes[i] for i, dim in enumerate(DIMENSION_NAMES)}
            for votes in votes_per_judge
        ]

    def test_two_a_one_tie_gives_a(self):
        resolved = [
            {d: "a" for d in DIMENSION_NAMES},
            {d: "a" for d in DIMENSION_NAMES},
            {d: "tie" for d in DIMENSION_NAMES},
        ]
        dim_winners, a, b, ties = _aggregate(resolved)
        assert a == 9 and b == 0 and ties == 0
        for d in DIMENSION_NAMES:
            assert dim_winners[d] == "a"

    def test_one_a_one_b_one_tie_gives_tie(self):
        resolved = [
            {d: "a" for d in DIMENSION_NAMES},
            {d: "b" for d in DIMENSION_NAMES},
            {d: "tie" for d in DIMENSION_NAMES},
        ]
        dim_winners, a, b, ties = _aggregate(resolved)
        assert a == 0 and b == 0 and ties == 9

    def test_one_a_two_ties_gives_a(self):
        """Single non-tie vote still wins — abstention does not veto."""
        resolved = [
            {d: "a" for d in DIMENSION_NAMES},
            {d: "tie" for d in DIMENSION_NAMES},
            {d: "tie" for d in DIMENSION_NAMES},
        ]
        dim_winners, a, b, ties = _aggregate(resolved)
        assert a == 9 and b == 0 and ties == 0

    def test_all_ties_gives_all_tie(self):
        resolved = [{d: "tie" for d in DIMENSION_NAMES} for _ in range(3)]
        dim_winners, a, b, ties = _aggregate(resolved)
        assert a == 0 and b == 0 and ties == 9
