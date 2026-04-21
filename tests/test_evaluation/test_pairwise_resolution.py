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
    _select_winner,
)

# Uniform weights for legacy aggregation tests — matches pre-reweighting behavior.
_UNIFORM_WEIGHTS = {d: 1.0 for d in DIMENSION_NAMES}


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
    """Tie = abstention, not veto. Majority of non-tie votes per dim wins.

    These tests use uniform weights (1.0 everywhere) so integer counts match
    the weighted totals — the underlying resolution logic is unchanged by
    weighting; it's the selection step (`_select_winner`) that weights matter for.
    """

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
        dim_winners, a, b, ties, a_w, b_w, tie_w = _aggregate(resolved, _UNIFORM_WEIGHTS)
        assert a == 9 and b == 0 and ties == 0
        assert a_w == 9.0 and b_w == 0.0 and tie_w == 0.0
        for d in DIMENSION_NAMES:
            assert dim_winners[d] == "a"

    def test_one_a_one_b_one_tie_gives_tie(self):
        resolved = [
            {d: "a" for d in DIMENSION_NAMES},
            {d: "b" for d in DIMENSION_NAMES},
            {d: "tie" for d in DIMENSION_NAMES},
        ]
        dim_winners, a, b, ties, a_w, b_w, tie_w = _aggregate(resolved, _UNIFORM_WEIGHTS)
        assert a == 0 and b == 0 and ties == 9
        assert a_w == 0.0 and b_w == 0.0 and tie_w == 9.0

    def test_one_a_two_ties_gives_a(self):
        """Single non-tie vote still wins — abstention does not veto."""
        resolved = [
            {d: "a" for d in DIMENSION_NAMES},
            {d: "tie" for d in DIMENSION_NAMES},
            {d: "tie" for d in DIMENSION_NAMES},
        ]
        dim_winners, a, b, ties, a_w, b_w, tie_w = _aggregate(resolved, _UNIFORM_WEIGHTS)
        assert a == 9 and b == 0 and ties == 0
        assert a_w == 9.0 and b_w == 0.0

    def test_all_ties_gives_all_tie(self):
        resolved = [{d: "tie" for d in DIMENSION_NAMES} for _ in range(3)]
        dim_winners, a, b, ties, a_w, b_w, tie_w = _aggregate(resolved, _UNIFORM_WEIGHTS)
        assert a == 0 and b == 0 and ties == 9
        assert tie_w == 9.0

    def test_weights_propagate_to_weighted_totals(self, default_pairwise_cfg):
        """Under production weights, a win on Indelibility contributes 2.0,
        not 1.0 — this is the actual behavioral change under test."""
        resolved = [
            {d: ("a" if d == "indelibility" else "b") for d in DIMENSION_NAMES}
        ]
        dim_winners, a, b, ties, a_w, b_w, tie_w = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert a == 1 and b == 8
        # A won only Indelibility → a_w = 2.00. B won the other 8 dims.
        assert a_w == 2.00
        # B's weight = sum of all except indelibility = 10.75 - 2.00 = 8.75
        assert b_w == pytest.approx(8.75, abs=1e-6)


class TestSelectWinner:
    """Option E: weighted aggregate + asymmetric tiebreaker. All 7 adversarial
    scenarios from lab/issues/2026-04-21-rubric-reweighting.md."""

    def _dim_winners(self, a_dims: list[str], b_dims: list[str]) -> dict[str, str]:
        """Build a dim_winners dict from lists of dims each side won. Remaining
        dims are ties."""
        out = {d: "tie" for d in DIMENSION_NAMES}
        for d in a_dims:
            out[d] = "a"
        for d in b_dims:
            out[d] = "b"
        return out

    def test_scenario_1_spiky_vs_mid(self, default_pairwise_cfg):
        """Spiky-great (wins I+G+N, 5.50) vs. mid-everywhere (wins rest, 5.25).
        Gap = 0.25, within threshold → Indelibility tiebreaker → A wins."""
        dim_winners = self._dim_winners(
            a_dims=["indelibility", "grip", "novelty"],
            b_dims=[
                "generative_fertility", "tension_architecture", "emotional_depth",
                "thematic_resonance", "concept_coherence", "scope_calibration",
            ],
        )
        winner, tb = _select_winner(
            a_weighted=5.50, b_weighted=5.25,
            dim_winners=dim_winners, champion_label="b",
            tiebreaker_threshold=default_pairwise_cfg.tiebreaker_threshold,
            tiebreaker_dims=default_pairwise_cfg.tiebreaker_dims,
        )
        assert winner == "a"
        assert tb == "indelibility"

    def test_scenario_2_spiky_vs_solidly_good(self, default_pairwise_cfg):
        """A wins I+N (3.75). B wins all other 7 (7.00). Gap = 3.25 → weighted wins."""
        dim_winners = self._dim_winners(
            a_dims=["indelibility", "novelty"],
            b_dims=[
                "grip", "generative_fertility", "tension_architecture",
                "emotional_depth", "thematic_resonance",
                "concept_coherence", "scope_calibration",
            ],
        )
        winner, tb = _select_winner(
            a_weighted=3.75, b_weighted=7.00,
            dim_winners=dim_winners, champion_label="b",
            tiebreaker_threshold=default_pairwise_cfg.tiebreaker_threshold,
            tiebreaker_dims=default_pairwise_cfg.tiebreaker_dims,
        )
        assert winner == "b"
        assert tb is None

    def test_scenario_3_spiky_with_tension(self, default_pairwise_cfg):
        """A wins I+G+N+TA+ED (7.50). B wins TR+GF+CC+SC (3.25). Gap = 4.25 → weighted."""
        dim_winners = self._dim_winners(
            a_dims=["indelibility", "grip", "novelty", "tension_architecture", "emotional_depth"],
            b_dims=["thematic_resonance", "generative_fertility", "concept_coherence", "scope_calibration"],
        )
        winner, tb = _select_winner(
            a_weighted=7.50, b_weighted=3.25,
            dim_winners=dim_winners, champion_label="b",
            tiebreaker_threshold=default_pairwise_cfg.tiebreaker_threshold,
            tiebreaker_dims=default_pairwise_cfg.tiebreaker_dims,
        )
        assert winner == "a"
        assert tb is None

    def test_scenario_4_coherent_conventional_vs_rough_original(self, default_pairwise_cfg):
        """A wins CC+SC+TA+TR (3.00). B wins I+G+N+GF+ED (7.75). Gap = 4.75 → weighted."""
        dim_winners = self._dim_winners(
            a_dims=["concept_coherence", "scope_calibration", "tension_architecture", "thematic_resonance"],
            b_dims=["indelibility", "grip", "novelty", "generative_fertility", "emotional_depth"],
        )
        winner, tb = _select_winner(
            a_weighted=3.00, b_weighted=7.75,
            dim_winners=dim_winners, champion_label="a",
            tiebreaker_threshold=default_pairwise_cfg.tiebreaker_threshold,
            tiebreaker_dims=default_pairwise_cfg.tiebreaker_dims,
        )
        assert winner == "b"
        assert tb is None

    def test_scenario_5_gap_above_threshold_no_tiebreaker(self, default_pairwise_cfg):
        """Gap 1.75 — above 1.0 threshold, no tiebreaker. B wins on weight."""
        dim_winners = self._dim_winners(
            a_dims=["indelibility", "scope_calibration", "concept_coherence"],
            b_dims=["grip", "tension_architecture", "emotional_depth", "thematic_resonance"],
        )
        winner, tb = _select_winner(
            a_weighted=3.00, b_weighted=4.75,
            dim_winners=dim_winners, champion_label="b",
            tiebreaker_threshold=default_pairwise_cfg.tiebreaker_threshold,
            tiebreaker_dims=default_pairwise_cfg.tiebreaker_dims,
        )
        assert winner == "b"
        assert tb is None

    def test_scenario_6_sub_threshold_indelibility_breaks(self, default_pairwise_cfg):
        """Gap 0.75, below 1.0 → tiebreaker → A wins Indelibility."""
        dim_winners = self._dim_winners(
            a_dims=["indelibility", "grip"],
            b_dims=["novelty", "generative_fertility"],
        )
        winner, tb = _select_winner(
            a_weighted=3.75, b_weighted=3.00,
            dim_winners=dim_winners, champion_label="b",
            tiebreaker_threshold=default_pairwise_cfg.tiebreaker_threshold,
            tiebreaker_dims=default_pairwise_cfg.tiebreaker_dims,
        )
        assert winner == "a"
        assert tb == "indelibility"

    def test_scenario_7_full_stalemate_incumbent(self, default_pairwise_cfg):
        """All 9 dims tied → gap 0 → tiebreaker → both top dims tied → incumbent."""
        dim_winners = {d: "tie" for d in DIMENSION_NAMES}
        winner, tb = _select_winner(
            a_weighted=0.0, b_weighted=0.0,
            dim_winners=dim_winners, champion_label="b",
            tiebreaker_threshold=default_pairwise_cfg.tiebreaker_threshold,
            tiebreaker_dims=default_pairwise_cfg.tiebreaker_dims,
        )
        assert winner == "b"
        assert tb == "incumbent"

    def test_threshold_boundary_inclusive(self, default_pairwise_cfg):
        """Gap exactly at threshold (1.0) should fire tiebreaker (<=)."""
        dim_winners = self._dim_winners(
            a_dims=["indelibility"], b_dims=["grip"],
        )
        winner, tb = _select_winner(
            a_weighted=2.0, b_weighted=1.0,  # gap exactly 1.0
            dim_winners=dim_winners, champion_label="b",
            tiebreaker_threshold=1.0,
            tiebreaker_dims=["indelibility", "grip"],
        )
        # Tiebreaker fires → A wins Indelibility → A
        assert winner == "a"
        assert tb == "indelibility"

    def test_threshold_above_no_tiebreaker(self, default_pairwise_cfg):
        """Gap just above threshold → no tiebreaker."""
        dim_winners = self._dim_winners(
            a_dims=["indelibility"], b_dims=["grip"],
        )
        winner, tb = _select_winner(
            a_weighted=2.01, b_weighted=1.0,  # gap 1.01 > 1.0
            dim_winners=dim_winners, champion_label="b",
            tiebreaker_threshold=1.0,
            tiebreaker_dims=["indelibility", "grip"],
        )
        assert winner == "a"
        assert tb is None

    def test_tiebreaker_falls_through_to_grip(self, default_pairwise_cfg):
        """Indelibility tied → falls through to Grip."""
        dim_winners = self._dim_winners(
            a_dims=["novelty"],  # neither Indelibility nor Grip
            b_dims=["grip"],     # B wins Grip
        )
        # Indelibility is "tie" by default
        winner, tb = _select_winner(
            a_weighted=1.75, b_weighted=1.75,  # gap 0
            dim_winners=dim_winners, champion_label="a",
            tiebreaker_threshold=1.0,
            tiebreaker_dims=["indelibility", "grip"],
        )
        assert winner == "b"
        assert tb == "grip"


class TestRegressionUniformWeights:
    """When weights are all 1.0 and threshold is 0.0, selection must reproduce
    the pre-reweighting behavior (integer majority, ties-favor-champion)."""

    def test_uniform_matches_integer_majority(self, uniform_pairwise_cfg):
        """A wins 5 dims, B wins 4 → A wins under uniform weights."""
        a_dims = ["indelibility", "grip", "novelty", "generative_fertility", "tension_architecture"]
        b_dims = ["emotional_depth", "thematic_resonance", "concept_coherence", "scope_calibration"]
        dim_winners = {d: "tie" for d in DIMENSION_NAMES}
        for d in a_dims:
            dim_winners[d] = "a"
        for d in b_dims:
            dim_winners[d] = "b"

        winner, tb = _select_winner(
            a_weighted=5.0, b_weighted=4.0,
            dim_winners=dim_winners, champion_label="b",
            tiebreaker_threshold=uniform_pairwise_cfg.tiebreaker_threshold,
            tiebreaker_dims=uniform_pairwise_cfg.tiebreaker_dims,
        )
        # Gap = 1.0, threshold = 0.0 → gap > threshold → no tiebreaker → A.
        assert winner == "a"
        assert tb is None

    def test_uniform_exact_tie_favors_incumbent(self, uniform_pairwise_cfg):
        """Exact tie on integer counts → threshold=0.0 gap=0.0, gap <= threshold
        triggers tiebreaker. With empty tiebreaker_dims, incumbent wins."""
        dim_winners = {d: "tie" for d in DIMENSION_NAMES}
        winner, tb = _select_winner(
            a_weighted=4.5, b_weighted=4.5,
            dim_winners=dim_winners, champion_label="b",
            tiebreaker_threshold=uniform_pairwise_cfg.tiebreaker_threshold,
            tiebreaker_dims=uniform_pairwise_cfg.tiebreaker_dims,
        )
        assert winner == "b"
        assert tb == "incumbent"
