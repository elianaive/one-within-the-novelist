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
    """Build a PairwiseJudgment with per-dim votes; unspecified dims default to 'a_clear'."""
    full = {dim: votes.get(dim, "a_clear") for dim in DIMENSION_NAMES}
    return PairwiseJudgment(reasoning="stub", **full)


class TestFlipVotes:
    def test_flip_a_and_b_preserves_magnitude(self):
        j = _judgment(novelty="a_decisive", grip="b_narrow")
        flipped = _flip_votes(j)
        assert flipped["novelty"] == "b_decisive"
        assert flipped["grip"] == "a_narrow"

    def test_tie_is_identity(self):
        j = _judgment(novelty="tie")
        flipped = _flip_votes(j)
        assert flipped["novelty"] == "tie"


class TestResolveVotes:
    """Dual-ordering resolution — side must agree, magnitude takes min."""

    @pytest.mark.parametrize("fwd,rev,expected", [
        # Same side + same magnitude → identity
        ("a_clear", "a_clear", "a_clear"),
        ("b_decisive", "b_decisive", "b_decisive"),
        ("a_narrow", "a_narrow", "a_narrow"),
        # Same side + different magnitude → min (conservative)
        ("a_decisive", "a_narrow", "a_narrow"),
        ("a_clear", "a_decisive", "a_clear"),
        ("b_decisive", "b_clear", "b_clear"),
        # Same tie → tie
        ("tie", "tie", "tie"),
        # Side disagreement → tie (resolution-tie)
        ("a_clear", "b_clear", "tie"),
        ("b_decisive", "a_narrow", "tie"),
        # One-side uncertainty → tie (soft-tie; conservative)
        ("a_clear", "tie", "tie"),
        ("tie", "a_decisive", "tie"),
        ("b_narrow", "tie", "tie"),
        ("tie", "b_clear", "tie"),
    ])
    def test_case_table(self, fwd, rev, expected):
        fwd_votes = {d: fwd for d in DIMENSION_NAMES}
        rev_votes = {d: rev for d in DIMENSION_NAMES}
        resolved = _resolve_votes(fwd_votes, rev_votes)
        for d in DIMENSION_NAMES:
            assert resolved[d] == expected, f"({fwd}, {rev}) → {resolved[d]}, expected {expected}"


class TestClassifyResolution:
    @pytest.mark.parametrize("fwd,rev,expected", [
        # Same side + same magnitude → confident-<side>
        ("a_clear", "a_clear", "confident-a"),
        ("b_decisive", "b_decisive", "confident-b"),
        ("a_narrow", "a_narrow", "confident-a"),
        # Same side + magnitude mismatch → agreed-<side>-min-<label>
        ("a_decisive", "a_narrow", "agreed-a-min-narrow"),
        ("a_clear", "a_decisive", "agreed-a-min-clear"),
        ("b_decisive", "b_clear", "agreed-b-min-clear"),
        # Same tie
        ("tie", "tie", "confident-tie"),
        # Side disagreement
        ("a_clear", "b_clear", "resolution-tie"),
        ("b_decisive", "a_narrow", "resolution-tie"),
        # Soft-tie
        ("a_clear", "tie", "soft-tie"),
        ("tie", "a_decisive", "soft-tie"),
        ("b_narrow", "tie", "soft-tie"),
        ("tie", "b_clear", "soft-tie"),
    ])
    def test_classifications(self, fwd, rev, expected):
        assert _classify_resolution(fwd, rev) == expected

    def test_classify_votes_covers_all_dims(self):
        fwd = {d: "a_clear" for d in DIMENSION_NAMES}
        rev = {d: "a_clear" for d in DIMENSION_NAMES}
        fwd["novelty"] = "tie"
        rev["novelty"] = "tie"
        fwd["grip"] = "a_clear"
        rev["grip"] = "b_clear"
        classes = _classify_votes(fwd, rev)
        assert classes["novelty"] == "confident-tie"
        assert classes["grip"] == "resolution-tie"
        assert classes["indelibility"] == "confident-a"


class TestAggregate:
    """Tie = abstention, not veto. Majority of non-tie votes picks the winning
    side; weighted contribution = dim_weight × mean magnitude on winning side.

    These tests use uniform weights (1.0 everywhere). All votes are `_clear`
    (magnitude 0.75) unless otherwise noted, so each dim contributes 0.75 to
    the winning side's weighted total.
    """

    def _mk(self, *votes_per_judge) -> list[dict[str, str]]:
        """votes_per_judge: list of per-judge vote lists, one value per dim."""
        return [
            {dim: votes[i] for i, dim in enumerate(DIMENSION_NAMES)}
            for votes in votes_per_judge
        ]

    def test_two_a_one_tie_gives_a(self):
        resolved = [
            {d: "a_clear" for d in DIMENSION_NAMES},
            {d: "a_clear" for d in DIMENSION_NAMES},
            {d: "tie" for d in DIMENSION_NAMES},
        ]
        dim_winners, a, b, ties, a_w, b_w, tie_w = _aggregate(resolved, _UNIFORM_WEIGHTS)
        assert a == 9 and b == 0 and ties == 0
        # Mean magnitude among A-voters (both voted _clear) = 0.75. Per dim
        # contribution = 1.0 × 0.75 = 0.75. 9 dims → 6.75.
        assert a_w == pytest.approx(6.75) and b_w == 0.0 and tie_w == 0.0
        for d in DIMENSION_NAMES:
            assert dim_winners[d] == "a"

    def test_one_a_one_b_one_tie_gives_tie(self):
        resolved = [
            {d: "a_clear" for d in DIMENSION_NAMES},
            {d: "b_clear" for d in DIMENSION_NAMES},
            {d: "tie" for d in DIMENSION_NAMES},
        ]
        dim_winners, a, b, ties, a_w, b_w, tie_w = _aggregate(resolved, _UNIFORM_WEIGHTS)
        assert a == 0 and b == 0 and ties == 9
        assert a_w == 0.0 and b_w == 0.0 and tie_w == 9.0

    def test_one_a_two_ties_gives_a(self):
        """Single non-tie vote still wins — abstention does not veto."""
        resolved = [
            {d: "a_clear" for d in DIMENSION_NAMES},
            {d: "tie" for d in DIMENSION_NAMES},
            {d: "tie" for d in DIMENSION_NAMES},
        ]
        dim_winners, a, b, ties, a_w, b_w, tie_w = _aggregate(resolved, _UNIFORM_WEIGHTS)
        assert a == 9 and b == 0 and ties == 0
        # Single A-voter at _clear → mean magnitude 0.75 → 9 × 0.75 = 6.75.
        assert a_w == pytest.approx(6.75) and b_w == 0.0

    def test_all_ties_gives_all_tie(self):
        resolved = [{d: "tie" for d in DIMENSION_NAMES} for _ in range(3)]
        dim_winners, a, b, ties, a_w, b_w, tie_w = _aggregate(resolved, _UNIFORM_WEIGHTS)
        assert a == 0 and b == 0 and ties == 9
        assert tie_w == 9.0

    def test_weights_propagate_to_weighted_totals(self, default_pairwise_cfg):
        """Under production weights with all votes at `_clear`, each dim
        contributes weight × 0.75 to the winning side."""
        resolved = [
            {d: ("a_clear" if d == "indelibility" else "b_clear") for d in DIMENSION_NAMES}
        ]
        dim_winners, a, b, ties, a_w, b_w, tie_w = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert a == 1 and b == 8
        # A won only Indelibility → a_w = 2.00 × 0.75 = 1.5
        assert a_w == pytest.approx(1.5, abs=1e-6)
        # B's weight = sum of all except indelibility = 10.75 - 2.00 = 8.75
        # At _clear magnitude: b_w = 8.75 × 0.75 = 6.5625
        assert b_w == pytest.approx(6.5625, abs=1e-6)


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
