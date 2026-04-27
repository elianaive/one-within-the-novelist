"""Tests for magnitude-aware pairwise voting.

Covers the new helpers (`parse_vote`, `encode_vote`) and the magnitude
semantics of `_aggregate` (mean-of-magnitudes on the winning side ×
dim weight). Dual-ordering resolution and classification under magnitudes
live in `test_pairwise_resolution.py`. See
`lab/issues/2026-04-22-pairwise-win-margin.md` for the design.
"""

from __future__ import annotations

import pytest

from owtn.evaluation.models import (
    DIMENSION_NAMES,
    MAGNITUDE_VALUE,
    encode_vote,
    parse_vote,
)
from owtn.evaluation.pairwise import (
    _aggregate,
    _format_dim_summary,
    _resolve_votes,
)


_UNIFORM_WEIGHTS = {d: 1.0 for d in DIMENSION_NAMES}


class TestParseEncodeVote:
    """Round-trip invariants for parse_vote / encode_vote, plus legacy map."""

    @pytest.mark.parametrize("raw,expected_side,expected_mag", [
        ("a_narrow", "a", 0.5),
        ("a_clear", "a", 0.75),
        ("a_decisive", "a", 1.0),
        ("b_narrow", "b", 0.5),
        ("b_clear", "b", 0.75),
        ("b_decisive", "b", 1.0),
        ("tie", "tie", 0.0),
    ])
    def test_parse_magnitude_forms(self, raw, expected_side, expected_mag):
        side, mag = parse_vote(raw)
        assert side == expected_side
        assert mag == expected_mag

    @pytest.mark.parametrize("legacy,expected_side,expected_mag", [
        ("a", "a", 0.75),   # legacy "a" → infer as "a_clear"
        ("b", "b", 0.75),
        ("tie", "tie", 0.0),
    ])
    def test_parse_legacy_binary_votes(self, legacy, expected_side, expected_mag):
        """Match-history JSON written before magnitudes existed must still
        parse — legacy binary votes are treated as `_clear` (middle magnitude,
        a deliberate but non-decisive pick)."""
        side, mag = parse_vote(legacy)
        assert side == expected_side
        assert mag == expected_mag

    @pytest.mark.parametrize("raw", list(MAGNITUDE_VALUE.keys()))
    def test_encode_decode_round_trip_sides(self, raw):
        """encode_vote(parse_vote(v)) == v for all magnitude-encoded votes."""
        for side in ("a", "b"):
            vote = f"{side}_{raw}"
            parsed_side, parsed_mag = parse_vote(vote)
            assert encode_vote(parsed_side, parsed_mag) == vote

    def test_encode_tie(self):
        assert encode_vote("tie", 0.0) == "tie"
        # Side == 'tie' with nonzero magnitude is degenerate; also yields "tie"
        assert encode_vote("tie", 0.5) == "tie"
        # 'a' with 0.0 magnitude is degenerate; treated as tie
        assert encode_vote("a", 0.0) == "tie"


class TestResolveMagnitudeMin:
    """Specific regression cases for the min-of-magnitudes rule."""

    def test_same_side_different_magnitudes_takes_min(self):
        fwd = {d: "a_decisive" for d in DIMENSION_NAMES}
        rev = {d: "a_narrow" for d in DIMENSION_NAMES}
        resolved = _resolve_votes(fwd, rev)
        for d in DIMENSION_NAMES:
            assert resolved[d] == "a_narrow"

    def test_min_across_all_three_levels(self):
        fwd = {"novelty": "b_decisive", "grip": "b_clear", "indelibility": "b_narrow"}
        rev = {"novelty": "b_clear", "grip": "b_narrow", "indelibility": "b_decisive"}
        # Other dims default to tie in both orderings
        for d in DIMENSION_NAMES:
            fwd.setdefault(d, "tie")
            rev.setdefault(d, "tie")
        resolved = _resolve_votes(fwd, rev)
        assert resolved["novelty"] == "b_clear"       # min(decisive, clear) = clear
        assert resolved["grip"] == "b_narrow"         # min(clear, narrow) = narrow
        assert resolved["indelibility"] == "b_narrow" # min(narrow, decisive) = narrow


class TestAggregateMagnitude:
    """Mean-of-magnitudes on winning side × dim_weight = contribution."""

    def test_all_judges_decisive_gives_full_weight(self, default_pairwise_cfg):
        """Three judges all at decisive → mean mag = 1.0 → contribution =
        dim_weight × 1.0 (today's pre-magnitude behavior as an upper bound)."""
        resolved = [
            {d: "a_decisive" for d in DIMENSION_NAMES} for _ in range(3)
        ]
        _, a, b, _, a_w, b_w, _ = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert a == 9 and b == 0
        # Sum of all dim weights in default config = 10.75
        assert a_w == pytest.approx(10.75, abs=1e-6)

    def test_all_judges_narrow_halves_contribution(self, default_pairwise_cfg):
        """Three judges all at narrow → mean = 0.5 → contribution halved."""
        resolved = [
            {d: "a_narrow" for d in DIMENSION_NAMES} for _ in range(3)
        ]
        _, _, _, _, a_w, _, _ = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert a_w == pytest.approx(10.75 * 0.5, abs=1e-6)

    def test_mixed_magnitudes_mean_on_winning_side(self, default_pairwise_cfg):
        """On indelibility (weight 2.0), three judges vote a_decisive /
        a_clear / a_narrow. Mean magnitude = (1.0 + 0.75 + 0.5) / 3 = 0.75.
        Contribution = 2.0 × 0.75 = 1.5."""
        ind = "indelibility"
        resolved = [
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_decisive"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_clear"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_narrow"},
        ]
        _, _, _, _, a_w, _, _ = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert a_w == pytest.approx(1.5, abs=1e-6)

    def test_minority_magnitude_ignored(self, default_pairwise_cfg):
        """Two judges vote a_decisive, one judge votes b_narrow on indelibility.
        Majority = a; mean magnitude on A side = (1.0 + 1.0)/2 = 1.0.
        B's magnitude is ignored — it lost the side vote."""
        ind = "indelibility"
        resolved = [
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_decisive"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_decisive"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_narrow"},
        ]
        _, a, b, _, a_w, b_w, _ = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        # A majority on indelibility → 2.0 × 1.0 = 2.0
        assert a == 1 and b == 0
        assert a_w == pytest.approx(2.0, abs=1e-6)
        assert b_w == 0.0


class TestAggregateMagnitudeTiebreaker:
    """With 4-judge panels, 2-2 dim-level splits are a structural case that
    3-judge panels never produced. If one side has stronger magnitudes
    (gap ≥ 0.25), the dim goes to the higher-magnitude side instead of
    dropping both sides' signal into tie_weighted. See
    `lab/issues/2026-04-24-aggregate-magnitude-tiebreaker.md`.
    """

    def test_2_2_decisive_vs_narrow_flips_to_decisive(self, default_pairwise_cfg):
        """2 a_decisive + 2 b_narrow → gap 0.5 → dim goes to A."""
        ind = "indelibility"
        resolved = [
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_decisive"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_decisive"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_narrow"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_narrow"},
        ]
        dim_winners, a, b, _, a_w, b_w, tie_w = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert dim_winners[ind] == "a"
        assert a == 1 and b == 0
        # mean a = 1.0, weight = 2.0 → a_w contribution 2.0 on indelibility.
        # Other 8 dims are all-tie (b=0, a=0) → go through the all-tie branch.
        assert a_w == pytest.approx(2.0, abs=1e-6)
        assert b_w == 0.0
        # Indelibility's weight does NOT go to tie bucket — it went to A.
        tie_weight_total = sum(
            w for d, w in default_pairwise_cfg.dim_weights.items() if d != ind
        )
        assert tie_w == pytest.approx(tie_weight_total, abs=1e-6)

    def test_2_2_clear_vs_narrow_at_threshold_flips(self, default_pairwise_cfg):
        """2 a_clear (0.75) + 2 b_narrow (0.5) → gap 0.25 (at threshold).
        Threshold is ≥ 0.25 so this flips to A."""
        ind = "indelibility"
        resolved = [
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_clear"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_clear"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_narrow"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_narrow"},
        ]
        dim_winners, a, b, _, a_w, _, _ = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert dim_winners[ind] == "a"
        assert a_w == pytest.approx(2.0 * 0.75, abs=1e-6)

    def test_2_2_symmetric_magnitudes_stays_tie(self, default_pairwise_cfg):
        """2 a_clear + 2 b_clear → gap 0 → stays tie (magnitudes dropped)."""
        ind = "indelibility"
        resolved = [
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_clear"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_clear"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_clear"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_clear"},
        ]
        dim_winners, a, b, _, a_w, b_w, _ = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert dim_winners[ind] == "tie"
        assert a == 0 and b == 0
        assert a_w == 0.0 and b_w == 0.0

    def test_2_2_same_bucket_both_narrow_stays_tie(self, default_pairwise_cfg):
        """2 a_narrow + 2 b_narrow → gap 0 → stays tie."""
        ind = "indelibility"
        resolved = [
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_narrow"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_narrow"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_narrow"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_narrow"},
        ]
        dim_winners, _, _, _, a_w, b_w, _ = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert dim_winners[ind] == "tie"
        assert a_w == 0.0 and b_w == 0.0

    def test_2_2_mixed_magnitudes_mean_gap_above_threshold(self, default_pairwise_cfg):
        """1 a_decisive + 1 a_clear + 2 b_narrow → means (0.875, 0.5) gap 0.375
        → flips to A at mean 0.875."""
        ind = "indelibility"
        resolved = [
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_decisive"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_clear"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_narrow"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_narrow"},
        ]
        dim_winners, _, _, _, a_w, _, _ = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert dim_winners[ind] == "a"
        assert a_w == pytest.approx(2.0 * 0.875, abs=1e-6)

    def test_2_2_mixed_magnitudes_mean_gap_below_threshold_stays_tie(
        self, default_pairwise_cfg,
    ):
        """1 a_decisive + 1 a_narrow + 2 b_clear → means (0.75, 0.75) gap 0
        → stays tie."""
        ind = "indelibility"
        resolved = [
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_decisive"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_narrow"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_clear"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_clear"},
        ]
        dim_winners, _, _, _, a_w, b_w, _ = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert dim_winners[ind] == "tie"
        assert a_w == 0.0 and b_w == 0.0

    def test_all_tie_unchanged(self, default_pairwise_cfg):
        """0-0-4 all-tie case falls through the magnitude tiebreaker — no
        magnitude signal exists, weight goes to tie_weighted as before."""
        resolved = [
            {d: "tie" for d in DIMENSION_NAMES} for _ in range(4)
        ]
        dim_winners, a, b, ties, a_w, b_w, tie_w = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        for d in DIMENSION_NAMES:
            assert dim_winners[d] == "tie"
        assert a == 0 and b == 0
        assert ties == len(DIMENSION_NAMES)
        assert a_w == 0.0 and b_w == 0.0
        assert tie_w == pytest.approx(
            sum(default_pairwise_cfg.dim_weights.values()), abs=1e-6,
        )

    def test_3_1_split_unchanged_by_magnitude_path(self, default_pairwise_cfg):
        """3-1 majority case is unchanged by the new tiebreaker code path."""
        ind = "indelibility"
        resolved = [
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_decisive"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_clear"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_clear"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_decisive"},
        ]
        dim_winners, a, b, _, a_w, _, _ = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert dim_winners[ind] == "a"
        assert a == 1 and b == 0
        # Mean magnitude on A = (1.0 + 0.75 + 0.75) / 3
        expected = 2.0 * (1.0 + 0.75 + 0.75) / 3
        assert a_w == pytest.approx(expected, abs=1e-6)

    def test_1_1_split_with_two_ties_applies_tiebreaker(
        self, default_pairwise_cfg,
    ):
        """1-1-2 split with magnitude differential ≥ 0.25 on the two voting
        judges still triggers the magnitude tiebreaker (generalizes beyond
        2-2)."""
        ind = "indelibility"
        resolved = [
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "a_decisive"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "b_narrow"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "tie"},
            {d: "tie" for d in DIMENSION_NAMES} | {ind: "tie"},
        ]
        dim_winners, _, _, _, a_w, _, _ = _aggregate(
            resolved, default_pairwise_cfg.dim_weights,
        )
        assert dim_winners[ind] == "a"
        assert a_w == pytest.approx(2.0 * 1.0, abs=1e-6)


class TestLegacyReplay:
    """Replay safety: old match_history JSON with binary votes must aggregate
    without error, treating each "a"/"b" as `_clear` (0.75 magnitude)."""

    def test_legacy_votes_aggregate_as_clear(self):
        # Three judges, all binary-only (pre-magnitude era)
        resolved = [
            {d: "a" for d in DIMENSION_NAMES} for _ in range(3)
        ]
        _, a, b, _, a_w, b_w, _ = _aggregate(resolved, _UNIFORM_WEIGHTS)
        assert a == 9 and b == 0
        # Each A-voter inferred at _clear (0.75) → 9 × 1.0 × 0.75 = 6.75
        assert a_w == pytest.approx(6.75, abs=1e-6)
        assert b_w == 0.0


class TestFormatDimSummary:
    """Log-line formatting: dim=<A|B><1|2|3> or dim==."""

    def test_decisive_sweep_renders_A3(self):
        dim_winners = {d: "a" for d in DIMENSION_NAMES}
        resolved = [{d: "a_decisive" for d in DIMENSION_NAMES}]
        summary = _format_dim_summary(dim_winners, resolved)
        for d in DIMENSION_NAMES:
            assert f"{d}=A3" in summary

    def test_narrow_b_wins_render_B1(self):
        dim_winners = {d: "b" for d in DIMENSION_NAMES}
        resolved = [{d: "b_narrow" for d in DIMENSION_NAMES}]
        summary = _format_dim_summary(dim_winners, resolved)
        for d in DIMENSION_NAMES:
            assert f"{d}=B1" in summary

    def test_ties_render_double_equals(self):
        dim_winners = {d: "tie" for d in DIMENSION_NAMES}
        resolved = [{d: "tie" for d in DIMENSION_NAMES}]
        summary = _format_dim_summary(dim_winners, resolved)
        for d in DIMENSION_NAMES:
            assert f"{d}==" in summary
