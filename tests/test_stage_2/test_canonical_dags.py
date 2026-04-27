"""The Phase 1 hard gate: each canonical DAG passes every Tier 0 validator
and serializes hash-stably.

If any of these tests fail, fix the canonical or the validator before
proceeding to Phase 2 (rendering). Per the implementation plan, this is a
hard block — Stage 2's test infrastructure rests on the canonicals being
known-valid regression anchors.
"""

from __future__ import annotations

import hashlib

from owtn.models.stage_2.dag import DAG


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class TestCanonicalsPassValidation:
    """Each rev-7 canonical must validate without error."""

    def test_lottery(self, canonical_lottery: DAG) -> None:
        assert canonical_lottery.concept_id == "canonical_lottery"
        assert len(canonical_lottery.nodes) == 5
        assert len(canonical_lottery.edges) == 6
        assert len(canonical_lottery.story_constraints) == 2

    def test_hemingway(self, canonical_hemingway: DAG) -> None:
        assert canonical_hemingway.concept_id == "canonical_hemingway"
        assert len(canonical_hemingway.nodes) == 4
        assert len(canonical_hemingway.character_arcs) == 2
        # Whole-story not-naming is wholesale; lifts_at is None
        sc = canonical_hemingway.story_constraints[0]
        assert sc.lifts_at is None

    def test_chiang(self, canonical_chiang: DAG) -> None:
        assert canonical_chiang.concept_id == "canonical_chiang"
        assert len(canonical_chiang.nodes) == 9
        # Chiang is the one canonical with a non-empty concept_demand
        assert len(canonical_chiang.concept_demands) == 1
        assert "frame-narrative" in canonical_chiang.concept_demands[0]

    def test_oconnor(self, canonical_oconnor: DAG) -> None:
        assert canonical_oconnor.concept_id == "canonical_oconnor"
        # Multi-role node: grace is both climax and pivot
        grace = next(n for n in canonical_oconnor.nodes if n.id == "grace")
        assert grace.role == ["climax", "pivot"]
        # Multi-audience disclosure: opening → arrival reveals to reader AND grandmother
        opening_to_arrival = next(
            e for e in canonical_oconnor.edges
            if e.src == "opening" and e.dst == "arrival" and e.type == "disclosure"
        )
        assert opening_to_arrival.disclosed_to == ["reader", "the grandmother"]


class TestExactlyOneRoleBearer:
    """Per docs/stage-2/overview.md §Nodes: exactly one node carries a role."""

    def test_each_canonical_has_one_anchor(self, any_canonical: DAG) -> None:
        role_bearers = [n for n in any_canonical.nodes if n.role is not None]
        assert len(role_bearers) == 1, (
            f"{any_canonical.concept_id}: expected exactly 1 role-bearing "
            f"node, found {len(role_bearers)}: {[n.id for n in role_bearers]}"
        )


class TestSerializationStability:
    """`model_dump_json()` must produce byte-identical output across two
    parses of the same fixture. Phase 2's renderer will rely on stable
    serialization for snapshot tests."""

    def test_round_trip_is_stable(self, any_canonical: DAG) -> None:
        first = any_canonical.model_dump_json()
        round_tripped = DAG.model_validate_json(first).model_dump_json()
        assert _hash(first) == _hash(round_tripped), (
            "round-trip serialization is not stable"
        )

    def test_double_parse_yields_equal_models(self, any_canonical: DAG) -> None:
        text = any_canonical.model_dump_json()
        first = DAG.model_validate_json(text)
        second = DAG.model_validate_json(text)
        assert first == second


class TestMotifTaxonomy:
    """Verify rev-7 motif-mode coverage is preserved across the canonical set.
    Regression-protects against silent changes in the modes the validators accept.
    """

    def test_all_six_modes_appear_across_set(
        self,
        canonical_lottery: DAG,
        canonical_hemingway: DAG,
        canonical_chiang: DAG,
        canonical_oconnor: DAG,
    ) -> None:
        seen_modes: set[str] = set()
        for dag in (canonical_lottery, canonical_hemingway,
                    canonical_chiang, canonical_oconnor):
            for n in dag.nodes:
                for m in n.motifs:
                    seen_modes.add(m.mode)
        expected = {"introduced", "embodied", "performed", "agent", "echoed", "inverted"}
        assert seen_modes == expected, (
            f"motif modes used across canonicals: {sorted(seen_modes)}; "
            f"missing: {sorted(expected - seen_modes)}"
        )
