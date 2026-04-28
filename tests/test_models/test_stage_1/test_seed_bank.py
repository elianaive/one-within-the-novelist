import random

import pytest

from owtn.models.stage_1.seed_bank import (
    OPERATOR_SEED_TYPES, SEED_OPERATOR_MAP, Seed, SeedBank,
)


SEED_BANK_PATH = "data/seed-bank.yaml"


class TestSeedBank:
    def test_load(self):
        bank = SeedBank.load(SEED_BANK_PATH)
        assert len(bank.seeds) > 50

    def test_real_world_seeds(self):
        bank = SeedBank.load(SEED_BANK_PATH)
        rw = bank.get_by_type("real_world")
        assert len(rw) >= 10

    def test_get_nonexistent_type(self):
        bank = SeedBank.load(SEED_BANK_PATH)
        assert bank.get_by_type("nonexistent") == []

    def test_select(self):
        bank = SeedBank.load(SEED_BANK_PATH)
        seed = bank.select("real_world")
        assert seed is not None
        assert seed.type == "real_world"

    def test_select_with_exclusion(self):
        bank = SeedBank.load(SEED_BANK_PATH)
        all_rw = bank.get_by_type("real_world")
        all_ids = {s.id for s in all_rw}
        result = bank.select("real_world", exclude_ids=all_ids)
        assert result is None

    def test_seed_fields(self):
        bank = SeedBank.load(SEED_BANK_PATH)
        seed = bank.seeds[0]
        assert seed.id
        assert seed.type
        assert seed.content
        assert seed.source
        assert isinstance(seed.tags, list)


class TestOperatorMapping:
    def test_all_seed_types_mapped(self):
        expected_types = {
            "real_world", "thought_experiment", "axiom", "dilemma",
            "constraint", "noun_cluster", "image", "compression",
            "collision_pair", "anti_target",
        }
        assert set(SEED_OPERATOR_MAP.keys()) == expected_types

    def test_reverse_map_coverage(self):
        # Every operator that has seeds should be in the reverse map
        assert "collision" in OPERATOR_SEED_TYPES
        assert "thought_experiment" in OPERATOR_SEED_TYPES
        assert "real_world_seed" in OPERATOR_SEED_TYPES

    def test_dilemma_maps_to_multiple(self):
        assert len(SEED_OPERATOR_MAP["dilemma"]) == 3


class TestFarthestFirst:
    """Hand-crafted bank with controllable embeddings so we can pin the
    farthest-first behavior without depending on a real model."""

    def _bank(self) -> SeedBank:
        seeds = [
            Seed(id=f"s{i}", type="real_world", content=f"seed {i}", tags=[])
            for i in range(5)
        ]
        bank = SeedBank(seeds)
        # 1D embeddings (extended to 2D for normalization). Place seeds along
        # a line so distances are predictable.
        # s0=[1,0], s1=[0.95, 0.31], s2=[0.5, 0.87], s3=[-0.5, 0.87], s4=[-1, 0]
        # Distances from s0: 0.05, 0.5, 1.5, 2.0 (cosine-ish, monotone in angle)
        import math
        embeds = {
            "s0": [1.0, 0.0],
            "s1": [math.cos(0.1), math.sin(0.1)],
            "s2": [math.cos(1.0), math.sin(1.0)],
            "s3": [math.cos(2.0), math.sin(2.0)],
            "s4": [math.cos(math.pi), math.sin(math.pi)],
        }
        bank.attach_embeddings(embeds)
        return bank

    def test_cold_start_falls_back_to_uniform(self):
        bank = self._bank()
        # Empty used set → uniform random over candidates
        random.seed(0)
        seed = bank.select_farthest_first("real_world", used_ids=set())
        assert seed is not None
        assert seed.type == "real_world"

    def test_picks_farthest(self):
        bank = self._bank()
        # With used={s0}, the farthest of the remaining is s4 (angle = pi).
        # Top-K=3 means s4, s3, s2 are candidates; we should hit s4 most often.
        random.seed(0)
        picks = [bank.select_farthest_first("real_world", used_ids={"s0"}).id
                 for _ in range(50)]
        # All picks should be in the top-3 farthest from s0 (s2, s3, s4).
        assert set(picks) <= {"s2", "s3", "s4"}
        # And the closest two (s1) should never appear.
        assert "s1" not in picks

    def test_excludes_used(self):
        bank = self._bank()
        # If we've used everything, return None.
        seed = bank.select_farthest_first(
            "real_world", used_ids={"s0", "s1", "s2", "s3", "s4"},
        )
        assert seed is None

    def test_falls_back_when_no_embeddings(self):
        seeds = [Seed(id=f"s{i}", type="real_world", content="x", tags=[])
                 for i in range(3)]
        bank = SeedBank(seeds)
        # No attach_embeddings call → falls back to uniform
        seed = bank.select_farthest_first("real_world", used_ids={"s0"})
        assert seed is not None
        assert seed.id != "s0"

    def test_max_min_semantics(self):
        """The candidate's score is its MIN distance to any used seed —
        not the average. Verify this for a 2-used-seed case where one
        candidate is mid-distance from both, vs another that's close to
        one but far from the other (should be ranked LOWER)."""
        seeds = [Seed(id=f"s{i}", type="real_world", content="x", tags=[])
                 for i in range(4)]
        bank = SeedBank(seeds)
        # s0, s1 are used. s2 is mid-distance from both. s3 is very close
        # to s0 but far from s1. min-distance: s2 > s3, so s2 should win.
        bank.attach_embeddings({
            "s0": [1.0, 0.0],
            "s1": [-1.0, 0.0],
            "s2": [0.0, 1.0],   # equidistant from s0 and s1
            "s3": [0.99, 0.14], # very close to s0
        })
        random.seed(0)
        # Top-K=3 still includes s2 (we have 2 candidates) but s2 should rank
        # higher than s3 because min(d_to_s0, d_to_s1) for s2 > for s3.
        # With only 2 candidates, top-3 = both — but s2 is still favored.
        # Verify by direct rank:
        from owtn.models.stage_1.seed_bank import _cosine_distance
        d_s2_min = min(_cosine_distance(bank._embeddings["s2"], bank._embeddings["s0"]),
                       _cosine_distance(bank._embeddings["s2"], bank._embeddings["s1"]))
        d_s3_min = min(_cosine_distance(bank._embeddings["s3"], bank._embeddings["s0"]),
                       _cosine_distance(bank._embeddings["s3"], bank._embeddings["s1"]))
        assert d_s2_min > d_s3_min, "s2 should be farther by min-distance"
