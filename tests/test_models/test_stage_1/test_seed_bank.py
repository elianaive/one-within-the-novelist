from owtn.models.stage_1.seed_bank import OPERATOR_SEED_TYPES, SEED_OPERATOR_MAP, SeedBank


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
