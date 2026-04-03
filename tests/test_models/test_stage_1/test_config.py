import pytest
from pydantic import ValidationError

from owtn.models.stage_1.config import StageConfig


CONFIG_PATH = "configs/stage_1/medium.yaml"


class TestStageConfig:
    def test_load_default(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert config.stage == 1

    def test_evolution(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert config.evolution.num_generations == 20
        assert config.evolution.language == "json"
        assert len(config.evolution.patch_types) == 11
        assert len(config.evolution.patch_type_probs) == 11
        assert config.evolution.llm_dynamic_selection == "ucb"
        assert config.evolution.use_text_feedback is True

    def test_database(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert config.database.num_islands == 10
        assert config.database.archive_selection_strategy == "map_elites"
        assert config.database.migration_rate == 0.20

    def test_operator_bandit(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert config.operator_bandit.enabled is True
        assert config.operator_bandit.warmup_generations == 3
        assert config.operator_bandit.min_probability_floor == 0.02

    def test_llm(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert "claude-sonnet-4-6" in config.llm.generation_models
        assert config.llm.generation_model_family == "anthropic"
        assert config.llm.embedding_model == "text-embedding-3-small"

    def test_judges(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert len(config.judges.panel) == 3
        assert "mira-okonkwo" in config.judges.panel
        assert config.judges.min_demanding_ratio == 0.3

    def test_evaluation(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert config.evaluation.holder_p == 0.4
        assert config.evaluation.diversity_weight == 0.15
        assert config.evaluation.tier_a_enabled is False
        assert config.evaluation.anti_cliche.similarity_threshold == 0.85
        assert config.evaluation.anti_cliche.elevated_originality_threshold == 3.5

    def test_handoff(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert config.handoff.strategy == "top_per_cell"
        assert config.handoff.max_concepts == 6

    def test_paths(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert config.paths.seed_bank == "data/seed-bank.yaml"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            StageConfig.from_yaml("nonexistent.yaml")
