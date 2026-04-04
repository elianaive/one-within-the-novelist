import pytest
from pydantic import ValidationError

from owtn.models.stage_1.config import StageConfig


CONFIG_PATH = "configs/stage_1/medium.yaml"


class TestStageConfig:
    def test_load_default(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert config.stage == 1

    def test_evolution_loads(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert config.evolution.language == "json"
        assert len(config.evolution.patch_types) > 0
        assert len(config.evolution.patch_types) == len(config.evolution.patch_type_probs)
        assert config.evolution.num_generations > 0

    def test_database_loads(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert config.database.num_islands > 0
        assert config.database.archive_selection_strategy in ("map_elites", "fitness")

    def test_operator_bandit_loads(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert isinstance(config.operator_bandit.enabled, bool)
        assert config.operator_bandit.min_probability_floor >= 0

    def test_llm_loads(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert len(config.llm.generation_models) > 0
        assert config.llm.generation_model_family != ""

    def test_judges_loads(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert len(config.judges.panel) > 0

    def test_evaluation_loads(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert 0 < config.evaluation.holder_p <= 1.0
        assert config.evaluation.anti_cliche.similarity_threshold > 0

    def test_handoff_loads(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert config.handoff.strategy != ""
        assert config.handoff.max_concepts > 0

    def test_paths_loads(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert config.paths.seed_bank.endswith(".yaml")

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            StageConfig.from_yaml("nonexistent.yaml")
