import pytest
from pydantic import ValidationError

from owtn.evaluation.models import DIMENSION_NAMES
from owtn.models.stage_1.config import PairwiseAggregationConfig, StageConfig


CONFIG_PATH = "configs/stage_1/medium.yaml"


def _full_weights() -> dict[str, float]:
    """Complete dim_weights dict matching production values."""
    return {
        "indelibility": 2.00,
        "grip": 1.75,
        "novelty": 1.75,
        "generative_fertility": 1.25,
        "tension_architecture": 1.00,
        "emotional_depth": 1.00,
        "thematic_resonance": 1.00,
        "concept_coherence": 0.50,
        "scope_calibration": 0.50,
    }


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
        assert config.llm.generation_models[0].name

    def test_judges_loads(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert len(config.judges.panel) > 0

    def test_evaluation_loads(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        assert 0 < config.evaluation.holder_p <= 1.0
        assert config.evaluation.anti_cliche.similarity_threshold > 0

    def test_pairwise_loads(self):
        config = StageConfig.from_yaml(CONFIG_PATH)
        pw = config.evaluation.pairwise
        assert set(pw.dim_weights) == set(DIMENSION_NAMES)
        assert pw.tiebreaker_threshold >= 0
        assert all(d in DIMENSION_NAMES for d in pw.tiebreaker_dims)

    @pytest.mark.parametrize("path", [
        "configs/stage_1/dry_run.yaml",
        "configs/stage_1/light.yaml",
        "configs/stage_1/medium.yaml",
    ])
    def test_all_stage_1_yamls_load(self, path):
        """All three YAMLs must parse cleanly — the new pairwise block is required."""
        config = StageConfig.from_yaml(path)
        assert config.evaluation.pairwise.dim_weights

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


class TestPairwiseAggregationConfigValidation:
    """Validator enforces structural correctness at load time — YAML typos
    and misconfigurations fail immediately, not at aggregation time."""

    def test_valid_config(self):
        cfg = PairwiseAggregationConfig(
            dim_weights=_full_weights(),
            tiebreaker_threshold=1.0,
            tiebreaker_dims=["indelibility", "grip"],
        )
        assert cfg.dim_weights["indelibility"] == 2.00

    def test_missing_dim_rejected(self):
        weights = _full_weights()
        del weights["indelibility"]
        with pytest.raises(ValidationError, match="missing=.*indelibility"):
            PairwiseAggregationConfig(
                dim_weights=weights,
                tiebreaker_threshold=1.0,
                tiebreaker_dims=["grip"],
            )

    def test_extra_dim_rejected(self):
        weights = _full_weights()
        weights["typo_dimension"] = 1.0
        with pytest.raises(ValidationError, match="extra=.*typo_dimension"):
            PairwiseAggregationConfig(
                dim_weights=weights,
                tiebreaker_threshold=1.0,
                tiebreaker_dims=["indelibility"],
            )

    def test_unknown_tiebreaker_dim_rejected(self):
        with pytest.raises(ValidationError, match="tiebreaker_dims contains unknown dim"):
            PairwiseAggregationConfig(
                dim_weights=_full_weights(),
                tiebreaker_threshold=1.0,
                tiebreaker_dims=["not_a_real_dim"],
            )

    def test_negative_weight_rejected(self):
        weights = _full_weights()
        weights["indelibility"] = -0.5
        with pytest.raises(ValidationError, match="non-negative"):
            PairwiseAggregationConfig(
                dim_weights=weights,
                tiebreaker_threshold=1.0,
                tiebreaker_dims=["grip"],
            )

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValidationError, match="tiebreaker_threshold must be non-negative"):
            PairwiseAggregationConfig(
                dim_weights=_full_weights(),
                tiebreaker_threshold=-0.1,
                tiebreaker_dims=["indelibility"],
            )

    def test_empty_tiebreaker_dims_allowed(self):
        """An empty tiebreaker list is legal — used for uniform-weights regression."""
        cfg = PairwiseAggregationConfig(
            dim_weights=_full_weights(),
            tiebreaker_threshold=0.0,
            tiebreaker_dims=[],
        )
        assert cfg.tiebreaker_dims == []
