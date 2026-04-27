"""Verify _build_shinka_configs threads per-model generation params into
evo_config.llm_kwargs as parallel lists indexed by model position. See
``GenerationModelConfig`` docstring for the schema.
"""

from __future__ import annotations

from owtn.models.stage_1.config import StageConfig
from owtn.runner import _build_shinka_configs


def _load(path: str = "configs/stage_1/dry_run.yaml") -> StageConfig:
    return StageConfig.from_yaml(path)


class TestLLMKwargsPlumbing:
    def test_temperatures_parallel_list(self):
        cfg = _load()
        evo, _, _ = _build_shinka_configs(cfg, "configs/stage_1/dry_run.yaml")
        assert evo.llm_kwargs["temperatures"] == [
            m.temperature for m in cfg.llm.generation_models
        ]

    def test_reasoning_efforts_parallel_list(self):
        cfg = _load()
        evo, _, _ = _build_shinka_configs(cfg, "configs/stage_1/dry_run.yaml")
        assert evo.llm_kwargs["reasoning_efforts"] == [
            m.reasoning_effort for m in cfg.llm.generation_models
        ]

    def test_sampler_params_parallel_lists(self):
        cfg = _load()
        evo, _, _ = _build_shinka_configs(cfg, "configs/stage_1/dry_run.yaml")
        assert evo.llm_kwargs["top_p"] == [m.top_p for m in cfg.llm.generation_models]
        assert evo.llm_kwargs["top_k"] == [m.top_k for m in cfg.llm.generation_models]
        assert evo.llm_kwargs["min_p"] == [m.min_p for m in cfg.llm.generation_models]

    def test_llm_models_is_name_list(self):
        cfg = _load()
        evo, _, _ = _build_shinka_configs(cfg, "configs/stage_1/dry_run.yaml")
        assert evo.llm_models == [m.name for m in cfg.llm.generation_models]

    def test_all_three_stage_1_configs_build(self):
        """Every bundled stage_1 config must round-trip through
        _build_shinka_configs without error."""
        for path in (
            "configs/stage_1/dry_run.yaml",
            "configs/stage_1/light.yaml",
            "configs/stage_1/medium.yaml",
        ):
            cfg = StageConfig.from_yaml(path)
            evo, db, job = _build_shinka_configs(cfg, path)
            assert evo.llm_models == [m.name for m in cfg.llm.generation_models]
