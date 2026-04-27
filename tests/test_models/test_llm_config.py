"""Tests for LLMConfig + GenerationModelConfig field validation + defaults.

Context: lab/issues/2026-04-22-generation-thinking-mode.md.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from owtn.llm.kwargs import THINKING_TOKENS
from owtn.models.config import GenerationModelConfig, LLMConfig


def _base() -> dict:
    return {
        "generation_models": [{"name": "claude-sonnet-4-6"}],
        "classifier_model": "gpt-4.1-mini",
        "embedding_model": "text-embedding-3-small",
    }


class TestGenerationModelDefaults:
    def test_temperature_default_is_1(self):
        m = GenerationModelConfig(name="claude-sonnet-4-6")
        assert m.temperature == 1.0

    def test_weight_default_is_1(self):
        m = GenerationModelConfig(name="claude-sonnet-4-6")
        assert m.weight == 1.0

    def test_reasoning_effort_default_disabled(self):
        m = GenerationModelConfig(name="claude-sonnet-4-6")
        assert m.reasoning_effort == "disabled"

    def test_samplers_default_none(self):
        m = GenerationModelConfig(name="claude-sonnet-4-6")
        assert m.top_p is None
        assert m.top_k is None
        assert m.min_p is None

    def test_self_critic_default_false(self):
        m = GenerationModelConfig(name="claude-sonnet-4-6")
        assert m.self_critic is False

    def test_self_critic_accepts_true(self):
        m = GenerationModelConfig(name="claude-sonnet-4-6", self_critic=True)
        assert m.self_critic is True

    def test_self_critic_reasoning_effort_default_disabled(self):
        m = GenerationModelConfig(name="claude-sonnet-4-6")
        assert m.self_critic_reasoning_effort == "disabled"

    def test_self_critic_reasoning_effort_accepts_other_efforts(self):
        m = GenerationModelConfig(
            name="claude-sonnet-4-6", self_critic_reasoning_effort="low"
        )
        assert m.self_critic_reasoning_effort == "low"

    def test_self_critic_reasoning_effort_rejects_bogus(self):
        with pytest.raises(ValidationError):
            GenerationModelConfig(
                name="x", self_critic_reasoning_effort="intense"
            )


class TestReasoningEffortValidator:
    def test_accepts_disabled(self):
        m = GenerationModelConfig(name="claude-sonnet-4-6", reasoning_effort="disabled")
        assert m.reasoning_effort == "disabled"

    @pytest.mark.parametrize("effort", list(THINKING_TOKENS.keys()))
    def test_accepts_all_preset_keys(self, effort: str):
        m = GenerationModelConfig(name="claude-sonnet-4-6", reasoning_effort=effort)
        assert m.reasoning_effort == effort

    def test_rejects_bogus(self):
        with pytest.raises(ValidationError):
            GenerationModelConfig(name="x", reasoning_effort="intense")

    def test_rejects_empty_string(self):
        with pytest.raises(ValidationError):
            GenerationModelConfig(name="x", reasoning_effort="")


class TestWeightValidator:
    def test_rejects_zero(self):
        with pytest.raises(ValidationError):
            GenerationModelConfig(name="x", weight=0)

    def test_rejects_negative(self):
        with pytest.raises(ValidationError):
            GenerationModelConfig(name="x", weight=-1)


class TestLLMConfigMultiModel:
    def test_per_model_params(self):
        cfg = LLMConfig(
            generation_models=[
                {"name": "deepseek-v4-pro", "temperature": 1.2, "reasoning_effort": "high"},
                {"name": "claude-sonnet-4-6", "temperature": 1.0, "reasoning_effort": "disabled"},
            ],
            classifier_model="gpt-4.1-mini",
            embedding_model="text-embedding-3-small",
        )
        assert len(cfg.generation_models) == 2
        assert cfg.generation_models[0].temperature == 1.2
        assert cfg.generation_models[0].reasoning_effort == "high"
        assert cfg.generation_models[1].temperature == 1.0
        assert cfg.generation_models[1].reasoning_effort == "disabled"
