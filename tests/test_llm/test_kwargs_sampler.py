"""Tests for sample_model_kwargs sampler-param routing + temp-forcing.

Covers:
- Narrowed temp-forcing condition: forces 1.0 only under active reasoning
  on think_temp_fixed models (not on openai-family with reasoning off).
- top_p dropped for OpenAI reasoning and under Anthropic thinking.
- top_k dropped for any OpenAI-family provider and under Anthropic thinking.
- min_p forwarded only via OpenRouter; dropped on native APIs.

Context: lab/issues/2026-04-22-generation-thinking-mode.md.
"""

from __future__ import annotations

from owtn.llm.kwargs import sample_model_kwargs


class TestTempForcing:
    def test_anthropic_thinking_forces_temp_1(self):
        kw = sample_model_kwargs(
            model_names="claude-sonnet-4-6",
            temperatures=0.3,
            reasoning_efforts="medium",
        )
        assert kw["temperature"] == 1.0

    def test_anthropic_no_thinking_honors_temp(self):
        kw = sample_model_kwargs(
            model_names="claude-sonnet-4-6",
            temperatures=0.3,
            reasoning_efforts="disabled",
        )
        assert kw["temperature"] == 0.3

    def test_openai_reasoning_forces_temp_1(self):
        """OpenAI reasoning models: requires_reasoning coerces effort → temp=1.0."""
        kw = sample_model_kwargs(
            model_names="gpt-5.4-mini",
            temperatures=0.3,
            reasoning_efforts="disabled",  # coerced to "low" for gpt-5.x
        )
        assert kw["temperature"] == 1.0

    def test_non_reasoning_openai_honors_temp(self):
        kw = sample_model_kwargs(
            model_names="gpt-4.1-mini",
            temperatures=0.3,
        )
        assert kw["temperature"] == 0.3


class TestTopPFilter:
    def test_top_p_passes_for_anthropic_no_thinking(self):
        kw = sample_model_kwargs(
            model_names="claude-sonnet-4-6",
            temperatures=0.7,
            reasoning_efforts="disabled",
            top_p=0.9,
        )
        assert kw["top_p"] == 0.9

    def test_top_p_dropped_under_anthropic_thinking(self):
        kw = sample_model_kwargs(
            model_names="claude-sonnet-4-6",
            temperatures=1.0,
            reasoning_efforts="medium",
            top_p=0.9,
        )
        assert "top_p" not in kw

    def test_top_p_dropped_for_openai_reasoning(self):
        kw = sample_model_kwargs(
            model_names="gpt-5.4-mini",
            temperatures=1.0,
            reasoning_efforts="low",
            top_p=0.9,
        )
        assert "top_p" not in kw

    def test_top_p_passes_for_non_reasoning_openai(self):
        kw = sample_model_kwargs(
            model_names="gpt-4.1-mini",
            temperatures=0.7,
            top_p=0.9,
        )
        assert kw["top_p"] == 0.9

    def test_top_p_none_omitted(self):
        kw = sample_model_kwargs(model_names="gpt-4.1-mini", temperatures=0.7)
        assert "top_p" not in kw


class TestTopKFilter:
    def test_top_k_passes_for_anthropic_no_thinking(self):
        kw = sample_model_kwargs(
            model_names="claude-sonnet-4-6",
            temperatures=0.7,
            reasoning_efforts="disabled",
            top_k=40,
        )
        assert kw["top_k"] == 40

    def test_top_k_dropped_for_openai(self):
        kw = sample_model_kwargs(
            model_names="gpt-4.1-mini",
            temperatures=0.7,
            top_k=40,
        )
        assert "top_k" not in kw

    def test_top_k_dropped_for_openrouter(self):
        kw = sample_model_kwargs(
            model_names="moonshotai/kimi-k2-0905",
            temperatures=0.7,
            top_k=40,
        )
        assert "top_k" not in kw

    def test_top_k_dropped_under_anthropic_thinking(self):
        kw = sample_model_kwargs(
            model_names="claude-sonnet-4-6",
            temperatures=1.0,
            reasoning_efforts="low",
            top_k=40,
        )
        assert "top_k" not in kw


class TestMinPFilter:
    def test_min_p_forwarded_on_openrouter(self):
        kw = sample_model_kwargs(
            model_names="moonshotai/kimi-k2-0905",
            temperatures=0.7,
            min_p=0.1,
        )
        assert kw["min_p"] == 0.1

    def test_min_p_dropped_on_native_anthropic(self):
        kw = sample_model_kwargs(
            model_names="claude-sonnet-4-6",
            temperatures=0.7,
            reasoning_efforts="disabled",
            min_p=0.1,
        )
        assert "min_p" not in kw

    def test_min_p_dropped_on_native_openai(self):
        kw = sample_model_kwargs(
            model_names="gpt-4.1-mini",
            temperatures=0.7,
            min_p=0.1,
        )
        assert "min_p" not in kw
