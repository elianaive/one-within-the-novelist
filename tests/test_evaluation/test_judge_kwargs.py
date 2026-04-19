"""Tests for _build_judge_kwargs — reasoning-cap + routing kwargs for judge calls.

Context: lab/issues/2026-04-18-judge-latency-variance.md.
"""

from __future__ import annotations

from owtn.evaluation.pairwise import _JUDGE_MAX_OUTPUT_TOKENS, _build_judge_kwargs


class TestOpenAIReasoningJudge:
    def test_gpt_5_4_mini_gets_reasoning_cap(self):
        kwargs = _build_judge_kwargs("gpt-5.4-mini")
        assert kwargs["reasoning"] == {"effort": "low"}
        assert kwargs["max_output_tokens"] == _JUDGE_MAX_OUTPUT_TOKENS

    def test_gpt_5_4_mini_has_no_extra_body(self):
        kwargs = _build_judge_kwargs("gpt-5.4-mini")
        assert "extra_body" not in kwargs


class TestOpenRouterReasoningJudge:
    def test_glm_gets_reasoning_cap(self):
        kwargs = _build_judge_kwargs("z-ai/glm-5.1")
        assert kwargs["reasoning"] == {"effort": "low"}
        assert kwargs["max_output_tokens"] == _JUDGE_MAX_OUTPUT_TOKENS

    def test_kimi_gets_reasoning_cap(self):
        kwargs = _build_judge_kwargs("moonshotai/kimi-k2.5")
        assert kwargs["reasoning"] == {"effort": "low"}
        assert kwargs["max_output_tokens"] == _JUDGE_MAX_OUTPUT_TOKENS

    def test_no_extra_body_routing(self):
        """require_parameters=true forced Kimi onto an EOS-leaking upstream.
        Routing kwargs removed until a clean recipe is found."""
        assert "extra_body" not in _build_judge_kwargs("z-ai/glm-5.1")
        assert "extra_body" not in _build_judge_kwargs("moonshotai/kimi-k2.5")


class TestNonReasoningModel:
    def test_deepseek_chat_gets_nothing(self):
        """DeepSeek provider is neither openai/openrouter/azure so no kwargs."""
        kwargs = _build_judge_kwargs("deepseek-chat")
        assert kwargs == {}

    def test_non_reasoning_openrouter_gets_only_output_cap(self):
        """Non-reasoning OpenRouter models still get max_output_tokens to
        override tight provider defaults that can truncate judgments."""
        kwargs = _build_judge_kwargs("moonshotai/kimi-k2-0905")
        assert kwargs == {"max_output_tokens": _JUDGE_MAX_OUTPUT_TOKENS}
        assert "reasoning" not in kwargs

    def test_non_reasoning_openai_gets_only_output_cap(self):
        kwargs = _build_judge_kwargs("gpt-4.1-mini")
        assert kwargs == {"max_output_tokens": _JUDGE_MAX_OUTPUT_TOKENS}
        assert "reasoning" not in kwargs
