"""Tests for _build_judge_kwargs — reasoning-cap + sampler kwargs for judge calls.

Context: lab/issues/2026-04-18-judge-latency-variance.md,
lab/issues/2026-04-22-generation-thinking-mode.md.
"""

from __future__ import annotations

from owtn.evaluation.pairwise import _JUDGE_MAX_OUTPUT_TOKENS, _build_judge_kwargs
from owtn.models.judge import JudgePersona


def _judge(
    model: str,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    reasoning_effort: str = "disabled",
) -> JudgePersona:
    return JudgePersona(
        id="test",
        name="Test",
        identity="test",
        values=["test"],
        exemplars=["test"],
        lean_in_signals=["test"],
        harshness="standard",
        priority="primary",
        model=[model],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        reasoning_effort=reasoning_effort,
    )


class TestOpenAIReasoningJudge:
    def test_gpt_5_4_mini_gets_reasoning_cap(self):
        _, kwargs = _build_judge_kwargs(_judge("gpt-5.4-mini"))
        assert kwargs["reasoning"] == {"effort": "low"}
        assert kwargs["max_output_tokens"] == _JUDGE_MAX_OUTPUT_TOKENS

    def test_gpt_5_4_mini_temp_forced_to_1(self):
        """OpenAI reasoning + think_temp_fixed → API requires temp=1.0."""
        _, kwargs = _build_judge_kwargs(_judge("gpt-5.4-mini", temperature=0.3))
        assert kwargs["temperature"] == 1.0

    def test_gpt_5_4_mini_top_p_dropped_under_reasoning(self):
        _, kwargs = _build_judge_kwargs(_judge("gpt-5.4-mini", top_p=0.9))
        assert "top_p" not in kwargs

    def test_gpt_5_4_mini_has_no_extra_body(self):
        _, kwargs = _build_judge_kwargs(_judge("gpt-5.4-mini"))
        assert "extra_body" not in kwargs


class TestOpenRouterReasoningJudge:
    def test_glm_default_disables_reasoning(self):
        """Default reasoning_effort=disabled → no reasoning kwarg for opt-in reasoning model."""
        _, kwargs = _build_judge_kwargs(_judge("z-ai/glm-5.1"))
        assert "reasoning" not in kwargs
        assert kwargs["max_output_tokens"] == _JUDGE_MAX_OUTPUT_TOKENS

    def test_glm_opt_in_low_effort(self):
        _, kwargs = _build_judge_kwargs(_judge("z-ai/glm-5.1", reasoning_effort="low"))
        assert kwargs["reasoning"] == {"effort": "low"}
        assert kwargs["max_output_tokens"] == _JUDGE_MAX_OUTPUT_TOKENS

    def test_kimi_default_disables_reasoning(self):
        _, kwargs = _build_judge_kwargs(_judge("moonshotai/kimi-k2.5"))
        assert "reasoning" not in kwargs
        assert kwargs["max_output_tokens"] == _JUDGE_MAX_OUTPUT_TOKENS

    def test_kimi_opt_in_low_effort(self):
        _, kwargs = _build_judge_kwargs(_judge("moonshotai/kimi-k2.5", reasoning_effort="low"))
        assert kwargs["reasoning"] == {"effort": "low"}

    def test_extra_body_reasoning_disabled(self):
        """OpenRouter reasoning models get extra_body.reasoning.enabled=False
        when reasoning_effort=disabled — the top-level Responses API reasoning
        kwarg isn't honored for GLM/Kimi, so this is how we actually suppress
        default reasoning-by-default. See 2026-04-23 diagnosis."""
        _, glm = _build_judge_kwargs(_judge("z-ai/glm-5.1"))
        _, kimi = _build_judge_kwargs(_judge("moonshotai/kimi-k2.5"))
        assert glm["extra_body"] == {"reasoning": {"enabled": False}}
        assert kimi["extra_body"] == {"reasoning": {"enabled": False}}

    def test_extra_body_reasoning_enabled(self):
        """When reasoning_effort is set, extra_body carries both enabled=True
        and the effort — in addition to the top-level reasoning kwarg."""
        _, glm = _build_judge_kwargs(
            _judge("z-ai/glm-5.1", reasoning_effort="low")
        )
        assert glm["reasoning"] == {"effort": "low"}
        assert glm["extra_body"] == {"reasoning": {"enabled": True, "effort": "low"}}

    def test_no_extra_body_routing_params(self):
        """We intentionally do NOT use extra_body.provider.sort routing — a
        previous attempt (require_parameters=true) forced Kimi onto an
        EOS-leaking upstream. extra_body is now scoped to reasoning only."""
        _, glm = _build_judge_kwargs(_judge("z-ai/glm-5.1"))
        assert "provider" not in glm.get("extra_body", {})


class TestNonReasoningModel:
    def test_deepseek_chat_uses_judge_temperature(self):
        """DeepSeek: no OpenAI caps; judge temperature flows through."""
        _, kwargs = _build_judge_kwargs(_judge("deepseek-chat", temperature=0.4))
        assert kwargs == {"temperature": 0.4}

    def test_non_reasoning_openrouter_gets_only_output_cap_and_temp(self):
        """Non-reasoning OpenRouter models still get max_output_tokens to
        override tight provider defaults that can truncate judgments."""
        _, kwargs = _build_judge_kwargs(_judge("moonshotai/kimi-k2-0905"))
        assert kwargs == {
            "max_output_tokens": _JUDGE_MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
        }
        assert "reasoning" not in kwargs

    def test_non_reasoning_openai_gets_output_cap_and_temp(self):
        _, kwargs = _build_judge_kwargs(_judge("gpt-4.1-mini", temperature=0.2))
        assert kwargs == {
            "max_output_tokens": _JUDGE_MAX_OUTPUT_TOKENS,
            "temperature": 0.2,
        }
        assert "reasoning" not in kwargs


class TestSamplerFiltering:
    def test_top_k_dropped_for_openai_family(self):
        """OpenAI API rejects top_k."""
        _, kwargs = _build_judge_kwargs(_judge("gpt-4.1-mini", top_k=40))
        assert "top_k" not in kwargs

    def test_top_k_dropped_for_openrouter(self):
        _, kwargs = _build_judge_kwargs(_judge("moonshotai/kimi-k2-0905", top_k=40))
        assert "top_k" not in kwargs

    def test_top_p_passes_through_for_non_reasoning(self):
        _, kwargs = _build_judge_kwargs(_judge("gpt-4.1-mini", top_p=0.9))
        assert kwargs["top_p"] == 0.9

    def test_top_k_passes_through_for_deepseek(self):
        """Non-openai-family provider — top_k allowed."""
        _, kwargs = _build_judge_kwargs(_judge("deepseek-chat", top_k=40))
        assert kwargs["top_k"] == 40

    def test_min_p_forwarded_only_on_openrouter(self):
        _, kwargs = _build_judge_kwargs(_judge("moonshotai/kimi-k2-0905", min_p=0.1))
        assert kwargs["min_p"] == 0.1

    def test_min_p_dropped_on_native_openai(self):
        _, kwargs = _build_judge_kwargs(_judge("gpt-4.1-mini", min_p=0.1))
        assert "min_p" not in kwargs

    def test_min_p_dropped_on_deepseek(self):
        _, kwargs = _build_judge_kwargs(_judge("deepseek-chat", min_p=0.1))
        assert "min_p" not in kwargs


class TestSignatureReturnsModelName:
    def test_returns_model_name_tuple(self):
        model, kwargs = _build_judge_kwargs(_judge("gpt-4.1"))
        assert model == "gpt-4.1"
        assert isinstance(kwargs, dict)
