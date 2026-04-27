"""Native structured output across providers — replaces what instructor used to do.

For every provider's structured-output path:
- the schema for production output models compiles successfully
- a mocked response with the right shape parses into a pydantic instance
- system_prefix is wired correctly (cache_control for Anthropic, concat elsewhere)
- recovery (Kimi [EOS] suffix, malformed keys) catches expected failure modes
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from owtn.evaluation.models import PairwiseJudgment, Stage2PairwiseJudgment
from owtn.models.stage_2.actions import ExpansionProposals
from owtn.optimizer.models import LineageBrief
from owtn.optimizer.population_brief import PopulationBrief
from owtn.stage_2.operators import SeedExtractionResult


_PRODUCTION_OUTPUT_MODELS = [
    PairwiseJudgment,
    Stage2PairwiseJudgment,
    LineageBrief,
    PopulationBrief,
    SeedExtractionResult,
    ExpansionProposals,
]


_PAIRWISE_INSTANCE = PairwiseJudgment(
    reasoning="FP1 vs FP2: tied across all dims.",
    novelty="tie", grip="tie", tension_architecture="tie",
    emotional_depth="tie", thematic_resonance="tie",
    concept_coherence="tie", generative_fertility="tie",
    scope_calibration="tie", indelibility="tie",
)


# ---------- Schema compatibility (no API calls) ----------


class TestSchemaBuildsForAllProviders:
    """Each production output model must build a schema each provider accepts."""

    @pytest.mark.parametrize("output_model", _PRODUCTION_OUTPUT_MODELS,
                             ids=lambda m: m.__name__)
    def test_anthropic_tool_schema_serializable(self, output_model):
        """Anthropic accepts JSON Schema Draft-07 with $defs/oneOf — verify
        the tool payload is JSON-serializable and well-formed."""
        import json
        from owtn.llm.providers.anthropic import _structured_output_tool

        tool = _structured_output_tool(output_model)
        json.dumps(tool)  # must round-trip
        assert tool["name"] == output_model.__name__
        assert tool["input_schema"]["type"] == "object"

    @pytest.mark.parametrize("output_model", _PRODUCTION_OUTPUT_MODELS,
                             ids=lambda m: m.__name__)
    def test_openai_strict_schema_builds(self, output_model):
        """OpenAI's strict mode is the most demanding (no Optional defaults,
        additionalProperties=False at every level). Verify the SDK's
        Pydantic→strict-schema transformer accepts the model."""
        from openai.lib._parsing._completions import type_to_response_format_param

        fmt = type_to_response_format_param(output_model)
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["strict"] is True
        assert fmt["json_schema"]["name"] == output_model.__name__

    @pytest.mark.parametrize("output_model", _PRODUCTION_OUTPUT_MODELS,
                             ids=lambda m: m.__name__)
    def test_deepseek_schema_instruction_well_formed(self, output_model):
        """DeepSeek path injects its own schema instruction. Verify the
        instruction includes the schema, mentions JSON, and is non-empty."""
        from owtn.llm.providers.deepseek import _schema_instruction

        text = _schema_instruction(output_model)
        assert "JSON" in text
        assert "Schema:" in text
        assert "properties" in text  # the schema dump itself

    @pytest.mark.parametrize("output_model", _PRODUCTION_OUTPUT_MODELS,
                             ids=lambda m: m.__name__)
    def test_gemini_config_builds_or_clear_error(self, output_model):
        """Gemini schema converter rejects discriminated unions; we should
        raise NotImplementedError up-front rather than emit a cryptic SDK
        ValidationError. Plain models build cleanly."""
        from owtn.llm.providers.gemini import GeminiProvider

        provider = GeminiProvider()
        try:
            config = provider._build_generation_config(
                system_msg="x", kwargs={}, output_model=output_model,
            )
        except NotImplementedError as e:
            # Discriminated unions are caught up-front with a clear message.
            assert "discriminated unions" in str(e)
            assert output_model is ExpansionProposals
            return
        # Plain models reach config construction.
        assert config.response_schema is output_model
        assert config.response_mime_type == "application/json"


# ---------- Mock-response parsing ----------


class TestAnthropicParsing:
    def test_tool_use_block_parses_to_pydantic_instance(self):
        from owtn.llm.providers.anthropic import AnthropicProvider

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "PairwiseJudgment"
        tool_block.input = _PAIRWISE_INSTANCE.model_dump()
        response = MagicMock()
        response.content = [tool_block]
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50
        response.usage.cache_read_input_tokens = 0
        response.usage.cache_creation_input_tokens = 0
        client = MagicMock()
        client.messages.create.return_value = response

        result = AnthropicProvider().query(
            model="claude-sonnet-4-6", msg="user", system_msg="sys",
            msg_history=[], system_prefix=None,
            output_model=PairwiseJudgment, kwargs={"max_tokens": 4096},
            client=client,
        )
        assert isinstance(result.content, PairwiseJudgment)
        assert result.content.novelty == "tie"

    def test_forces_tool_choice_on_structured_call(self):
        """Verify the SDK is told to force-emit the tool, not 'auto'."""
        from owtn.llm.providers.anthropic import AnthropicProvider

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "PairwiseJudgment"
        tool_block.input = _PAIRWISE_INSTANCE.model_dump()
        response = MagicMock()
        response.content = [tool_block]
        response.usage.input_tokens = 1
        response.usage.output_tokens = 1
        response.usage.cache_read_input_tokens = 0
        response.usage.cache_creation_input_tokens = 0
        client = MagicMock()
        client.messages.create.return_value = response

        AnthropicProvider().query(
            model="claude-sonnet-4-6", msg="m", system_msg="s",
            msg_history=[], system_prefix=None,
            output_model=PairwiseJudgment, kwargs={"max_tokens": 100},
            client=client,
        )
        kw = client.messages.create.call_args.kwargs
        assert kw["tool_choice"] == {"type": "tool", "name": "PairwiseJudgment"}
        assert len(kw["tools"]) == 1
        assert kw["tools"][0]["name"] == "PairwiseJudgment"


class TestOpenAIParsing:
    def test_responses_parse_returns_parsed_instance(self):
        from owtn.llm.providers.openai import OpenAIProvider

        response = MagicMock()
        response.output_parsed = _PAIRWISE_INSTANCE
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50
        response.usage.output_tokens_details = MagicMock(reasoning_tokens=0)
        response.usage.input_tokens_details = MagicMock(cached_tokens=0)
        response.usage.cost_details = None
        client = MagicMock()
        client.responses.parse.return_value = response

        result = OpenAIProvider().query(
            model="gpt-5.4-mini", msg="m", system_msg="s",
            msg_history=[], system_prefix=None,
            output_model=PairwiseJudgment, kwargs={"max_output_tokens": 4096},
            client=client,
        )
        assert isinstance(result.content, PairwiseJudgment)
        assert client.responses.parse.call_args.kwargs["text_format"] is PairwiseJudgment

    def test_recovery_on_eos_suffix(self):
        """The shared recovery module catches Kimi's [EOS] suffix and
        re-parses the cleaned content."""
        from owtn.llm.providers.openai import OpenAIProvider

        class _Tiny(BaseModel):
            x: str

        err = ValidationError.from_exception_data("_Tiny", [{
            "type": "json_invalid", "loc": (),
            "input": '{"x":"hello"}[EOS]',
            "ctx": {"error": "trailing characters"},
        }])
        client = MagicMock()
        client.responses.parse.side_effect = err

        result = OpenAIProvider().query(
            model="gpt-4.1-mini", msg="?", system_msg="s",
            msg_history=[], system_prefix=None,
            output_model=_Tiny, kwargs={"max_output_tokens": 100},
            client=client,
        )
        assert isinstance(result.content, _Tiny)
        assert result.content.x == "hello"


class TestDeepSeekParsing:
    def test_json_object_response_parses(self):
        from owtn.llm.providers.deepseek import DeepSeekProvider

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = _PAIRWISE_INSTANCE.model_dump_json()
        response.choices[0].message.reasoning_content = ""
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        response.usage.completion_tokens_details = None
        response.usage.prompt_tokens_details = None
        response.usage.prompt_cache_hit_tokens = 0
        client = MagicMock()
        client.chat.completions.create.return_value = response

        result = DeepSeekProvider().query(
            model="deepseek-v4-pro", msg="m", system_msg="s",
            msg_history=[], system_prefix=None,
            output_model=PairwiseJudgment, kwargs={},
            client=client,
        )
        assert isinstance(result.content, PairwiseJudgment)

    def test_response_format_set_on_structured_call(self):
        from owtn.llm.providers.deepseek import DeepSeekProvider

        class _Tiny(BaseModel):
            v: int

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"v":1}'
        response.choices[0].message.reasoning_content = ""
        response.usage.prompt_tokens = 1
        response.usage.completion_tokens = 1
        response.usage.completion_tokens_details = None
        response.usage.prompt_tokens_details = None
        response.usage.prompt_cache_hit_tokens = 0
        client = MagicMock()
        client.chat.completions.create.return_value = response

        DeepSeekProvider().query(
            model="deepseek-chat", msg="?", system_msg="s",
            msg_history=[], system_prefix=None,
            output_model=_Tiny, kwargs={},
            client=client,
        )
        kw = client.chat.completions.create.call_args.kwargs
        assert kw["response_format"] == {"type": "json_object"}
        # Schema instruction must reach the system message.
        sys_msg = kw["messages"][0]["content"]
        assert "Respond with a single valid JSON object" in sys_msg
        assert "Schema:" in sys_msg

    def test_recovery_on_eos_suffix(self):
        from owtn.llm.providers.deepseek import DeepSeekProvider

        class _Tiny(BaseModel):
            v: int

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"v":42}[EOS]'
        response.choices[0].message.reasoning_content = ""
        response.usage.prompt_tokens = 1
        response.usage.completion_tokens = 1
        response.usage.completion_tokens_details = None
        response.usage.prompt_tokens_details = None
        response.usage.prompt_cache_hit_tokens = 0
        client = MagicMock()
        client.chat.completions.create.return_value = response

        result = DeepSeekProvider().query(
            model="deepseek-chat", msg="?", system_msg="s",
            msg_history=[], system_prefix=None,
            output_model=_Tiny, kwargs={},
            client=client,
        )
        assert result.content.v == 42


class TestGeminiParsing:
    def test_json_text_part_parses(self):
        from owtn.llm.providers.gemini import GeminiProvider

        response = MagicMock()
        candidate = MagicMock()
        json_part = MagicMock(text=_PAIRWISE_INSTANCE.model_dump_json(), thought=False)
        candidate.content.parts = [json_part]
        response.candidates = [candidate]
        response.text = _PAIRWISE_INSTANCE.model_dump_json()
        response.usage_metadata.prompt_token_count = 100
        response.usage_metadata.candidates_token_count = 50
        response.usage_metadata.thoughts_token_count = 0
        response.usage_metadata.cached_content_token_count = 0
        client = MagicMock()
        client.models.generate_content.return_value = response

        result = GeminiProvider().query(
            model="gemini-3-flash-preview", msg="m", system_msg="s",
            msg_history=[], system_prefix=None,
            output_model=PairwiseJudgment, kwargs={"max_tokens": 4096},
            client=client,
        )
        assert isinstance(result.content, PairwiseJudgment)
        kw = client.models.generate_content.call_args.kwargs
        assert kw["config"].response_mime_type == "application/json"
        assert kw["config"].response_schema is PairwiseJudgment


# ---------- system_prefix wiring ----------


class TestSystemPrefixWiring:
    def test_anthropic_uses_cache_control_block(self):
        from owtn.llm.providers.anthropic import AnthropicProvider

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ok"
        response = MagicMock()
        response.content = [text_block]
        response.usage.input_tokens = 1
        response.usage.output_tokens = 1
        response.usage.cache_read_input_tokens = 0
        response.usage.cache_creation_input_tokens = 0
        client = MagicMock()
        client.messages.create.return_value = response

        AnthropicProvider().query(
            model="claude-sonnet-4-6", msg="m", system_msg="suffix",
            msg_history=[], system_prefix="cached prefix",
            output_model=None, kwargs={"max_tokens": 100},
            client=client,
        )
        sys_arg = client.messages.create.call_args.kwargs["system"]
        assert isinstance(sys_arg, list)
        assert sys_arg[0]["cache_control"] == {"type": "ephemeral"}
        assert sys_arg[0]["text"] == "cached prefix"
        assert sys_arg[1]["text"] == "suffix"

    def test_openai_concatenates_prefix(self):
        from owtn.llm.providers.openai import OpenAIProvider

        response = MagicMock()
        response.output = [MagicMock()]
        response.output[0].content = [MagicMock(text="ok")]
        response.output[0].summary = []
        response.usage.input_tokens = 1
        response.usage.output_tokens = 1
        response.usage.output_tokens_details = MagicMock(reasoning_tokens=0)
        response.usage.input_tokens_details = MagicMock(cached_tokens=0)
        response.usage.cost_details = None
        client = MagicMock()
        client.responses.create.return_value = response

        OpenAIProvider().query(
            model="gpt-4.1-mini", msg="m", system_msg="suffix",
            msg_history=[], system_prefix="cached prefix",
            output_model=None, kwargs={"max_output_tokens": 100},
            client=client,
        )
        sys_msg = client.responses.create.call_args.kwargs["input"][0]["content"]
        assert sys_msg.startswith("cached prefix")
        assert "suffix" in sys_msg

    def test_deepseek_concatenates_prefix_with_schema_instruction(self):
        """All three (prefix, system_msg, schema instruction) must appear."""
        from owtn.llm.providers.deepseek import DeepSeekProvider

        class _Tiny(BaseModel):
            v: int

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"v":1}'
        response.choices[0].message.reasoning_content = ""
        response.usage.prompt_tokens = 1
        response.usage.completion_tokens = 1
        response.usage.completion_tokens_details = None
        response.usage.prompt_tokens_details = None
        response.usage.prompt_cache_hit_tokens = 0
        client = MagicMock()
        client.chat.completions.create.return_value = response

        DeepSeekProvider().query(
            model="deepseek-chat", msg="m", system_msg="judge instructions",
            msg_history=[], system_prefix="cached prefix",
            output_model=_Tiny, kwargs={},
            client=client,
        )
        sys_msg = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert sys_msg.startswith("cached prefix")
        assert "judge instructions" in sys_msg
        assert "Respond with a single valid JSON object" in sys_msg


# ---------- Explicit thinking_tokens override ----------


class TestExplicitThinkingTokens:
    """`thinking_tokens` is the preferred way to set Anthropic/Gemini
    reasoning budget — overrides the legacy THINKING_TOKENS[effort] mapping.
    Has no effect on OpenAI/DeepSeek (those APIs take an effort string)."""

    def test_anthropic_thinking_tokens_overrides_effort_lookup(self):
        from owtn.llm.providers.anthropic import AnthropicProvider

        out = AnthropicProvider().build_call_kwargs(
            api_model="claude-sonnet-4-6",
            requested={
                "reasoning_effort": "low",   # would map to 2048
                "thinking_tokens": 6000,     # but explicit wins
                "max_tokens": 16384,
            },
        )
        assert out["thinking"]["budget_tokens"] == 6000

    def test_anthropic_thinking_tokens_clamped_under_max(self):
        """When the explicit budget exceeds max_tokens, fall back to 1024."""
        from owtn.llm.providers.anthropic import AnthropicProvider

        out = AnthropicProvider().build_call_kwargs(
            api_model="claude-sonnet-4-6",
            requested={
                "reasoning_effort": "low",
                "thinking_tokens": 20000,    # > max_tokens (4096)
                "max_tokens": 4096,
            },
        )
        assert out["thinking"]["budget_tokens"] == 1024

    def test_anthropic_falls_back_to_effort_when_thinking_tokens_unset(self):
        """Backward-compat: existing configs without thinking_tokens keep
        getting the THINKING_TOKENS[effort] mapping."""
        from owtn.llm.providers.anthropic import AnthropicProvider
        from owtn.llm.providers.base import THINKING_TOKENS

        out = AnthropicProvider().build_call_kwargs(
            api_model="claude-sonnet-4-6",
            requested={"reasoning_effort": "high", "max_tokens": 16384},
        )
        assert out["thinking"]["budget_tokens"] == THINKING_TOKENS["high"]

    def test_gemini_thinking_tokens_overrides_effort_lookup(self):
        from owtn.llm.providers.gemini import GeminiProvider

        out = GeminiProvider().build_call_kwargs(
            api_model="gemini-3-pro-preview",
            requested={
                "reasoning_effort": "low",
                "thinking_tokens": 3000,
                "max_tokens": 16384,
            },
        )
        assert out["thinking_budget"] == 3000

    def test_openai_ignores_thinking_tokens(self):
        """OpenAI takes effort as a string; thinking_tokens is silently
        ignored (no provider knob to set it)."""
        from owtn.llm.providers.openai import OpenAIProvider

        out = OpenAIProvider().build_call_kwargs(
            api_model="gpt-5.4-mini",
            requested={
                "reasoning_effort": "low",
                "thinking_tokens": 9999,
                "max_tokens": 4096,
            },
        )
        assert out["reasoning"] == {"effort": "low"}
        # No thinking_budget anywhere in OpenAI shape.
        assert "thinking_budget" not in out
        assert "thinking_tokens" not in out

    def test_deepseek_ignores_thinking_tokens(self):
        from owtn.llm.providers.deepseek import DeepSeekProvider

        out = DeepSeekProvider().build_call_kwargs(
            api_model="deepseek-v4-pro",
            requested={
                "reasoning_effort": "medium",
                "thinking_tokens": 9999,
            },
        )
        assert "thinking_budget" not in out
        assert "thinking_tokens" not in out
        assert out["reasoning_effort"] == "medium"

    def test_anthropic_thinking_tokens_only_no_effort(self):
        """thinking_tokens alone (effort='disabled') still enables thinking."""
        from owtn.llm.providers.anthropic import AnthropicProvider

        out = AnthropicProvider().build_call_kwargs(
            api_model="claude-sonnet-4-6",
            requested={
                "reasoning_effort": "disabled",
                "thinking_tokens": 4000,
                "max_tokens": 16384,
            },
        )
        assert out["thinking"]["budget_tokens"] == 4000

    def test_anthropic_no_thinking_when_neither_set(self):
        from owtn.llm.providers.anthropic import AnthropicProvider

        out = AnthropicProvider().build_call_kwargs(
            api_model="claude-sonnet-4-6",
            requested={"reasoning_effort": "disabled", "max_tokens": 16384},
        )
        assert "thinking" not in out
