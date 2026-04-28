"""End-to-end live-API tests for native structured output.

Verifies the *actual* claim of this refactor: each provider's structured-
output path produces a valid pydantic instance against real APIs, without
instructor doing the work.

Marked live_api so it's deselected by default. Run explicitly:
    uv run pytest tests/test_llm/test_integration_structured.py -m live_api
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from owtn.llm.api import query, query_async


pytestmark = pytest.mark.live_api


class _Verdict(BaseModel):
    """Simple model: pick a side and explain. Tests structured-output
    enforcement without piggybacking on the production schemas (which
    are large and could mask which call path is being exercised)."""
    reasoning: str = Field(description="One-sentence explanation.")
    pick: Literal["a", "b"] = Field(description="Either 'a' or 'b'.")


PROMPT_SYSTEM = "You compare two short statements. Pick the one with more vowels."
PROMPT_USER = "A: 'aeiou'\nB: 'xyz'"


# ============================================================
# Sync — one per provider that supports structured output
# ============================================================


class TestSyncStructuredOutput:
    """For each provider, run a structured-output call and assert the
    response is a valid _Verdict instance with pick='a' (the obvious
    answer; if the model can read, it'll pick a)."""

    def test_anthropic_forced_tool_use(self):
        result = query(
            "claude-haiku-4-5-20251001", PROMPT_USER, PROMPT_SYSTEM,
            output_model=_Verdict,
        )
        assert isinstance(result.content, _Verdict), \
            f"expected _Verdict, got {type(result.content).__name__}"
        assert result.content.pick == "a"
        assert len(result.content.reasoning) > 0
        assert result.input_tokens > 0
        assert result.output_tokens > 0

    def test_openai_responses_parse(self):
        result = query(
            "gpt-4.1-nano", PROMPT_USER, PROMPT_SYSTEM,
            output_model=_Verdict,
        )
        assert isinstance(result.content, _Verdict)
        assert result.content.pick == "a"
        assert result.input_tokens > 0

    def test_deepseek_json_object(self):
        result = query(
            "deepseek-chat", PROMPT_USER, PROMPT_SYSTEM,
            output_model=_Verdict,
        )
        assert isinstance(result.content, _Verdict)
        assert result.content.pick == "a"
        assert result.input_tokens > 0

    def test_gemini_response_schema(self):
        result = query(
            "gemini-2.5-flash-lite", PROMPT_USER, PROMPT_SYSTEM,
            output_model=_Verdict,
        )
        assert isinstance(result.content, _Verdict)
        assert result.content.pick == "a"
        assert result.input_tokens > 0


# ============================================================
# Async — one per provider
# ============================================================


class TestAsyncStructuredOutput:
    async def test_anthropic_forced_tool_use_async(self):
        result = await query_async(
            "claude-haiku-4-5-20251001", PROMPT_USER, PROMPT_SYSTEM,
            output_model=_Verdict,
        )
        assert isinstance(result.content, _Verdict)
        assert result.content.pick == "a"

    async def test_openai_responses_parse_async(self):
        result = await query_async(
            "gpt-4.1-nano", PROMPT_USER, PROMPT_SYSTEM,
            output_model=_Verdict,
        )
        assert isinstance(result.content, _Verdict)
        assert result.content.pick == "a"

    async def test_deepseek_async(self):
        result = await query_async(
            "deepseek-chat", PROMPT_USER, PROMPT_SYSTEM,
            output_model=_Verdict,
        )
        assert isinstance(result.content, _Verdict)
        assert result.content.pick == "a"

    async def test_gemini_async(self):
        result = await query_async(
            "gemini-2.5-flash-lite", PROMPT_USER, PROMPT_SYSTEM,
            output_model=_Verdict,
        )
        assert isinstance(result.content, _Verdict)
        assert result.content.pick == "a"


# ============================================================
# Real production model — PairwiseJudgment via Anthropic tool use
# ============================================================


class TestProductionShapeStructuredOutput:
    """Use the real PairwiseJudgment schema (the same one Stage 1 judges
    use in production) to verify the wire-level structured output works
    for our actual workload, not just toy schemas."""

    async def test_pairwise_judgment_via_anthropic(self):
        from owtn.evaluation.models import PairwiseJudgment

        # A trivial pairwise prompt — we don't care about the verdict,
        # only that the schema round-trips.
        sys_msg = (
            "Compare two simple concepts on each of these dimensions: "
            "novelty, grip, tension_architecture, emotional_depth, "
            "thematic_resonance, concept_coherence, generative_fertility, "
            "scope_calibration, indelibility. For each dimension, return "
            "one of: a_narrow, a_clear, a_decisive, b_narrow, b_clear, "
            "b_decisive, or tie. Also produce a 'reasoning' string."
        )
        user_msg = (
            "Concept A: A boy meets a fox.\n"
            "Concept B: A girl meets a wolf.\n\n"
            "Compare on every dimension."
        )
        result = await query_async(
            "claude-haiku-4-5-20251001", user_msg, sys_msg,
            output_model=PairwiseJudgment,
        )
        assert isinstance(result.content, PairwiseJudgment)
        # The 9 dimension fields all produced valid Vote literal values:
        for dim_name in [
            "novelty", "grip", "tension_architecture", "emotional_depth",
            "thematic_resonance", "concept_coherence", "generative_fertility",
            "scope_calibration", "indelibility",
        ]:
            v = getattr(result.content, dim_name)
            assert v in {"a_narrow", "a_clear", "a_decisive", "tie",
                         "b_narrow", "b_clear", "b_decisive"}, \
                f"{dim_name} returned invalid vote: {v!r}"
        assert len(result.content.reasoning) > 0
