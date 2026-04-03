"""Integration tests for the LLM client — makes real API calls.

Uses the cheapest model from each provider to minimize cost.
Requires API keys in .env.
"""

import pytest

from owtn.llm.query import query, query_async


pytestmark = pytest.mark.live_api

# Cheapest models per provider
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = "gpt-4.1-nano"
GEMINI_MODEL = "gemini-2.5-flash-lite"
DEEPSEEK_MODEL = "deepseek-chat"

SYSTEM_MSG = "You are a helpful assistant. Respond in one short sentence."
USER_MSG = "What is 2+2?"


class TestSyncQueries:
    """Synchronous query tests — one per provider."""

    def test_anthropic(self):
        result = query(ANTHROPIC_MODEL, USER_MSG, SYSTEM_MSG)
        assert result.content
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.cost > 0
        assert result.model_name == ANTHROPIC_MODEL

    def test_openai(self):
        result = query(OPENAI_MODEL, USER_MSG, SYSTEM_MSG)
        assert result.content
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.model_name == OPENAI_MODEL

    def test_gemini(self):
        result = query(GEMINI_MODEL, USER_MSG, SYSTEM_MSG)
        assert result.content
        assert result.input_tokens > 0
        assert result.output_tokens > 0

    def test_deepseek(self):
        result = query(DEEPSEEK_MODEL, USER_MSG, SYSTEM_MSG)
        assert result.content
        assert result.input_tokens > 0
        assert result.output_tokens > 0

    def test_openai_with_system_prefix(self):
        result = query(
            OPENAI_MODEL, USER_MSG, "Respond briefly.",
            system_prefix="You are a math tutor.",
        )
        assert result.content
        assert result.input_tokens > 0

    def test_deepseek_with_system_prefix(self):
        result = query(
            DEEPSEEK_MODEL, USER_MSG, "Respond briefly.",
            system_prefix="You are a math tutor.",
        )
        assert result.content


class TestAsyncQueries:
    """Async query tests — one per provider."""

    async def test_anthropic_async(self):
        result = await query_async(ANTHROPIC_MODEL, USER_MSG, SYSTEM_MSG)
        assert result.content
        assert result.input_tokens > 0
        assert result.cost > 0

    async def test_openai_async(self):
        result = await query_async(OPENAI_MODEL, USER_MSG, SYSTEM_MSG)
        assert result.content
        assert result.input_tokens > 0

    async def test_gemini_async(self):
        result = await query_async(GEMINI_MODEL, USER_MSG, SYSTEM_MSG)
        assert result.content
        assert result.input_tokens > 0

    async def test_deepseek_async(self):
        result = await query_async(DEEPSEEK_MODEL, USER_MSG, SYSTEM_MSG)
        assert result.content
        assert result.input_tokens > 0


class TestAnthropicPromptCaching:
    """Verify system_prefix produces cache_control content blocks.

    Anthropic prompt caching requires a minimum prefix size to activate.
    Haiku needs >2048 tokens in the cached prefix.
    """

    CACHE_PREFIX = (
        "You are an expert literary critic with deep knowledge of narrative "
        "structure, character development, emotional resonance, and stylistic "
        "innovation across centuries of world literature. You evaluate prose on "
        "precise craft elements including dialogue authenticity, sensory detail, "
        "pacing rhythm, metaphor originality, and thematic coherence. You draw "
        "from canonical and contemporary examples to anchor your evaluations. "
    ) * 80

    def test_prompt_cache_create_then_read(self):
        """First call creates cache, second call reads it."""
        suffix = "Be concise. Score 1-5."

        r1 = query(
            ANTHROPIC_MODEL,
            "Rate: 'Stars blinked overhead.'",
            suffix,
            system_prefix=self.CACHE_PREFIX,
        )
        assert r1.content
        assert r1.cache_creation_tokens > 0 or r1.cache_read_tokens > 0

        r2 = query(
            ANTHROPIC_MODEL,
            "Rate: 'The wind howled through empty streets.'",
            suffix,
            system_prefix=self.CACHE_PREFIX,
        )
        assert r2.content
        assert r2.cache_read_tokens > 0
