"""The Provider abstraction.

Every backend (Anthropic, OpenAI, Gemini, etc.) is a Provider class with:

- **`make_async_client()` / `make_sync_client()`**: build the SDK client once
  and cache it. Reusing the same client across calls preserves the httpx
  connection pool — meaningful latency win when fanning out judges.

- **`query_async(...)` / `query(...)`**: the actual call. Knows how to
  invoke its SDK natively, including structured output (tool use for
  Anthropic, response_format for OpenAI/DeepSeek, response_schema for
  Gemini). No instructor.

- **`build_call_kwargs(api_model, requested)`**: rewrite generic requested
  kwargs (temperature, max_tokens, reasoning_effort, top_p/k/min_p) into
  this provider's API shape. The single home for "Anthropic uses
  max_tokens, OpenAI uses max_output_tokens, DeepSeek wants
  extra_body.thinking" knowledge.

- **`make_call_kwargs(api_model, requested)`** is the shared helper:
  resolves reasoning effort + temperature against the model's flags
  (is_reasoning_model, requires_reasoning, has_fixed_temperature) before
  the per-provider rewriter sees them.

System-prefix handling is per-provider — Anthropic uses cache_control
content blocks, everyone else just concatenates.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Type

from pydantic import BaseModel

from ..result import QueryResult
from .pricing import (
    has_fixed_temperature,
    is_reasoning_model,
    requires_reasoning,
)


TIMEOUT = 600  # 10 minutes — covers slow reasoning models on first token

# Reasoning-effort → token budget mapping. Used by every provider that
# accepts an explicit thinking budget (Anthropic, Gemini). OpenAI just
# passes the effort label string through; DeepSeek toggles a flag.
THINKING_TOKENS = {
    "min": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "max": 16384,
}


class Provider(Protocol):
    """The contract every provider implements. Stateful (caches its SDK
    client) but otherwise pure."""

    name: str

    async def query_async(
        self,
        *,
        model: str,
        msg: str,
        system_msg: str,
        msg_history: list[dict],
        system_prefix: Optional[str],
        output_model: Optional[Type[BaseModel]],
        kwargs: dict,
    ) -> QueryResult: ...

    def query(
        self,
        *,
        model: str,
        msg: str,
        system_msg: str,
        msg_history: list[dict],
        system_prefix: Optional[str],
        output_model: Optional[Type[BaseModel]],
        kwargs: dict,
    ) -> QueryResult: ...

    def build_call_kwargs(self, *, api_model: str, requested: Mapping[str, Any]) -> dict: ...


def resolve_effort(api_model: str, requested_effort: str) -> str:
    """Coerce reasoning_effort against the model's flags.

    Non-reasoning models always return "disabled". Reasoning models that
    require reasoning (some OpenRouter-hosted ones) coerce "disabled" up
    to "low" — they reject explicit-disabled requests.
    """
    if not is_reasoning_model(api_model):
        return "disabled"
    if requested_effort == "disabled" and requires_reasoning(api_model):
        return "low"
    return requested_effort


def resolve_temperature(
    api_model: str, requested_temp: Optional[float], effort: str
) -> Optional[float]:
    """Force temp=1.0 when the model has think_temp_fixed=1 AND reasoning
    is active. Anthropic extended-thinking and OpenAI reasoning models
    rejection-fail on any other temperature."""
    if has_fixed_temperature(api_model) and effort != "disabled":
        return 1.0
    return requested_temp
