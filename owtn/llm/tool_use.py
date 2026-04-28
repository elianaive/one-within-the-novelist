"""Tool-use loop on top of provider tool-call APIs.

`query_async_with_tools` resolves the model's provider, hands off to that
provider's tool-use loop method, and returns a single QueryResult with
cumulative cost. The provider is responsible for the message-history
shape during dispatch (Anthropic uses tool_use/tool_result content blocks;
OpenAI uses tool_calls/tool message roles); this module is the dispatch
seam.

Currently implemented for: anthropic, bedrock, deepseek.
Stub for everything else — extend per-provider as Stage 3 testing requires.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Mapping

from .api import _log
from .providers import PROVIDERS
from .providers.model_resolver import resolve_model_backend
from .result import QueryResult

logger = logging.getLogger(__name__)


ToolDispatch = Callable[[str, Mapping[str, Any]], Awaitable[str]]
"""Async callback: (tool_name, params) -> stringified tool result."""


_PROVIDERS_WITH_TOOL_USE = ("anthropic", "bedrock", "deepseek")


async def query_async_with_tools(
    *,
    model_name: str,
    msg: str,
    system_msg: str,
    tools: list[dict],
    dispatch: ToolDispatch,
    msg_history: list[dict] | None = None,
    max_iters: int = 10,
    **kwargs: Any,
) -> QueryResult:
    """Run a multi-turn tool-use loop against `model_name`.

    `tools` is a list of provider-shape tool schemas (name, description,
    input_schema) — the orchestrator typically obtains these from
    `ToolRegistry.schemas_for(agent.tools, phase.name)`. `dispatch` runs
    one tool-call at a time and returns the string result the model sees
    in the next turn.

    Returns a single QueryResult summing cost across all iterations and
    carrying the full multi-turn `new_msg_history`.

    Raises NotImplementedError for providers without tool-use support yet
    (extend `owtn/llm/providers/<provider>.py` then add to
    `_PROVIDERS_WITH_TOOL_USE`).
    """
    resolved = resolve_model_backend(model_name)
    if resolved.provider not in _PROVIDERS_WITH_TOOL_USE:
        raise NotImplementedError(
            f"tool-use loop not implemented for provider {resolved.provider!r}; "
            f"available: {_PROVIDERS_WITH_TOOL_USE}"
        )

    provider = PROVIDERS[resolved.provider]
    system_prefix = kwargs.pop("system_prefix", None)

    t0 = time.perf_counter()
    result = await provider.query_async_with_tools(
        model=resolved.api_model_name,
        msg=msg,
        system_msg=system_msg,
        msg_history=list(msg_history or []),
        system_prefix=system_prefix,
        tools=tools,
        dispatch=dispatch,
        max_iters=max_iters,
        kwargs=kwargs,
    )
    _log(result, resolved.provider, msg, system_msg, time.perf_counter() - t0, kwargs)
    return result
