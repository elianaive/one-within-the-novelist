"""Anthropic + Bedrock providers.

Native structured output via forced tool use. Each `output_model` is
declared as a tool whose `input_schema` is the Pydantic JSON schema;
`tool_choice` forces the model to emit a tool call in that exact shape.
No instructor injection — Anthropic's tool-use prompt internals are the
only "schema-related text" the model sees, and that's the API contract,
not our prompt.

system_prefix is sent as a separate cache_control content block so
prompt caching activates (saves ~80% input cost on repeated prefixes).

Both AnthropicProvider and BedrockProvider share the call shape; only
the SDK client construction differs (one uses ANTHROPIC_API_KEY, the
other uses AWS credentials).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional, Type

import anthropic
import backoff
from pydantic import BaseModel

from ..result import QueryResult
from .base import THINKING_TOKENS, TIMEOUT, resolve_effort, resolve_temperature
from .pricing import calculate_cost, is_reasoning_model

logger = logging.getLogger(__name__)

MAX_TRIES = 20
MAX_VALUE = 20
MAX_TIME = 600
ANTHROPIC_DEFAULT_MAX_TOKENS = 16384
ANTHROPIC_MAX_TOKENS_CEILING = 64000

_RETRY_EXCEPTIONS = (
    anthropic.APIConnectionError,
    anthropic.APIStatusError,
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
)


def _resolve_thinking_budget(
    *,
    api_model: str,
    effort: str,
    explicit_tokens: Optional[int],
    ceiling: int,
) -> Optional[int]:
    """Decide the thinking budget for an Anthropic call.

    Returns None when thinking should be off (non-reasoning model, or
    effort=disabled and no explicit override). Otherwise returns an int,
    clamped below `ceiling` (Anthropic rejects budget >= max_tokens).
    """
    if not is_reasoning_model(api_model):
        return None
    if explicit_tokens is not None:
        return explicit_tokens if explicit_tokens < ceiling else 1024
    if effort != "disabled":
        t = THINKING_TOKENS[effort]
        return t if t < ceiling else 1024
    return None


def _build_system(system_msg: str, system_prefix: Optional[str]):
    """Build the `system` parameter. With a prefix, returns content blocks
    with cache_control on the prefix block; without, returns the plain string.
    """
    if system_prefix:
        blocks = [
            {"type": "text", "text": system_prefix, "cache_control": {"type": "ephemeral"}}
        ]
        if system_msg:
            blocks.append({"type": "text", "text": system_msg})
        return blocks
    return system_msg


def get_anthropic_costs(response, model: str) -> dict:
    """Token counts and dollar costs from a messages.create response."""
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    input_cost, output_cost = calculate_cost(model, input_tokens, output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": 0,  # Anthropic doesn't expose a separate thinking-token count today
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cost": input_cost + output_cost,
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
    }


def _backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Anthropic - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


def _structured_output_tool(output_model: Type[BaseModel]) -> dict:
    """Build the forced-tool-use payload for a Pydantic output model.

    The tool name is the class name; the schema is the model's JSON schema
    (Anthropic accepts Draft-07 with $defs/$ref/oneOf, so no inlining needed).
    `description` falls back to a generic message if the model lacks a docstring.
    """
    return {
        "name": output_model.__name__,
        "description": (output_model.__doc__ or f"Return a {output_model.__name__}").strip(),
        "input_schema": output_model.model_json_schema(),
    }


def _extract_tool_input(response, output_model: Type[BaseModel]) -> BaseModel:
    """Find the tool_use block in the response and parse its input."""
    for block in response.content:
        if block.type == "tool_use" and block.name == output_model.__name__:
            return output_model.model_validate(block.input)
    raise ValueError(
        f"Anthropic structured-output response missing tool_use for {output_model.__name__}"
    )


def _extract_text(response) -> tuple[str, str]:
    """Return (content, thought). Anthropic puts thinking blocks before
    text blocks when extended thinking is enabled."""
    thought = ""
    content = ""
    for block in response.content:
        if block.type == "thinking":
            thought = block.thinking
        elif block.type == "text":
            content = block.text
    return content, thought


class AnthropicProvider:
    """Anthropic provider. Singleton clients per (sync/async)."""

    name = "anthropic"

    def __init__(self) -> None:
        self._sync_client: Optional[anthropic.Anthropic] = None
        self._async_client: Optional[anthropic.AsyncAnthropic] = None

    def _make_sync_client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(timeout=TIMEOUT)

    def _make_async_client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(timeout=TIMEOUT)

    def _sync(self) -> anthropic.Anthropic:
        if self._sync_client is None:
            self._sync_client = self._make_sync_client()
        return self._sync_client

    def _async(self) -> anthropic.AsyncAnthropic:
        if self._async_client is None:
            self._async_client = self._make_async_client()
        return self._async_client

    def build_call_kwargs(self, *, api_model: str, requested: Mapping[str, Any]) -> dict:
        """Anthropic shape: max_tokens (capped at 64k), thinking dict for
        extended-thinking models, top_p/top_k forbidden under thinking.

        Thinking budget resolution:
          - explicit `thinking_tokens` int wins (preferred, fine-grained)
          - else falls back to THINKING_TOKENS[reasoning_effort] for back-compat
          - else thinking is off
        """
        effort = resolve_effort(api_model, requested.get("reasoning_effort", "disabled"))
        out: dict = {}
        if (v := requested.get("max_tokens")) is not None:
            out["max_tokens"] = min(v, ANTHROPIC_MAX_TOKENS_CEILING)

        temp = resolve_temperature(api_model, requested.get("temperature"), effort)
        if temp is not None:
            out["temperature"] = temp

        budget = _resolve_thinking_budget(
            api_model=api_model,
            effort=effort,
            explicit_tokens=requested.get("thinking_tokens"),
            ceiling=out.get("max_tokens", ANTHROPIC_DEFAULT_MAX_TOKENS),
        )
        if budget is not None:
            out["thinking"] = {"type": "enabled", "budget_tokens": budget}
        else:
            # top_p/top_k are forbidden when thinking is enabled.
            if (v := requested.get("top_p")) is not None:
                out["top_p"] = v
            if (v := requested.get("top_k")) is not None:
                out["top_k"] = v
        return out

    @backoff.on_exception(
        backoff.expo, _RETRY_EXCEPTIONS,
        max_tries=MAX_TRIES, max_value=MAX_VALUE, max_time=MAX_TIME,
        on_backoff=_backoff_handler,
    )
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
        client: Optional[anthropic.Anthropic] = None,
    ) -> QueryResult:
        client = client or self._sync()
        new_msg_history = msg_history + [
            {"role": "user", "content": [{"type": "text", "text": msg}]}
        ]
        call_kwargs = self._prepare_call_kwargs(kwargs, output_model)
        response = client.messages.create(
            model=model,
            system=_build_system(system_msg, system_prefix),
            messages=new_msg_history,
            **call_kwargs,
        )
        return self._build_result(
            response=response, model=model, msg=msg, system_msg=system_msg,
            new_msg_history=new_msg_history, output_model=output_model, kwargs=kwargs,
        )

    @backoff.on_exception(
        backoff.expo, _RETRY_EXCEPTIONS,
        max_tries=MAX_TRIES, max_value=MAX_VALUE, max_time=MAX_TIME,
        on_backoff=_backoff_handler,
    )
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
        client: Optional[anthropic.AsyncAnthropic] = None,
    ) -> QueryResult:
        client = client or self._async()
        new_msg_history = msg_history + [
            {"role": "user", "content": [{"type": "text", "text": msg}]}
        ]
        call_kwargs = self._prepare_call_kwargs(kwargs, output_model)
        response = await client.messages.create(
            model=model,
            system=_build_system(system_msg, system_prefix),
            messages=new_msg_history,
            **call_kwargs,
        )
        return self._build_result(
            response=response, model=model, msg=msg, system_msg=system_msg,
            new_msg_history=new_msg_history, output_model=output_model, kwargs=kwargs,
        )

    def _prepare_call_kwargs(
        self, kwargs: dict, output_model: Optional[Type[BaseModel]]
    ) -> dict:
        """Inject the forced-tool-use payload when output_model is set.

        Forced tool use can't combine with extended thinking — drop the
        `thinking` kwarg so callers don't accidentally request both."""
        out = dict(kwargs)
        out.setdefault("max_tokens", ANTHROPIC_DEFAULT_MAX_TOKENS)
        if output_model is not None:
            out.pop("thinking", None)
            out["tools"] = [_structured_output_tool(output_model)]
            out["tool_choice"] = {"type": "tool", "name": output_model.__name__}
        return out

    def _build_result(
        self,
        *,
        response,
        model: str,
        msg: str,
        system_msg: str,
        new_msg_history: list[dict],
        output_model: Optional[Type[BaseModel]],
        kwargs: dict,
    ) -> QueryResult:
        if output_model is not None:
            content = _extract_tool_input(response, output_model)
            thought = ""
            new_msg_history.append(
                {"role": "assistant", "content": [{"type": "text", "text": str(content)}]}
            )
        else:
            content, thought = _extract_text(response)
            new_msg_history.append(
                {"role": "assistant", "content": [{"type": "text", "text": content}]}
            )
        cost_results = get_anthropic_costs(response, model)
        return QueryResult(
            content=content,
            msg=msg,
            system_msg=system_msg,
            new_msg_history=new_msg_history,
            model_name=model,
            kwargs=kwargs,
            **cost_results,
            thought=thought,
        )


    @backoff.on_exception(
        backoff.expo, _RETRY_EXCEPTIONS,
        max_tries=MAX_TRIES, max_value=MAX_VALUE, max_time=MAX_TIME,
        on_backoff=_backoff_handler,
    )
    async def query_async_with_tools(
        self,
        *,
        model: str,
        msg: str,
        system_msg: str,
        msg_history: list[dict],
        system_prefix: Optional[str],
        tools: list[dict],
        dispatch: Any,  # async (name, params) -> str
        max_iters: int = 10,
        kwargs: dict,
        client: Optional[anthropic.AsyncAnthropic] = None,
    ) -> QueryResult:
        """Tool-use loop. Sends tool schemas; on tool_use blocks, runs the
        dispatcher for each, appends tool_result blocks, re-calls until the
        model responds without tool_use or `max_iters` is hit.

        Returns a QueryResult whose `content` is the final assistant text,
        whose token/cost fields sum across all loop iterations, and whose
        `new_msg_history` is the full multi-turn transcript including all
        tool_use/tool_result rounds.
        """
        client = client or self._async()
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("max_tokens", ANTHROPIC_DEFAULT_MAX_TOKENS)
        # Translate neutral {name, description, parameters} → Anthropic's
        # {name, description, input_schema}. Allows the orchestrator to stay
        # provider-agnostic.
        call_kwargs["tools"] = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema") or t.get("parameters") or {"type": "object"},
            }
            for t in tools
        ]

        history: list[dict] = list(msg_history) + [
            {"role": "user", "content": [{"type": "text", "text": msg}]}
        ]

        totals = {
            "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0,
            "input_cost": 0.0, "output_cost": 0.0, "cost": 0.0,
            "cache_read_tokens": 0, "cache_creation_tokens": 0,
        }
        last_text = ""
        last_thought = ""

        for _ in range(max_iters):
            response = await client.messages.create(
                model=model,
                system=_build_system(system_msg, system_prefix),
                messages=list(history),
                **call_kwargs,
            )
            costs = get_anthropic_costs(response, model)
            for k in totals:
                totals[k] += costs[k]

            assistant_blocks = [
                _block_to_dict(b) for b in response.content
            ]
            history.append({"role": "assistant", "content": assistant_blocks})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                last_text, last_thought = _extract_text(response)
                break

            tool_results = []
            for use in tool_uses:
                try:
                    result_str = await dispatch(use.name, dict(use.input))
                except Exception as e:
                    logger.warning("tool %s dispatch failed: %s", use.name, e)
                    result_str = f"tool error: {e}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": use.id,
                    "content": result_str,
                })
            history.append({"role": "user", "content": tool_results})
        else:
            logger.warning(
                "tool-use loop hit max_iters=%d without final text response",
                max_iters,
            )
            last_text, last_thought = _extract_text(response)

        return QueryResult(
            content=last_text,
            msg=msg,
            system_msg=system_msg,
            new_msg_history=history,
            model_name=model,
            kwargs=kwargs,
            input_tokens=totals["input_tokens"],
            output_tokens=totals["output_tokens"],
            thinking_tokens=totals["thinking_tokens"],
            cost=totals["cost"],
            input_cost=totals["input_cost"],
            output_cost=totals["output_cost"],
            cache_read_tokens=totals["cache_read_tokens"],
            cache_creation_tokens=totals["cache_creation_tokens"],
            thought=last_thought,
        )


def _block_to_dict(block: Any) -> dict:
    """Serialize an Anthropic content block back to its API-shape dict so
    it can be replayed in the next call's message history."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": getattr(block, "signature", "")}
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": dict(block.input),
        }
    return {"type": block.type}


class BedrockProvider(AnthropicProvider):
    """Bedrock-hosted Claude. Same call shape, different SDK client."""

    name = "bedrock"

    def _make_sync_client(self) -> anthropic.AnthropicBedrock:
        return anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
            timeout=TIMEOUT,
        )

    def _make_async_client(self) -> anthropic.AsyncAnthropicBedrock:
        return anthropic.AsyncAnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
            timeout=TIMEOUT,
        )


# Module-level singletons.
ANTHROPIC = AnthropicProvider()
BEDROCK = BedrockProvider()


# Back-compat shim: existing tests import these names from this module.
# Keep them as thin wrappers around the singleton until the test refactor.
def query_anthropic(client, model, msg, system_msg, msg_history, output_model,
                    model_posteriors=None, **kwargs) -> QueryResult:
    """Legacy free-function form. Routes to AnthropicProvider.query, passing
    the caller-supplied client through. Kept for backwards compatibility."""
    system_prefix = kwargs.pop("system_prefix", None)
    result = ANTHROPIC.query(
        model=model, msg=msg, system_msg=system_msg, msg_history=msg_history,
        system_prefix=system_prefix, output_model=output_model, kwargs=kwargs,
        client=client,
    )
    if model_posteriors is not None:
        # The old shape included model_posteriors; reconstruct the result with it.
        from dataclasses import replace
        return replace(result, model_posteriors=model_posteriors)
    return result


async def query_anthropic_async(client, model, msg, system_msg, msg_history, output_model,
                                model_posteriors=None, **kwargs) -> QueryResult:
    system_prefix = kwargs.pop("system_prefix", None)
    result = await ANTHROPIC.query_async(
        model=model, msg=msg, system_msg=system_msg, msg_history=msg_history,
        system_prefix=system_prefix, output_model=output_model, kwargs=kwargs,
        client=client,
    )
    if model_posteriors is not None:
        from dataclasses import replace
        return replace(result, model_posteriors=model_posteriors)
    return result
