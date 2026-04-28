"""DeepSeek provider.

Uses the OpenAI SDK against DeepSeek's chat-completions endpoint (DeepSeek
doesn't expose the Responses API). For structured output we ask for
`response_format={"type":"json_object"}` and inject our own schema-in-prompt
instruction — we own the wording, no instructor injection. Parse failures
go through the shared recovery module.

Reasoning models (deepseek-v4-pro, deepseek-reasoner, etc.) are toggled
via `extra_body.thinking` and `reasoning_effort` — DeepSeek's chat API
accepts these as JSON-body extras.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping, Optional, Type

import backoff
import openai
from pydantic import BaseModel, ValidationError

from ..recovery import recover_from_validation_error
from ..result import QueryResult
from .base import TIMEOUT, resolve_effort, resolve_temperature
from .pricing import calculate_cost, is_reasoning_model

logger = logging.getLogger(__name__)

MAX_TRIES = 20
MAX_VALUE = 20
MAX_TIME = 600
PARSE_RETRIES = 3

# DeepSeek's max_tokens covers reasoning + visible output (unlike OpenAI's
# max_output_tokens which is visible-only). 32K covers high-effort reasoning
# (~16K) plus a substantial structured response.
_REASONING_BUDGET_FLOOR = 32768

_RETRY_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APIStatusError,
    openai.RateLimitError,
    openai.APITimeoutError,
)


def _backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"DeepSeek - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


def _schema_instruction(output_model: Type[BaseModel]) -> str:
    """The schema-in-prompt instruction we control. Concise and direct —
    not the verbose 'As a genius expert' that instructor injected."""
    schema_json = json.dumps(output_model.model_json_schema(), indent=2, ensure_ascii=False)
    return (
        f"Respond with a single valid JSON object that conforms to this schema. "
        f"Output the JSON object only, no other text or markdown fencing.\n\n"
        f"Schema:\n{schema_json}"
    )


def _parse_json_with_recovery(
    raw: str, output_model: Type[BaseModel], model: str
) -> BaseModel:
    """Try to parse `raw` as the given output_model. Falls back through the
    shared recovery helpers (trailing-garbage strip, key normalization)."""
    try:
        return output_model.model_validate_json(raw)
    except ValidationError as e:
        recovered = recover_from_validation_error(e, output_model, model)
        if recovered is not None:
            return recovered
        raise


def _get_costs(response, model: str) -> dict:
    in_tokens = response.usage.prompt_tokens
    all_out_tokens = response.usage.completion_tokens
    try:
        thinking_tokens = response.usage.completion_tokens_details.reasoning_tokens
    except Exception:
        thinking_tokens = 0
    out_tokens = all_out_tokens - thinking_tokens
    # Prefer prompt_tokens_details.cached_tokens; fall back to DeepSeek's
    # legacy prompt_cache_hit_tokens. We always use the DeepSeek total to
    # keep cache-hit pricing accurate (cached_input_tokens=… below).
    details = getattr(response.usage, "prompt_tokens_details", None)
    cached_tokens = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    if not cached_tokens:
        cached_tokens = getattr(response.usage, "prompt_cache_hit_tokens", 0) or 0
    input_cost, output_cost = calculate_cost(
        model, in_tokens, all_out_tokens, cached_input_tokens=cached_tokens
    )
    return {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "thinking_tokens": thinking_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cost": input_cost + output_cost,
        "cache_read_tokens": cached_tokens,
    }


class DeepSeekProvider:
    """DeepSeek provider — OpenAI SDK against the deepseek.com endpoint."""

    name = "deepseek"

    def __init__(self) -> None:
        self._sync_client: Optional[openai.OpenAI] = None
        self._async_client: Optional[openai.AsyncOpenAI] = None

    def _make_sync_client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=TIMEOUT,
        )

    def _make_async_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=TIMEOUT,
        )

    def _sync(self) -> openai.OpenAI:
        if self._sync_client is None:
            self._sync_client = self._make_sync_client()
        return self._sync_client

    def _async(self) -> openai.AsyncOpenAI:
        if self._async_client is None:
            self._async_client = self._make_async_client()
        return self._async_client

    def build_call_kwargs(self, *, api_model: str, requested: Mapping[str, Any]) -> dict:
        """DeepSeek shape: max_tokens (legacy chat-completions name) covers
        reasoning + output, so reasoning models get a generous floor.
        Thinking toggle via extra_body. top_p/top_k supported."""
        effort = resolve_effort(api_model, requested.get("reasoning_effort", "disabled"))
        out: dict = {}
        if (v := requested.get("max_tokens")) is not None:
            out["max_tokens"] = v

        temp = resolve_temperature(api_model, requested.get("temperature"), effort)
        if temp is not None:
            out["temperature"] = temp

        if is_reasoning_model(api_model):
            if effort == "disabled":
                out["extra_body"] = {"thinking": {"type": "disabled"}}
            else:
                out["extra_body"] = {"thinking": {"type": "enabled"}}
                out["reasoning_effort"] = effort
                out["max_tokens"] = max(out.get("max_tokens") or 0, _REASONING_BUDGET_FLOOR)

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
        client: Optional[openai.OpenAI] = None,
    ) -> QueryResult:
        client = client or self._sync()
        merged_system = _merge_prefix(system_msg, system_prefix, output_model)
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        messages = [{"role": "system", "content": merged_system}, *new_msg_history]

        call_kwargs = dict(kwargs)
        if output_model is not None:
            call_kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(PARSE_RETRIES if output_model else 1):
            response = client.chat.completions.create(
                model=model, messages=messages, n=1, stop=None, **call_kwargs,
            )
            raw_content = response.choices[0].message.content
            if output_model is not None:
                try:
                    content = _parse_json_with_recovery(raw_content, output_model, model)
                    break
                except ValidationError:
                    if attempt == PARSE_RETRIES - 1:
                        raise
                    logger.warning(
                        "DeepSeek parse failed for %s (attempt %d/%d); retrying.",
                        model, attempt + 1, PARSE_RETRIES,
                    )
            else:
                content = raw_content
                break

        thought = getattr(response.choices[0].message, "reasoning_content", "") or ""
        new_msg_history.append(
            {"role": "assistant", "content": str(content) if output_model else content}
        )
        return QueryResult(
            content=content, msg=msg, system_msg=system_msg,
            new_msg_history=new_msg_history, model_name=model, kwargs=kwargs,
            **_get_costs(response, model), thought=thought,
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
        client: Optional[openai.AsyncOpenAI] = None,
    ) -> QueryResult:
        client = client or self._async()
        merged_system = _merge_prefix(system_msg, system_prefix, output_model)
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        messages = [{"role": "system", "content": merged_system}, *new_msg_history]

        call_kwargs = dict(kwargs)
        if output_model is not None:
            call_kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(PARSE_RETRIES if output_model else 1):
            response = await client.chat.completions.create(
                model=model, messages=messages, n=1, stop=None, **call_kwargs,
            )
            raw_content = response.choices[0].message.content
            if output_model is not None:
                try:
                    content = _parse_json_with_recovery(raw_content, output_model, model)
                    break
                except ValidationError:
                    if attempt == PARSE_RETRIES - 1:
                        raise
                    logger.warning(
                        "DeepSeek parse failed for %s (attempt %d/%d); retrying.",
                        model, attempt + 1, PARSE_RETRIES,
                    )
            else:
                content = raw_content
                break

        thought = getattr(response.choices[0].message, "reasoning_content", "") or ""
        new_msg_history.append(
            {"role": "assistant", "content": str(content) if output_model else content}
        )
        return QueryResult(
            content=content, msg=msg, system_msg=system_msg,
            new_msg_history=new_msg_history, model_name=model, kwargs=kwargs,
            **_get_costs(response, model), thought=thought,
        )


def _merge_prefix(
    system_msg: str, system_prefix: Optional[str], output_model: Optional[Type[BaseModel]]
) -> str:
    """Concatenate prefix + system_msg + (schema instruction if structured).

    DeepSeek doesn't have a cache_control content-block API; matching string
    prefixes still get cached server-side. Schema instruction goes last so
    the model sees it just before producing output.
    """
    parts = []
    if system_prefix:
        parts.append(system_prefix)
    if system_msg:
        parts.append(system_msg)
    if output_model is not None:
        parts.append(_schema_instruction(output_model))
    return "\n\n".join(parts)


# Module-level singleton.
DEEPSEEK = DeepSeekProvider()


# Back-compat shims.
def query_deepseek(client, model, msg, system_msg, msg_history, output_model,
                   model_posteriors=None, **kwargs) -> QueryResult:
    system_prefix = kwargs.pop("system_prefix", None)
    result = DEEPSEEK.query(
        model=model, msg=msg, system_msg=system_msg, msg_history=msg_history,
        system_prefix=system_prefix, output_model=output_model, kwargs=kwargs,
        client=client,
    )
    if model_posteriors is not None:
        from dataclasses import replace
        return replace(result, model_posteriors=model_posteriors)
    return result


async def query_deepseek_async(client, model, msg, system_msg, msg_history, output_model,
                               model_posteriors=None, **kwargs) -> QueryResult:
    system_prefix = kwargs.pop("system_prefix", None)
    result = await DEEPSEEK.query_async(
        model=model, msg=msg, system_msg=system_msg, msg_history=msg_history,
        system_prefix=system_prefix, output_model=output_model, kwargs=kwargs,
        client=client,
    )
    if model_posteriors is not None:
        from dataclasses import replace
        return replace(result, model_posteriors=model_posteriors)
    return result
