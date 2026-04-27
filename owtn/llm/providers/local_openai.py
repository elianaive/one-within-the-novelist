"""Local OpenAI-compatible provider (vLLM, llama.cpp, etc.).

Uses the OpenAI SDK's chat-completions endpoint against a user-provided
base_url. No structured output (those backends rarely implement
response_format=json_object reliably).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional, Type

import backoff
import openai
from pydantic import BaseModel

from ..result import QueryResult
from .base import TIMEOUT, resolve_effort, resolve_temperature
from .model_resolver import ResolvedModel
from .pricing import calculate_cost, model_exists

logger = logging.getLogger(__name__)

MAX_TRIES = 20
MAX_VALUE = 20
MAX_TIME = 600

_RETRY_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APIStatusError,
    openai.RateLimitError,
    openai.APITimeoutError,
)


def _backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"Local OpenAI - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


def _extract_costs(model: str, in_tokens: int, all_out_tokens: int) -> tuple[float, float]:
    if model_exists(model):
        return calculate_cost(model, in_tokens, all_out_tokens)
    return 0.0, 0.0


def _extract_usage(response) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    in_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    all_out_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    completion_details = getattr(usage, "completion_tokens_details", None)
    thinking_tokens = 0
    if completion_details is not None:
        thinking_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
    return in_tokens, all_out_tokens, thinking_tokens


class LocalOpenAIProvider:
    """Self-hosted OpenAI-compatible backends. Per-call base_url since
    different model identifiers may resolve to different upstream URLs."""

    name = "local_openai"

    def __init__(self) -> None:
        self._sync_clients: dict[str, openai.OpenAI] = {}
        self._async_clients: dict[str, openai.AsyncOpenAI] = {}

    def _sync(self, base_url: str) -> openai.OpenAI:
        if base_url not in self._sync_clients:
            self._sync_clients[base_url] = openai.OpenAI(
                api_key=os.getenv("LOCAL_OPENAI_API_KEY", "local"),
                base_url=base_url,
                timeout=TIMEOUT,
            )
        return self._sync_clients[base_url]

    def _async(self, base_url: str) -> openai.AsyncOpenAI:
        if base_url not in self._async_clients:
            self._async_clients[base_url] = openai.AsyncOpenAI(
                api_key=os.getenv("LOCAL_OPENAI_API_KEY", "local"),
                base_url=base_url,
                timeout=TIMEOUT,
            )
        return self._async_clients[base_url]

    def build_call_kwargs(self, *, api_model: str, requested: Mapping[str, Any]) -> dict:
        """Local backends have no provider-specific quirks beyond what was
        passed in. We just forward the standard kwargs."""
        effort = resolve_effort(api_model, requested.get("reasoning_effort", "disabled"))
        out: dict = {"max_tokens": requested.get("max_tokens", 4096)}
        temp = resolve_temperature(api_model, requested.get("temperature"), effort)
        if temp is not None:
            out["temperature"] = temp
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
        base_url: Optional[str] = None,
    ) -> QueryResult:
        if output_model is not None:
            raise NotImplementedError(
                "Structured output is not supported for local OpenAI-compatible backends."
            )
        client = client or self._sync(base_url or "")
        merged_system = _merge_prefix(system_msg, system_prefix)
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": merged_system}, *new_msg_history],
            n=1,
            **kwargs,
        )
        return self._build_result(response, model, msg, system_msg, new_msg_history, kwargs)

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
        base_url: Optional[str] = None,
    ) -> QueryResult:
        if output_model is not None:
            raise NotImplementedError(
                "Structured output is not supported for local OpenAI-compatible backends."
            )
        client = client or self._async(base_url or "")
        merged_system = _merge_prefix(system_msg, system_prefix)
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": merged_system}, *new_msg_history],
            n=1,
            **kwargs,
        )
        return self._build_result(response, model, msg, system_msg, new_msg_history, kwargs)

    def _build_result(self, response, model, msg, system_msg, new_msg_history, kwargs) -> QueryResult:
        content = response.choices[0].message.content or ""
        thought = getattr(response.choices[0].message, "reasoning_content", "") or ""
        new_msg_history.append({"role": "assistant", "content": content})

        in_tokens, all_out_tokens, thinking_tokens = _extract_usage(response)
        out_tokens = max(all_out_tokens - thinking_tokens, 0)
        input_cost, output_cost = _extract_costs(model, in_tokens, all_out_tokens)
        return QueryResult(
            content=content,
            msg=msg,
            system_msg=system_msg,
            new_msg_history=new_msg_history,
            model_name=model,
            kwargs=kwargs,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            thinking_tokens=thinking_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            cost=input_cost + output_cost,
            thought=thought,
        )


def _merge_prefix(system_msg: str, system_prefix: Optional[str]) -> str:
    if system_prefix:
        return system_prefix + "\n\n" + system_msg
    return system_msg


# Module-level singleton.
LOCAL_OPENAI = LocalOpenAIProvider()


# Back-compat shims.
def query_local_openai(client, model, msg, system_msg, msg_history, output_model,
                       model_posteriors=None, **kwargs) -> QueryResult:
    system_prefix = kwargs.pop("system_prefix", None)
    result = LOCAL_OPENAI.query(
        model=model, msg=msg, system_msg=system_msg, msg_history=msg_history,
        system_prefix=system_prefix, output_model=output_model, kwargs=kwargs,
        client=client,
    )
    if model_posteriors is not None:
        from dataclasses import replace
        return replace(result, model_posteriors=model_posteriors)
    return result


async def query_local_openai_async(client, model, msg, system_msg, msg_history, output_model,
                                   model_posteriors=None, **kwargs) -> QueryResult:
    system_prefix = kwargs.pop("system_prefix", None)
    result = await LOCAL_OPENAI.query_async(
        model=model, msg=msg, system_msg=system_msg, msg_history=msg_history,
        system_prefix=system_prefix, output_model=output_model, kwargs=kwargs,
        client=client,
    )
    if model_posteriors is not None:
        from dataclasses import replace
        return replace(result, model_posteriors=model_posteriors)
    return result
