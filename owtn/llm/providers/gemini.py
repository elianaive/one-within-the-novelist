"""Gemini provider.

Native structured output via `response_mime_type="application/json"` +
`response_schema=<pydantic_model>`. The Google GenAI SDK accepts a Pydantic
class directly as the schema; the response is JSON text we parse into the
model.

Caveats: Gemini's schema parser doesn't handle some Pydantic constructs
(unions with $defs, certain Optional patterns). For models that hit those
limits, callers should drop output_model and parse the JSON themselves.
None of our current consumers route discriminated-union outputs through
Gemini today.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional, Type, cast

import backoff
from google import genai
from google.genai import types
from pydantic import BaseModel

from ..result import QueryResult
from .base import THINKING_TOKENS, resolve_effort, resolve_temperature
from .pricing import calculate_cost, is_reasoning_model

logger = logging.getLogger(__name__)

MAX_TRIES = 20
MAX_VALUE = 20
MAX_TIME = 600


def _build_thinking_config(thinking_budget: int):
    """Build Gemini ThinkingConfig across SDK versions."""
    model_fields = getattr(types.ThinkingConfig, "model_fields", {})
    config_kwargs: dict[str, object] = {"include_thoughts": True}
    if "thinking_budget" in model_fields:
        config_kwargs["thinking_budget"] = int(thinking_budget)
    elif "thinkingBudget" in model_fields:
        config_kwargs["thinkingBudget"] = int(thinking_budget)
    return cast(Any, types.ThinkingConfig)(**config_kwargs)


def _build_afc_config():
    """Disable automatic function calling without SDK warnings."""
    model_fields = getattr(types.AutomaticFunctionCallingConfig, "model_fields", {})
    config_kwargs: dict[str, object] = {"disable": True}
    if "maximum_remote_calls" in model_fields:
        config_kwargs["maximum_remote_calls"] = None
    elif "maximumRemoteCalls" in model_fields:
        config_kwargs["maximumRemoteCalls"] = None
    return cast(Any, types.AutomaticFunctionCallingConfig)(**config_kwargs)


def get_gemini_costs(response, model: str) -> dict:
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        in_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
        candidates_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
        thoughts_tokens = getattr(usage_metadata, "thoughts_token_count", 0) or 0
        cached_tokens = getattr(usage_metadata, "cached_content_token_count", 0) or 0
    else:
        in_tokens = candidates_tokens = thoughts_tokens = cached_tokens = 0

    out_tokens = candidates_tokens
    thinking_tokens = thoughts_tokens
    input_cost, output_cost = calculate_cost(model, in_tokens, out_tokens + thinking_tokens)
    return {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "thinking_tokens": thinking_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cost": input_cost + output_cost,
        "cache_read_tokens": cached_tokens,
    }


def _backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Gemini - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


def _build_contents(msg_history: list[dict], msg: str) -> list[dict]:
    """Convert msg_history + current msg to Gemini's contents format."""
    contents = []
    for hist_msg in msg_history:
        role = hist_msg["role"]
        gemini_role = "model" if role == "assistant" else role
        contents.append({"role": gemini_role, "parts": [{"text": hist_msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": msg}]})
    return contents


def _extract_thoughts_and_content(response) -> tuple[str, str]:
    """Pull (thought, content) from response.candidates[0].content.parts."""
    thoughts = []
    content_parts = []
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            for part in candidate.content.parts:
                part_text = getattr(part, "text", "")
                if not part_text:
                    continue
                if getattr(part, "thought", False):
                    thoughts.append(part_text)
                else:
                    content_parts.append(part_text)
    thought = "\n".join(thoughts) if thoughts else ""
    content = "\n".join(content_parts) if content_parts else ""
    if not content and hasattr(response, "text"):
        content = response.text or ""
    return thought, content


class GeminiProvider:
    """Gemini provider. Singleton client; the genai.Client is used for both
    sync (.models.generate_content) and async (.aio.models.generate_content)."""

    name = "google"

    def __init__(self) -> None:
        self._client: Optional[genai.Client] = None

    def _make_client(self) -> genai.Client:
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _client_or_make(self) -> genai.Client:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def build_call_kwargs(self, *, api_model: str, requested: Mapping[str, Any]) -> dict:
        """Gemini shape: max_tokens (used as max_output_tokens at call time),
        thinking_budget int (not a dict). top_p/top_k supported."""
        effort = resolve_effort(api_model, requested.get("reasoning_effort", "disabled"))
        out: dict = {}
        if (v := requested.get("max_tokens")) is not None:
            out["max_tokens"] = v

        temp = resolve_temperature(api_model, requested.get("temperature"), effort)
        if temp is not None:
            out["temperature"] = temp

        if is_reasoning_model(api_model):
            ceiling = out.get("max_tokens", 4096)
            if effort != "disabled":
                t = THINKING_TOKENS[effort]
                out["thinking_budget"] = t if t < ceiling else 1024
            else:
                # Some Gemini reasoning models can't fully disable thinking;
                # keep a tiny budget so the call doesn't error.
                if api_model in ("gemini-2.5-pro", "gemini-3-pro-preview"):
                    out["thinking_budget"] = 128
                else:
                    out["thinking_budget"] = 0

        if (v := requested.get("top_p")) is not None:
            out["top_p"] = v
        if (v := requested.get("top_k")) is not None:
            out["top_k"] = v
        return out

    def _build_generation_config(
        self, *, system_msg: str, kwargs: dict, output_model: Optional[Type[BaseModel]]
    ) -> types.GenerateContentConfig:
        """Build the GenerateContentConfig from our kwargs shape. Includes
        the schema if structured output is requested."""
        config_kwargs: dict[str, Any] = {
            "temperature": float(kwargs.get("temperature", 0.8)),
            "top_p": float(kwargs.get("top_p", 1.0)),
            "max_output_tokens": int(kwargs.get("max_tokens", 2048)),
            "system_instruction": system_msg if system_msg else None,
            "automatic_function_calling": _build_afc_config(),
            "thinking_config": _build_thinking_config(kwargs.get("thinking_budget", 0)),
        }
        if (top_k := kwargs.get("top_k")) is not None:
            config_kwargs["top_k"] = int(top_k)
        if output_model is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = output_model
        return types.GenerateContentConfig(**config_kwargs)

    @backoff.on_exception(
        backoff.expo, (Exception,),  # Gemini SDK throws unique exception types
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
        client: Optional[genai.Client] = None,
    ) -> QueryResult:
        client = client or self._client_or_make()
        merged_system = _merge_prefix(system_msg, system_prefix)
        contents = _build_contents(msg_history, msg)
        config = self._build_generation_config(
            system_msg=merged_system, kwargs=kwargs, output_model=output_model,
        )
        response = client.models.generate_content(model=model, contents=contents, config=config)
        return self._build_result(
            response=response, model=model, msg=msg, system_msg=system_msg,
            msg_history=msg_history, output_model=output_model, kwargs=kwargs,
        )

    @backoff.on_exception(
        backoff.expo, (Exception,),
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
        client: Optional[genai.Client] = None,
    ) -> QueryResult:
        client = client or self._client_or_make()
        merged_system = _merge_prefix(system_msg, system_prefix)
        contents = _build_contents(msg_history, msg)
        config = self._build_generation_config(
            system_msg=merged_system, kwargs=kwargs, output_model=output_model,
        )
        response = await client.aio.models.generate_content(model=model, contents=contents, config=config)
        return self._build_result(
            response=response, model=model, msg=msg, system_msg=system_msg,
            msg_history=msg_history, output_model=output_model, kwargs=kwargs,
        )

    def _build_result(
        self,
        *,
        response,
        model: str,
        msg: str,
        system_msg: str,
        msg_history: list[dict],
        output_model: Optional[Type[BaseModel]],
        kwargs: dict,
    ) -> QueryResult:
        thought, raw_content = _extract_thoughts_and_content(response)
        if output_model is not None:
            content = output_model.model_validate_json(raw_content)
            display = str(content)
        else:
            content = raw_content
            display = raw_content

        new_msg_history = msg_history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": display},
        ]
        cost_results = get_gemini_costs(response, model)
        return QueryResult(
            content=content, msg=msg, system_msg=system_msg,
            new_msg_history=new_msg_history, model_name=model, kwargs=kwargs,
            **cost_results, thought=thought,
        )


def _merge_prefix(system_msg: str, system_prefix: Optional[str]) -> str:
    """Gemini doesn't have separate cache_control blocks for system prompts
    (only via cached_content); plain concat is fine for now."""
    if system_prefix:
        return system_prefix + "\n\n" + system_msg
    return system_msg


# Module-level singleton.
GEMINI = GeminiProvider()


# Back-compat shims (mostly used via shinka). Kept thin.
def query_gemini(client, model, msg, system_msg, msg_history, output_model,
                 model_posteriors=None, **kwargs) -> QueryResult:
    system_prefix = kwargs.pop("system_prefix", None)
    result = GEMINI.query(
        model=model, msg=msg, system_msg=system_msg, msg_history=msg_history,
        system_prefix=system_prefix, output_model=output_model, kwargs=kwargs,
        client=client,
    )
    if model_posteriors is not None:
        from dataclasses import replace
        return replace(result, model_posteriors=model_posteriors)
    return result


async def query_gemini_async(client, model, msg, system_msg, msg_history, output_model,
                             model_posteriors=None, **kwargs) -> QueryResult:
    system_prefix = kwargs.pop("system_prefix", None)
    result = await GEMINI.query_async(
        model=model, msg=msg, system_msg=system_msg, msg_history=msg_history,
        system_prefix=system_prefix, output_model=output_model, kwargs=kwargs,
        client=client,
    )
    if model_posteriors is not None:
        from dataclasses import replace
        return replace(result, model_posteriors=model_posteriors)
    return result
