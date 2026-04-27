"""OpenAI + Azure + OpenRouter providers.

All three use OpenAI's Responses API (`responses.parse` for structured
output, `responses.create` for free-form). Native structured output via
`text_format=<pydantic_model>` — no instructor injection.

The three classes differ only in client construction (different api_key,
api_base, etc.). The call shape and kwargs rewriter are shared.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional, Type

import backoff
import openai
from pydantic import BaseModel, ValidationError

from ..recovery import recover_from_validation_error
from ..result import QueryResult
from .base import TIMEOUT, resolve_effort, resolve_temperature
from .pricing import calculate_cost, is_reasoning_model, model_exists

logger = logging.getLogger(__name__)

MAX_TRIES = 20
MAX_VALUE = 20
MAX_TIME = 600

# Some providers (notably Kimi via OpenRouter) occasionally return a 200
# response where `output_parsed` is None — the JSON didn't coerce to the
# pydantic model. Retry those a few times before failing the whole call.
PARSE_RETRIES = 3

_RETRY_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APIStatusError,
    openai.RateLimitError,
    openai.APITimeoutError,
)


class _RecoveredResponseStub:
    """Stand-in for an openai responses object when we recovered content from
    a ValidationError — the original response is lost inside the SDK, so
    cost/token accounting for this call reports zeros. Accepted trade for
    making the call succeed at all.
    """

    class _Usage:
        input_tokens = 0
        output_tokens = 0
        output_tokens_details = None
        input_tokens_details = None
        cost_details = None

    usage = _Usage()


def _parse_with_retry(call_fn, output_model: Type[BaseModel], model: str):
    """Run a structured-output call with automatic recovery for known failure
    modes (Kimi `[EOS]` suffix, malformed keys). Retries up to PARSE_RETRIES.
    """
    for attempt in range(PARSE_RETRIES):
        if attempt > 0:
            logger.warning("Retrying structured parse for %s: attempt %d/%d", model, attempt + 1, PARSE_RETRIES)
        try:
            response = call_fn()
        except ValidationError as e:
            recovered = recover_from_validation_error(e, output_model, model)
            if recovered is not None:
                return _RecoveredResponseStub(), recovered
            logger.warning(
                "OpenAI/OpenRouter parse raised unrecoverable ValidationError for %s "
                "(attempt %d/%d); retrying.", model, attempt + 1, PARSE_RETRIES,
            )
            continue
        content = response.output_parsed
        if content is not None:
            return response, content
        logger.warning(
            "OpenAI/OpenRouter parse returned None for %s (attempt %d/%d); retrying.",
            model, attempt + 1, PARSE_RETRIES,
        )
    raise ValueError(f"Structured parse failed after {PARSE_RETRIES} attempts for model {model}.")


async def _parse_with_retry_async(call_fn, output_model: Type[BaseModel], model: str):
    """Async variant of _parse_with_retry."""
    for attempt in range(PARSE_RETRIES):
        if attempt > 0:
            logger.warning("Retrying structured parse for %s: attempt %d/%d", model, attempt + 1, PARSE_RETRIES)
        try:
            response = await call_fn()
        except ValidationError as e:
            recovered = recover_from_validation_error(e, output_model, model)
            if recovered is not None:
                return _RecoveredResponseStub(), recovered
            logger.warning(
                "OpenAI/OpenRouter parse raised unrecoverable ValidationError for %s "
                "(attempt %d/%d); retrying.", model, attempt + 1, PARSE_RETRIES,
            )
            continue
        content = response.output_parsed
        if content is not None:
            return response, content
        logger.warning(
            "OpenAI/OpenRouter parse returned None for %s (attempt %d/%d); retrying.",
            model, attempt + 1, PARSE_RETRIES,
        )
    raise ValueError(f"Structured parse failed after {PARSE_RETRIES} attempts for model {model}.")


def _backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenAI - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


def get_openai_costs(response, model: str) -> dict:
    """Token counts and dollar costs from a Responses API response.

    Prefers OpenRouter-supplied `cost_details.upstream_inference_*` when
    available; otherwise falls back to our pricing table. Models without a
    pricing entry get a zero cost and a warning.
    """
    in_tokens = response.usage.input_tokens
    try:
        thinking_tokens = response.usage.output_tokens_details.reasoning_tokens
    except Exception:
        thinking_tokens = 0
    all_out_tokens = response.usage.output_tokens
    out_tokens = all_out_tokens - thinking_tokens

    cached_tokens = 0
    details = getattr(response.usage, "input_tokens_details", None)
    if details is not None:
        cached_tokens = getattr(details, "cached_tokens", 0) or 0

    cost_details = getattr(response.usage, "cost_details", None)
    if cost_details:
        if isinstance(cost_details, dict):
            input_cost = float(cost_details.get("upstream_inference_input_cost", 0.0))
            output_cost = float(cost_details.get("upstream_inference_output_cost", 0.0))
        else:
            input_cost = float(getattr(cost_details, "upstream_inference_input_cost", 0.0) or 0.0)
            output_cost = float(getattr(cost_details, "upstream_inference_output_cost", 0.0) or 0.0)
    elif model_exists(model):
        input_cost, output_cost = calculate_cost(model, in_tokens, all_out_tokens)
    else:
        logger.warning(
            "Model '%s' has no pricing entry and response cost metadata is absent. "
            "Defaulting query cost to 0.", model,
        )
        input_cost, output_cost = 0.0, 0.0

    return {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "thinking_tokens": thinking_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cost": input_cost + output_cost,
        "cache_read_tokens": cached_tokens,
    }


def _extract_text_and_thought(response) -> tuple[str, str]:
    """Pull (content, thought) from a Responses API response. Reasoning
    models put the message at output[1] (output[0] is the reasoning summary)."""
    content = ""
    thought = ""
    try:
        content = response.output[0].content[0].text
    except Exception:
        try:
            content = response.output[1].content[0].text
        except Exception:
            content = ""
    try:
        thought = response.output[0].summary[0].text
    except Exception:
        pass
    return content, thought


class OpenAIProvider:
    """OpenAI provider. Singleton clients per (sync/async).

    Subclassed for Azure and OpenRouter — same call shape, different client
    construction.
    """

    name = "openai"

    def __init__(self) -> None:
        self._sync_client: Optional[openai.OpenAI] = None
        self._async_client: Optional[openai.AsyncOpenAI] = None

    def _make_sync_client(self) -> openai.OpenAI:
        return openai.OpenAI(timeout=TIMEOUT)

    def _make_async_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(timeout=TIMEOUT)

    def _sync(self) -> openai.OpenAI:
        if self._sync_client is None:
            self._sync_client = self._make_sync_client()
        return self._sync_client

    def _async(self) -> openai.AsyncOpenAI:
        if self._async_client is None:
            self._async_client = self._make_async_client()
        return self._async_client

    def build_call_kwargs(self, *, api_model: str, requested: Mapping[str, Any]) -> dict:
        """OpenAI shape: max_output_tokens (not max_tokens), reasoning dict
        for reasoning models, no top_k support, top_p dropped under reasoning."""
        effort = resolve_effort(api_model, requested.get("reasoning_effort", "disabled"))
        out: dict = {}
        max_tokens = requested.get("max_tokens", 4096)
        reasoning = is_reasoning_model(api_model)

        if reasoning:
            out["max_output_tokens"] = max_tokens
            if effort == "disabled":
                out["reasoning"] = {"effort": None}
            elif effort == "min":
                out["reasoning"] = {"effort": "low"}
            elif effort == "max":
                out["reasoning"] = {"effort": "high"}
            else:
                out["reasoning"] = {"effort": effort}
            if self.name == "openai" and effort != "disabled":
                out["reasoning"]["summary"] = "auto"
        else:
            out["max_output_tokens"] = max_tokens

        temp = resolve_temperature(api_model, requested.get("temperature"), effort)
        if temp is not None:
            out["temperature"] = temp

        # top_p forbidden under reasoning; top_k unsupported by OpenAI family.
        if not reasoning and (v := requested.get("top_p")) is not None:
            out["top_p"] = v
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
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        merged_system = _merge_prefix(system_msg, system_prefix)
        input_msgs = [{"role": "system", "content": merged_system}, *new_msg_history]

        if output_model is None:
            response = client.responses.create(model=model, input=input_msgs, **kwargs)
            content, thought = _extract_text_and_thought(response)
            new_msg_history.append({"role": "assistant", "content": content})
        else:
            response, content = _parse_with_retry(
                lambda: client.responses.parse(
                    model=model, input=input_msgs, text_format=output_model, **kwargs,
                ),
                output_model=output_model, model=model,
            )
            thought = ""
            new_msg_history.append({"role": "assistant", "content": str(content)})

        cost_results = get_openai_costs(response, model)
        return QueryResult(
            content=content, msg=msg, system_msg=system_msg,
            new_msg_history=new_msg_history, model_name=model, kwargs=kwargs,
            **cost_results, thought=thought,
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
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        merged_system = _merge_prefix(system_msg, system_prefix)
        input_msgs = [{"role": "system", "content": merged_system}, *new_msg_history]

        if output_model is None:
            response = await client.responses.create(model=model, input=input_msgs, **kwargs)
            content, thought = _extract_text_and_thought(response)
            new_msg_history.append({"role": "assistant", "content": content})
        else:
            response, content = await _parse_with_retry_async(
                lambda: client.responses.parse(
                    model=model, input=input_msgs, text_format=output_model, **kwargs,
                ),
                output_model=output_model, model=model,
            )
            thought = ""
            new_msg_history.append({"role": "assistant", "content": str(content)})

        cost_results = get_openai_costs(response, model)
        return QueryResult(
            content=content, msg=msg, system_msg=system_msg,
            new_msg_history=new_msg_history, model_name=model, kwargs=kwargs,
            **cost_results, thought=thought,
        )


def _merge_prefix(system_msg: str, system_prefix: Optional[str]) -> str:
    """OpenAI auto-caches matching string prefixes, so just concatenate."""
    if system_prefix:
        return system_prefix + "\n\n" + system_msg
    return system_msg


def _build_azure_endpoint() -> str:
    endpoint = os.getenv("AZURE_API_ENDPOINT")
    if not endpoint:
        raise ValueError("AZURE_API_ENDPOINT is required for Azure OpenAI models.")
    if not endpoint.endswith("/"):
        endpoint += "/"
    return endpoint + "openai/v1/"


class AzureOpenAIProvider(OpenAIProvider):
    """Azure-hosted OpenAI. Same call shape, different client construction."""

    name = "azure_openai"

    def _make_sync_client(self) -> openai.AzureOpenAI:
        return openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=_build_azure_endpoint(),
            timeout=TIMEOUT,
        )

    def _make_async_client(self) -> openai.AsyncAzureOpenAI:
        return openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=_build_azure_endpoint(),
            timeout=TIMEOUT,
        )


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter — OpenAI-compatible proxy. Different client + extras for
    min_p (passed through to upstream provider)."""

    name = "openrouter"

    def _make_sync_client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )

    def _make_async_client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=TIMEOUT,
        )

    def build_call_kwargs(self, *, api_model: str, requested: Mapping[str, Any]) -> dict:
        out = super().build_call_kwargs(api_model=api_model, requested=requested)
        # OpenRouter forwards arbitrary sampler params to the upstream.
        if (v := requested.get("min_p")) is not None and "reasoning" not in out:
            out["min_p"] = v
        return out


# Module-level singletons.
OPENAI = OpenAIProvider()
AZURE_OPENAI = AzureOpenAIProvider()
OPENROUTER = OpenRouterProvider()


# Back-compat shims for old free-function imports + tests.
def query_openai(client, model, msg, system_msg, msg_history, output_model,
                 model_posteriors=None, **kwargs) -> QueryResult:
    system_prefix = kwargs.pop("system_prefix", None)
    result = OPENAI.query(
        model=model, msg=msg, system_msg=system_msg, msg_history=msg_history,
        system_prefix=system_prefix, output_model=output_model, kwargs=kwargs,
        client=client,
    )
    if model_posteriors is not None:
        from dataclasses import replace
        return replace(result, model_posteriors=model_posteriors)
    return result


async def query_openai_async(client, model, msg, system_msg, msg_history, output_model,
                             model_posteriors=None, **kwargs) -> QueryResult:
    system_prefix = kwargs.pop("system_prefix", None)
    result = await OPENAI.query_async(
        model=model, msg=msg, system_msg=system_msg, msg_history=msg_history,
        system_prefix=system_prefix, output_model=output_model, kwargs=kwargs,
        client=client,
    )
    if model_posteriors is not None:
        from dataclasses import replace
        return replace(result, model_posteriors=model_posteriors)
    return result
