"""The single LLM entry point.

`query` and `query_async` resolve a model name → provider, ask that
provider's class to do the work, log the call, and return the result.
Self-critic, retries, batching, etc. all live elsewhere — this module
is purely the dispatch + telemetry layer.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from . import call_logger
from .cache import cached
from .providers import PROVIDERS, QueryResult
from .providers.model_resolver import resolve_model_backend

logger = logging.getLogger(__name__)


def _log(result: QueryResult, provider: str, msg: str, system_msg: str, duration: float, kwargs: dict) -> None:
    try:
        content = result.content if isinstance(result.content, str) else str(result.content)
        call_logger.log_call(
            model=result.model_name,
            provider=provider,
            system_msg=system_msg,
            user_msg=msg,
            content=content,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            thinking_tokens=result.thinking_tokens,
            cache_read_tokens=result.cache_read_tokens,
            cache_creation_tokens=result.cache_creation_tokens,
            cost=result.cost or 0.0,
            duration_s=duration,
            thought=result.thought or "",
            kwargs=kwargs,
        )
    except Exception as e:
        logger.warning("LLM call logging failed: %s", e)


@cached
def query(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[Type[BaseModel]] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> QueryResult:
    """Synchronous LLM call. Resolves the model, dispatches to its Provider."""
    resolved = resolve_model_backend(model_name)
    provider = PROVIDERS[resolved.provider]
    system_prefix = kwargs.pop("system_prefix", None)

    # local_openai needs the per-call base_url from the resolved model.
    extra: dict = {}
    if resolved.provider == "local_openai":
        extra["base_url"] = resolved.base_url

    t0 = time.perf_counter()
    result = provider.query(
        model=resolved.api_model_name, msg=msg, system_msg=system_msg,
        msg_history=msg_history, system_prefix=system_prefix,
        output_model=output_model, kwargs=kwargs, **extra,
    )
    if model_posteriors is not None:
        from dataclasses import replace
        result = replace(result, model_posteriors=model_posteriors)
    _log(result, resolved.provider, msg, system_msg, time.perf_counter() - t0, kwargs)
    return result


@cached
async def query_async(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[Type[BaseModel]] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> QueryResult:
    """Asynchronous LLM call. Resolves the model, dispatches to its Provider."""
    resolved = resolve_model_backend(model_name)
    provider = PROVIDERS[resolved.provider]
    system_prefix = kwargs.pop("system_prefix", None)

    extra: dict = {}
    if resolved.provider == "local_openai":
        extra["base_url"] = resolved.base_url

    t0 = time.perf_counter()
    result = await provider.query_async(
        model=resolved.api_model_name, msg=msg, system_msg=system_msg,
        msg_history=msg_history, system_prefix=system_prefix,
        output_model=output_model, kwargs=kwargs, **extra,
    )
    if model_posteriors is not None:
        from dataclasses import replace
        result = replace(result, model_posteriors=model_posteriors)
    _log(result, resolved.provider, msg, system_msg, time.perf_counter() - t0, kwargs)
    return result
