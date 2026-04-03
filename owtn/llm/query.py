import hashlib
import json
import logging
import os
import time
from datetime import timedelta
from typing import Dict, List, Optional

from pydantic import BaseModel

from . import call_logger
from .client import get_async_client_llm, get_client_llm
from .providers import (
    QueryResult,
    query_anthropic,
    query_anthropic_async,
    query_deepseek,
    query_deepseek_async,
    query_gemini,
    query_gemini_async,
    query_local_openai,
    query_local_openai_async,
    query_openai,
    query_openai_async,
)

logger = logging.getLogger(__name__)


def _log_result(result: QueryResult, provider: str, msg: str, system_msg: str, duration: float, kwargs: dict) -> None:
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


CACHE_ENABLED = os.environ.get("OWTN_CACHE_ENABLED", "").lower() in ("1", "true")


def _query_cache_key(args, kwargs):
    """Deterministic cache key from query inputs."""
    key_data = json.dumps(
        {
            "model": args[0] if args else kwargs.get("model_name"),
            "msg": args[1] if len(args) > 1 else kwargs.get("msg"),
            "system_msg": args[2] if len(args) > 2 else kwargs.get("system_msg"),
            "msg_history": args[3] if len(args) > 3 else kwargs.get("msg_history", []),
            "system_prefix": kwargs.get("system_prefix"),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()


def _make_cached(fn):
    """Wrap a query function with cachier if caching is enabled."""
    if not CACHE_ENABLED:
        return fn
    from cachier import cachier

    return cachier(
        cache_dir=os.environ.get("OWTN_CACHE_DIR", ".cache/llm"),
        hash_func=_query_cache_key,
        stale_after=timedelta(days=7),
    )(fn)


def _merge_system_prefix(kwargs, system_msg, provider):
    """Handle system_prefix for non-Anthropic providers.

    Anthropic providers handle system_prefix internally via cache_control
    content blocks. For other providers, merge it into system_msg as a
    plain string prefix (OpenAI auto-caches matching prefixes).
    """
    system_prefix = kwargs.get("system_prefix")
    if system_prefix and provider not in ("anthropic", "bedrock"):
        kwargs.pop("system_prefix")
        system_msg = system_prefix + "\n\n" + system_msg
    return system_msg


def query(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Query the LLM."""
    client, model_name, provider = get_client_llm(
        model_name, structured_output=output_model is not None
    )
    system_msg = _merge_system_prefix(kwargs, system_msg, provider)
    if provider in ("anthropic", "bedrock"):
        query_fn = query_anthropic
    elif provider in ("openai", "azure_openai", "openrouter"):
        query_fn = query_openai
    elif provider == "deepseek":
        query_fn = query_deepseek
    elif provider == "google":
        query_fn = query_gemini
    elif provider == "local_openai":
        query_fn = query_local_openai
    else:
        raise ValueError(f"Model {model_name} not supported.")
    t0 = time.perf_counter()
    result = query_fn(
        client,
        model_name,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=model_posteriors,
        **kwargs,
    )
    _log_result(result, provider, msg, system_msg, time.perf_counter() - t0, kwargs)
    return result


async def query_async(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Query the LLM asynchronously."""
    client, model_name, provider = get_async_client_llm(
        model_name, structured_output=output_model is not None
    )
    system_msg = _merge_system_prefix(kwargs, system_msg, provider)
    if provider in ("anthropic", "bedrock"):
        query_fn = query_anthropic_async
    elif provider in ("openai", "azure_openai", "openrouter"):
        query_fn = query_openai_async
    elif provider == "deepseek":
        query_fn = query_deepseek_async
    elif provider == "google":
        query_fn = query_gemini_async
    elif provider == "local_openai":
        query_fn = query_local_openai_async
    else:
        raise ValueError(f"Model {model_name} not supported.")
    t0 = time.perf_counter()
    result = await query_fn(
        client,
        model_name,
        msg,
        system_msg,
        msg_history,
        output_model,
        model_posteriors=model_posteriors,
        **kwargs,
    )
    _log_result(result, provider, msg, system_msg, time.perf_counter() - t0, kwargs)
    return result


# Apply cachier decorator if caching is enabled
query = _make_cached(query)
query_async = _make_cached(query_async)
