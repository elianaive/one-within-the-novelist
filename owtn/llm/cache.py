"""Optional disk cache for LLM query results.

Off by default (zero overhead). Enable with `OWTN_CACHE_ENABLED=1` —
useful for replaying expensive runs against fixed inputs while developing.
The cache key hashes only inputs that affect the output (model, msgs,
temperature, max_tokens, system_prefix), so kwargs-rotation noise
(reasoning_effort, top_p, etc.) doesn't bust the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from typing import Callable, TypeVar

CACHE_ENABLED = os.environ.get("OWTN_CACHE_ENABLED", "").lower() in ("1", "true")

F = TypeVar("F", bound=Callable)


def query_cache_key(args, kwargs) -> str:
    """Deterministic SHA-256 of (model, msg, system_msg, msg_history,
    system_prefix, temperature, max_tokens). Anything else in kwargs is
    treated as noise that shouldn't bust the cache."""
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


def cached(fn: F) -> F:
    """Wrap a query function with cachier when caching is enabled.
    No-op (returns fn unchanged) when OWTN_CACHE_ENABLED is unset."""
    if not CACHE_ENABLED:
        return fn
    from cachier import cachier

    return cachier(
        cache_dir=os.environ.get("OWTN_CACHE_DIR", ".cache/llm"),
        hash_func=query_cache_key,
        stale_after=timedelta(days=7),
    )(fn)
