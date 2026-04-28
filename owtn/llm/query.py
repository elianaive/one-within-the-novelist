"""Legacy entry point — preserved for back-compat with shinka and any
external import that hasn't migrated to `owtn.llm.api`.

The pure dispatch lives in `owtn.llm.api`; the self-critic critique-revise
cycle lives in `owtn.optimizer.self_critic`. This module re-exports both
plus the backward-compatible test helpers (`_query_cache_key`,
`_merge_system_prefix`).

New code should import directly from `owtn.llm.api` (no self-critic) or
`owtn.optimizer.self_critic` (with self-critic gating).
"""

from __future__ import annotations

from typing import Optional, Type

from pydantic import BaseModel

# Public API re-exports.
from .api import query  # noqa: F401  (used by tests and shinka shim)
from .cache import query_cache_key as _query_cache_key  # noqa: F401  (test surface)
from .providers import QueryResult  # noqa: F401  (test surface)


# Self-critic — wraps query_async with the critique-revise cycle gate.
# Lazy import so owtn.llm doesn't pull owtn.optimizer at module load.
def query_async(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: list = [],
    output_model: Optional[Type[BaseModel]] = None,
    model_posteriors: Optional[dict] = None,
    **kwargs,
) -> "QueryResult":
    from owtn.optimizer.self_critic import query_async as _wrapped
    return _wrapped(
        model_name=model_name, msg=msg, system_msg=system_msg,
        msg_history=msg_history, output_model=output_model,
        model_posteriors=model_posteriors, **kwargs,
    )


def register_self_critic_models(*args, **kwargs):
    from owtn.optimizer.self_critic import register_models
    return register_models(*args, **kwargs)


# Backward-compat: this used to merge system_prefix into system_msg for
# non-Anthropic providers. Now each provider handles it internally; this
# stub remains for the test that pins the legacy contract.
def _merge_system_prefix(kwargs, system_msg, provider):
    """Legacy helper. Anthropic/Bedrock keep system_prefix as a separate
    cache_control content block; everyone else concatenates it into
    system_msg. Returns the (possibly modified) system_msg; mutates
    kwargs in-place."""
    system_prefix = kwargs.get("system_prefix")
    if system_prefix and provider not in ("anthropic", "bedrock"):
        kwargs.pop("system_prefix")
        system_msg = system_prefix + "\n\n" + system_msg
    return system_msg
