"""Legacy client factory — preserved for back-compat with shinka tests.

Returns the (client, api_model_name, provider) triple by delegating to the
appropriate Provider's cached singleton. New code should use
`owtn.llm.api.query`/`query_async` directly; they handle client management
internally.
"""

from __future__ import annotations

from typing import Any, Tuple

from .providers import PROVIDERS
from .providers.local_openai import LocalOpenAIProvider
from .providers.model_resolver import resolve_model_backend


def get_client_llm(model_name: str, structured_output: bool = False) -> Tuple[Any, str, str]:
    """Return (sync client, api_model_name, provider). `structured_output`
    is accepted for back-compat but is now a no-op — providers handle
    structured output internally via their own paths."""
    resolved = resolve_model_backend(model_name)
    provider = PROVIDERS[resolved.provider]
    if isinstance(provider, LocalOpenAIProvider):
        client = provider._sync(resolved.base_url or "")
    else:
        client = provider._sync()
    return client, resolved.api_model_name, resolved.provider


def get_async_client_llm(model_name: str, structured_output: bool = False) -> Tuple[Any, str, str]:
    """Return (async client, api_model_name, provider). `structured_output`
    is accepted for back-compat but is now a no-op."""
    resolved = resolve_model_backend(model_name)
    provider = PROVIDERS[resolved.provider]
    if isinstance(provider, LocalOpenAIProvider):
        client = provider._async(resolved.base_url or "")
    else:
        client = provider._async()
    return client, resolved.api_model_name, resolved.provider
