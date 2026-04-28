"""Provider registry.

Each backend (Anthropic, OpenAI, Gemini, DeepSeek, etc.) is a `Provider`
class living in one file. `PROVIDERS` maps the resolved-provider string
to the singleton instance — `api.py` uses this for dispatch.

Free-function exports (`query_anthropic`, etc.) remain for backwards
compatibility with shinka and any external import that still uses them.
"""

from ..result import QueryResult

from .anthropic import (
    ANTHROPIC,
    BEDROCK,
    AnthropicProvider,
    BedrockProvider,
    query_anthropic,
    query_anthropic_async,
)
from .base import Provider
from .deepseek import DEEPSEEK, DeepSeekProvider, query_deepseek, query_deepseek_async
from .gemini import GEMINI, GeminiProvider, query_gemini, query_gemini_async
from .local_openai import (
    LOCAL_OPENAI,
    LocalOpenAIProvider,
    query_local_openai,
    query_local_openai_async,
)
from .openai import (
    AZURE_OPENAI,
    OPENAI,
    OPENROUTER,
    AzureOpenAIProvider,
    OpenAIProvider,
    OpenRouterProvider,
    query_openai,
    query_openai_async,
)


PROVIDERS: dict[str, Provider] = {
    "anthropic": ANTHROPIC,
    "bedrock": BEDROCK,
    "openai": OPENAI,
    "azure_openai": AZURE_OPENAI,
    "openrouter": OPENROUTER,
    "deepseek": DEEPSEEK,
    "google": GEMINI,
    "local_openai": LOCAL_OPENAI,
}


__all__ = [
    "Provider",
    "PROVIDERS",
    "QueryResult",
    "AnthropicProvider", "BedrockProvider",
    "OpenAIProvider", "AzureOpenAIProvider", "OpenRouterProvider",
    "DeepSeekProvider", "GeminiProvider", "LocalOpenAIProvider",
    "ANTHROPIC", "BEDROCK", "OPENAI", "AZURE_OPENAI", "OPENROUTER",
    "DEEPSEEK", "GEMINI", "LOCAL_OPENAI",
    # Legacy free-function entry points
    "query_anthropic", "query_anthropic_async",
    "query_openai", "query_openai_async",
    "query_deepseek", "query_deepseek_async",
    "query_gemini", "query_gemini_async",
    "query_local_openai", "query_local_openai_async",
]
