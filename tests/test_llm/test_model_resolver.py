"""Tests for model name resolution to provider backends."""

import pytest

from owtn.llm.providers.model_resolver import resolve_model_backend


class TestResolveModelBackend:
    def test_known_anthropic_model(self):
        resolved = resolve_model_backend("claude-4-sonnet-20250514")
        assert resolved.provider == "anthropic"
        assert resolved.api_model_name == "claude-4-sonnet-20250514"

    def test_azure_prefix(self):
        resolved = resolve_model_backend("azure-gpt-4o")
        assert resolved.provider == "azure_openai"

    def test_openrouter_prefix(self):
        resolved = resolve_model_backend("openrouter/meta-llama/llama-3")
        assert resolved.provider == "openrouter"

    def test_local_model(self):
        resolved = resolve_model_backend("local/llama@http://localhost:8080/v1")
        assert resolved.provider == "local_openai"
        assert resolved.base_url == "http://localhost:8080/v1"

    def test_unknown_model_raises(self):
        with pytest.raises((ValueError, KeyError)):
            resolve_model_backend("completely-fake-model-12345")
