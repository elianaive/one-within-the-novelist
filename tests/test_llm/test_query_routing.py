"""Tests for system_prefix merging in query routing."""

from owtn.llm.query import _merge_system_prefix


class TestMergeSystemPrefix:
    def test_non_anthropic_merges_prefix(self):
        kwargs = {"system_prefix": "cached prefix", "temperature": 0.7}
        result = _merge_system_prefix(kwargs, "suffix", "openai")
        assert result == "cached prefix\n\nsuffix"
        assert "system_prefix" not in kwargs  # removed from kwargs

    def test_anthropic_preserves_prefix_in_kwargs(self):
        kwargs = {"system_prefix": "cached prefix", "temperature": 0.7}
        result = _merge_system_prefix(kwargs, "suffix", "anthropic")
        assert result == "suffix"  # unchanged
        assert kwargs["system_prefix"] == "cached prefix"  # still in kwargs

    def test_bedrock_preserves_prefix_in_kwargs(self):
        kwargs = {"system_prefix": "cached prefix"}
        result = _merge_system_prefix(kwargs, "suffix", "bedrock")
        assert result == "suffix"
        assert kwargs["system_prefix"] == "cached prefix"

    def test_no_prefix_returns_unchanged(self):
        kwargs = {"temperature": 0.7}
        result = _merge_system_prefix(kwargs, "system msg", "openai")
        assert result == "system msg"

    def test_deepseek_merges_prefix(self):
        kwargs = {"system_prefix": "prefix"}
        result = _merge_system_prefix(kwargs, "suffix", "deepseek")
        assert result == "prefix\n\nsuffix"

    def test_google_merges_prefix(self):
        kwargs = {"system_prefix": "prefix"}
        result = _merge_system_prefix(kwargs, "suffix", "google")
        assert result == "prefix\n\nsuffix"
