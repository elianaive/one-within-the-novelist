"""Tests for trailing-garbage cleanup on structured-output responses.

Context: lab/issues/2026-04-18-judge-latency-variance.md. Some OpenRouter
upstreams (Kimi k2.5 via :nitro routing) emit special-token markers as
literal text suffix. Pydantic rejects otherwise-valid JSON with "trailing
characters" — we strip and re-parse.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from owtn.llm.recovery import (
    clean_trailing_garbage as _clean_trailing_garbage,
    normalize_json_keys as _normalize_json_keys,
    normalize_key as _normalize_key,
    recover_from_validation_error as _recover_from_validation_error,
)


class _Sample(BaseModel):
    reasoning: str
    winner: str


class TestCleanTrailingGarbage:
    def test_strips_eos_suffix(self):
        assert _clean_trailing_garbage('{"x":1}[EOS]') == '{"x":1}'

    def test_strips_eos_with_space(self):
        assert _clean_trailing_garbage('{"x":1} [EOS]') == '{"x":1}'

    def test_strips_other_special_tokens(self):
        assert _clean_trailing_garbage('{"x":1}<|endoftext|>') == '{"x":1}'
        assert _clean_trailing_garbage('{"x":1}</s>') == '{"x":1}'

    def test_strips_multiple_stacked_markers(self):
        assert _clean_trailing_garbage('{"x":1}[EOS][EOS]') == '{"x":1}'

    def test_truncates_after_last_brace(self):
        assert _clean_trailing_garbage('{"x":1}garbage') == '{"x":1}'

    def test_idempotent_on_clean_input(self):
        clean = '{"x":1}'
        assert _clean_trailing_garbage(clean) == clean

    def test_preserves_nested_braces(self):
        nested = '{"a":{"b":1}}'
        assert _clean_trailing_garbage(nested) == nested
        assert _clean_trailing_garbage(nested + "[EOS]") == nested


class TestRecoverFromValidationError:
    def test_recovers_eos_suffixed_response(self):
        raw = '{"reasoning":"x","winner":"a"}[EOS]'
        try:
            _Sample.model_validate_json(raw)
        except ValidationError as e:
            recovered = _recover_from_validation_error(e, _Sample, "test-model")
            assert recovered is not None
            assert recovered.reasoning == "x"
            assert recovered.winner == "a"
        else:
            pytest.fail("expected ValidationError")

    def test_returns_none_for_genuinely_malformed(self):
        """Missing-required-field errors can't be recovered by stripping."""
        raw = '{"reasoning":"x"}'  # winner missing
        try:
            _Sample.model_validate_json(raw)
        except ValidationError as e:
            recovered = _recover_from_validation_error(e, _Sample, "test-model")
            assert recovered is None
        else:
            pytest.fail("expected ValidationError")

    def test_returns_none_when_no_string_input(self):
        """Errors whose `input` isn't a string (dict-level errors) skip cleanly."""
        class _Broken(BaseModel):
            value: int
        try:
            _Broken.model_validate({"value": "not-a-number"})
        except ValidationError as e:
            recovered = _recover_from_validation_error(e, _Broken, "test-model")
            assert recovered is None
        else:
            pytest.fail("expected ValidationError")

    def test_recovers_newline_prefixed_keys(self):
        """Kimi k2-0905:nitro emits `"\\nnovelty":"a"` instead of `"novelty"`."""
        raw = '{"reasoning":"x","\\nwinner":"a"}'
        try:
            _Sample.model_validate_json(raw)
        except ValidationError as e:
            recovered = _recover_from_validation_error(e, _Sample, "test-model")
            assert recovered is not None
            assert recovered.reasoning == "x"
            assert recovered.winner == "a"

    def test_recovers_uppercase_keys(self):
        raw = '{"REASONING":"x","WINNER":"a"}'
        try:
            _Sample.model_validate_json(raw)
        except ValidationError as e:
            recovered = _recover_from_validation_error(e, _Sample, "test-model")
            assert recovered is not None
            assert recovered.winner == "a"

    def test_recovers_numbered_key_prefix(self):
        raw = '{"reasoning":"x","3. winner":"a"}'
        try:
            _Sample.model_validate_json(raw)
        except ValidationError as e:
            recovered = _recover_from_validation_error(e, _Sample, "test-model")
            assert recovered is not None
            assert recovered.winner == "a"

    def test_recovers_combined_prefix_and_numbered(self):
        raw = '{"reasoning":"x","\\n3. winner":"a"}'
        try:
            _Sample.model_validate_json(raw)
        except ValidationError as e:
            recovered = _recover_from_validation_error(e, _Sample, "test-model")
            assert recovered is not None
            assert recovered.winner == "a"


class TestNormalizeKey:
    def test_strips_leading_newline(self):
        assert _normalize_key("\\nnovelty") == "\\nnovelty"  # literal backslash-n
        assert _normalize_key("\nnovelty") == "novelty"  # actual newline

    def test_lowercases(self):
        assert _normalize_key("NOVELTY") == "novelty"

    def test_strips_numeric_prefix(self):
        assert _normalize_key("3. tension_architecture") == "tension_architecture"
        assert _normalize_key("4) emotional_depth") == "emotional_depth"
        assert _normalize_key("5: thematic_resonance") == "thematic_resonance"

    def test_combined(self):
        assert _normalize_key("\n3. TENSION_ARCHITECTURE") == "tension_architecture"


class TestNormalizeJsonKeys:
    def test_drops_unknown_keys(self):
        raw = '{"reasoning":"x","winner":"a","junk":"ignore_me"}'
        result = _normalize_json_keys(raw, {"reasoning", "winner"})
        assert result == {"reasoning": "x", "winner": "a"}

    def test_drops_bare_newline_key(self):
        """The `"\\n":"a"` junk-key pattern shouldn't clobber anything."""
        raw = '{"reasoning":"x","\\n":"a","winner":"b"}'
        result = _normalize_json_keys(raw, {"reasoning", "winner"})
        assert result == {"reasoning": "x", "winner": "b"}

    def test_returns_none_for_non_object(self):
        assert _normalize_json_keys('"just a string"', {"reasoning"}) is None
        assert _normalize_json_keys("not json at all", {"reasoning"}) is None
