"""JSON-validation recovery shared across providers.

Some upstream providers leak special-token markers (`[EOS]`, `<|endoftext|>`)
or emit malformed keys (newline prefixes, numeric prefixes, upper-cased
field names) into otherwise-valid structured-output responses. Pydantic
rejects those as ValidationError; this module recovers what it can before
the caller decides to retry the whole call.

Lifted from `providers/openai.py` because the same patterns appear in
DeepSeek's native JSON path post-instructor-removal — every provider that
hands us raw JSON will benefit.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

logger = logging.getLogger(__name__)


# Special-token markers some upstreams leak into completion text as literal
# strings (e.g. Kimi k2.5 via OpenRouter's :nitro routing). When these trail
# an otherwise-valid JSON response, pydantic rejects it as `json_invalid:
# trailing characters`. We strip them and retry the parse.
_TRAILING_TOKEN_MARKERS = ("[EOS]", "<|endoftext|>", "<|im_end|>", "</s>")

# Matches a leading "N." or "N)" or "N:" style numeric prefix on a key,
# e.g. "3. tension_architecture" → "tension_architecture".
_NUMERIC_KEY_PREFIX = re.compile(r"^\s*\d+\s*[\.\)\:]\s*")


def clean_trailing_garbage(text: str) -> str:
    """Strip known special-token markers and anything after the last JSON '}'.
    Idempotent — safe to call on already-clean text.
    """
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for marker in _TRAILING_TOKEN_MARKERS:
            if cleaned.endswith(marker):
                cleaned = cleaned[: -len(marker)].rstrip()
                changed = True
    last_brace = cleaned.rfind("}")
    if last_brace != -1 and last_brace + 1 < len(cleaned):
        cleaned = cleaned[: last_brace + 1]
    return cleaned


def normalize_key(key: str) -> str:
    """Normalize a JSON dict key for fuzzy-match against a pydantic schema.

    Kimi k2-0905:nitro routinely emits keys like '\\nnovelty', 'NOVELTY',
    '3. tension_architecture', '\\n4. emotional_depth'. None of these match
    the actual pydantic field names. Stripping newlines, numeric prefixes,
    and lower-casing recovers the vast majority.
    """
    k = key.strip().lstrip("\r\n\t ")
    k = _NUMERIC_KEY_PREFIX.sub("", k)
    return k.lower()


def normalize_json_keys(raw: str, expected_fields: set[str]) -> dict | None:
    """Parse raw as JSON dict, normalize keys, keep only expected fields.

    Returns the normalized dict on success, None if the raw text isn't a
    JSON object at all. Drops keys that don't match expected_fields after
    normalization (e.g. the `"\\n":"a"` junk key pattern).
    """
    try:
        # strict=False accepts unescaped control chars (raw newlines) inside
        # JSON string values. Kimi k2-0905:nitro routinely emits these; strict
        # parsing rejects the entire response otherwise. Pydantic's own JSON
        # parser is lenient here, so matching its behavior restores recovery.
        parsed = json.loads(raw, strict=False)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    out: dict = {}
    for key, value in parsed.items():
        norm = normalize_key(key)
        if norm in expected_fields and norm not in out:
            out[norm] = value
    return out


def recover_from_validation_error(err: ValidationError, output_model, model: str):
    """Try to recover a parsed model from a ValidationError.

    Three strategies, in order:
    1. Strip trailing garbage (`[EOS]` and friends, or text after last `}`)
       and re-validate. Handles Kimi k2.5 :nitro [EOS] suffix.
    2. Parse raw as a JSON dict, normalize keys (strip \\n, numeric prefixes,
       lowercase), keep only schema-expected fields, validate. Handles Kimi
       k2-0905:nitro's malformed-key patterns.
    3. If neither works, return None; caller logs and retries or raises.
    """
    best_raw: str | None = None  # longest raw string we saw, for diagnostics
    expected_fields = set(output_model.model_fields.keys())

    for detail in err.errors():
        raw_input = detail.get("input")

        # Case 1: input is a dict (pydantic already parsed the JSON but can't
        # match the fields — malformed keys). Normalize keys in-place.
        if isinstance(raw_input, dict):
            normalized = {}
            for key, value in raw_input.items():
                norm = normalize_key(key)
                if norm in expected_fields and norm not in normalized:
                    normalized[norm] = value
            try:
                parsed = output_model.model_validate(normalized)
            except ValidationError:
                continue
            logger.info(
                "Recovered %s response via key normalization (%d fields mapped).",
                model, len(normalized),
            )
            return parsed

        # Case 2: input is a string (JSON couldn't parse at all).
        if not isinstance(raw_input, str):
            continue
        raw = raw_input
        if best_raw is None or len(raw) > len(best_raw):
            best_raw = raw

        # Strategy 1: trailing-garbage strip
        cleaned = clean_trailing_garbage(raw)
        if cleaned != raw.strip():
            try:
                parsed = output_model.model_validate_json(cleaned)
            except ValidationError:
                pass
            else:
                logger.info(
                    "Recovered %s response by stripping %d trailing chars.",
                    model, len(raw.strip()) - len(cleaned),
                )
                return parsed

        # Strategy 2: key normalization on raw JSON text. Try both original
        # and trailing-stripped variants — both fixes may be needed at once.
        for candidate in (raw, cleaned):
            normalized = normalize_json_keys(candidate, expected_fields)
            if normalized is None:
                continue
            try:
                parsed = output_model.model_validate(normalized)
            except ValidationError:
                continue
            logger.info(
                "Recovered %s response via key normalization (%d fields mapped).",
                model, len(normalized),
            )
            return parsed

    if best_raw is not None:
        preview_head = best_raw[:200].replace("\n", "\\n")
        preview_tail = best_raw[-200:].replace("\n", "\\n")
        logger.warning(
            "Unrecoverable parse for %s. Raw length=%d. Head=%r  Tail=%r",
            model, len(best_raw), preview_head, preview_tail,
        )
    return None
