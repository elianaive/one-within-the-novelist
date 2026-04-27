import json
import re
import backoff
import openai
from pydantic import ValidationError
from .pricing import calculate_cost, model_exists
from ..result import QueryResult
import logging

logger = logging.getLogger(__name__)

MAX_TRIES = 20
MAX_VALUE = 20
MAX_TIME = 600

# Some providers (notably Kimi via OpenRouter) occasionally return a 200
# response where `output_parsed` is None — the JSON didn't coerce to the
# pydantic model. Retry those a few times before failing the whole judge call.
PARSE_RETRIES = 3

# Special-token markers some upstreams leak into completion text as literal
# strings (e.g. Kimi k2.5 via OpenRouter's :nitro routing). When these trail
# an otherwise-valid JSON response, pydantic rejects it as `json_invalid:
# trailing characters`. We strip them and retry the parse.
_TRAILING_TOKEN_MARKERS = ("[EOS]", "<|endoftext|>", "<|im_end|>", "</s>")


def _clean_trailing_garbage(text: str) -> str:
    """Strip known special-token markers and anything after the last JSON '}'.

    Idempotent — safe to call on already-clean text (returns input unchanged).
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


# Matches a leading "N." or "N)" or "N:" style numeric prefix on a key,
# e.g. "3. tension_architecture" → "tension_architecture".
_NUMERIC_KEY_PREFIX = re.compile(r"^\s*\d+\s*[\.\)\:]\s*")


def _normalize_key(key: str) -> str:
    """Normalize a JSON dict key for fuzzy-match against a pydantic schema.

    Kimi k2-0905:nitro routinely emits keys like:
    - '\\nnovelty' (leading newline)
    - 'NOVELTY' (upper-cased)
    - '3. tension_architecture' (numeric prefix)
    - '\\n4. emotional_depth' (both)
    None of these match the actual pydantic field names. Stripping newlines,
    numeric prefixes, and lower-casing recovers the vast majority.
    """
    k = key.strip().lstrip("\r\n\t ")
    k = _NUMERIC_KEY_PREFIX.sub("", k)
    return k.lower()


def _normalize_json_keys(raw: str, expected_fields: set[str]) -> dict | None:
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
        norm = _normalize_key(key)
        if norm in expected_fields and norm not in out:
            out[norm] = value
    return out


def _recover_from_validation_error(err, output_model, model: str):
    """Try to recover a parsed model from a ValidationError.

    Three strategies, in order:
    1. Strip trailing garbage (`[EOS]` and friends, or text after last `}`)
       and re-validate. Handles Kimi k2.5 :nitro [EOS] suffix.
    2. Parse raw as a JSON dict, normalize keys (strip \n, numeric prefixes,
       lowercase), keep only schema-expected fields, validate. Handles Kimi
       k2-0905:nitro's malformed-key patterns (e.g. `"\\nnovelty":"a"`,
       `"NOVELTY":"a"`, `"3. tension_architecture":"b"`).
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
                norm = _normalize_key(key)
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

        # Case 2: input is a string (JSON couldn't parse at all — trailing
        # garbage or other surface-level corruption).
        if not isinstance(raw_input, str):
            continue
        raw = raw_input
        if best_raw is None or len(raw) > len(best_raw):
            best_raw = raw

        # Strategy 1: trailing-garbage strip
        cleaned = _clean_trailing_garbage(raw)
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
            normalized = _normalize_json_keys(candidate, expected_fields)
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

    # No recovery possible. Log a bounded preview of the raw payload so we can
    # see what pattern needs handling next.
    if best_raw is not None:
        preview_head = best_raw[:200].replace("\n", "\\n")
        preview_tail = best_raw[-200:].replace("\n", "\\n")
        logger.warning(
            "Unrecoverable parse for %s. Raw length=%d. Head=%r  Tail=%r",
            model, len(best_raw), preview_head, preview_tail,
        )
    return None


class _RecoveredResponseStub:
    """Stand-in for an openai responses object when we recovered content from
    a ValidationError — the original response is lost inside the SDK, so
    cost/token accounting for this call reports zeros. Accepted trade for
    making the call succeed at all.
    """

    class _Usage:
        input_tokens = 0
        output_tokens = 0
        output_tokens_details = None
        input_tokens_details = None
        cost_details = None

    usage = _Usage()


def _parse_with_retry(call_fn, output_model, model: str):
    """Call the parse function, retry if output_parsed is None or if the
    response parsed to a ValidationError recoverable by trailing-garbage strip.
    """
    for attempt in range(PARSE_RETRIES):
        if attempt > 0:
            logger.warning(
                "Retrying structured parse for %s: attempt %d/%d",
                model, attempt + 1, PARSE_RETRIES,
            )
        try:
            response = call_fn()
        except ValidationError as e:
            recovered = _recover_from_validation_error(e, output_model, model)
            if recovered is not None:
                return _RecoveredResponseStub(), recovered
            logger.warning(
                "OpenAI/OpenRouter parse raised unrecoverable ValidationError "
                "for %s (attempt %d/%d); retrying.", model, attempt + 1, PARSE_RETRIES,
            )
            continue
        content = response.output_parsed
        if content is not None:
            return response, content
        logger.warning(
            "OpenAI/OpenRouter parse returned None for %s (attempt %d/%d); retrying.",
            model, attempt + 1, PARSE_RETRIES,
        )
    raise ValueError(
        f"Structured parse failed after {PARSE_RETRIES} attempts for model {model}."
    )


async def _parse_with_retry_async(call_fn, output_model, model: str):
    """Async variant of _parse_with_retry."""
    for attempt in range(PARSE_RETRIES):
        if attempt > 0:
            logger.warning(
                "Retrying structured parse for %s: attempt %d/%d",
                model, attempt + 1, PARSE_RETRIES,
            )
        try:
            response = await call_fn()
        except ValidationError as e:
            recovered = _recover_from_validation_error(e, output_model, model)
            if recovered is not None:
                return _RecoveredResponseStub(), recovered
            logger.warning(
                "OpenAI/OpenRouter parse raised unrecoverable ValidationError "
                "for %s (attempt %d/%d); retrying.", model, attempt + 1, PARSE_RETRIES,
            )
            continue
        content = response.output_parsed
        if content is not None:
            return response, content
        logger.warning(
            "OpenAI/OpenRouter parse returned None for %s (attempt %d/%d); retrying.",
            model, attempt + 1, PARSE_RETRIES,
        )
    raise ValueError(
        f"Structured parse failed after {PARSE_RETRIES} attempts for model {model}."
    )


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenAI - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


def get_openai_costs(response, model):
    # Get token counts and costs
    in_tokens = response.usage.input_tokens
    try:
        thinking_tokens = response.usage.output_tokens_details.reasoning_tokens
    except Exception:
        thinking_tokens = 0
    all_out_tokens = response.usage.output_tokens
    out_tokens = response.usage.output_tokens - thinking_tokens
    cached_tokens = 0
    details = getattr(response.usage, "input_tokens_details", None)
    if details is not None:
        cached_tokens = getattr(details, "cached_tokens", 0) or 0

    # Get actual costs from OpenRouter API if available -- if not use OAI
    cost_details = getattr(response.usage, "cost_details", None)
    if cost_details:
        if isinstance(cost_details, dict):
            input_cost = float(cost_details.get("upstream_inference_input_cost", 0.0))
            output_cost = float(cost_details.get("upstream_inference_output_cost", 0.0))
        else:
            input_cost = float(
                getattr(cost_details, "upstream_inference_input_cost", 0.0) or 0.0
            )
            output_cost = float(
                getattr(cost_details, "upstream_inference_output_cost", 0.0) or 0.0
            )
    elif model_exists(model):
        input_cost, output_cost = calculate_cost(model, in_tokens, all_out_tokens)
    else:
        logger.warning(
            "Model '%s' has no pricing entry and response cost metadata is absent. "
            "Defaulting query cost to 0.",
            model,
        )
        input_cost, output_cost = 0.0, 0.0
    return {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "thinking_tokens": thinking_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cost": input_cost + output_cost,
        "cache_read_tokens": cached_tokens,
    }


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
def query_openai(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query OpenAI model."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    thought = ""
    if output_model is None:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            **kwargs,
        )
        try:
            content = response.output[0].content[0].text
        except Exception:
            # Reasoning models - ResponseOutputMessage
            content = response.output[1].content[0].text

        try:
            thought = response.output[0].summary[0].text
        except Exception:
            pass
        new_msg_history.append({"role": "assistant", "content": content})
    else:
        response, content = _parse_with_retry(
            lambda: client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system_msg},
                    *new_msg_history,
                ],
                text_format=output_model,
                **kwargs,
            ),
            output_model=output_model,
            model=model,
        )
        new_content = ""
        for i in content:
            new_content += str(i[0]) + ":" + str(i[1]) + "\n"
        new_msg_history.append({"role": "assistant", "content": new_content})

    # Get token counts and costs
    cost_results = get_openai_costs(response, model)

    # Collect all results
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        **cost_results,
        thought=thought,
        model_posteriors=model_posteriors,
    )
    return result


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
async def query_openai_async(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query OpenAI model asynchronously."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    thought = ""
    if output_model is None:
        response = await client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            **kwargs,
        )
        try:
            content = response.output[0].content[0].text
        except Exception:
            # Reasoning models - ResponseOutputMessage
            content = response.output[1].content[0].text
        try:
            thought = response.output[0].summary[0].text
        except Exception:
            pass
        new_msg_history.append({"role": "assistant", "content": content})
    else:
        response, content = await _parse_with_retry_async(
            lambda: client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system_msg},
                    *new_msg_history,
                ],
                text_format=output_model,
                **kwargs,
            ),
            output_model=output_model,
            model=model,
        )
        new_content = ""
        for i in content:
            new_content += str(i[0]) + ":" + str(i[1]) + "\n"
        new_msg_history.append({"role": "assistant", "content": new_content})
    cost_results = get_openai_costs(response, model)
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        **cost_results,
        thought=thought,
        model_posteriors=model_posteriors,
    )
    return result
