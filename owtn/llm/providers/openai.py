import backoff
import openai
from pydantic import ValidationError
from .pricing import calculate_cost, model_exists
from ..recovery import recover_from_validation_error
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
            recovered = recover_from_validation_error(e, output_model, model)
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
            recovered = recover_from_validation_error(e, output_model, model)
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
