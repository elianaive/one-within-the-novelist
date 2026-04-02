import backoff
import anthropic
from .pricing import calculate_cost
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


MAX_TRIES = 20
MAX_VALUE = 20
MAX_TIME = 600
ANTHROPIC_DEFAULT_MAX_TOKENS = 16384


def _build_system(system_msg, system_prefix):
    """Build the system parameter for Anthropic API calls.

    When system_prefix is provided, returns a list of content blocks with
    cache_control on the prefix block. Otherwise returns the plain string.
    """
    if system_prefix:
        return [
            {
                "type": "text",
                "text": system_prefix,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": system_msg,
            },
        ]
    return system_msg


def get_anthropic_costs(response, model):
    """Get the costs for the given response and model."""
    input_tokens = response.usage.input_tokens
    all_out_tokens = response.usage.output_tokens
    thinking_tokens = 0
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    input_cost, output_cost = calculate_cost(model, input_tokens, all_out_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": all_out_tokens,
        "thinking_tokens": thinking_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cost": input_cost + output_cost,
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
    }


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Anthropic - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
    ),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
def query_anthropic(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Anthropic/Bedrock model."""
    system_prefix = kwargs.pop("system_prefix", None)
    system = _build_system(system_msg, system_prefix)
    new_msg_history = msg_history + [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": msg,
                }
            ],
        }
    ]
    kwargs.setdefault("max_tokens", ANTHROPIC_DEFAULT_MAX_TOKENS)
    thought = ""
    if output_model is not None:
        # Instructor-wrapped client.
        content, response = client.create_with_completion(
            model=model,
            max_tokens=kwargs.get("max_tokens", ANTHROPIC_DEFAULT_MAX_TOKENS),
            messages=[{"role": "system", "content": system_msg}, *new_msg_history],
            response_model=output_model,
        )
        new_msg_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": str(content)}]}
        )
    else:
        response = client.messages.create(
            model=model,
            system=system,
            messages=new_msg_history,
            **kwargs,
        )
        # Separate thinking from non-thinking content
        if len(response.content) == 1:
            content = response.content[0].text
        else:
            thought = response.content[0].thinking
            content = response.content[1].text
        new_msg_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": content}]}
        )
    cost_results = get_anthropic_costs(response, model)
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
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
    ),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
async def query_anthropic_async(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Anthropic/Bedrock model asynchronously."""
    system_prefix = kwargs.pop("system_prefix", None)
    system = _build_system(system_msg, system_prefix)
    kwargs.setdefault("max_tokens", ANTHROPIC_DEFAULT_MAX_TOKENS)
    new_msg_history = msg_history + [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": msg,
                }
            ],
        }
    ]
    thought = ""
    if output_model is not None:
        # Instructor-wrapped async client.
        content, response = await client.create_with_completion(
            model=model,
            max_tokens=kwargs.get("max_tokens", ANTHROPIC_DEFAULT_MAX_TOKENS),
            messages=[{"role": "system", "content": system_msg}, *new_msg_history],
            response_model=output_model,
        )
        new_msg_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": str(content)}]}
        )
    else:
        response = await client.messages.create(
            model=model,
            system=system,
            messages=new_msg_history,
            **kwargs,
        )
        # Separate thinking from non-thinking content
        if len(response.content) == 1:
            content = response.content[0].text
        else:
            thought = response.content[0].thinking
            content = response.content[1].text
        new_msg_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": content}]}
        )
    cost_results = get_anthropic_costs(response, model)
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
