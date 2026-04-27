from typing import List, Union, Optional
import random
from .providers.pricing import (
    is_reasoning_model,
    has_fixed_temperature,
    requires_reasoning,
)
from .providers.model_resolver import resolve_model_backend
import logging

logger = logging.getLogger(__name__)

THINKING_TOKENS = {
    "min": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "max": 16384,
}


def sample_batch_kwargs(
    num_samples: int,
    model_names: Union[List[str], str] = "gpt-4o-mini-2024-07-18",
    temperatures: Union[List[float], float] = 0.0,
    max_tokens: Union[List[int], int] = 4096,
    reasoning_efforts: Union[List[str], str] = "",
    model_sample_probs: Optional[List[float]] = None,
    unique_filter: bool = False,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
):
    """Sample a dictionary of kwargs for a given model."""
    all_kwargs = []
    attempts = 0
    max_attempts = num_samples * 10  # Prevent infinite loops

    while len(all_kwargs) < num_samples and attempts < max_attempts:
        kwargs_dict = sample_model_kwargs(
            model_names=model_names,
            temperatures=temperatures,
            max_tokens=max_tokens,
            reasoning_efforts=reasoning_efforts,
            model_sample_probs=model_sample_probs,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

        if unique_filter:
            if kwargs_dict not in all_kwargs:
                all_kwargs.append(kwargs_dict)
        else:
            all_kwargs.append(kwargs_dict)

        attempts += 1

    if len(all_kwargs) < num_samples:
        logger.info(
            f"Could not generate {num_samples} unique kwargs combinations "
            f"after {max_attempts} attempts"
        )
        logger.info(f"Returning {len(all_kwargs)} unique kwargs combinations.")

    return all_kwargs


def sample_model_kwargs(
    model_names: Union[List[str], str] = "gpt-4o-mini-2024-07-18",
    temperatures: Union[List[float], float] = 0.0,
    max_tokens: Union[List[int], int] = 4096,
    reasoning_efforts: Union[List[str], str] = "",
    model_sample_probs: Optional[List[float]] = None,
    top_p: Union[List[Optional[float]], Optional[float]] = None,
    top_k: Union[List[Optional[int]], Optional[int]] = None,
    min_p: Union[List[Optional[float]], Optional[float]] = None,
):
    """Sample a dictionary of kwargs for a given model.

    When a param list has the same length as ``model_names``, it is treated as
    a parallel list indexed by the sampled model. When it has length 1 or is
    a scalar, it broadcasts to every model. This lets callers pass per-model
    params (e.g. ``temperatures=[1.2, 1.0]`` alongside ``model_names=["ds", "claude"]``)
    without independent sampling breaking the per-model intent.
    """
    # Make all inputs lists
    if isinstance(model_names, str):
        model_names = [model_names]
    if isinstance(temperatures, (int, float)):
        temperatures = [float(temperatures)]
    if isinstance(max_tokens, int):
        max_tokens = [max_tokens]
    if isinstance(reasoning_efforts, str):
        reasoning_efforts = [reasoning_efforts]
    if not isinstance(top_p, list):
        top_p = [top_p]
    if not isinstance(top_k, list):
        top_k = [top_k]
    if not isinstance(min_p, list):
        min_p = [min_p]

    kwargs_dict = {}

    # 1. SAMPLE: model index (then look up name + all parallel params by index)
    n = len(model_names)
    if model_sample_probs is not None:
        if len(model_sample_probs) != n:
            raise ValueError(
                "model_sample_probs must have the same length as model_names"
            )
        if not abs(sum(model_sample_probs) - 1.0) < 1e-9:
            raise ValueError("model_sample_probs must sum to 1")
        model_idx = random.choices(range(n), weights=model_sample_probs, k=1)[0]
    else:
        model_idx = random.randrange(n)
    kwargs_dict["model_name"] = model_names[model_idx]

    def _pick(values):
        """Parallel-index if list matches model count, else random.choice."""
        if len(values) == n:
            return values[model_idx]
        return random.choice(values)

    model_name = kwargs_dict["model_name"]
    resolved_model = resolve_model_backend(model_name)
    api_model_name = resolved_model.api_model_name
    provider = resolved_model.provider

    # 2. SAMPLE: reasoning effort (per-model when list matches model count)
    if is_reasoning_model(api_model_name):
        r_effort = _pick(reasoning_efforts)
    else:
        r_effort = "disabled"

    # Some opennrouter models only support running with reasoning effort
    if requires_reasoning(api_model_name) and r_effort == "disabled":
        r_effort = "low"

    # 3. SAMPLE: temperature (per-model when list matches model count). Force
    # 1.0 only when reasoning is actually active on a think_temp_fixed model
    # (Anthropic extended-thinking API requirement; OpenAI reasoning models
    # with requires_reasoning=1 are coerced above so this clause still fires
    # for them). Non-thinking calls honor the config temperature regardless
    # of provider gateway.
    if has_fixed_temperature(api_model_name) and r_effort != "disabled":
        kwargs_dict["temperature"] = 1.0
    else:
        kwargs_dict["temperature"] = _pick(temperatures)

    # 4.a) SET: max_output_tokens for OpenAI reasoning effort
    if provider in ("openai", "openrouter", "azure_openai") and is_reasoning_model(
        api_model_name
    ):
        kwargs_dict["max_output_tokens"] = random.choice(max_tokens)
        if r_effort == "disabled":
            kwargs_dict["reasoning"] = {"effort": None}
        elif r_effort == "min":
            kwargs_dict["reasoning"] = {"effort": "low"}
        elif r_effort == "max":
            kwargs_dict["reasoning"] = {"effort": "high"}
        else:
            kwargs_dict["reasoning"] = {"effort": r_effort}

        # 4.b.1) SET: auto-summarization for OpenAI reasoning effort
        if provider == "openai" and r_effort != "disabled":
            kwargs_dict["reasoning"]["summary"] = "auto"

    # 4.b) SET: max_tokens for Google reasoning effort
    elif provider == "google" and is_reasoning_model(api_model_name):
        kwargs_dict["max_tokens"] = random.choice(max_tokens)
        think_bool = r_effort != "disabled"
        if think_bool:
            t = THINKING_TOKENS[r_effort]
            thinking_tokens = t if t < kwargs_dict["max_tokens"] else 1024
            kwargs_dict["thinking_budget"] = thinking_tokens
        else:
            if api_model_name in ("gemini-2.5-pro", "gemini-3-pro-preview"):
                kwargs_dict["thinking_budget"] = 128
            else:
                kwargs_dict["thinking_budget"] = 0

    # 4.b.2) SET: reasoning controls for DeepSeek reasoning models.
    # DeepSeek's OpenAI-format API separates two params:
    #   - `thinking={"type": "enabled/disabled"}` — mode toggle
    #   - `reasoning_effort` — effort level when enabled
    # The OpenAI SDK's Chat Completions `create()` doesn't accept `thinking`
    # as a top-level kwarg, so we pass it via `extra_body`. `reasoning_effort`
    # IS accepted at top-level and flows through as a JSON body field.
    # Kwarg-omission leaves the server default (reasoning-on), so we must
    # explicitly send `thinking={"type": "disabled"}` to actually turn off.
    elif provider == "deepseek" and is_reasoning_model(api_model_name):
        kwargs_dict["max_tokens"] = random.choice(max_tokens)
        if r_effort == "disabled":
            kwargs_dict["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs_dict["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs_dict["reasoning_effort"] = r_effort

    # 4.c) SET: max_tokens for Anthropic or Bedrock reasoning effort
    elif provider in ("anthropic", "bedrock") and is_reasoning_model(api_model_name):
        kwargs_dict["max_tokens"] = min(random.choice(max_tokens), 64000)
        think_bool = r_effort != "disabled"
        if think_bool:
            # filter thinking tokens to be smaller than max_tokens
            # not auto THINKING_TOKENS
            t = THINKING_TOKENS[r_effort]
            thinking_tokens = t if t < kwargs_dict["max_tokens"] else 1024
            # sample only from thinking tokens that are valid
            kwargs_dict["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_tokens,
            }

    # 4.d) SET: max_tokens for all other models
    else:
        # Non-reasoning models or other providers
        if provider in ("anthropic", "bedrock", "deepseek", "local_openai"):
            kwargs_dict["max_tokens"] = random.choice(max_tokens)
        else:
            kwargs_dict["max_output_tokens"] = random.choice(max_tokens)

    # 5. SET: top_p / top_k when supported. Dropped for OpenAI reasoning
    # (top_p ignored/disallowed, top_k unsupported) and for Anthropic under
    # extended thinking (both forbidden when thinking.type=enabled).
    openai_family = provider in ("openai", "openrouter", "azure_openai")
    openai_reasoning = openai_family and is_reasoning_model(api_model_name)
    anthropic_thinking = "thinking" in kwargs_dict

    top_p_val = _pick(top_p)
    top_k_val = _pick(top_k)
    min_p_val = _pick(min_p)

    if top_p_val is not None and not openai_reasoning and not anthropic_thinking:
        kwargs_dict["top_p"] = top_p_val

    if top_k_val is not None and not openai_family and not anthropic_thinking:
        kwargs_dict["top_k"] = top_k_val

    # min_p: only forwarded on OpenRouter (which passes arbitrary sampler
    # params to the upstream). Native Anthropic/OpenAI/DeepSeek/Google APIs
    # reject unknown kwargs, so drop there.
    if min_p_val is not None and provider == "openrouter" and not anthropic_thinking:
        kwargs_dict["min_p"] = min_p_val

    return kwargs_dict
