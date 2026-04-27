import functools
import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from pydantic import BaseModel

from . import call_logger
from .client import get_async_client_llm, get_client_llm
from .providers import (
    QueryResult,
    query_anthropic,
    query_anthropic_async,
    query_deepseek,
    query_deepseek_async,
    query_gemini,
    query_gemini_async,
    query_local_openai,
    query_local_openai_async,
    query_openai,
    query_openai_async,
)

logger = logging.getLogger(__name__)


# Provider → query function. Same shape sync + async; one dispatch table per
# call style. Replaces an if/elif chain that needed to be kept in sync across
# both `query` and `_query_async_single`.
_PROVIDER_QUERY_FNS = {
    "anthropic": query_anthropic,
    "bedrock": query_anthropic,
    "openai": query_openai,
    "azure_openai": query_openai,
    "openrouter": query_openai,
    "deepseek": query_deepseek,
    "google": query_gemini,
    "local_openai": query_local_openai,
}
_PROVIDER_QUERY_FNS_ASYNC = {
    "anthropic": query_anthropic_async,
    "bedrock": query_anthropic_async,
    "openai": query_openai_async,
    "azure_openai": query_openai_async,
    "openrouter": query_openai_async,
    "deepseek": query_deepseek_async,
    "google": query_gemini_async,
    "local_openai": query_local_openai_async,
}


# Generation roles that trigger a self-critic cycle when the model is opted in.
# Other roles (pairwise_judge, classifier, embedding, run_brief, tournament,
# etc.) are evaluative or meta; never self-critique them.
_GENERATION_ROLES = frozenset({"generation", "genesis", "mutation"})

# Self-critic registry. Keys are model names with self-critic enabled; values
# are the reasoning_effort override for the critic sub-call. Populated at
# runner init from the StageConfig. Module-level so the query path can
# consult it without threading config through every call site.
_self_critic_config: dict[str, str] = {}


def register_self_critic_models(configs: Mapping[str, str]) -> None:
    """Mark which generation model names should run through a self-critic
    critique-revise cycle.

    ``configs`` maps model_name → reasoning_effort for the critic sub-call.
    "disabled" (the default) strips thinking/reasoning kwargs before the
    critic call; other values pass kwargs through unchanged. Safe to call
    multiple times — replaces the registry.
    """
    global _self_critic_config
    _self_critic_config = dict(configs)


def _strip_reasoning_kwargs(kwargs: dict) -> dict:
    """Return a copy of kwargs with thinking/reasoning toggles removed.

    Covers every provider's path to reasoning:
      - ``reasoning_effort`` (OpenAI reasoning models, top-level kwarg)
      - ``thinking`` (Anthropic, dict)
      - ``extra_body.thinking`` (DeepSeek — OpenAI-compat thinking toggle)
      - ``extra_body.reasoning`` (OpenRouter — GLM and friends)
    """
    out = dict(kwargs)
    out.pop("reasoning_effort", None)
    out.pop("thinking", None)
    extra = out.get("extra_body")
    if isinstance(extra, dict):
        extra = {k: v for k, v in extra.items() if k not in ("thinking", "reasoning")}
        out["extra_body"] = extra if extra else None
        if out["extra_body"] is None:
            out.pop("extra_body")
    return out


@functools.lru_cache(maxsize=8)
def _load_self_critic_prompt(name: str) -> str:
    """Read a self-critic prompt file. Cached for the life of the process —
    these files don't change between calls."""
    return (Path(__file__).resolve().parents[1] / "prompts" / "stage_1" / name).read_text()


@contextmanager
def _scoped_llm_role(role: str):
    """Set llm_context.role for the duration of the block, preserving any
    other ctx fields (operator, generation, parent_code, ...) and restoring
    on exit. Used to tag critic + revise sub-calls with their distinct
    roles without leaking those roles back to the outer caller's ctx.
    """
    base = call_logger.llm_context.get({})
    token = call_logger.llm_context.set({**base, "role": role})
    try:
        yield
    finally:
        call_logger.llm_context.reset(token)


def _log_result(result: QueryResult, provider: str, msg: str, system_msg: str, duration: float, kwargs: dict) -> None:
    try:
        content = result.content if isinstance(result.content, str) else str(result.content)
        call_logger.log_call(
            model=result.model_name,
            provider=provider,
            system_msg=system_msg,
            user_msg=msg,
            content=content,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            thinking_tokens=result.thinking_tokens,
            cache_read_tokens=result.cache_read_tokens,
            cache_creation_tokens=result.cache_creation_tokens,
            cost=result.cost or 0.0,
            duration_s=duration,
            thought=result.thought or "",
            kwargs=kwargs,
        )
    except Exception as e:
        logger.warning("LLM call logging failed: %s", e)


CACHE_ENABLED = os.environ.get("OWTN_CACHE_ENABLED", "").lower() in ("1", "true")


def _query_cache_key(args, kwargs):
    """Deterministic cache key from query inputs."""
    key_data = json.dumps(
        {
            "model": args[0] if args else kwargs.get("model_name"),
            "msg": args[1] if len(args) > 1 else kwargs.get("msg"),
            "system_msg": args[2] if len(args) > 2 else kwargs.get("system_msg"),
            "msg_history": args[3] if len(args) > 3 else kwargs.get("msg_history", []),
            "system_prefix": kwargs.get("system_prefix"),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()


def _make_cached(fn):
    """Wrap a query function with cachier if caching is enabled."""
    if not CACHE_ENABLED:
        return fn
    from cachier import cachier

    return cachier(
        cache_dir=os.environ.get("OWTN_CACHE_DIR", ".cache/llm"),
        hash_func=_query_cache_key,
        stale_after=timedelta(days=7),
    )(fn)


def _merge_system_prefix(kwargs, system_msg, provider):
    """Handle system_prefix for non-Anthropic providers.

    Anthropic providers handle system_prefix internally via cache_control
    content blocks. For other providers, merge it into system_msg as a
    plain string prefix (OpenAI auto-caches matching prefixes).
    """
    system_prefix = kwargs.get("system_prefix")
    if system_prefix and provider not in ("anthropic", "bedrock"):
        kwargs.pop("system_prefix")
        system_msg = system_prefix + "\n\n" + system_msg
    return system_msg


@_make_cached
def query(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Query the LLM."""
    client, model_name, provider = get_client_llm(
        model_name, structured_output=output_model is not None
    )
    system_msg = _merge_system_prefix(kwargs, system_msg, provider)
    query_fn = _PROVIDER_QUERY_FNS.get(provider)
    if query_fn is None:
        raise ValueError(f"Model {model_name} not supported.")
    t0 = time.perf_counter()
    result = query_fn(
        client, model_name, msg, system_msg, msg_history, output_model,
        model_posteriors=model_posteriors, **kwargs,
    )
    _log_result(result, provider, msg, system_msg, time.perf_counter() - t0, kwargs)
    return result


async def _query_async_single(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List,
    output_model: Optional[BaseModel],
    model_posteriors: Optional[Dict[str, float]],
    **kwargs,
) -> QueryResult:
    """One provider call + call-log. No self-critic logic — used both by the
    top-level query_async and by the self-critic sub-calls to avoid recursion
    into the critic gate.
    """
    client, model_name, provider = get_async_client_llm(
        model_name, structured_output=output_model is not None
    )
    system_msg = _merge_system_prefix(kwargs, system_msg, provider)
    query_fn = _PROVIDER_QUERY_FNS_ASYNC.get(provider)
    if query_fn is None:
        raise ValueError(f"Model {model_name} not supported.")
    t0 = time.perf_counter()
    result = await query_fn(
        client, model_name, msg, system_msg, msg_history, output_model,
        model_posteriors=model_posteriors, **kwargs,
    )
    _log_result(result, provider, msg, system_msg, time.perf_counter() - t0, kwargs)
    return result


def _reconstruct_genome_for_critic(
    *,
    operator: str | None,
    parent_code: str | None,
    raw_output: str,
) -> str | None:
    """Return the resulting genome JSON for the critic's user message.

    Routes through the same extractors the pipeline uses to produce
    Program.code, so the critic sees exactly what becomes the genome:
      - full routing  → extract the ``` ```json ``` ``` fence (apply_full_patch)
      - diff routing  → apply SEARCH/REPLACE blocks against parent_code
                         (apply_diff_patch)

    Returns None if extraction fails (malformed output) — the caller skips
    the critique-revise cycle in that case; the pipeline's own apply step
    will hit the same failure and retry.
    """
    from owtn.prompts.stage_1.registry import OPERATOR_DEFS
    from shinka.edit.apply_diff import apply_diff_patch
    from shinka.edit.apply_full import apply_full_patch

    routing = OPERATOR_DEFS.get(operator or "", {}).get("routing", "full")

    if routing == "diff":
        if not parent_code:
            return None
        updated, num_applied, _, error, _, _ = apply_diff_patch(
            patch_str=raw_output,
            original_str=parent_code,
            language="json",
            verbose=False,
        )
        if num_applied == 0 or error:
            return None
        return updated

    # full routing (genesis + most mutations)
    updated, num_applied, _, error, _, _ = apply_full_patch(
        patch_str=raw_output,
        original_str=parent_code or "{}",
        language="json",
        verbose=False,
    )
    if num_applied == 0 or error:
        return None
    return updated


async def _run_self_critic(
    *,
    model_name: str,
    original_msg: str,
    original_system_msg: str,
    original_result: QueryResult,
    output_model: Optional[BaseModel],
    model_posteriors: Optional[Dict[str, float]],
    **kwargs,
) -> QueryResult:
    """Run one critique-revise round. Three LLM calls total: caller's initial
    (already done), critic, revise. Returns the revise result, which becomes
    the genome downstream.

    Critic call has thinking kwargs stripped (when configured "disabled" —
    the default; critique IS the thinking). Revise call inherits the
    generator's original kwargs — it's doing the structured genome
    regeneration and benefits from extended reasoning.

    If genome reconstruction fails (malformed output), the cycle is skipped
    and the original result is returned unchanged — the pipeline's own
    apply step will hit the same failure and retry.
    """
    base_ctx = call_logger.llm_context.get({})
    critic_user_msg = _reconstruct_genome_for_critic(
        operator=base_ctx.get("operator"),
        parent_code=base_ctx.get("parent_code"),
        raw_output=original_result.content,
    )
    if critic_user_msg is None:
        return original_result

    effort = _self_critic_config.get(model_name, "disabled")
    critic_kwargs = _strip_reasoning_kwargs(kwargs) if effort == "disabled" else kwargs

    with _scoped_llm_role("self_critic_review"):
        critic_result = await _query_async_single(
            model_name=model_name,
            msg=critic_user_msg,
            system_msg=_load_self_critic_prompt("self_critic_system.txt"),
            msg_history=[],
            output_model=None,  # free-form critique, not structured
            model_posteriors=model_posteriors,
            **critic_kwargs,
        )

    with _scoped_llm_role("self_critic_revise"):
        return await _query_async_single(
            model_name=model_name,
            msg=(
                _load_self_critic_prompt("self_critic_revise.txt").strip()
                + "\n\n" + (critic_result.content or "")
            ),
            system_msg=original_system_msg,
            msg_history=[
                {"role": "user", "content": original_msg},
                {"role": "assistant", "content": original_result.content},
            ],
            output_model=output_model,
            model_posteriors=model_posteriors,
            **kwargs,
        )


@_make_cached
async def query_async(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Query the LLM asynchronously.

    If the current ``llm_context.role`` is a generation role AND the model is
    registered via ``register_self_critic_models``, fires a critique-revise
    cycle after the initial call and returns the revised result. The initial
    call is still logged (with its original role) — its tokens and cost are
    part of the run total.
    """
    result = await _query_async_single(
        model_name=model_name,
        msg=msg,
        system_msg=system_msg,
        msg_history=msg_history,
        output_model=output_model,
        model_posteriors=model_posteriors,
        **kwargs,
    )

    ctx = call_logger.llm_context.get({})
    role = ctx.get("role")
    if role in _GENERATION_ROLES and model_name in _self_critic_config:
        result = await _run_self_critic(
            model_name=model_name,
            original_msg=msg,
            original_system_msg=system_msg,
            original_result=result,
            output_model=output_model,
            model_posteriors=model_posteriors,
            **kwargs,
        )
    return result


