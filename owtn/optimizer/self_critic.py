"""Self-critic critique-revise cycle for generation calls.

Lives in `owtn/optimizer/` because it's a Stage-1-generation-specific
post-process — it imports `owtn.prompts.stage_1.registry` and shinka's
patch appliers. It is NOT a generic LLM utility; that's why it's not in
`owtn/llm/`.

Public surface: `query_async` (a wrapper over `owtn.llm.api.query_async`
that fires the critique-revise cycle when (a) the call's role is in
`_GENERATION_ROLES` AND (b) the model is registered) and `register_models`.

Currently bound to Stage-1 prompts (`owtn/prompts/stage_1/self_critic_*.txt`)
and the Stage-1 operator registry's full/diff routing. Stage-2's expansion
calls don't go through this — the user-message reconstruction depends on
the operator routing model, which is Stage-1-specific.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Mapping, Optional, Type

from pydantic import BaseModel

from owtn.llm import call_logger
from owtn.llm.api import query_async as _llm_query_async
from owtn.llm.providers import QueryResult

logger = logging.getLogger(__name__)


# Generation roles that trigger the cycle when the model is opted in.
# Other roles (pairwise_judge, classifier, run_brief, tournament, etc.)
# are evaluative or meta — never self-critique them.
_GENERATION_ROLES = frozenset({"generation", "genesis", "mutation"})

# {model_name: critic-call reasoning_effort override}. "disabled" strips
# thinking kwargs before the critic sub-call (the default — critique IS
# the thinking); other values pass kwargs through unchanged.
_self_critic_config: Dict[str, str] = {}


def register_models(configs: Mapping[str, str]) -> None:
    """Mark which generation model names should run through the cycle.

    `configs` maps model_name → reasoning_effort for the critic sub-call.
    Safe to call multiple times — replaces the registry."""
    _self_critic_config.clear()
    _self_critic_config.update(configs)


# Legacy alias retained for runner.py and tests that haven't migrated.
register_self_critic_models = register_models


def _strip_reasoning_kwargs(kwargs: dict) -> dict:
    """Return a copy of kwargs with thinking/reasoning toggles removed.

    Covers every provider's path to reasoning:
      - `reasoning_effort` (OpenAI, top-level)
      - `thinking` (Anthropic, dict)
      - `extra_body.thinking` (DeepSeek)
      - `extra_body.reasoning` (OpenRouter — GLM and friends)
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
def _load_prompt(name: str) -> str:
    """Read a self-critic prompt file from owtn/prompts/stage_1/."""
    path = Path(__file__).resolve().parents[1] / "prompts" / "stage_1" / name
    return path.read_text()


# Public alias used by tests that pin the legacy contract.
def _load_self_critic_prompt(name: str) -> str:
    return _load_prompt(name)


@contextmanager
def _scoped_role(role: str):
    """Set llm_context.role for the duration of the block, preserving
    other ctx fields (operator, generation, parent_code, ...) and
    restoring on exit."""
    base = call_logger.llm_context.get({})
    token = call_logger.llm_context.set({**base, "role": role})
    try:
        yield
    finally:
        call_logger.llm_context.reset(token)


def _reconstruct_genome(
    *, operator: Optional[str], parent_code: Optional[str], raw_output: str
) -> Optional[str]:
    """Return the resulting genome JSON for the critic's user message.

    Routes through the same extractors the pipeline uses to produce
    Program.code, so the critic sees exactly what becomes the genome:
      - full routing → extract the ```json``` fence (apply_full_patch)
      - diff routing → apply SEARCH/REPLACE blocks against parent_code

    Returns None if extraction fails — the caller skips the cycle in
    that case; the pipeline's own apply step will hit the same failure.
    """
    from owtn.prompts.stage_1.registry import OPERATOR_DEFS
    from shinka.edit.apply_diff import apply_diff_patch
    from shinka.edit.apply_full import apply_full_patch

    routing = OPERATOR_DEFS.get(operator or "", {}).get("routing", "full")

    if routing == "diff":
        if not parent_code:
            return None
        updated, num_applied, _, error, _, _ = apply_diff_patch(
            patch_str=raw_output, original_str=parent_code,
            language="json", verbose=False,
        )
        if num_applied == 0 or error:
            return None
        return updated

    updated, num_applied, _, error, _, _ = apply_full_patch(
        patch_str=raw_output, original_str=parent_code or "{}",
        language="json", verbose=False,
    )
    if num_applied == 0 or error:
        return None
    return updated


# Indirection so tests can patch `_query_async_single` and mock the actual
# LLM dispatch without making real API calls.
async def _query_async_single(
    *,
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: list,
    output_model: Optional[Type[BaseModel]],
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Single LLM call that bypasses the self-critic gate. Used by both the
    cycle (for critic + revise sub-calls) and the top-level wrapper (for
    the initial call). Tests patch this to count calls + inspect kwargs."""
    return await _llm_query_async(
        model_name=model_name, msg=msg, system_msg=system_msg,
        msg_history=msg_history, output_model=output_model,
        model_posteriors=model_posteriors, **kwargs,
    )


async def _run_cycle(
    *,
    model_name: str,
    original_msg: str,
    original_system_msg: str,
    original_result: QueryResult,
    output_model: Optional[Type[BaseModel]],
    model_posteriors: Optional[Dict[str, float]],
    **kwargs,
) -> QueryResult:
    """Run one critique-revise round. Three LLM calls total: caller's
    initial (already done), critic, revise. Returns the revise result.

    Critic call has thinking kwargs stripped (when configured "disabled").
    Revise call inherits the generator's original kwargs.

    If genome reconstruction fails, the cycle is skipped and the original
    result is returned unchanged."""
    base_ctx = call_logger.llm_context.get({})
    critic_user_msg = _reconstruct_genome(
        operator=base_ctx.get("operator"),
        parent_code=base_ctx.get("parent_code"),
        raw_output=original_result.content,
    )
    if critic_user_msg is None:
        return original_result

    effort = _self_critic_config.get(model_name, "disabled")
    critic_kwargs = _strip_reasoning_kwargs(kwargs) if effort == "disabled" else kwargs

    with _scoped_role("self_critic_review"):
        critic_result = await _query_async_single(
            model_name=model_name,
            msg=critic_user_msg,
            system_msg=_load_prompt("self_critic_system.txt"),
            msg_history=[],
            output_model=None,
            model_posteriors=model_posteriors,
            **critic_kwargs,
        )

    with _scoped_role("self_critic_revise"):
        return await _query_async_single(
            model_name=model_name,
            msg=(_load_prompt("self_critic_revise.txt").strip() + "\n\n" + (critic_result.content or "")),
            system_msg=original_system_msg,
            msg_history=[
                {"role": "user", "content": original_msg},
                {"role": "assistant", "content": original_result.content},
            ],
            output_model=output_model,
            model_posteriors=model_posteriors,
            **kwargs,
        )


async def query_async(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: list = [],
    output_model: Optional[Type[BaseModel]] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Generation entry point. Falls through to the raw primitive when the
    current call doesn't qualify (non-generation role, or unregistered model).

    *Implicit state:*
    - `llm_context.role` (contextvar) decides whether this is a generation call.
    - `_self_critic_config` (mutated by `register_models`) decides which models opt in.
    """
    result = await _query_async_single(
        model_name=model_name, msg=msg, system_msg=system_msg,
        msg_history=msg_history, output_model=output_model,
        model_posteriors=model_posteriors, **kwargs,
    )

    role = call_logger.llm_context.get({}).get("role")
    if role in _GENERATION_ROLES and model_name in _self_critic_config:
        result = await _run_cycle(
            model_name=model_name, original_msg=msg, original_system_msg=system_msg,
            original_result=result, output_model=output_model,
            model_posteriors=model_posteriors, **kwargs,
        )
    return result
