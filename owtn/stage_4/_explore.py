"""Shared explore-loop helper for Stage 4 phases.

The three phases (PreThink, DownDraft, Revise sub-phases) all run the
same shape: a single writer agent in a tool-use loop until it calls a
`finalize_*` tool. This helper does the dispatch wiring + push_llm_context
+ cost accounting + sampler translation; phases own the per-phase user
prompt and per-phase reasoning/temperature.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from owtn.llm.kwargs import sample_model_kwargs
from owtn.llm.tool_use import query_async_with_tools
from owtn.orchestration import (
    Agent,
    SessionState,
    ToolContext,
    ToolRegistry,
    push_llm_context,
)


def phase_sampler(
    model: str, *, reasoning_effort: str, temperature: float, max_tokens: int = 32768,
) -> dict[str, Any]:
    """Translate `reasoning_effort + temperature` to provider-specific
    sampler kwargs. With reasoning ON, Claude forces temperature=1.0;
    DeepSeek allows the full sampler. Caller passes desired values; the
    underlying `sample_model_kwargs` enforces provider rules.
    """
    if reasoning_effort == "disabled":
        return {"temperature": temperature, "max_tokens": max_tokens}
    out = sample_model_kwargs(
        model_names=[model],
        reasoning_efforts=[reasoning_effort],
        temperatures=[temperature],
        max_tokens=[max_tokens],
    )
    out.pop("model_name", None)
    return out


async def run_writer_loop(
    *,
    agent: Agent,
    user_msg: str,
    phase_name: str,
    state: SessionState,
    registry: ToolRegistry,
    max_iters: int,
    sampler: dict[str, Any],
    nudge_msg: str | None = None,
    nudge_commit_check: Any = None,
    nudge_max_iters: int = 6,
    nudge_tool_names: tuple[str, ...] | None = None,
):
    """Run a tool-use loop for the writer agent.

    `phase_name` keys into `registry`'s per-phase allowlist — the tools
    the agent can call in this loop. Returns the QueryResult; the phase
    inspects state.payload for the agent's finalize commit afterward.

    `nudge_msg` and `nudge_commit_check` enable the loop-exit-without-commit
    fallback. If the loop returns without the agent committing (the
    classic "agent produced prose as chat text" failure), the same
    msg_history is re-prompted with `nudge_msg` and the same tool surface,
    giving the agent a chance to write to a file and then finalize.
    `nudge_commit_check(state)` returns True iff the commit landed.

    `nudge_tool_names` restricts the tool surface during the nudge to a
    subset of the phase's allowlist — typically just the phase's finalize
    tool. Without this, the writer often keeps calling write/edit tools
    during the nudge (it's already done writing; the nudge needs it to
    commit) and exhausts the nudge budget without finalizing. Mirrors
    Stage 3's Phase 1 nudge which restricts to `finalize_voice_genome`
    only.
    """
    with push_llm_context(agent_id=agent.id):
        tool_schemas = registry.schemas_for(agent.tools, phase_name)

        async def dispatch(tool_name: str, params: dict) -> str:
            ctx = ToolContext(
                session_id=state.session_id,
                phase_id=phase_name,
                agent_id=agent.id,
                state_view=state.payload,
            )
            return await registry.dispatch(tool_name, params, ctx)

        result = await query_async_with_tools(
            model_name=agent.model,
            msg=user_msg,
            system_msg=agent.system_prompt,
            tools=tool_schemas,
            dispatch=dispatch,
            max_iters=max_iters,
            **sampler,
        )
        state.cost_usd += result.cost

        if nudge_msg is not None and nudge_commit_check is not None and not nudge_commit_check(state):
            nudge_tools = tool_schemas
            if nudge_tool_names is not None:
                nudge_tools = [t for t in tool_schemas if t["name"] in nudge_tool_names]
            nudge_result = await query_async_with_tools(
                model_name=agent.model,
                msg=nudge_msg,
                system_msg=agent.system_prompt,
                msg_history=result.new_msg_history,
                tools=nudge_tools,
                dispatch=dispatch,
                max_iters=nudge_max_iters,
                **sampler,
            )
            state.cost_usd += nudge_result.cost
            # Stitch the histories so the transcript captures both passes.
            # QueryResult is frozen; rebuild via dataclasses.replace.
            result = replace(
                result,
                new_msg_history=list(nudge_result.new_msg_history),
                cost=result.cost + nudge_result.cost,
            )
    return result
