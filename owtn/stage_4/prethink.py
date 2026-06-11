"""Phase 1 — PreThink.

Single-agent tool-use loop. The writer agent walks each scene of the
structural plan in natural language and writes per-scene plans into
`pre_think.md`. Loop terminates when the agent calls `finalize_pre_think`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from owtn.orchestration import Agent, SessionState, ToolRegistry, write_transcript
from owtn.prompts.stage_4 import build_prethink_user_prompt
from owtn.stage_4._explore import phase_sampler, run_writer_loop


logger = logging.getLogger(__name__)


@dataclass
class PreThinkPhase:
    """Phase 1 implementation. Mirrors the explore-then-commit shape from
    Stage 3, single-agent."""

    name: str = "phase_1_prethink"
    explore_max_iters: int = 20
    reasoning_effort: str = "medium"
    temperature: float = 1.0

    async def run(
        self,
        agents: list[Agent],
        state: SessionState,
        registry: ToolRegistry,
    ) -> SessionState:
        if len(agents) != 1:
            raise ValueError(
                f"PreThinkPhase expects exactly one agent (the writer); got {len(agents)}"
            )
        agent = agents[0]
        user_msg = build_prethink_user_prompt()
        sampler = phase_sampler(
            agent.model,
            reasoning_effort=self.reasoning_effort,
            temperature=self.temperature,
        )

        result = await run_writer_loop(
            agent=agent,
            user_msg=user_msg,
            phase_name=self.name,
            state=state,
            registry=registry,
            max_iters=self.explore_max_iters,
            sampler=sampler,
            nudge_msg=(
                "Your PreThink pass is over and `pre_think.md` already holds "
                "your per-scene plans. Your only remaining action is to call "
                "`finalize_pre_think` now to commit and release the phase. "
                "Do not write or edit further; the only available tool is "
                "`finalize_pre_think`."
            ),
            nudge_commit_check=lambda s: bool(
                s.payload.get("phase_1_prethink", {}).get("committed_by")
            ),
            nudge_tool_names=("finalize_pre_think",),
        )

        committed = state.payload.get("phase_1_prethink", {}).get("committed_by")
        if not committed:
            raise RuntimeError(
                f"PreThink ended without finalize_pre_think being called "
                f"after nudge (agent {agent.id}, {len(result.new_msg_history)} turns)"
            )

        state.payload[self.name].setdefault("turns", len(result.new_msg_history))
        write_transcript(
            agent_id=agent.id,
            label=self.name,
            title=f"Phase 1 (pre-think) — {agent.id}",
            system_msg=agent.system_prompt,
            msg_history=result.new_msg_history,
        )
        logger.info(
            "phase_1_prethink: %s committed (cost $%.4f, %d turns)",
            agent.id, result.cost, len(result.new_msg_history),
        )
        return state


