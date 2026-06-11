"""Phase 2 — DownDraft.

Single-agent tool-use loop. Per-scene write to `story.md` in topological
order with prompt-side read-as-reader micro-loop. Reasoning is OFF in
this phase — thinking mode is "more detached" / unhelpful for prose
generation per `project_voice_api_techniques.md`. Loop terminates when
the agent calls `finalize_down_draft`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from owtn.orchestration import Agent, SessionState, ToolRegistry, write_transcript
from owtn.prompts.stage_4 import build_downdraft_user_prompt
from owtn.stage_4._explore import phase_sampler, run_writer_loop


logger = logging.getLogger(__name__)


@dataclass
class DownDraftPhase:
    name: str = "phase_2_downdraft"
    explore_max_iters: int = 80
    temperature: float = 1.0
    reasoning_effort: str = "disabled"

    async def run(
        self,
        agents: list[Agent],
        state: SessionState,
        registry: ToolRegistry,
    ) -> SessionState:
        if len(agents) != 1:
            raise ValueError(
                f"DownDraftPhase expects exactly one agent (the writer); got {len(agents)}"
            )
        agent = agents[0]
        user_msg = build_downdraft_user_prompt()
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
                "Your DownDraft pass is over and `story.md` already holds "
                "the manuscript you wrote. Your only remaining action is to "
                "call `finalize_down_draft` now to commit the down-draft and "
                "release the phase. Do not write or edit further; the only "
                "available tool is `finalize_down_draft`."
            ),
            nudge_commit_check=lambda s: bool(
                s.payload.get("phase_2_downdraft", {}).get("committed_by")
            ),
            nudge_tool_names=("finalize_down_draft",),
        )

        committed = state.payload.get("phase_2_downdraft", {}).get("committed_by")
        if not committed:
            raise RuntimeError(
                f"DownDraft ended without finalize_down_draft being called "
                f"after nudge (agent {agent.id}, {len(result.new_msg_history)} turns)"
            )

        state.payload[self.name].setdefault("turns", len(result.new_msg_history))
        write_transcript(
            agent_id=agent.id,
            label=self.name,
            title=f"Phase 2 (down draft) — {agent.id}",
            system_msg=agent.system_prompt,
            msg_history=result.new_msg_history,
        )
        logger.info(
            "phase_2_downdraft: %s committed (cost $%.4f, %d turns)",
            agent.id, result.cost, len(result.new_msg_history),
        )
        return state
