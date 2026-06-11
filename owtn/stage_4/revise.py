"""Phase 3 — Revise.

Cycled sub-phases A (gather) and B (revise) until plateau, cycle cap,
call ceiling, or the agent calls `finalize_stage_4`. Owns the cycle
loop internally — the umbrella phase name (`phase_3_revise`) is the log
directory key; sub-phase tool allowlists key off `phase_3a_gather` and
`phase_3b_revise`.

At the start of each cycle's sub-phase A, `prelaunch_critics` kicks off
the cycle's mandatory critics as background tasks. Cycle 0 pre-launches
Tier A + Tier B (the full sweep — matches the architecture's "down
draft → sweep → revise" pattern). Cycles 1+ pre-launch Tier A only;
Tier B is on-demand. The agent's `call_critic` calls await pre-launched
tasks transparently; cancel-on-divergent-focus avoids waste.

At the end of sub-phase A, any prelaunched task the agent didn't await
gets drained — its report stashed on the cycle for plateau detection
even if the agent didn't look at it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from owtn.orchestration import Agent, SessionState, ToolRegistry, write_transcript
from owtn.prompts.stage_4 import (
    build_revise_apply_user_prompt,
    build_revise_gather_user_prompt,
)
from owtn.stage_4._explore import phase_sampler, run_writer_loop
from owtn.stage_4.critics import CriticRegistry, drain_prelaunched, prelaunch_critics
from owtn.stage_4.plateau import PlateauDetector, PlateauVerdict


logger = logging.getLogger(__name__)


GATHER_PHASE_KEY = "phase_3a_gather"
REVISE_PHASE_KEY = "phase_3b_revise"


@dataclass
class RevisePhase:
    name: str = "phase_3_revise"
    cycle_cap: int = 6
    call_ceiling: int = 800
    gather_max_iters: int = 40
    revise_max_iters: int = 40
    tier_b_in_cycle_zero: bool = True
    reasoning_effort: str = "medium"
    temperature: float = 1.0
    plateau_detector: PlateauDetector = field(default_factory=PlateauDetector)

    async def run(
        self,
        agents: list[Agent],
        state: SessionState,
        registry: ToolRegistry,
    ) -> SessionState:
        if len(agents) != 1:
            raise ValueError(
                f"RevisePhase expects exactly one agent (the writer); got {len(agents)}"
            )
        agent = agents[0]

        critic_registry = state.payload.get("critic_registry")
        if not isinstance(critic_registry, CriticRegistry):
            raise RuntimeError("critic_registry not in state.payload")

        state.payload[self.name] = {"cycles": []}
        critic_lines = _render_critic_list(critic_registry)
        sampler = phase_sampler(
            agent.model,
            reasoning_effort=self.reasoning_effort,
            temperature=self.temperature,
        )

        cycle_idx = 0
        total_calls = 0
        while True:
            cycle_record = _begin_cycle(state, cycle_idx)
            await self._prelaunch_for_cycle(state, critic_registry, cycle_idx)

            gather_result = await run_writer_loop(
                agent=agent,
                user_msg=build_revise_gather_user_prompt(critic_lines=critic_lines),
                phase_name=GATHER_PHASE_KEY,
                state=state,
                registry=registry,
                max_iters=self.gather_max_iters,
                sampler=sampler,
            )
            total_calls += len(gather_result.new_msg_history)
            write_transcript(
                agent_id=agent.id,
                label="gather",
                title=f"Revise — gather, cycle {cycle_idx} — {agent.id}",
                system_msg=agent.system_prompt,
                msg_history=gather_result.new_msg_history,
                sub_dirs=(self.name, f"cycle_{cycle_idx}"),
            )
            if cycle_record["plan"] is None:
                raise RuntimeError(
                    f"sub-phase A ended without finalize_critique_plan "
                    f"(cycle {cycle_idx}, {len(gather_result.new_msg_history)} turns)"
                )

            await drain_prelaunched(state.payload)

            apply_result = await run_writer_loop(
                agent=agent,
                user_msg=build_revise_apply_user_prompt(
                    plan_summary=cycle_record["plan"]["plan_summary"],
                    intended_revisions=cycle_record["plan"].get("intended_revisions", []),
                ),
                phase_name=REVISE_PHASE_KEY,
                state=state,
                registry=registry,
                max_iters=self.revise_max_iters,
                sampler=sampler,
            )
            total_calls += len(apply_result.new_msg_history)
            write_transcript(
                agent_id=agent.id,
                label="apply",
                title=f"Revise — apply, cycle {cycle_idx} — {agent.id}",
                system_msg=agent.system_prompt,
                msg_history=apply_result.new_msg_history,
                sub_dirs=(self.name, f"cycle_{cycle_idx}"),
            )

            stage_committed = bool(state.payload[self.name].get("committed_by"))
            if stage_committed:
                logger.info(
                    "phase_3_revise: finalize_stage_4 by %s after cycle %d",
                    agent.id, cycle_idx,
                )
                break

            if not cycle_record["completed"]:
                raise RuntimeError(
                    f"sub-phase B ended without finalize_cycle "
                    f"(cycle {cycle_idx}, {len(apply_result.new_msg_history)} turns)"
                )

            cycle_idx += 1

            if cycle_idx >= self.cycle_cap:
                logger.info("phase_3_revise: cycle cap reached (%d)", self.cycle_cap)
                state.payload[self.name]["exit_reason"] = "cycle_cap"
                break
            if total_calls >= self.call_ceiling:
                logger.info("phase_3_revise: call ceiling reached (%d turns)", total_calls)
                state.payload[self.name]["exit_reason"] = "call_ceiling"
                break

            verdict = self.plateau_detector.check(state.payload[self.name]["cycles"])
            if verdict.plateaued:
                logger.info("phase_3_revise: plateau (%s)", verdict.reason)
                state.payload[self.name]["exit_reason"] = f"plateau:{verdict.reason}"
                state.payload[self.name]["plateau_verdict"] = _verdict_dict(verdict)
                break

        state.payload[self.name]["cycles_completed"] = sum(
            1 for c in state.payload[self.name]["cycles"] if c.get("completed")
        )
        state.payload[self.name]["total_subphase_turns"] = total_calls
        state.payload[self.name].setdefault(
            "exit_reason",
            "finalize_stage_4" if state.payload[self.name].get("committed_by") else "unknown",
        )
        return state

    async def _prelaunch_for_cycle(
        self,
        state: SessionState,
        critic_registry: CriticRegistry,
        cycle_idx: int,
    ) -> None:
        ids = list(critic_registry.tier_a_ids())
        if cycle_idx == 0 and self.tier_b_in_cycle_zero:
            ids.extend(critic_registry.tier_b_ids())
        await prelaunch_critics(ids, state_payload=state.payload, cycle=cycle_idx)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _begin_cycle(state: SessionState, cycle_idx: int) -> dict:
    """Append a fresh cycle record to state.payload's Phase 3 cycle list."""
    cycle_record = {
        "cycle": cycle_idx,
        "critic_calls": [],
        "plan": None,
        "completed": False,
        "reports_seen": {},
        "prelaunched": {},
    }
    state.payload["phase_3_revise"]["cycles"].append(cycle_record)
    return cycle_record


def _render_critic_list(registry: CriticRegistry) -> list[str]:
    """Bullet lines naming each critic id and tier — rendered into the
    sub-phase A user prompt so the agent knows what `call_critic` accepts.

    Domain-expert critics are surfaced with their domain descriptor;
    without it the writer sees opaque ids like `domain_expert_quantum_optics`
    and has no basis for choosing whether to consult them.
    """
    lines: list[str] = []
    for cid in registry.tier_a_ids():
        lines.append(f"- {cid} (tier_a, mandatory)")
    for cid in registry.tier_b_ids():
        lines.append(f"- {cid} (tier_b)")
    for persona in registry.personas():
        if persona.tier != "domain":
            continue
        descriptor = persona.name or persona.id
        lines.append(f"- {persona.id} ({descriptor})")
    return lines


def _verdict_dict(verdict: PlateauVerdict) -> dict:
    return asdict(verdict)
