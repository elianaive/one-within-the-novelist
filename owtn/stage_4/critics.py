"""Critic system — registry, dispatch, concurrent pre-launch.

Critics are subagents. Each one reads the manuscript and returns a
`CriticReport` (issue list + severity tags); they never rewrite. Per
`feedback_judge_query_scope.md` and the Stage 3 `ask_judge` rule.

Two dispatch shapes:

- **Single-turn structured output** for criteria-direct critics. The
  critic's persona + mechanism + work context live in the system prompt
  (cached prefix); the user prompt carries the manuscript and an
  optional focus.
- **Tool-using explore-then-commit** for critics with metric or
  external tools (voice_fidelity, domain_expert). The critic runs a
  tool-use loop with `read_file` + the tools its persona declared,
  then commits via `finalize_critic_report`. Mirrors Stage 3's
  finalize_voice_genome shape.

The agent's `call_critic` tool routes to whichever shape the critic's
persona declares (empty `tools` → single-turn; non-empty → tool-using).

Concurrency: at the start of Phase 3 sub-phase A (gather), the
orchestrator calls `prelaunch_critics(...)` to kick off the cycle's
mandatory critics as background tasks. When the agent then calls
`call_critic` without a focus, the handler awaits the pre-launched
task. With a focus, the handler cancels the unfocused pre-launch and
runs a fresh dispatch — focused critique is more useful than waste.

Tier A enforcement is also routed here: every successful `call_critic`
appends `critic_id` to the active cycle's `critic_calls` list. The
`finalize_critique_plan` handler (in `tools.py`) reads this list and
rejects the agent's commit when any of the four Tier A critics is
missing.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from owtn.llm.kwargs import sample_model_kwargs
from owtn.llm.query import query_async
from owtn.llm.tool_use import query_async_with_tools
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_3 import VoiceGenome
from owtn.stage_3.tools import SLOP_SCORE, STYLOMETRY, WRITING_STYLE
from owtn.models.stage_4 import (
    AudienceFraming,
    CriticPersona,
    CriticReport,
    CriticReportBody,
)
from owtn.orchestration import ToolContext, ToolSpec, write_transcript
from owtn.prompts.stage_4 import (
    build_critic_system_prompt,
    build_critic_user_prompt,
)


logger = logging.getLogger(__name__)


# ─── Tier A constant ─────────────────────────────────────────────────────


TIER_A_CRITICS: frozenset[str] = frozenset({
    "voice_fidelity",
    "payload_enactment",
    "continuity",
    "motif_fidelity",
})
"""Critics whose firing is mandatory in every Phase 3 cycle.
`finalize_critique_plan` rejects the agent's commit when any are absent
from the active cycle's `critic_calls`. Per Huang ICLR 2024 — LLMs
cannot self-correct without external feedback."""


# ─── Registry ────────────────────────────────────────────────────────────


class CriticRegistry:
    """Map of critic_id → CriticPersona.

    Built once at session start from `load_critic_pool()` and stashed
    in `state.payload["critic_registry"]`. Read-only after construction.
    """

    def __init__(self, personas: list[CriticPersona]):
        self._by_id: dict[str, CriticPersona] = {}
        for p in personas:
            if p.id in self._by_id:
                raise ValueError(f"duplicate critic id: {p.id!r}")
            self._by_id[p.id] = p

    def __contains__(self, critic_id: str) -> bool:
        return critic_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, critic_id: str) -> CriticPersona:
        try:
            return self._by_id[critic_id]
        except KeyError as e:
            raise KeyError(
                f"unknown critic: {critic_id!r}; available: {sorted(self._by_id)}"
            ) from e

    def ids(self) -> list[str]:
        return sorted(self._by_id)

    def tier_a_ids(self) -> list[str]:
        return sorted(p.id for p in self._by_id.values() if p.is_tier_a)

    def tier_b_ids(self) -> list[str]:
        return sorted(p.id for p in self._by_id.values() if p.is_tier_b)

    def personas(self) -> list[CriticPersona]:
        """All registered personas, in id-sorted order."""
        return [self._by_id[cid] for cid in sorted(self._by_id)]

    def with_replaced(self, persona: CriticPersona) -> "CriticRegistry":
        """Return a new registry where `persona.id` is replaced with the
        given persona. Insert when not present. Existing registries are
        read-only after construction; this is the supported way to swap
        an entry (used by the voice_fidelity Stage 3 promotion)."""
        return CriticRegistry([
            p for p in self._by_id.values() if p.id != persona.id
        ] + [persona])


# ─── State-view helpers ──────────────────────────────────────────────────


def _active_cycle(state_view: Mapping[str, Any]) -> dict | None:
    """The Phase 3 cycle the agent is currently working in. None when no
    cycle is active. Mirrors the helper in `tools.py` so `critics.py`
    doesn't have to import it (avoids the circular path)."""
    p3 = state_view.get("phase_3_revise")
    if not isinstance(p3, dict):
        return None
    cycles = p3.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        return None
    last = cycles[-1]
    if isinstance(last, dict) and not last.get("completed"):
        return last
    return None


def _registry_from(state_view: Mapping[str, Any]) -> CriticRegistry | None:
    reg = state_view.get("critic_registry")
    return reg if isinstance(reg, CriticRegistry) else None


def _read_manuscript(state_view: Mapping[str, Any]) -> str:
    """Load the current manuscript text. Empty string when the file
    doesn't exist yet (e.g. Phase 1 calls before DownDraft has run)."""
    story_path = state_view.get("story_path")
    if story_path is None:
        run_dir = state_view.get("run_dir")
        if run_dir is None:
            return ""
        story_path = Path(run_dir) / "story.md"
    p = Path(story_path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


# ─── Critic tool registry (for tool-using critics) ──────────────────────


_CRITIC_REPORT_SLOT: ContextVar[dict | None] = ContextVar("_critic_report_slot", default=None)
"""Async-task-local slot dispatch_critic uses to capture the report a
tool-using critic commits via `finalize_critic_report`. Set on entry,
read after the explore loop returns, reset on exit. Concurrent
dispatches don't collide because ContextVars are async-task-local."""


async def _finalize_critic_report_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    """Tool handler critic subagents call to commit their report.

    Mirrors Stage 3's `finalize_voice_genome` shape — the model fills
    `CriticReportBody` fields as args; we validate; we stash for the
    dispatcher to extract. The dispatcher attaches `critic_id` and
    `cycle` post-parse so the critic doesn't invent them.
    """
    slot = _CRITIC_REPORT_SLOT.get()
    if slot is None:
        return (
            "ERROR: not in a critic dispatch context — finalize_critic_report "
            "is only available to tool-using critic subagents."
        )
    if slot.get("body") is not None:
        return (
            "ERROR: critic report already committed for this dispatch. "
            "Stop calling tools."
        )
    try:
        body = CriticReportBody.model_validate(params)
    except ValidationError as e:
        return (
            "ERROR: CriticReportBody validation failed. Fix the issues and "
            f"call finalize_critic_report again with corrected fields.\n\n{e}"
        )
    slot["body"] = body
    return (
        "Critic report committed. Stop calling tools — the explore loop "
        "ends when you produce a non-tool-call response."
    )


# Reused tool specs. The critic subagent's tool surface is a subset of
# these, gated by `persona.tools`. Generic tools live in `owtn/tools/`.
from owtn.tools.fetch_page import FETCH_PAGE  # noqa: E402
from owtn.tools.file_ops import READ_FILE  # noqa: E402
from owtn.tools.web_search import WEB_SEARCH  # noqa: E402


FINALIZE_CRITIC_REPORT = ToolSpec(
    name="finalize_critic_report",
    description=(
        "Submit your CriticReport. Pass `not_load_bearing=true` with a "
        "`pursuit_observation` when this dimension isn't load-bearing "
        "for this story; otherwise pass an `issues` list of structured "
        "observations. After this call succeeds, stop calling tools — "
        "the explore loop ends when you produce a non-tool-call response."
    ),
    parameters=CriticReportBody.model_json_schema(),
    handler=_finalize_critic_report_handler,
)


CRITIC_TOOL_REGISTRY: dict[str, ToolSpec] = {
    READ_FILE.name: READ_FILE,
    STYLOMETRY.name: STYLOMETRY,
    SLOP_SCORE.name: SLOP_SCORE,
    WRITING_STYLE.name: WRITING_STYLE,
    WEB_SEARCH.name: WEB_SEARCH,
    FETCH_PAGE.name: FETCH_PAGE,
    FINALIZE_CRITIC_REPORT.name: FINALIZE_CRITIC_REPORT,
}
"""Available tools for tool-using critics. Each persona's `tools` list
declares which subset this particular critic gets; `finalize_critic_report`
is always added so the critic can commit. Unknown names in a persona's
tools list are logged and dropped on dispatch.

`web_search` and `fetch_page` are included here so domain_expert critics
can opt in via their `tools` list — `web_search` finds candidate sources,
`fetch_page` reads them. The agent-level `STAGE_4_PHASE_ALLOW` does NOT
include either — the writer agent never calls them directly."""


def _build_sampler(persona: CriticPersona) -> dict:
    """Sampler kwargs for one critic call. Reasoning effort comes from
    the persona; temperature stays neutral for analytical work (mirrors
    Stage 3's `_analytical_sampler` shape)."""
    if persona.reasoning_effort == "disabled":
        return {"temperature": 0.6}
    out = sample_model_kwargs(
        model_names=[persona.model],
        reasoning_efforts=[persona.reasoning_effort],
        temperatures=[0.6],
        max_tokens=[16384],
    )
    out.pop("model_name", None)
    return out


# ─── Dispatch ────────────────────────────────────────────────────────────


async def dispatch_critic(
    critic_id: str,
    *,
    state_view: Mapping[str, Any],
    cycle: int,
    focus: str | None = None,
) -> CriticReport:
    """Run one critic and return its report.

    Branches on the persona: tool-using critics run an explore-then-commit
    loop with their declared tools; criteria-direct critics run a
    single-turn structured-output call. Either way, the dispatcher
    force-corrects `critic_id` and `cycle` on the returned report so the
    model can't change them.

    Writes a transcript per dispatch under
    `agents/<critic_id>/cycle_<n>_<seq>.transcript.md` so each critical
    read is inspectable post-run.
    """
    registry = _registry_from(state_view)
    if registry is None:
        raise RuntimeError("critic_registry not set in session state")
    persona = registry.get(critic_id)

    system_msg, user_msg = _build_critic_messages(
        persona, state_view, cycle=cycle, focus=focus,
    )

    try:
        if persona.is_tool_using:
            body, msg_history = await _dispatch_tool_using(
                persona,
                system_msg=system_msg,
                user_msg=user_msg,
                state_payload=state_view,
            )
            report = CriticReport(critic_id=critic_id, cycle=cycle, **body.model_dump())
        else:
            single = await _dispatch_single_turn(persona, system_msg=system_msg, user_msg=user_msg)
            report = single.model_copy(update={"critic_id": critic_id, "cycle": cycle})
            msg_history = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": report.model_dump_json(indent=2)},
            ]
    except Exception as e:
        logger.warning("critic %s call failed (%s: %s)", critic_id, type(e).__name__, e)
        raise

    seq = _next_dispatch_seq(state_view, critic_id, cycle)
    write_transcript(
        agent_id=critic_id,
        label=f"cycle_{cycle}_call_{seq}",
        title=f"Critic {critic_id} — cycle {cycle}, call {seq}{f' (focus: {focus})' if focus else ''}",
        system_msg=system_msg,
        msg_history=msg_history,
        final_output=report,
    )
    return report


def _next_dispatch_seq(state_view: Mapping[str, Any], critic_id: str, cycle: int) -> int:
    """Return a 1-indexed dispatch counter scoped to (cycle, critic_id).
    Stored under the active cycle's `dispatch_seq` map so transcripts
    produced by the same critic in the same cycle (e.g., prelaunched +
    focused follow-up) get distinct filenames."""
    cycle_record = _active_cycle(state_view)
    if cycle_record is None:
        return 1
    seq_map: dict = cycle_record.setdefault("dispatch_seq", {})
    seq_map[critic_id] = seq_map.get(critic_id, 0) + 1
    return seq_map[critic_id]


def _build_critic_messages(
    persona: CriticPersona,
    state_view: Mapping[str, Any],
    *,
    cycle: int,
    focus: str | None,
) -> tuple[str, str]:
    """Compose system and user prompts from session state. Shared between
    the two dispatch paths."""
    voice_genome = state_view.get("voice_genome")
    concept = state_view.get("concept")
    dag_rendering = state_view.get("dag_rendering", "")
    audience_framing = state_view.get("audience_framing")

    if not isinstance(voice_genome, VoiceGenome):
        raise RuntimeError("state_view['voice_genome'] missing or wrong type")
    if not isinstance(concept, ConceptGenome):
        raise RuntimeError("state_view['concept'] missing or wrong type")

    af = audience_framing if isinstance(audience_framing, AudienceFraming) else None

    system_msg = build_critic_system_prompt(
        persona,
        voice_genome=voice_genome,
        concept=concept,
        dag_rendering=dag_rendering,
        audience_framing=af,
    )
    user_msg = build_critic_user_prompt(
        _read_manuscript(state_view), cycle=cycle, focus=focus,
    )
    return system_msg, user_msg


async def _dispatch_single_turn(
    persona: CriticPersona, *, system_msg: str, user_msg: str,
) -> CriticReport:
    result = await query_async(
        model_name=persona.model,
        msg=user_msg,
        system_msg=system_msg,
        output_model=CriticReport,
        **_build_sampler(persona),
    )
    report = result.content
    if not isinstance(report, CriticReport):
        raise RuntimeError(
            f"critic {persona.id} returned {type(report).__name__}, expected CriticReport"
        )
    return report


async def _dispatch_tool_using(
    persona: CriticPersona,
    *,
    system_msg: str,
    user_msg: str,
    state_payload: Mapping[str, Any],
    max_iters: int = 10,
    nudge_max_iters: int = 2,
) -> tuple[CriticReportBody, list[dict]]:
    """Tool-using critic explore-then-commit loop. Returns the committed
    body plus the full tool-use msg_history so the caller can write a
    transcript. Caller wraps the body with `critic_id` and `cycle`.

    If the critic exhausts `max_iters` without calling
    `finalize_critic_report`, runs a short nudge pass with the tool
    surface restricted to `finalize_critic_report` only — same pattern
    as Stage 4 PreThink/DownDraft phases. Mirrors the load-bearing
    finding from `2026-04-30-stage-4-nudge-tools-not-restricted.md`:
    leaving the full tool surface available during the nudge lets the
    agent keep researching instead of committing.
    """
    spec_by_name: dict[str, ToolSpec] = {}
    for name in persona.tools:
        spec = CRITIC_TOOL_REGISTRY.get(name)
        if spec is None:
            logger.warning("critic %s: unknown tool %r; skipping", persona.id, name)
            continue
        spec_by_name[name] = spec
    spec_by_name[FINALIZE_CRITIC_REPORT.name] = FINALIZE_CRITIC_REPORT
    tool_schemas = [
        {"name": s.name, "description": s.description, "parameters": dict(s.parameters)}
        for s in spec_by_name.values()
    ]
    finalize_only_schemas = [
        {"name": s.name, "description": s.description, "parameters": dict(s.parameters)}
        for s in spec_by_name.values() if s.name == FINALIZE_CRITIC_REPORT.name
    ]

    async def dispatch(tool_name: str, params: dict) -> str:
        spec = spec_by_name.get(tool_name)
        if spec is None:
            return f"ERROR: tool {tool_name!r} not allowed for this critic"
        ctx = ToolContext(
            session_id="critic_subagent",
            phase_id=f"critic:{persona.id}",
            agent_id=persona.id,
            state_view=state_payload,
        )
        return await spec.handler(params, ctx)

    slot: dict[str, CriticReportBody | None] = {"body": None}
    token = _CRITIC_REPORT_SLOT.set(slot)
    msg_history: list[dict] = []
    try:
        result = await query_async_with_tools(
            model_name=persona.model,
            msg=user_msg,
            system_msg=system_msg,
            tools=tool_schemas,
            dispatch=dispatch,
            max_iters=max_iters,
            **_build_sampler(persona),
        )
        msg_history = list(result.new_msg_history)

        # Nudge fallback: critic ran out of explore budget without
        # committing. Re-prompt with only the finalize tool available so
        # the model can't keep researching — the only legal next move is
        # to commit the report it already has the material for.
        if slot.get("body") is None:
            nudge_result = await query_async_with_tools(
                model_name=persona.model,
                msg=(
                    "Your exploration budget is exhausted but you have not "
                    "yet committed your CriticReport. Call "
                    "`finalize_critic_report` now with the observations you "
                    "have gathered so far — even partial findings are more "
                    "useful than no report. The only available tool is "
                    "`finalize_critic_report`."
                ),
                system_msg=system_msg,
                msg_history=msg_history,
                tools=finalize_only_schemas,
                dispatch=dispatch,
                max_iters=nudge_max_iters,
                **_build_sampler(persona),
            )
            msg_history = list(nudge_result.new_msg_history)
    finally:
        _CRITIC_REPORT_SLOT.reset(token)

    body = slot.get("body")
    if body is None:
        raise RuntimeError(
            f"critic {persona.id} did not call finalize_critic_report within "
            f"{max_iters} iters (or nudge fallback)"
        )
    return body, msg_history


async def prelaunch_critics(
    critic_ids: list[str],
    *,
    state_payload: dict,
    cycle: int,
) -> None:
    """Kick off the listed critics as background tasks for the active
    cycle. Idempotent — already-prelaunched ids are skipped, unknown ids
    are logged and skipped. Tasks land under
    `state_payload["phase_3_revise"]["cycles"][-1]["prelaunched"]` for
    `_call_critic_handler` to await.

    The orchestrator calls this at the start of sub-phase A. By the time
    the agent makes its first `call_critic`, several critics may already
    be returning — the agent's wait time amortizes across the cycle.
    """
    registry = _registry_from(state_payload)
    if registry is None:
        raise RuntimeError("critic_registry not set; cannot prelaunch")
    cycle_record = _active_cycle(state_payload)
    if cycle_record is None:
        raise RuntimeError("no active Phase 3 cycle; cannot prelaunch critics")
    prelaunched: dict[str, asyncio.Task] = cycle_record.setdefault("prelaunched", {})
    for cid in critic_ids:
        if cid in prelaunched:
            continue
        if cid not in registry:
            logger.warning("prelaunch_critics: unknown critic %r; skipping", cid)
            continue
        prelaunched[cid] = asyncio.create_task(
            dispatch_critic(cid, state_view=state_payload, cycle=cycle),
            name=f"critic:{cid}:cycle_{cycle}",
        )


# ─── Tool handler (replaces the stub in tools.py) ────────────────────────


async def _call_critic_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    """Real `call_critic` handler.

    Awaits the cycle's pre-launched task when no focus is given;
    cancels the pre-launch and runs fresh when a focus is given.
    Tracks the call on the active cycle's `critic_calls` list so
    `finalize_critique_plan` can verify Tier A coverage.
    """
    critic_id = params.get("critic_id", "")
    focus = params.get("focus")
    if not critic_id:
        return "ERROR: call_critic requires non-empty `critic_id`"
    if focus is not None and not isinstance(focus, str):
        return "ERROR: call_critic `focus` must be a string"

    cycle_record = _active_cycle(ctx.state_view)
    if cycle_record is None:
        return (
            "ERROR: no active Phase 3 cycle; call_critic can only fire "
            "during a cycle's gather sub-phase. The orchestrator must "
            "initialize the cycle before tools fire."
        )

    registry = _registry_from(ctx.state_view)
    if registry is None:
        return "ERROR: critic_registry not set in session state"
    if critic_id not in registry:
        return f"ERROR: unknown critic_id {critic_id!r}; available: {registry.ids()}"

    cycle_idx = cycle_record.get("cycle", 0)
    prelaunched: dict[str, asyncio.Task] = cycle_record.setdefault("prelaunched", {})

    try:
        if critic_id in prelaunched:
            task = prelaunched.pop(critic_id)
            if focus is None:
                report = await task
            else:
                # Focused request supersedes the unfocused pre-launch.
                task.cancel()
                report = await dispatch_critic(
                    critic_id, state_view=ctx.state_view, cycle=cycle_idx, focus=focus,
                )
        else:
            report = await dispatch_critic(
                critic_id, state_view=ctx.state_view, cycle=cycle_idx, focus=focus,
            )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("call_critic %s failed: %s", critic_id, e)
        return f"ERROR: critic {critic_id!r} failed ({type(e).__name__}: {e})"

    if critic_id not in cycle_record["critic_calls"]:
        cycle_record["critic_calls"].append(critic_id)
    cycle_record.setdefault("reports_seen", {})[critic_id] = report

    return report.model_dump_json(indent=2)


async def drain_prelaunched(state_payload: dict) -> None:
    """Await any pre-launched critic tasks that the agent didn't explicitly
    call_critic on, and stash their results in `reports_seen`.

    Invoked by `RevisePhase` at the end of sub-phase A. The pre-launched
    sweep in cycle 0 is fire-and-forget by design — the orchestrator
    captures every result for plateau detection regardless of whether
    the agent looked at it. Tasks that raised propagate their exceptions
    here (logged-and-skipped); a partial sweep is acceptable, plateau
    detection just sees fewer signals.
    """
    cycle_record = _active_cycle(state_payload)
    if cycle_record is None:
        return
    prelaunched: dict[str, asyncio.Task] = cycle_record.get("prelaunched", {})
    for cid, task in list(prelaunched.items()):
        try:
            report = await task
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("drain: prelaunched %s failed (%s: %s)", cid, type(e).__name__, e)
            continue
        cycle_record.setdefault("reports_seen", {}).setdefault(cid, report)
        if cid not in cycle_record["critic_calls"]:
            cycle_record["critic_calls"].append(cid)
    prelaunched.clear()
