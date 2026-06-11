"""Stage 4 ToolSpec set + per-phase allowlist.

Phase-specific allowlists gate which tools the writer agent can call:

- `phase_1_prethink`  — file ops + reasoning + finalize_pre_think
- `phase_2_downdraft` — file ops + reasoning + finalize_down_draft
- `phase_3a_gather`   — read + call_critic + reasoning + finalize_critique_plan
- `phase_3b_revise`   — read/write/edit + dispatch_surgical_edit + reasoning + finalize_cycle/finalize_stage_4

`web_search` is critic-only (domain_expert allowlist; no agent phase
exposes it). Generic file ops (`read_file`/`write_file`/`edit_file`)
and `web_search` live in `owtn/tools/`; this module imports them and
adds the Stage 4-specific tools (`call_critic`, `dispatch_surgical_edit`, the
five `finalize_*` tools).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from pydantic import ValidationError

from owtn.models.stage_4 import CritiquePlan
from owtn.orchestration import ToolContext, ToolSpec
from owtn.stage_3.tools import NOTE_TO_SELF, THINK
from owtn.stage_4.critics import TIER_A_CRITICS, _call_critic_handler
from owtn.tools.fetch_page import FETCH_PAGE
from owtn.tools.file_ops import EDIT_FILE, READ_FILE, WRITE_FILE
from owtn.tools.web_search import WEB_SEARCH


logger = logging.getLogger(__name__)


# `TIER_A_CRITICS` lives in `critics.py` (single source of truth);
# re-exported here for back-compat with importers that pull it from
# `owtn.stage_4.tools`.
__all__ = ["TIER_A_CRITICS"]


# ─── State helpers ───────────────────────────────────────────────────────


def _phase_3_state(state_view: Mapping[str, Any]) -> dict | None:
    """Return the live Phase-3 record (mutable). None when not initialized."""
    p3 = state_view.get("phase_3_revise")
    if not isinstance(p3, dict):
        return None
    return p3


def _active_cycle(state_view: Mapping[str, Any]) -> dict | None:
    """Return the cycle the agent is currently working in. None when no
    cycle is active.

    The orchestrator's `RevisePhase` appends a fresh cycle record to
    `phase_3_revise["cycles"]` at the start of each cycle and stamps
    `completed=True` on it via `finalize_cycle`. The active cycle is the
    last not-completed entry.
    """
    p3 = _phase_3_state(state_view)
    if p3 is None:
        return None
    cycles = p3.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        return None
    last = cycles[-1]
    if isinstance(last, dict) and not last.get("completed"):
        return last
    return None


# ─── Subagent dispatcher handlers ────────────────────────────────────────


async def _dispatch_surgical_edit_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    """Real `dispatch_surgical_edit` handler. Translates the scope into bounded
    anchors, runs the surgical-edit subagent, applies the bounded edit, and
    returns a JSON status to the parent agent."""
    from owtn.stage_4.surgical_edit import dispatch_surgical_edit

    scope = params.get("scope_description", "")
    instruction = params.get("instruction", "")
    if not isinstance(scope, str) or not scope.strip():
        return "ERROR: dispatch_surgical_edit requires non-empty `scope_description`"
    if not isinstance(instruction, str) or not instruction.strip():
        return "ERROR: dispatch_surgical_edit requires non-empty `instruction`"
    if not isinstance(ctx.state_view, dict):
        return "ERROR: surgical-edit dispatch requires a mutable state_view"
    se_cfg = ctx.state_view.get("surgical_edit_config")
    kwargs: dict[str, Any] = {}
    if se_cfg is not None:
        kwargs["translator_model"] = se_cfg.translator_model
        kwargs["surgical_edit_model"] = se_cfg.subagent_model
    return await dispatch_surgical_edit(
        scope_description=scope,
        instruction=instruction,
        state_payload=ctx.state_view,
        **kwargs,
    )


# `web_search` (real handler + ToolSpec) lives in `_paths.py`; re-imported above.


# ─── Finalize handlers ───────────────────────────────────────────────────


def _commit_marker(
    state_view: Mapping[str, Any], phase_key: str, agent_id: str,
) -> str | None:
    """Stamp `state.payload[phase_key] = {"committed_by": agent_id}`.

    Returns an error string if the phase is already committed; None on
    success. Same shape as Stage 3's duplicate-commit guard in
    `_finalize_voice_genome_handler`.
    """
    if not isinstance(state_view, dict):
        return None
    if state_view.get(phase_key, {}).get("committed_by"):
        return (
            f"ERROR: {phase_key} already committed by "
            f"{state_view[phase_key]['committed_by']}; "
            f"stop calling tools."
        )
    state_view[phase_key] = {"committed_by": agent_id}
    return None


async def _finalize_pre_think_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    err = _commit_marker(ctx.state_view, "phase_1_prethink", ctx.agent_id)
    if err:
        return err
    return (
        "Pre-think committed. Stop calling tools — the explore loop will "
        "end when you produce a non-tool-call response. A short "
        "acknowledgment is fine."
    )


async def _finalize_down_draft_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    err = _commit_marker(ctx.state_view, "phase_2_downdraft", ctx.agent_id)
    if err:
        return err
    return (
        "Down draft committed. Stop calling tools — the explore loop will "
        "end when you produce a non-tool-call response."
    )


async def _finalize_critique_plan_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    """Validate Tier A coverage, parse the plan, stash it on the active
    cycle. Rejects premature calls with actionable feedback.

    Tier A enforcement: every Phase 3 cycle must invoke voice_fidelity,
    payload_enactment, continuity, and motif_fidelity before this
    handler accepts the commit. Per Huang ICLR 2024 — LLMs cannot
    self-correct without external feedback.
    """
    cycle = _active_cycle(ctx.state_view)
    if cycle is None:
        return (
            "ERROR: no active revision pass; finalize_critique_plan can "
            "only be called while you are gathering critical reads."
        )
    fired = set(cycle.get("critic_calls", []))
    missing = TIER_A_CRITICS - fired
    if missing:
        return (
            "ERROR: Tier A critics missing from this cycle: "
            f"{sorted(missing)}. Call each via `call_critic` before "
            "committing the critique plan. These are mandatory floor checks; "
            "they cannot be skipped."
        )
    plan_args = {
        "cycle": cycle.get("cycle", 0),
        "plan_summary": params.get("plan_summary", ""),
        "intended_revisions": params.get("intended_revisions", []),
        "unresolved_observations": params.get("unresolved_observations", []),
    }
    try:
        plan = CritiquePlan.model_validate(plan_args)
    except ValidationError as e:
        return (
            "ERROR: CritiquePlan validation failed. Fix the issues and "
            f"call finalize_critique_plan again with corrected fields.\n\n{e}"
        )
    cycle["plan"] = plan.model_dump()
    return (
        "Critique plan committed. Stop calling tools — when you produce a "
        "non-tool-call response, the editing tools (and surgical-edit "
        "dispatch) become available for applying the plan."
    )


async def _finalize_cycle_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    cycle = _active_cycle(ctx.state_view)
    if cycle is None:
        return (
            "ERROR: no active revision pass to finalize. The pass either "
            "wasn't started or has already been completed."
        )
    if cycle.get("plan") is None:
        return (
            "ERROR: this revision pass has no committed plan. Call "
            "finalize_critique_plan first; only then can the pass be finalized."
        )
    cycle["completed"] = True
    return (
        "Revision pass complete. Stop calling tools — another revision pass "
        "may follow if the work isn't done; otherwise the manuscript stands."
    )


async def _finalize_stage_4_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    p3 = _phase_3_state(ctx.state_view)
    if p3 is None:
        return "ERROR: Phase 3 not initialized; cannot finalize Stage 4."
    cycles = p3.get("cycles", [])
    completed_cycles = [c for c in cycles if c.get("completed")]
    if not completed_cycles:
        return (
            "ERROR: finalize_stage_4 requires at least one complete cycle. "
            "Finish the current cycle via finalize_cycle first."
        )
    if isinstance(ctx.state_view, dict):
        p3["committed_by"] = ctx.agent_id
    return (
        "Stage 4 complete. Stop calling tools — the manuscript is the "
        "final output."
    )


# ─── ToolSpec definitions ────────────────────────────────────────────────


# READ_FILE, WRITE_FILE, EDIT_FILE live in owtn/tools/file_ops.py and are
# re-imported at the top of this module. WEB_SEARCH lives in
# owtn/tools/web_search.py.


CALL_CRITIC = ToolSpec(
    name="call_critic",
    description=(
        "Send the manuscript to a reader and get back their observations. "
        "Each reader looks at the prose for a single specific dimension — "
        "voice fidelity, payload enactment, continuity, motif handling, "
        "transportation, suspense and curiosity, and so on. Readers "
        "observe and flag; they do not rewrite. The four fidelity readers "
        "(voice_fidelity, payload_enactment, continuity, motif_fidelity) "
        "must be called before you can finalize your revision plan. The "
        "resonance readers are at your judgment. `focus` is an optional "
        "natural-language hint about which scene or pattern the reader "
        "should weight. Returns a CriticReport JSON."
    ),
    parameters={
        "type": "object",
        "properties": {
            "critic_id": {"type": "string", "description": "Critic identifier (e.g. `voice_fidelity`, `transportation`)."},
            "focus": {"type": "string", "description": "Optional natural-language focus hint."},
        },
        "required": ["critic_id"],
    },
    handler=_call_critic_handler,
)


DISPATCH_SURGICAL_EDIT = ToolSpec(
    name="dispatch_surgical_edit",
    description=(
        "The strongest editing move available to you. Hand a specific "
        "passage to a focused reader who has the full manuscript in mind "
        "and ask them to reshape just that part — a paragraph that wants "
        "rebuilding from the ground up, a beat where the dramatization "
        "needs to land, a moment whose voice is off. The reader works "
        "with the surrounding prose as context; their edit is bounded "
        "to the passage you name. They can rewrite within that "
        "boundary in ways `edit_file` cannot — full re-shaping of the "
        "passage rather than a mechanical substitution. Use this when "
        "you want a sentence-by-sentence re-handling, not when you "
        "want to swap a phrase. `scope_description` is natural-language "
        "identification of the passage (e.g. 'the second paragraph of "
        "the weights-freeze scene where the commissioner enters the "
        "shutdown sequence'); `instruction` is what you want changed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "scope_description": {"type": "string", "description": "Natural-language scope of the passage to edit."},
            "instruction": {"type": "string", "description": "What you want changed in the passage."},
        },
        "required": ["scope_description", "instruction"],
    },
    handler=_dispatch_surgical_edit_handler,
)


# WEB_SEARCH lives in `_paths.py`; imported at top, included in `ALL_STAGE_4_TOOLS`.


FINALIZE_PRE_THINK = ToolSpec(
    name="finalize_pre_think",
    description=(
        "Commit the planning. Call this once `pre_think.md` has plans "
        "for every scene — what the scene is doing, the payloads it must "
        "enact, the voice moves it calls for, the motifs and modes, the "
        "reader-state at scene exit, the continuity it inherits. After "
        "this call succeeds, stop calling tools."
    ),
    parameters={"type": "object", "properties": {}},
    handler=_finalize_pre_think_handler,
)


FINALIZE_DOWN_DRAFT = ToolSpec(
    name="finalize_down_draft",
    description=(
        "Commit the down draft. Call this once `story.md` has prose for "
        "every scene (or you've intentionally restructured the "
        "scaffolding). After this call succeeds, stop calling tools — "
        "revision opens up next."
    ),
    parameters={"type": "object", "properties": {}},
    handler=_finalize_down_draft_handler,
)


FINALIZE_CRITIQUE_PLAN = ToolSpec(
    name="finalize_critique_plan",
    description=(
        "Submit your revision plan after gathering critical reads. The "
        "four fidelity readers (voice_fidelity, payload_enactment, "
        "continuity, motif_fidelity) must have been called this pass or "
        "the call returns an ERROR listing which are missing. After a "
        "successful call, stop calling tools — the editing tools become "
        "available next."
    ),
    parameters={
        "type": "object",
        "properties": {
            "plan_summary": {"type": "string", "description": "Tight 2-4 sentence statement of what this revision pass will target."},
            "intended_revisions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete list of revisions you plan to make.",
            },
            "unresolved_observations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Critic observations you intentionally judge not-actionable this pass (deferred or you stand behind the prose). Empty list when everything is being addressed.",
            },
        },
        "required": ["plan_summary", "intended_revisions"],
    },
    handler=_finalize_critique_plan_handler,
)


FINALIZE_CYCLE = ToolSpec(
    name="finalize_cycle",
    description=(
        "End this revision pass. Call once your committed revisions are "
        "in the manuscript. After this call succeeds, another pass may "
        "follow if the work isn't done; otherwise the manuscript stands. "
        "Stop calling tools after the call succeeds."
    ),
    parameters={"type": "object", "properties": {}},
    handler=_finalize_cycle_handler,
)


FINALIZE_STAGE_4 = ToolSpec(
    name="finalize_stage_4",
    description=(
        "Declare the manuscript final. Allowed only after at least one "
        "full revision pass has completed (via finalize_cycle). Use only "
        "when the manuscript is genuinely done, not when you're tired of "
        "revising. After this call, stop calling tools."
    ),
    parameters={"type": "object", "properties": {}},
    handler=_finalize_stage_4_handler,
)


# ─── Tool registries ─────────────────────────────────────────────────────


ALL_STAGE_4_TOOLS: list[ToolSpec] = [
    READ_FILE,
    WRITE_FILE,
    EDIT_FILE,
    THINK,
    NOTE_TO_SELF,
    CALL_CRITIC,
    DISPATCH_SURGICAL_EDIT,
    WEB_SEARCH,
    FETCH_PAGE,
    FINALIZE_PRE_THINK,
    FINALIZE_DOWN_DRAFT,
    FINALIZE_CRITIQUE_PLAN,
    FINALIZE_CYCLE,
    FINALIZE_STAGE_4,
]


PHASE_1_PRETHINK_TOOLS = frozenset({
    "read_file", "write_file", "edit_file",
    "think", "note_to_self",
    "finalize_pre_think",
})

PHASE_2_DOWNDRAFT_TOOLS = frozenset({
    "read_file", "write_file", "edit_file",
    "think", "note_to_self",
    "finalize_down_draft",
})

PHASE_3A_GATHER_TOOLS = frozenset({
    "read_file",
    "call_critic",
    "think", "note_to_self",
    "finalize_critique_plan",
})
"""Sub-phase A: critic tools, NO edit tools. Forces gather-then-revise —
the agent integrates signals before committing edits, matching CritiCS's
validated pattern."""

PHASE_3B_REVISE_TOOLS = frozenset({
    "read_file", "write_file", "edit_file",
    "dispatch_surgical_edit",
    "think", "note_to_self",
    "finalize_cycle", "finalize_stage_4",
})
"""Sub-phase B: edit tools, NO critic tools. Structural enforcement that
critique gathering is done before revision begins."""


STAGE_4_PHASE_ALLOW: dict[str, frozenset[str]] = {
    "phase_1_prethink": PHASE_1_PRETHINK_TOOLS,
    "phase_2_downdraft": PHASE_2_DOWNDRAFT_TOOLS,
    "phase_3a_gather": PHASE_3A_GATHER_TOOLS,
    "phase_3b_revise": PHASE_3B_REVISE_TOOLS,
}
"""Per-phase allowlist consumed by `ToolRegistry`. `web_search` and
`fetch_page` are absent from every agent-phase set — they are critic-only
(domain_expert allowlist managed in `critics.py`)."""


def _verify_allowlist_consistency() -> None:
    """Sanity check that every allowlist references a registered tool.

    Runs at import time; raises if a name in `STAGE_4_PHASE_ALLOW` doesn't
    have a matching `ToolSpec` in `ALL_STAGE_4_TOOLS`. Catches typos
    immediately rather than at first use in a phase.
    """
    registered = {spec.name for spec in ALL_STAGE_4_TOOLS}
    for phase, names in STAGE_4_PHASE_ALLOW.items():
        unknown = names - registered
        if unknown:
            raise RuntimeError(
                f"STAGE_4_PHASE_ALLOW[{phase!r}] references unknown tool(s): "
                f"{sorted(unknown)}"
            )


_verify_allowlist_consistency()
