"""Stage 4 — prose generation.

The agent reads, writes, and edits a manuscript file (`story.md`) via
generic file tools, calling critic and surgical-edit subagents as needed.
Architecture, philosophy, and per-decision rationale live in
`docs/stage-4/overview.md`.

Foundation slice (this issue) ships:
- `manuscript.py` — file ops on `story.md`
- `tools.py` — ToolSpec set + per-phase allowlist; finalize family
  has real handlers, critic / surgical-edit / web_search are stubbed for the
  child issues that follow

Net new modules to come (per `lab/issues/2026-04-29-stage-4-architecture-scoping.md`):
- `filter.py` — pre-stage classification (audience framing + expert needs)
- `personas.py` — critic persona registry
- `critics.py` — critic dispatch + Tier A enforcement at finalize_critique_plan
- `domain_expert.py` — runtime instantiation of domain-expert critics
- `surgical_edit.py` — surgical-edit subagent dispatch (Flavor B: scope-translate then constrain)
- `prethink.py`, `drafter.py`, `revise.py`, `plateau.py` — phase implementations
- `session.py` — composer
"""

from owtn.stage_4.critics import CriticRegistry, dispatch_critic, prelaunch_critics
from owtn.stage_4.drafter import DownDraftPhase
from owtn.stage_4.filter import run_stage_4_filter
from owtn.stage_4.manuscript import (
    EditError,
    apply_edit,
    parse,
    read_text,
    render,
    scaffold_from_dag,
    write_text,
)
from owtn.stage_4.personas import load_critic_pool
from owtn.stage_4.plateau import PlateauDetector, PlateauVerdict
from owtn.stage_4.prethink import PreThinkPhase
from owtn.stage_4.revise import RevisePhase
from owtn.stage_4.session import run_stage_4_session
from owtn.stage_4.tools import (
    ALL_STAGE_4_TOOLS,
    PHASE_1_PRETHINK_TOOLS,
    PHASE_2_DOWNDRAFT_TOOLS,
    PHASE_3A_GATHER_TOOLS,
    PHASE_3B_REVISE_TOOLS,
    STAGE_4_PHASE_ALLOW,
    TIER_A_CRITICS,
)


__all__ = [
    "ALL_STAGE_4_TOOLS",
    "CriticRegistry",
    "DownDraftPhase",
    "EditError",
    "PHASE_1_PRETHINK_TOOLS",
    "PHASE_2_DOWNDRAFT_TOOLS",
    "PHASE_3A_GATHER_TOOLS",
    "PHASE_3B_REVISE_TOOLS",
    "PlateauDetector",
    "PlateauVerdict",
    "PreThinkPhase",
    "RevisePhase",
    "STAGE_4_PHASE_ALLOW",
    "TIER_A_CRITICS",
    "apply_edit",
    "dispatch_critic",
    "load_critic_pool",
    "parse",
    "prelaunch_critics",
    "read_text",
    "render",
    "run_stage_4_filter",
    "run_stage_4_session",
    "scaffold_from_dag",
    "write_text",
]
