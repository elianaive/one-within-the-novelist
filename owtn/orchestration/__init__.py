"""Multi-agent session orchestration.

Stage-agnostic primitives for running phased multi-agent sessions over a
cast of agents with mediated tool access. Stage 3 (voice) and Stage 4
(prose) compose stage-specific phases on top of these primitives.

Public surface:
- `Agent` — one participant; persona prompt + model + tool allowlist
- `Phase` — protocol for one unit of session work
- `ToolSpec` / `ToolRegistry` / `ToolContext` — tool registry + dispatch
- `SessionState` / `run_session` — the orchestrator entry point
- `session_log_dir`, `push_llm_context`, `session_log_path`, `write_yaml` —
  tracing helpers (compose with the existing `llm_context` ContextVar)

Composition lives in stage-specific modules (`owtn.stage_3.session`,
`owtn.stage_4.session`); this package contains no voice- or
prose-specific code.
"""

from .agent import Agent
from .phase import Phase
from .session import (
    SessionState,
    push_llm_context,
    run_session,
    session_log_dir,
    session_log_path,
    write_phase_trace,
    write_session_manifest,
    write_yaml,
)
from .tools import ToolContext, ToolRegistry, ToolSpec

__all__ = [
    "Agent",
    "Phase",
    "SessionState",
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "push_llm_context",
    "run_session",
    "session_log_dir",
    "session_log_path",
    "write_phase_trace",
    "write_session_manifest",
    "write_yaml",
]
