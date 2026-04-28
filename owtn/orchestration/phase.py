"""The Phase protocol.

A Phase is one unit of session work over the cast: take agents + state,
do something (in parallel or sequentially), return updated state. Phases
are stage-specific concrete classes — Stage 3's private-brief, judge
consultation, reveal-critique, revise, and Borda phases all implement
this protocol.

Common patterns (parallel-independent calls, structured-output critique
passes, voting) get promoted into helper functions here only when a
second consumer arrives. Until then, each stage owns its phase shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .agent import Agent
    from .session import SessionState
    from .tools import ToolRegistry


@runtime_checkable
class Phase(Protocol):
    """One session phase.

    `name` doubles as the log-tree path component (e.g.
    `"phase_1_private_brief"` → `<session_log_dir>/phases/phase_1_private_brief.yaml`).
    Use snake_case with a phase-number prefix to keep traces ordered on disk.
    """

    name: str

    async def run(
        self,
        agents: list["Agent"],
        state: "SessionState",
        registry: "ToolRegistry",
    ) -> "SessionState": ...
