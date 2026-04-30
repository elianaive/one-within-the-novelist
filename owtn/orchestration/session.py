"""Session state, run_session, and tracing helpers.

`run_session(state, phases, registry)` is the orchestrator entry point.
It sequences phases in order, sets tracing breadcrumbs on the existing
`llm_context` ContextVar so per-LLM-call logs auto-tag with
session/phase/agent identity, and writes a structured YAML log tree
under `session_log_dir`.

Logging surface:
    <session_log_dir>/
    ├── session.yaml                # final manifest
    ├── phases/<phase.name>.yaml    # per-phase summary record
    ├── agents/<agent_id>/<phase.name>.yaml   # phases write per-agent records
    ├── tools/<tool_name>/...       # per tool-call records
    └── llm/<model>/NNNN.yaml       # existing per-LLM-call logs

Phases own how they format their per-agent records — orchestration only
provides path resolution and YAML writing helpers. Phase exceptions abort
the session (no automatic retry); phases are responsible for their own
per-agent retry logic.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import yaml

from owtn.llm.call_logger import _LiteralStr, _dumper, llm_context

from .agent import Agent
from .phase import Phase
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


# ─── ContextVars ──────────────────────────────────────────────────────────

session_log_dir: ContextVar[str | None] = ContextVar(
    "session_log_dir", default=None
)
"""Active session log directory. Mirror of `llm_log_dir`."""


# ─── Session state ────────────────────────────────────────────────────────


@dataclass
class SessionState:
    """Mutable session-state carrier.

    `payload` is the cross-phase scratchpad. Phases agree on key shape by
    convention; e.g. Stage 3's private-brief phase writes
    `payload["phase_1_private_brief"][agent.id] = proposal_dict`. The
    dict-style carrier is provisional — revisit when we want file-backed
    persistence or stricter typing.

    `pair_id` ties this session to its (concept, structure) source for
    Stage 3, or analogous upstream identity in other stages. Optional;
    when set, gets logged into the session manifest and llm_context.
    """
    session_id: str
    cast: list[Agent]
    pair_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    phases_completed: list[str] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        cast: list[Agent],
        *,
        pair_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionState:
        return cls(
            session_id=session_id or f"sess_{uuid.uuid4().hex[:12]}",
            cast=list(cast),
            pair_id=pair_id,
        )


# ─── Tracing helpers ──────────────────────────────────────────────────────


@contextmanager
def push_llm_context(**fields: Any) -> Iterator[None]:
    """Merge `fields` into `llm_context` for the duration of the block.

    Use inside per-agent coroutines (set `agent_id` *inside* the coroutine
    that asyncio.gather dispatches, not before) — ContextVars are
    async-task-local but only if the var is set within the task.
    """
    current = dict(llm_context.get({}))
    merged = {**current, **{k: v for k, v in fields.items() if v is not None}}
    token: Token = llm_context.set(merged)
    try:
        yield
    finally:
        llm_context.reset(token)


def session_log_path(*parts: str) -> Path | None:
    """Resolve a path within the active session log dir.

    Creates parent directories. Returns None when `session_log_dir` is
    unset (e.g., tests that don't care about the log tree).
    """
    base = session_log_dir.get(None)
    if base is None:
        return None
    p = Path(base, *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _yaml_safe(value: Any) -> Any:
    """Render multi-line strings with literal-block style (matches
    call_logger). Coerce numpy/pandas scalars to Python builtins —
    yaml.dump can't represent them otherwise. Recurses into nested
    dicts/lists."""
    if isinstance(value, str) and ("\n" in value or len(value) > 120):
        return _LiteralStr(value)
    if isinstance(value, dict):
        return {k: _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_yaml_safe(v) for v in value]
    # numpy / pandas scalars: type from a non-builtin module + .item() method
    if (
        hasattr(value, "item")
        and not isinstance(value, (str, bytes, list, tuple, dict))
        and type(value).__module__ != "builtins"
    ):
        try:
            return value.item()
        except Exception:
            return value
    return value


def write_yaml(path: Path, data: dict) -> None:
    """Write a dict to YAML using the literal-block style for prose fields."""
    rendered = _yaml_safe(data)
    try:
        path.write_text(
            yaml.dump(
                rendered,
                Dumper=_dumper,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
        )
    except Exception as e:
        logger.warning("Failed to write session log %s: %s", path, e)


def _summarize_payload_for_phase(state: SessionState, phase_name: str) -> dict:
    """Phase-trace payload summary: keys + agent ids only, not full content.

    Per-agent file output (full proposals, critiques) is the phase's
    responsibility; the phase-level record here is just an index.
    """
    bucket = state.payload.get(phase_name, {})
    if isinstance(bucket, dict):
        return {"agents": sorted(bucket.keys())}
    return {"shape": type(bucket).__name__}


def write_phase_trace(
    phase: Phase,
    state: SessionState,
    *,
    duration_s: float,
    cost_delta: float,
) -> None:
    """Write the per-phase summary record."""
    path = session_log_path("phases", f"{phase.name}.yaml")
    if path is None:
        return
    write_yaml(
        path,
        {
            "phase": phase.name,
            "session_id": state.session_id,
            "pair_id": state.pair_id,
            "duration_s": round(duration_s, 3),
            "cost_delta_usd": round(cost_delta, 6),
            "ended_at": datetime.now().isoformat(timespec="milliseconds"),
            "outputs": _summarize_payload_for_phase(state, phase.name),
        },
    )


def write_session_manifest(state: SessionState) -> None:
    """Final session record — manifest of cast, phases run, totals."""
    path = session_log_path("session.yaml")
    if path is None:
        return
    write_yaml(
        path,
        {
            "session_id": state.session_id,
            "pair_id": state.pair_id,
            "started_at": state.started_at.isoformat(timespec="milliseconds"),
            "ended_at": datetime.now().isoformat(timespec="milliseconds"),
            "cost_usd": round(state.cost_usd, 6),
            "phases_completed": list(state.phases_completed),
            "cast": [
                {"id": a.id, "model": a.model, "tools": sorted(a.tools)}
                for a in state.cast
            ],
        },
    )


# ─── Orchestrator ─────────────────────────────────────────────────────────


async def run_session(
    state: SessionState,
    phases: list[Phase],
    registry: ToolRegistry,
) -> SessionState:
    """Run a multi-agent session.

    Sequences `phases` in order. At each boundary: tags `llm_context` with
    `session_id`/`phase_id`, calls `phase.run`, records duration + cost
    delta, writes the phase trace. Phase exceptions propagate (session
    aborts; partial logs persist).

    Per-agent identity (`agent_id` in `llm_context`) is the phase's
    responsibility — the phase code wraps its per-agent coroutines with
    `push_llm_context(agent_id=...)`.
    """
    logger.info(
        "session %s: starting (cast=%s, phases=%s)",
        state.session_id,
        [a.id for a in state.cast],
        [p.name for p in phases],
    )

    with push_llm_context(
        session_id=state.session_id,
        pair_id=state.pair_id,
    ):
        for phase in phases:
            with push_llm_context(phase_id=phase.name):
                t0 = time.perf_counter()
                cost_before = state.cost_usd
                logger.info(
                    "session %s: phase %s start (n_agents=%d)",
                    state.session_id, phase.name, len(state.cast),
                )
                try:
                    state = await phase.run(state.cast, state, registry)
                except Exception:
                    logger.exception(
                        "session %s: phase %s failed; aborting",
                        state.session_id, phase.name,
                    )
                    raise
                duration = time.perf_counter() - t0
                cost_delta = state.cost_usd - cost_before
                state.phases_completed.append(phase.name)
                logger.info(
                    "session %s: phase %s done (%.2fs, $%.4f)",
                    state.session_id, phase.name, duration, cost_delta,
                )
                write_phase_trace(
                    phase, state, duration_s=duration, cost_delta=cost_delta,
                )

    write_session_manifest(state)
    logger.info(
        "session %s: complete (total $%.4f, %d phases)",
        state.session_id, state.cost_usd, len(state.phases_completed),
    )
    return state
