"""Stage 4 session output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Stage4SessionResult(BaseModel):
    """End-of-session record. The manuscript is the load-bearing artifact;
    everything else is bookkeeping the orchestrator already wrote to disk
    and surfaces here for the caller to consume directly."""
    tuple_id: str
    manuscript_path: str
    pre_think_path: str
    run_dir: str
    cost_usd: float = 0.0
    cycles_completed: int = 0
    exit_reason: str = "unknown"
    session_log_dir: str = ""

    @property
    def words(self) -> int:
        """Read the manuscript's current word count from disk. Cheap to
        compute on demand; the file IS the source of truth."""
        from pathlib import Path
        p = Path(self.manuscript_path)
        return len(p.read_text(encoding="utf-8").split()) if p.exists() else 0
