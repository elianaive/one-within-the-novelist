"""Per-generation state snapshots for debugging.

Writes MAP-Elites grid, archive, and island state to
<results_dir>/state/gen_N.json after each generation completes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def snapshot_generation(
    db,
    generation: int,
    results_dir: str | Path,
    total_api_cost: float = 0.0,
) -> None:
    """Write a state snapshot after a generation completes."""
    results_dir = Path(results_dir)
    state_dir = results_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # MAP-Elites cells
    cells = []
    try:
        db.cursor.execute("SELECT cell_key, program_id FROM map_elites_cells")
        for cell_key_str, program_id in db.cursor.fetchall():
            prog = db._get_program_by_id(program_id)
            holder = None
            if prog and prog.public_metrics:
                holder = prog.public_metrics.get("holder_score")
            cells.append({
                "cell_key": json.loads(cell_key_str),
                "program_id": program_id,
                "holder_score": holder,
            })
    except Exception as e:
        logger.debug("Failed to read MAP-Elites cells: %s", e)

    # Archive size
    archive_size = 0
    try:
        db.cursor.execute("SELECT COUNT(*) FROM archive")
        archive_size = db.cursor.fetchone()[0]
    except Exception:
        pass

    # Island populations
    islands = {}
    try:
        db.cursor.execute(
            "SELECT island_idx, COUNT(*) FROM programs GROUP BY island_idx"
        )
        for island_idx, count in db.cursor.fetchall():
            islands[str(island_idx)] = count
    except Exception:
        pass

    # Total programs and correct count
    total = correct = 0
    try:
        db.cursor.execute("SELECT COUNT(*) FROM programs")
        total = db.cursor.fetchone()[0]
        db.cursor.execute("SELECT COUNT(*) FROM programs WHERE correct = 1")
        correct = db.cursor.fetchone()[0]
    except Exception:
        pass

    record = {
        "generation": generation,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "programs": {"total": total, "correct": correct},
        "archive_size": archive_size,
        "map_elites": {
            "cells_occupied": len(cells),
            "cells": cells,
        },
        "islands": islands,
        "total_api_cost": round(total_api_cost, 6),
    }

    path = state_dir / f"gen_{generation}.json"
    try:
        path.write_text(json.dumps(record, indent=2))
    except OSError as e:
        logger.warning("Failed to write state snapshot: %s", e)
