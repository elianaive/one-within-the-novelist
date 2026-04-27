"""Stage 2 quality-diversity archive: write-only per-run grid.

Per `docs/stage-2/qd-archive.md`:
- 5×5 grid indexed by `(disclosure_ratio_bin, structural_density_bin)`.
- Both axes computed deterministically from the DAG (no LLM, no classifier
  drift). Bin boundaries default to the design-doc values; configurable via
  `Stage2Config.archive_bin_boundaries`.
- v1 is **write-only**: every terminal DAG passing validation gets appended
  to its cell. No competitive insertion (deferred to v1.5 once we have
  cross-run data showing churn matters).

What this module is NOT:
- Cross-run persistence — Stage 1's compost layer doesn't ship cross-run
  storage yet, so Stage 2 follows. Each run flushes its archive to a JSON
  file under the run directory. No SQLite, no shared store.
- A retrieval layer — v2 will inject archived DAGs as exemplars in
  expansion prompts. For now, the archive is pure write-only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from owtn.models.stage_2.dag import DAG


logger = logging.getLogger(__name__)


# Defaults from `docs/stage-2/qd-archive.md`. Each list is the 4 internal
# cuts that define 5 bins. Override via `Stage2Config.archive_bin_boundaries`
# at run time.
DEFAULT_DISCLOSURE_CUTS: tuple[float, ...] = (0.10, 0.25, 0.40, 0.55)
DEFAULT_DENSITY_CUTS: tuple[float, ...] = (1.2, 1.8, 2.5, 3.2)

# Human-readable bin labels for diagnostic output. Also from qd-archive.md.
DISCLOSURE_LABELS: tuple[str, ...] = ("None", "Light", "Balanced", "Heavy", "Dominant")
DENSITY_LABELS: tuple[str, ...] = ("Skeletal", "Simple", "Moderate", "Dense", "Very dense")


def _bin_index(value: float, cuts: tuple[float, ...]) -> int:
    """Return the bin index for `value` against `cuts` (4 internal cuts → 5 bins)."""
    for i, cut in enumerate(cuts):
        if value < cut:
            return i
    return len(cuts)  # values ≥ last cut land in the top bin


def disclosure_ratio(dag: DAG) -> float:
    """Fraction of edges that are `disclosure` edges. Returns 0 on empty DAGs."""
    if not dag.edges:
        return 0.0
    n_disclosure = sum(1 for e in dag.edges if e.type == "disclosure")
    return n_disclosure / len(dag.edges)


def structural_density(dag: DAG) -> float:
    """Edges per node. Returns 0 on empty DAGs."""
    if not dag.nodes:
        return 0.0
    return len(dag.edges) / len(dag.nodes)


def compute_cell(
    dag: DAG,
    *,
    disclosure_cuts: tuple[float, ...] = DEFAULT_DISCLOSURE_CUTS,
    density_cuts: tuple[float, ...] = DEFAULT_DENSITY_CUTS,
) -> tuple[int, int]:
    """Compute the (disclosure_bin, density_bin) cell for a DAG.

    Both bins are integers in [0, 4]. See `qd-archive.md` §Grid Axes for
    the binning rationale and bin labels.
    """
    return (
        _bin_index(disclosure_ratio(dag), disclosure_cuts),
        _bin_index(structural_density(dag), density_cuts),
    )


@dataclass
class ArchiveEntry:
    """One DAG's record in an archive cell.

    Stored as a structured dict on disk; this dataclass is just for in-memory
    convenience. The DAG itself is held by reference until flush, then dumped.
    """
    dag: DAG
    concept_id: str
    preset: str
    tournament_rank: int
    mcts_reward: float = 0.0


@dataclass
class Stage2Archive:
    """Per-run write-only QD archive.

    Construct one per Stage 2 run. Call `add(...)` for each terminal DAG —
    advancing or non-advancing — then `flush(run_dir)` at run end to produce
    `qd_archive.json`.

    Cell coords are computed deterministically; multiple DAGs may share a cell.
    """
    disclosure_cuts: tuple[float, ...] = DEFAULT_DISCLOSURE_CUTS
    density_cuts: tuple[float, ...] = DEFAULT_DENSITY_CUTS
    cells: dict[tuple[int, int], list[ArchiveEntry]] = field(
        default_factory=lambda: {},
    )

    def add(
        self,
        dag: DAG,
        *,
        concept_id: str,
        preset: str,
        tournament_rank: int,
        mcts_reward: float = 0.0,
    ) -> tuple[int, int]:
        """Add a DAG to the archive. Returns the cell coords it landed in."""
        cell = compute_cell(
            dag,
            disclosure_cuts=self.disclosure_cuts,
            density_cuts=self.density_cuts,
        )
        entry = ArchiveEntry(
            dag=dag,
            concept_id=concept_id,
            preset=preset,
            tournament_rank=tournament_rank,
            mcts_reward=mcts_reward,
        )
        self.cells.setdefault(cell, []).append(entry)
        logger.info(
            "Archive: %s (preset=%s, rank=%d) → cell %s",
            concept_id, preset, tournament_rank, cell,
        )
        return cell

    def to_dict(self, *, run_id: str) -> dict[str, Any]:
        """Serialize the archive to the JSON shape from qd-archive.md
        §Archive Visualization."""
        # Bin definitions: each pair is (lower, upper). The 4 internal cuts
        # produce 5 bins. We use 0.0 as the bottom and 1.0 (disclosure) /
        # ∞ (density, but represented as a large number) as the top.
        disclosure_bins = self._bin_ranges(
            self.disclosure_cuts, lower_bound=0.0, upper_bound=1.0,
        )
        density_bins = self._bin_ranges(
            self.density_cuts, lower_bound=0.0, upper_bound=100.0,
        )

        cells = []
        for (disc_bin, dens_bin), entries in sorted(self.cells.items()):
            cells.append({
                "disclosure_bin": disc_bin,
                "density_bin": dens_bin,
                "label": (
                    f"{DISCLOSURE_LABELS[disc_bin]} × {DENSITY_LABELS[dens_bin]}"
                ),
                "entries": [self._entry_to_dict(e) for e in entries],
            })

        return {
            "run_id": run_id,
            "grid_size": [5, 5],
            "axes": {
                "disclosure_ratio": {
                    "bins": disclosure_bins,
                    "labels": list(DISCLOSURE_LABELS),
                },
                "structural_density": {
                    "bins": density_bins,
                    "labels": list(DENSITY_LABELS),
                },
            },
            "cells": cells,
        }

    @staticmethod
    def _bin_ranges(
        cuts: tuple[float, ...], *, lower_bound: float, upper_bound: float,
    ) -> list[list[float]]:
        """Convert 4 internal cuts → 5 [lower, upper] bin ranges."""
        edges = [lower_bound, *cuts, upper_bound]
        return [[edges[i], edges[i + 1]] for i in range(len(edges) - 1)]

    @staticmethod
    def _entry_to_dict(entry: ArchiveEntry) -> dict[str, Any]:
        return {
            "concept_id": entry.concept_id,
            "preset": entry.preset,
            "tournament_rank": entry.tournament_rank,
            "mcts_reward": entry.mcts_reward,
            "genome": entry.dag.model_dump(),
        }

    def flush(self, run_dir: Path, *, run_id: str | None = None) -> Path:
        """Write the archive to `<run_dir>/qd_archive.json`. Returns the file path.

        `run_id` defaults to `run_dir.name` if not supplied.
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            run_id = run_dir.name
        out_path = run_dir / "qd_archive.json"
        out_path.write_text(json.dumps(self.to_dict(run_id=run_id), indent=2))
        logger.info("Archive: wrote %s with %d cells", out_path, len(self.cells))
        return out_path
