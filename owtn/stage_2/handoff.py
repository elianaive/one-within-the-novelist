"""Stage 2 → Stage 3 handoff manifest construction.

Glue between the within-concept tournament's ranking, the QD archive's cell
assignment, and the `Stage2Output` schema in `owtn.models.stage_2.handoff`.
The runner (Phase 9) consumes this module per concept; the resulting
manifest is serialized to `results/run_<ts>/stage_2/handoff_manifest.json`.

What `build_handoff_for_concept` does:
1. Filters the tournament's ranked entries by `top_k` (1 / 2 / all per
   light/medium/heavy config), with optional near-tie promotion at the
   K-boundary per `evaluation.md` §Top-K Advancement.
2. Computes each advancing DAG's QD cell.
3. Wraps each into a `Stage2Output` with stage_1_forwarded metadata.
4. Adds non-advancing DAGs to the archive (so structural ideas aren't lost).

The runner accumulates `Stage2Output`s across concepts and finally calls
`write_manifest(...)` once with the full list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from owtn.models.stage_2.dag import DAG
from owtn.models.stage_2.handoff import (
    Stage1Winner,
    Stage2HandoffManifest,
    Stage2Output,
)
from owtn.stage_2.archive import Stage2Archive, compute_cell
from owtn.stage_2.tournament import TournamentEntry


logger = logging.getLogger(__name__)


# Default near-tie threshold per `docs/stage-2/evaluation.md` §"Edge case:
# near-tie at the K boundary": if the K-th and (K+1)-th ranked DAGs differ
# by ≤1 dim-level wins across all matches, both advance.
DEFAULT_NEAR_TIE_DIM_WIN_GAP: int = 1


def _select_top_k(
    ranked: list[TournamentEntry],
    *,
    top_k: int | None,
    near_tie_promoted: bool,
    near_tie_gap: int,
) -> tuple[list[TournamentEntry], bool]:
    """Return (advancing, near_tie_fired).

    `top_k=None` means "all advance" (heavy.yaml mode).
    `near_tie_promoted=True` allows the K-th and (K+1)-th to both advance
    if their `dim_wins_total` differ by ≤ `near_tie_gap`.
    """
    if top_k is None or top_k >= len(ranked):
        return list(ranked), False
    if top_k <= 0:
        return [], False

    advancing = ranked[:top_k]
    near_tie_fired = False
    if near_tie_promoted and top_k < len(ranked):
        boundary = ranked[top_k - 1]
        next_after = ranked[top_k]
        if boundary.dim_wins_total - next_after.dim_wins_total <= near_tie_gap:
            advancing = ranked[: top_k + 1]
            near_tie_fired = True
            logger.info(
                "Top-K boundary near-tie: promoting both rank %d (%s) "
                "and rank %d (%s); dim_wins gap=%d",
                top_k, boundary.preset, top_k + 1, next_after.preset,
                boundary.dim_wins_total - next_after.dim_wins_total,
            )
    return advancing, near_tie_fired


def build_handoff_for_concept(
    *,
    concept_id: str,
    stage_1_forwarded: Stage1Winner,
    ranked_entries: list[TournamentEntry],
    archive: Stage2Archive,
    top_k: int | None,
    near_tie_promoted: bool = True,
    near_tie_gap: int = DEFAULT_NEAR_TIE_DIM_WIN_GAP,
    adaptation_permissions: list[str] | None = None,
) -> tuple[list[Stage2Output], bool]:
    """Per-concept handoff: filter ranked entries by top_k, build outputs,
    and side-effect the archive with all entries (advancing + non-advancing).

    Returns (advancing_outputs, near_tie_fired).
    """
    if not ranked_entries:
        return [], False

    # Archive ALL entries (advancing or not) per qd-archive.md §Population
    # Rules (write-only): "Non-advancing DAGs from a concept (those that
    # lost the within-concept tournament) are archived rather than discarded."
    for rank, entry in enumerate(ranked_entries, 1):
        archive.add(
            entry.dag,
            concept_id=concept_id,
            preset=entry.preset,
            tournament_rank=rank,
            mcts_reward=entry.mcts_reward,
        )

    advancing, near_tie_fired = _select_top_k(
        ranked_entries,
        top_k=top_k,
        near_tie_promoted=near_tie_promoted,
        near_tie_gap=near_tie_gap,
    )

    outputs: list[Stage2Output] = []
    perms = list(adaptation_permissions or [])
    for rank_idx, entry in enumerate(advancing, 1):
        cell = compute_cell(
            entry.dag,
            disclosure_cuts=archive.disclosure_cuts,
            density_cuts=archive.density_cuts,
        )
        outputs.append(Stage2Output(
            concept_id=concept_id,
            preset=entry.preset,
            tournament_rank=rank_idx,
            qd_cell=cell,
            genome=entry.dag,
            stage_1_forwarded=stage_1_forwarded,
            mcts_reward=entry.mcts_reward,
            adaptation_permissions=perms,
        ))
    return outputs, near_tie_fired


def write_manifest(
    *,
    run_id: str,
    advancing: list[Stage2Output],
    run_dir: Path,
    near_tie_promoted: bool = False,
) -> Path:
    """Serialize the handoff manifest to `<run_dir>/handoff_manifest.json`.

    `advancing` is the union of `build_handoff_for_concept(...)` outputs
    across every concept the run handled. The manifest is what Stage 3 reads.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = Stage2HandoffManifest(
        run_id=run_id,
        advancing=advancing,
        near_tie_promoted=near_tie_promoted,
    )
    out_path = run_dir / "handoff_manifest.json"
    out_path.write_text(manifest.model_dump_json(indent=2))
    logger.info(
        "Wrote handoff manifest: %s (%d advancing DAGs across %d concepts)",
        out_path, len(advancing), len({o.concept_id for o in advancing}),
    )
    return out_path
