"""Stage 2 top-level runner.

Reads a Stage 1 results directory + a Stage 2 config; runs `run_concept`
across all advancing concepts; writes the run-level QD archive +
handoff manifest.

Per the implementation plan's cleanliness flag: this file stays under 300
lines. Per-concept logic lives in `owtn.stage_2.orchestration`; per-tree
logic in `bidirectional.py` / `refinement.py`. The runner here only owns:
- Stage 1 result loading
- Concept-level concurrency
- Run-level archive + manifest output
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from owtn.llm.call_logger import llm_context, llm_log_dir
from owtn.models.judge import JudgePersona, load_panel
from owtn.models.stage_2.config import Stage2Config
from owtn.models.stage_2.handoff import Stage1Winner, Stage2Output
from owtn.stage_2.archive import Stage2Archive
from owtn.stage_2.handoff import write_manifest
from owtn.stage_2.orchestration import run_concept


logger = logging.getLogger(__name__)


# Default judges directory (Stage 1 convention; Stage 2 reuses).
DEFAULT_JUDGES_DIR = "configs/judges"


def _attach_run_log_handler(log_path: Path) -> None:
    """Tee root-logger output to a run-specific log file.

    Idempotent on the path: if a FileHandler for this exact path already
    exists, do nothing. The runner sets up the handler once per run; the
    console handler (set by `logging.basicConfig` in __main__.py) stays in
    place so live operators still see streaming output.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    target = str(log_path.resolve())
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == target:
            return
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)


def load_stage_1_winners(stage_1_dir: Path) -> list[Stage1Winner]:
    """Read every champions/island_*.json from the Stage 1 directory.

    Each champion is one Stage 1 winner; Stage 2 runs once per winner.
    `tournament.json` (sibling) is auto-loaded by `Stage1Winner.from_champion_file`.
    """
    champions_dir = stage_1_dir / "champions"
    if not champions_dir.exists():
        raise FileNotFoundError(f"Stage 1 champions dir not found: {champions_dir}")
    winners: list[Stage1Winner] = []
    for path in sorted(champions_dir.glob("island_*.json")):
        winners.append(Stage1Winner.from_champion_file(path))
    if not winners:
        raise RuntimeError(f"no champions found in {champions_dir}")
    logger.info("Loaded %d Stage 1 winners from %s", len(winners), champions_dir)
    return winners


def _resolve_cheap_judge(panel: list[JudgePersona], cheap_id: str) -> JudgePersona:
    for j in panel:
        if j.id == cheap_id:
            return j
    raise ValueError(
        f"cheap_judge_id {cheap_id!r} not found in panel "
        f"(known: {[j.id for j in panel]})"
    )


async def _run_concept_with_logging(
    *,
    winner: Stage1Winner,
    config: Stage2Config,
    config_tier: str,
    cheap_judge: JudgePersona,
    full_panel: list[JudgePersona] | None,
    classifier_model: str,
    archive: Stage2Archive,
) -> list[Stage2Output]:
    """Wrapper around `run_concept` that catches exceptions per concept so
    one bad concept doesn't kill the whole run."""
    try:
        return await run_concept(
            winner=winner,
            config=config,
            config_tier=config_tier,
            cheap_judge=cheap_judge,
            full_panel=full_panel,
            classifier_model=classifier_model,
            archive=archive,
        )
    except Exception as e:  # noqa: BLE001 — runner-level catchall
        logger.exception(
            "concept %s failed (%s); skipping in handoff",
            winner.program_id, type(e).__name__,
        )
        return []


async def run_stage_2(
    *,
    stage_1_dir: Path,
    config: Stage2Config,
    config_tier: str = "light",
    cheap_judge_id: str | None = None,
    judges_dir: str = DEFAULT_JUDGES_DIR,
    output_dir: Path | None = None,
    max_concepts: int | None = None,
) -> tuple[Path, Path]:
    """Top-level Stage 2 entry point.

    Returns (qd_archive_path, handoff_manifest_path).

    Args:
        stage_1_dir: Path to a `results/run_<ts>/stage_1/` directory.
        config: Loaded `Stage2Config` (use `Stage2Config.from_yaml` for files).
        config_tier: Which preset list to use ("light"/"medium"/"heavy").
        cheap_judge_id: Judge id from the panel to use as the cheap judge;
            defaults to `config.judges.cheap_judge_id` when None.
        judges_dir: Path to judge YAML configs.
        output_dir: Where to write `qd_archive.json` and `handoff_manifest.json`.
            Defaults to `stage_1_dir.parent / "stage_2"`.
        max_concepts: If set, cap the number of Stage 1 winners processed
            (for pilot / dry runs).
    """
    if cheap_judge_id is None:
        cheap_judge_id = config.judges.cheap_judge_id
    run_id = stage_1_dir.parent.name if stage_1_dir.parent.name.startswith("run_") else stage_1_dir.name

    if output_dir is None:
        output_dir = stage_1_dir.parent / "stage_2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wire run-level logging mirrored from Stage 1's pattern:
    # - Per-call LLM logs to <output_dir>/llm/<model>/NNNN.yaml.
    # - Run log to <output_dir>/stage_2_run.log (file handler on root logger).
    llm_log_dir.set(str(output_dir))
    llm_context.set({"role": "stage_2", "run_id": run_id})
    _attach_run_log_handler(output_dir / "stage_2_run.log")

    winners = load_stage_1_winners(stage_1_dir)
    if max_concepts is not None:
        winners = winners[:max_concepts]

    # Panel + cheap judge are sourced from `config.judges` so the same
    # IDs are visible in the YAML and the runner reads no magic constants.
    full_panel = load_panel(judges_dir, config.judges.full_panel_ids)
    if not full_panel:
        raise RuntimeError(f"no judges loaded from {judges_dir}")
    cheap_judge = _resolve_cheap_judge(full_panel, cheap_judge_id)

    archive = Stage2Archive(
        disclosure_cuts=tuple(config.archive_bin_boundaries.disclosure_ratio),
        density_cuts=tuple(config.archive_bin_boundaries.structural_density),
    )

    logger.info(
        "Stage 2 run %s: %d concepts, tier=%s, cheap_judge=%s, full_panel=%s",
        run_id, len(winners), config_tier, cheap_judge.id,
        [j.id for j in full_panel],
    )

    started_at = datetime.now(timezone.utc)

    concept_tasks = [
        _run_concept_with_logging(
            winner=w,
            config=config,
            config_tier=config_tier,
            cheap_judge=cheap_judge,
            full_panel=full_panel,
            classifier_model=config.classifier_model or cheap_judge.model[0],
            archive=archive,
        )
        for w in winners
    ]
    per_concept_outputs = await asyncio.gather(*concept_tasks)
    advancing: list[Stage2Output] = []
    for outs in per_concept_outputs:
        advancing.extend(outs)

    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    logger.info(
        "Stage 2 run %s: %d/%d concepts produced advancing outputs in %.1fs",
        run_id,
        sum(1 for outs in per_concept_outputs if outs),
        len(winners),
        elapsed,
    )

    archive_path = archive.flush(output_dir, run_id=run_id)
    manifest_path = write_manifest(
        run_id=run_id,
        advancing=advancing,
        run_dir=output_dir,
    )
    return archive_path, manifest_path
