"""Cross-stage pipeline: a single config wires Stages 1 → 4.

A *pipeline config* (see `configs/pipeline/dry_run.yaml`) points at one
per-stage YAML each and adds a few orchestration-only knobs (eval-job
concurrency, tier, max-concept caps). It does not duplicate any per-stage
schema — those stay authoritative.

Output layout under a single `results/run_<ts>/` root:
    stage_1/                  ConceptEvolutionRunner output
        best/main.json        ← concept handed to Stage 3 / Stage 4
        champions/island_*.json
    stage_2/
        handoff_manifest.json ← DAG list, top-ranked picked downstream
        qd_archive.json
    stage_3/by_pair/<pair_id>/
        winner.json           ← VoiceGenome handed to Stage 4
        session_result.json
    stage_4/by_tuple/<tuple_id>/
        sandbox/<writer>/story.md
        session_result.json

`pair_id` and `tuple_id` are derived from the top-ranked Stage 2 output
(`c_<concept_id[:8]>_<preset>` and `<pair_id>_v1`).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from owtn.llm.call_logger import llm_log_dir
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.config import Stage2Config
from owtn.models.stage_2.handoff import Stage2HandoffManifest, Stage2Output
from owtn.models.stage_3 import Stage3Config, VoiceGenome
from owtn.models.stage_4 import Stage4Config
from owtn.stage_1.runner import ConceptEvolutionRunner
from owtn.stage_2.rendering import render as render_dag
from owtn.stage_2.runner import run_stage_2
from owtn.stage_3.adjacent_scenes import generate_adjacent_scenes
from owtn.stage_3.casting import cast_voice_panel
from owtn.stage_3.personas import load_persona_pool
from owtn.stage_3.session import run_voice_session
from owtn.stage_4 import run_stage_4_session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StageRefs:
    """Per-stage entries in a pipeline config. Each `config` is a path to
    the per-stage YAML; remaining fields are orchestration-only knobs."""
    stage_1_config: Path
    stage_1_max_eval_jobs: int
    stage_1_max_proposal_jobs: int
    stage_2_config: Path
    stage_2_tier: str
    stage_2_max_concepts: int | None
    stage_3_config: Path
    stage_3_panel_size: int | None
    stage_4_config: Path


_STAGES = ("stage_1", "stage_2", "stage_3", "stage_4")


@dataclass(frozen=True)
class PipelineConfig:
    results_dir: Path | None
    start_at: str  # one of _STAGES; earlier stages must already exist on disk
    refs: _StageRefs

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        s1 = data.get("stage_1") or {}
        s2 = data.get("stage_2") or {}
        s3 = data.get("stage_3") or {}
        s4 = data.get("stage_4") or {}
        for label, block in (("stage_1", s1), ("stage_2", s2),
                             ("stage_3", s3), ("stage_4", s4)):
            if "config" not in block:
                raise ValueError(f"pipeline config: {label}.config is required")
        results_override = data.get("results_dir")
        start_at = str(data.get("start_at", "stage_1"))
        if start_at not in _STAGES:
            raise ValueError(f"start_at must be one of {_STAGES}, got {start_at!r}")
        if start_at != "stage_1" and not results_override:
            raise ValueError("start_at != stage_1 requires `results_dir` to point at an existing run")
        return cls(
            results_dir=Path(results_override) if results_override else None,
            start_at=start_at,
            refs=_StageRefs(
                stage_1_config=Path(s1["config"]),
                stage_1_max_eval_jobs=int(s1.get("max_eval_jobs", 2)),
                stage_1_max_proposal_jobs=int(s1.get("max_proposal_jobs", 1)),
                stage_2_config=Path(s2["config"]),
                stage_2_tier=str(s2.get("tier", "light")),
                stage_2_max_concepts=s2.get("max_concepts"),
                stage_3_config=Path(s3["config"]),
                stage_3_panel_size=s3.get("panel_size"),
                stage_4_config=Path(s4["config"]),
            ),
        )


def _resolve_results_dir(override: Path | None) -> Path:
    chosen = override or Path("results") / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    chosen.mkdir(parents=True, exist_ok=True)
    return chosen


def _select_top_dag(stage_2_dir: Path) -> Stage2Output:
    manifest_path = stage_2_dir / "handoff_manifest.json"
    manifest = Stage2HandoffManifest.model_validate_json(manifest_path.read_text())
    if not manifest.advancing:
        raise RuntimeError(f"no advancing DAGs in {manifest_path}")
    return min(manifest.advancing, key=lambda o: o.tournament_rank)


async def _run_stage_1(refs: _StageRefs, stage_1_dir: Path) -> None:
    runner = ConceptEvolutionRunner(
        config_path=str(refs.stage_1_config),
        results_dir=str(stage_1_dir),
        max_evaluation_jobs=refs.stage_1_max_eval_jobs,
        max_proposal_jobs=refs.stage_1_max_proposal_jobs,
        verbose=True,
    )
    await runner.run_async()


async def _run_stage_2(refs: _StageRefs, stage_1_dir: Path, stage_2_dir: Path) -> None:
    config = Stage2Config.from_yaml(refs.stage_2_config)
    await run_stage_2(
        stage_1_dir=stage_1_dir,
        config=config,
        config_tier=refs.stage_2_tier,
        output_dir=stage_2_dir,
        max_concepts=refs.stage_2_max_concepts,
    )


async def _run_stage_3(
    refs: _StageRefs,
    *,
    concept: ConceptGenome,
    chosen: Stage2Output,
    pair_id: str,
    session_dir: Path,
) -> Path:
    """Run Stage 3 against the top-ranked Stage 2 DAG. Mirrors
    `owtn.stage_3.__main__._run`'s body but with explicit paths so the
    pipeline owns directory layout and avoids the per-stage timestamp
    nesting that the standalone CLI applies."""
    session_dir.mkdir(parents=True, exist_ok=True)
    llm_log_dir.set(str(session_dir))

    cfg = Stage3Config.from_yaml(refs.stage_3_config)
    panel_size = refs.stage_3_panel_size or cfg.casting.panel_size
    dag_rendering = render_dag(chosen.genome)

    bench = await generate_adjacent_scenes(
        concept, dag_rendering,
        picker_model=cfg.adjacent_scene.picker_model,
        drafter_model=cfg.adjacent_scene.drafter_model,
        picker_kwargs={"reasoning_effort": cfg.adjacent_scene.picker_reasoning_effort},
    )
    if bench is None:
        raise RuntimeError("Stage 3 bench generation failed")
    (session_dir / "bench.json").write_text(bench.model_dump_json(indent=2))

    casting = await cast_voice_panel(
        concept, dag_rendering,
        panel_size=panel_size,
        model_name=cfg.casting.caster_model,
        classify_kwargs={"reasoning_effort": cfg.casting.reasoning_effort},
        argue_kwargs={"reasoning_effort": cfg.casting.reasoning_effort},
        select_kwargs={"reasoning_effort": cfg.casting.reasoning_effort},
    )
    if casting is None:
        raise RuntimeError("Stage 3 casting failed")
    (session_dir / "casting.json").write_text(casting.model_dump_json(indent=2))

    pool_by_id = {p.id: p for p in load_persona_pool()}
    voice_cast = [pool_by_id[c.persona_id] for c in casting.cast]

    result = await run_voice_session(
        cast=voice_cast,
        bench=bench,
        concept=concept,
        dag_rendering=dag_rendering,
        pair_id=pair_id,
        session_log_dir=session_dir,
        config=cfg.voice_session,
    )
    (session_dir / "winner.json").write_text(result.winner.model_dump_json(indent=2))
    (session_dir / "session_result.json").write_text(result.model_dump_json(indent=2))
    return session_dir


async def _run_stage_4(
    refs: _StageRefs,
    *,
    concept: ConceptGenome,
    chosen: Stage2Output,
    voice_genome: VoiceGenome,
    tuple_id: str,
    session_dir: Path,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    llm_log_dir.set(str(session_dir))

    cfg = Stage4Config.from_yaml(refs.stage_4_config)
    result = await run_stage_4_session(
        concept=concept,
        dag=chosen.genome,
        voice_genome=voice_genome,
        tuple_id=tuple_id,
        run_dir=session_dir,
        session_log_dir=session_dir,
        config=cfg,
        stage_3_winner_persona_id=voice_genome.agent_id,
    )
    (session_dir / "session_result.json").write_text(result.model_dump_json(indent=2))


async def run_pipeline_async(pipeline_cfg: PipelineConfig) -> Path:
    results_dir = _resolve_results_dir(pipeline_cfg.results_dir)
    refs = pipeline_cfg.refs
    skip_until = _STAGES.index(pipeline_cfg.start_at)
    logger.info("Pipeline run: %s (start_at=%s)", results_dir, pipeline_cfg.start_at)

    stage_1_dir = results_dir / "stage_1"
    if skip_until <= 0:
        logger.info("=== Stage 1 → %s ===", stage_1_dir)
        await _run_stage_1(refs, stage_1_dir)
    else:
        logger.info("Skipping Stage 1 (resume; expects %s)", stage_1_dir)

    stage_2_dir = results_dir / "stage_2"
    if skip_until <= 1:
        logger.info("=== Stage 2 → %s ===", stage_2_dir)
        await _run_stage_2(refs, stage_1_dir, stage_2_dir)
    else:
        logger.info("Skipping Stage 2 (resume; expects %s)", stage_2_dir)

    concept_path = stage_1_dir / "best" / "main.json"
    concept = ConceptGenome.model_validate_json(concept_path.read_text())
    chosen = _select_top_dag(stage_2_dir)
    pair_id = f"c_{chosen.concept_id[:8]}_{chosen.preset}"
    tuple_id = f"{pair_id}_v1"

    stage_3_dir = results_dir / "stage_3" / "by_pair" / pair_id
    if skip_until <= 2:
        logger.info("=== Stage 3 → %s (pair_id=%s) ===", stage_3_dir, pair_id)
        await _run_stage_3(
            refs,
            concept=concept, chosen=chosen, pair_id=pair_id, session_dir=stage_3_dir,
        )
    else:
        logger.info("Skipping Stage 3 (resume; expects %s)", stage_3_dir)
    voice_genome = VoiceGenome.model_validate_json(
        (stage_3_dir / "winner.json").read_text()
    )

    stage_4_dir = results_dir / "stage_4" / "by_tuple" / tuple_id
    logger.info("=== Stage 4 → %s (tuple_id=%s) ===", stage_4_dir, tuple_id)
    await _run_stage_4(
        refs,
        concept=concept, chosen=chosen, voice_genome=voice_genome,
        tuple_id=tuple_id, session_dir=stage_4_dir,
    )

    logger.info("Pipeline complete: %s", results_dir)
    return results_dir


def run_pipeline(pipeline_cfg: PipelineConfig) -> Path:
    return asyncio.run(run_pipeline_async(pipeline_cfg))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Stages 1→4 in sequence under a single results dir.",
    )
    parser.add_argument(
        "--pipeline-config", required=True, type=Path,
        help="Pipeline YAML pointing to per-stage configs (e.g. configs/pipeline/dry_run.yaml).",
    )
    parser.add_argument(
        "--results-dir", default=None, type=Path,
        help="Override the run's root results dir (default: results/run_<ts>/).",
    )
    parser.add_argument(
        "--start-at", default=None, choices=_STAGES,
        help="Resume from this stage; earlier stages must already exist on disk "
             "under --results-dir (overrides the YAML's start_at).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.quiet:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    if not args.pipeline_config.exists():
        print(f"error: pipeline config {args.pipeline_config} not found", file=sys.stderr)
        return 1

    try:
        cfg = PipelineConfig.from_yaml(args.pipeline_config)
    except (ValidationError, ValueError) as e:
        print(f"pipeline config invalid: {e}", file=sys.stderr)
        return 1

    overrides = {}
    if args.results_dir is not None:
        overrides["results_dir"] = args.results_dir
    if args.start_at is not None:
        overrides["start_at"] = args.start_at
    if overrides:
        from dataclasses import replace
        cfg = replace(cfg, **overrides)
    if cfg.start_at != "stage_1" and cfg.results_dir is None:
        print("error: --start-at != stage_1 requires --results-dir", file=sys.stderr)
        return 1

    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
