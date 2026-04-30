"""Stage 3 CLI entry point.

Usage:
    uv run python -m owtn.stage_3 --config configs/stage_3/demo.yaml \\
        --concept-path results/run_<ts>/stage_1/best/main.json \\
        --stage-2-results results/run_<ts>/stage_2/ \\
        --pair-id c_<concept_slug>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from owtn.llm.call_logger import llm_log_dir
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.handoff import Stage2HandoffManifest
from owtn.models.stage_3 import Stage3Config
from owtn.stage_2.rendering import render as render_dag
from owtn.stage_3.adjacent_scenes import AdjacentSceneBench, generate_adjacent_scenes
from owtn.stage_3.casting import cast_voice_panel
from owtn.stage_3.personas import load_persona_pool
from owtn.stage_3.session import run_voice_session


def _setup_logging(run_dir: Path) -> None:
    log_path = run_dir / "stage_3_run.log"
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    file_h = logging.FileHandler(log_path)
    file_h.setFormatter(fmt)
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_h)
    root.addHandler(stream_h)


async def _run(args: argparse.Namespace) -> int:
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    session_dir = args.results_dir / run_id / "stage_3" / "by_pair" / args.pair_id
    session_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(session_dir)
    llm_log_dir.set(str(session_dir))

    cfg = Stage3Config.from_yaml(args.config)
    panel_size = args.panel_size if args.panel_size is not None else cfg.casting.panel_size
    logging.info(
        "config %s: picker=%s caster=%s gen=%s panel_size=%d",
        args.config, cfg.adjacent_scene.picker_model,
        cfg.casting.caster_model, cfg.voice_session.generator_model, panel_size,
    )

    concept = ConceptGenome.model_validate_json(args.concept_path.read_text())
    logging.info("loaded concept: premise=%s...", concept.premise[:120])

    if args.stage_2_results is None:
        raise SystemExit(
            "--stage-2-results is required (the canonical entry needs a real "
            "DAG; use lab/scripts/stage_3_smoke_real.py for hand-crafted-DAG "
            "smoke tests)"
        )
    manifest_path = args.stage_2_results / "handoff_manifest.json"
    manifest = Stage2HandoffManifest.model_validate_json(manifest_path.read_text())
    if not manifest.advancing:
        raise SystemExit(f"no advancing DAGs in {manifest_path}")
    chosen = min(manifest.advancing, key=lambda o: o.tournament_rank)
    if len(manifest.advancing) > 1:
        logging.info(
            "%d advancing DAGs in manifest; picking top-ranked",
            len(manifest.advancing),
        )
    logging.info(
        "using Stage 2 DAG: concept=%s preset=%s rank=%d reward=%.3f",
        chosen.concept_id, chosen.preset, chosen.tournament_rank, chosen.mcts_reward,
    )
    dag_rendering = render_dag(chosen.genome)
    logging.info("dag rendering ready (%d chars)", len(dag_rendering))

    if args.bench_from is not None:
        logging.info("step 1/3: loading precomputed bench from %s", args.bench_from)
        bench = AdjacentSceneBench.model_validate_json(args.bench_from.read_text())
    else:
        logging.info("step 1/3: generate_adjacent_scenes — picking + drafting")
        bench = await generate_adjacent_scenes(
            concept, dag_rendering,
            picker_model=cfg.adjacent_scene.picker_model,
            drafter_model=cfg.adjacent_scene.drafter_model,
            picker_kwargs={"reasoning_effort": cfg.adjacent_scene.picker_reasoning_effort},
        )
        if bench is None:
            raise SystemExit("bench generation failed")
    (session_dir / "bench.json").write_text(bench.model_dump_json(indent=2))
    for d in bench.drafts:
        logging.info("  bench scene: %s — demand: %s", d.scene_id, d.demand[:80])

    logging.info("step 2/3: cast_voice_panel")
    casting = await cast_voice_panel(
        concept, dag_rendering,
        panel_size=panel_size,
        model_name=cfg.casting.caster_model,
        classify_kwargs={"reasoning_effort": cfg.casting.reasoning_effort},
        argue_kwargs={"reasoning_effort": cfg.casting.reasoning_effort},
        select_kwargs={"reasoning_effort": cfg.casting.reasoning_effort},
    )
    if casting is None:
        raise SystemExit("casting failed")
    (session_dir / "casting.json").write_text(casting.model_dump_json(indent=2))
    logging.info("casting picked %d: %s",
                 len(casting.cast),
                 [c.persona_id for c in casting.cast])

    pool = load_persona_pool()
    by_id = {p.id: p for p in pool}
    voice_cast = [by_id[c.persona_id] for c in casting.cast]

    logging.info("step 3/3: run_voice_session with cast=%s",
                 [p.id for p in voice_cast])
    result = await run_voice_session(
        cast=voice_cast,
        bench=bench,
        concept=concept,
        dag_rendering=dag_rendering,
        pair_id=args.pair_id,
        session_log_dir=session_dir,
        config=cfg.voice_session,
    )

    (session_dir / "winner.json").write_text(result.winner.model_dump_json(indent=2))
    (session_dir / "session_result.json").write_text(result.model_dump_json(indent=2))
    print(
        f"\n== stage 3 complete ==\n"
        f"winner:        {result.winner.agent_id}\n"
        f"borda_points:  {result.borda_points}\n"
        f"cost:          ${result.cost_usd:.4f}\n"
        f"artifacts:     {session_dir}\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 3: voice evolution from a (concept, structure) pair.",
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Stage 3 config YAML (configs/stage_3/*.yaml).")
    parser.add_argument("--concept-path", required=True, type=Path,
                        help="Path to stage_1/best/main.json.")
    parser.add_argument("--stage-2-results", required=True, type=Path,
                        help="Stage 2 run dir containing handoff_manifest.json.")
    parser.add_argument("--pair-id", required=True,
                        help="Identifier for this (concept, structure) pair.")
    parser.add_argument("--panel-size", type=int, default=None,
                        help="Override config.casting.panel_size for this run.")
    parser.add_argument("--bench-from", type=Path, default=None,
                        help="Precomputed bench.json (skip live bench generation).")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"error: {args.config} not found", file=sys.stderr)
        return 1
    try:
        Stage3Config.from_yaml(args.config)
    except ValidationError as e:
        print(f"config validation failed:\n{e}", file=sys.stderr)
        return 1

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
