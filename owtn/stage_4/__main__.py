"""Stage 4 CLI entry point.

Usage:
    uv run python -m owtn.stage_4 --config configs/stage_4/light.yaml \\
        --concept-path results/run_<ts>/stage_1/best/main.json \\
        --stage-2-results results/run_<ts>/stage_2/ \\
        --stage-3-results results/run_<ts>/stage_3/by_pair/<pair_id>/ \\
        --tuple-id c_<concept_slug>_<voice_iter>
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
from owtn.models.stage_2.dag import DAG
from owtn.models.stage_2.handoff import Stage2HandoffManifest
from owtn.models.stage_3 import VoiceGenome
from owtn.models.stage_4 import Stage4Config
from owtn.stage_4 import run_stage_4_session


def _setup_logging(run_dir: Path) -> None:
    log_path = run_dir / "stage_4_run.log"
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    file_h = logging.FileHandler(log_path)
    file_h.setFormatter(fmt)
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_h)
    root.addHandler(stream_h)


def _load_dag_from_stage_2_results(stage_2_dir: Path) -> DAG:
    """Mirrors Stage 3's manifest-loading: pick the top-ranked advancing
    DAG from the Stage 2 handoff manifest. Caller can override by passing
    --dag-path to bypass this lookup."""
    manifest_path = stage_2_dir / "handoff_manifest.json"
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
    return chosen.genome


def _load_voice_from_stage_3_results(stage_3_dir: Path) -> VoiceGenome:
    """Stage 3 writes `winner.json` per-pair under `by_pair/<pair>/`. The
    --stage-3-results arg points directly at that pair dir."""
    winner_path = stage_3_dir / "winner.json"
    if not winner_path.exists():
        raise SystemExit(f"winner.json not found at {winner_path}")
    return VoiceGenome.model_validate_json(winner_path.read_text())


async def _run(args: argparse.Namespace) -> int:
    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    session_dir = (
        args.results_dir / run_id / "stage_4" / "by_tuple" / args.tuple_id
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(session_dir)
    llm_log_dir.set(str(session_dir))

    cfg = Stage4Config.from_yaml(args.config)
    logging.info(
        "config %s: generator=%s cycle_cap=%d call_ceiling=%d",
        args.config, cfg.generator_model, cfg.revise.cycle_cap, cfg.revise.call_ceiling,
    )

    concept = ConceptGenome.model_validate_json(args.concept_path.read_text())
    logging.info("loaded concept: premise=%s...", concept.premise[:120])

    if args.dag_path is not None:
        dag = DAG.model_validate_json(args.dag_path.read_text())
        logging.info("loaded DAG from --dag-path: %d nodes", len(dag.nodes))
    else:
        if args.stage_2_results is None:
            raise SystemExit("--stage-2-results or --dag-path is required")
        dag = _load_dag_from_stage_2_results(args.stage_2_results)
        logging.info("loaded DAG from stage_2 handoff: %d nodes", len(dag.nodes))

    voice_genome = _load_voice_from_stage_3_results(args.stage_3_results)
    logging.info(
        "loaded voice genome: agent_id=%s pair_id=%s",
        voice_genome.agent_id, voice_genome.pair_id,
    )

    result = await run_stage_4_session(
        concept=concept,
        dag=dag,
        voice_genome=voice_genome,
        tuple_id=args.tuple_id,
        run_dir=session_dir,
        session_log_dir=session_dir,
        config=cfg,
        stage_3_winner_persona_id=voice_genome.agent_id,
    )

    (session_dir / "session_result.json").write_text(
        result.model_dump_json(indent=2)
    )
    print(
        f"\n== stage 4 complete ==\n"
        f"manuscript:       {result.manuscript_path}\n"
        f"pre_think:        {result.pre_think_path}\n"
        f"cycles_completed: {result.cycles_completed}\n"
        f"exit_reason:      {result.exit_reason}\n"
        f"cost_usd:         ${result.cost_usd:.4f}\n"
        f"word count:       {result.words}\n"
        f"artifacts:        {session_dir}\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4: prose generation from a (concept, structure, voice) tuple.",
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Stage 4 config YAML (configs/stage_4/*.yaml).")
    parser.add_argument("--concept-path", required=True, type=Path,
                        help="Path to stage_1/best/main.json.")
    parser.add_argument("--stage-2-results", type=Path, default=None,
                        help="Stage 2 run dir containing handoff_manifest.json.")
    parser.add_argument("--dag-path", type=Path, default=None,
                        help="Direct path to a DAG JSON; overrides --stage-2-results.")
    parser.add_argument("--stage-3-results", required=True, type=Path,
                        help="Stage 3 by_pair/<pair> dir containing winner.json.")
    parser.add_argument("--tuple-id", required=True,
                        help="Identifier for this (concept, structure, voice) tuple.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-id", type=str, default=None,
                        help=("Override auto-generated run_id so artifacts land "
                              "in an existing pipeline run dir (e.g. "
                              "results/<run-id>/stage_4/by_tuple/<tuple-id>/) "
                              "instead of creating a fresh dir."))
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"error: {args.config} not found", file=sys.stderr)
        return 1
    try:
        Stage4Config.from_yaml(args.config)
    except ValidationError as e:
        print(f"config validation failed:\n{e}", file=sys.stderr)
        return 1

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
