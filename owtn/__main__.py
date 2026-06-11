"""Cross-stage pipeline entry point.

Two modes:

  Pipeline (Stages 1→4 under a single results dir):
      uv run python -m owtn --pipeline-config configs/pipeline/dry_run.yaml

  Stage-1-only (legacy; equivalent to `python -m owtn.stage_1`):
      uv run python -m owtn --stage-1-config configs/stage_1/medium.yaml

The pipeline form is preferred — it threads the per-stage results-dir
handoff (Stage 1 best/main.json → Stage 3, Stage 2 manifest → Stages 3/4,
Stage 3 winner.json → Stage 4) without per-CLI plumbing. See
`owtn/pipeline.py` for the orchestrator and output layout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from owtn.pipeline import _STAGES, PipelineConfig, run_pipeline
from owtn.stage_1.runner import ConceptEvolutionRunner

logger = logging.getLogger(__name__)


def _run_stage_1_only(args: argparse.Namespace) -> int:
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results") / f"run_{timestamp}"
    else:
        results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    runner = ConceptEvolutionRunner(
        config_path=args.stage_1_config,
        verbose=not args.quiet,
        max_evaluation_jobs=args.max_eval_jobs,
        max_proposal_jobs=args.max_proposal_jobs,
        results_dir=str(results_dir / "stage_1"),
    )
    runner.run()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OWTN pipeline.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--pipeline-config", type=Path, default=None,
        help="Pipeline YAML wiring Stages 1→4 (e.g. configs/pipeline/dry_run.yaml).",
    )
    mode.add_argument(
        "--stage-1-config", default=None,
        help="Stage-1-only mode: path to a Stage 1 config YAML.",
    )
    parser.add_argument(
        "--results-dir", default=None, type=Path,
        help="Override the run's root results dir (default: results/run_<ts>/).",
    )
    parser.add_argument(
        "--start-at", default=None, choices=_STAGES,
        help="Pipeline mode: resume from this stage; earlier stages must "
             "already exist on disk under --results-dir.",
    )
    parser.add_argument("--max-eval-jobs", type=int, default=2,
                        help="Stage-1-only: max concurrent evaluation jobs.")
    parser.add_argument("--max-proposal-jobs", type=int, default=1,
                        help="Stage-1-only: max concurrent proposal tasks.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    if args.pipeline_config is not None:
        if not args.pipeline_config.exists():
            print(f"error: {args.pipeline_config} not found", file=sys.stderr)
            return 1
        cfg = PipelineConfig.from_yaml(args.pipeline_config)
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

    return _run_stage_1_only(args)


if __name__ == "__main__":
    sys.exit(main())
