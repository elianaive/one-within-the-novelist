"""Cross-stage pipeline orchestrator.

Runs the OWTN stages sequentially against a single results directory.
Stages 1-6 each evolve a different genome (concept → structure → voice →
prose → refinement → archive) — see `docs/stages.md` for the full spec.

Currently wired: Stage 1.
Stubbed: Stages 2-6. Stage 2 has its own runner (`python -m owtn.stage_2`)
but is not yet plumbed into this orchestrator. As subsequent stages come
online, register them in `_STAGES` below.

Per-stage CLIs remain the canonical way to run a single stage:
    uv run python -m owtn.stage_1 --config configs/stage_1/medium.yaml
    uv run python -m owtn.stage_2 --config configs/stage_2/light.yaml \\
        --stage-1-results results/run_<ts>/stage_1/

Pipeline usage:
    uv run python -m owtn --stage-1-config configs/stage_1/medium.yaml
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from owtn.stage_1.runner import ConceptEvolutionRunner

logger = logging.getLogger(__name__)


def _run_stage_1(config_path: str, results_dir: Path, args) -> None:
    runner = ConceptEvolutionRunner(
        config_path=config_path,
        verbose=not args.quiet,
        max_evaluation_jobs=args.max_eval_jobs,
        max_proposal_jobs=args.max_proposal_jobs,
        results_dir=str(results_dir / "stage_1"),
    )
    runner.run()


# Registry of (name, runner). Add stages here as their pipeline integration
# is wired up. Stage 2+ runners need a config arg AND awareness of the prior
# stage's output dir; design that interface when Stage 2 lands.
_STAGES = [
    ("stage_1", _run_stage_1),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OWTN cross-stage pipeline (currently runs Stage 1 only).",
    )
    parser.add_argument(
        "--stage-1-config", default="configs/stage_1/medium.yaml",
        help="Path to Stage 1 config YAML",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Override the run's root results dir (default: results/run_<ts>/)",
    )
    parser.add_argument("--max-eval-jobs", type=int, default=2)
    parser.add_argument("--max-proposal-jobs", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results") / f"run_{timestamp}"
    else:
        results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, runner_fn in _STAGES:
        logger.info("=" * 60)
        logger.info("  Running %s", name)
        logger.info("=" * 60)
        if name == "stage_1":
            runner_fn(args.stage_1_config, results_dir, args)
        else:
            runner_fn(results_dir, args)

    logger.info(
        "Pipeline complete. Stages 2-6 not yet wired — invoke them "
        "individually against %s/",
        results_dir,
    )


if __name__ == "__main__":
    main()
