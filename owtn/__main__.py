"""CLI entry point for Stage 1 concept evolution.

Usage:
    uv run python -m owtn --config configs/stage_1/medium.yaml
    uv run python -m owtn --config configs/stage_1/dry_run.yaml
"""

import argparse
import logging
import sys

from owtn.runner import ConceptEvolutionRunner


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Evolutionary concept generation",
    )
    parser.add_argument(
        "--config", default="configs/stage_1/medium.yaml",
        help="Path to stage config YAML",
    )
    parser.add_argument(
        "--max-eval-jobs", type=int, default=2,
        help="Max concurrent evaluation subprocesses",
    )
    parser.add_argument(
        "--max-proposal-jobs", type=int, default=1,
        help="Max concurrent proposal (mutation) tasks",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Override results directory (default: auto-generated)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )

    runner = ConceptEvolutionRunner(
        config_path=args.config,
        verbose=verbose,
        max_evaluation_jobs=args.max_eval_jobs,
        max_proposal_jobs=args.max_proposal_jobs,
        results_dir=args.results_dir,
    )
    runner.run()


if __name__ == "__main__":
    main()
