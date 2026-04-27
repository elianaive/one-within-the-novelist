"""Stage 2 CLI entry point.

Usage:
    uv run python -m owtn.stage_2 --config configs/stage_2/light.yaml \\
        --stage-1-results results/run_<ts>/stage_1/ [--max-concepts 3]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from owtn.models.stage_2.config import Stage2Config
from owtn.stage_2.runner import (
    DEFAULT_JUDGES_DIR,
    run_stage_2,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2: structural evolution from Stage 1 winners.",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to a Stage 2 config YAML (e.g. configs/stage_2/light.yaml)",
    )
    parser.add_argument(
        "--stage-1-results", required=True, type=Path,
        help="Path to a Stage 1 results directory (e.g. results/run_<ts>/stage_1/)",
    )
    parser.add_argument(
        "--tier", default="light", choices=("light", "medium", "heavy"),
        help="Which preset list from the config to use",
    )
    parser.add_argument(
        "--cheap-judge-id", default=None,
        help="Judge id for cheap-judge rollout signal (default: config.judges.cheap_judge_id)",
    )
    parser.add_argument(
        "--judges-dir", default=DEFAULT_JUDGES_DIR, type=Path,
        help="Directory of judge YAMLs",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: <stage_1_results>/../stage_2/)",
    )
    parser.add_argument(
        "--max-concepts", type=int, default=None,
        help="Cap the number of Stage 1 winners processed (pilot mode)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    verbose = args.verbose and not args.quiet
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    if not args.config.exists():
        print(f"error: config {args.config} not found", file=sys.stderr)
        return 1
    if not args.stage_1_results.exists():
        print(f"error: stage 1 results {args.stage_1_results} not found", file=sys.stderr)
        return 1

    try:
        config = Stage2Config.from_yaml(args.config)
    except ValidationError as e:
        print(f"config validation failed:\n{e}", file=sys.stderr)
        return 1

    archive_path, manifest_path = asyncio.run(run_stage_2(
        stage_1_dir=args.stage_1_results,
        config=config,
        config_tier=args.tier,
        cheap_judge_id=args.cheap_judge_id,
        judges_dir=str(args.judges_dir),
        output_dir=args.output_dir,
        max_concepts=args.max_concepts,
    ))
    print(f"qd_archive: {archive_path}")
    print(f"handoff_manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
