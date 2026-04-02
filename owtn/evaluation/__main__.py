"""ShinkaEvolve subprocess entry point.

Usage: python -m owtn.evaluation --program_path <path> --results_dir <path> [--config_path <path>]
"""

import argparse
import asyncio

from owtn.evaluation.stage_1 import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Stage 1 concept genome."
    )
    parser.add_argument("--program_path", required=True, help="Path to concept genome JSON file")
    parser.add_argument("--results_dir", required=True, help="Directory to write metrics.json and correct.json")
    parser.add_argument("--config_path", default="configs/stage_1_default.yaml", help="Path to stage config YAML")
    args = parser.parse_args()
    asyncio.run(evaluate(args.program_path, args.results_dir, args.config_path))


if __name__ == "__main__":
    main()
