"""ShinkaEvolve subprocess entry point.

Usage: python -m owtn.evaluation --program_path <path> --results_dir <path> [--config_path <path>]
"""

import argparse
import asyncio
from pathlib import Path

from owtn.evaluation.stage_1 import evaluate
from owtn.llm.call_logger import llm_log_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Stage 1 concept genome."
    )
    parser.add_argument("--program_path", required=True, help="Path to concept genome JSON file")
    parser.add_argument("--results_dir", required=True, help="Directory to write metrics.json and correct.json")
    parser.add_argument("--config_path", default="configs/stage_1/medium.yaml", help="Path to stage config YAML")
    parser.add_argument("--log_dir", default=None, help="Directory for LLM call logs")
    args = parser.parse_args()

    if args.log_dir:
        llm_log_dir.set(args.log_dir)
    else:
        # Derive from results_dir: gen_N/results -> stage_1 root
        stage_dir = Path(args.results_dir).parent.parent
        llm_log_dir.set(str(stage_dir))

    asyncio.run(evaluate(args.program_path, args.results_dir, args.config_path))


if __name__ == "__main__":
    main()
