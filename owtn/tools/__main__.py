"""Unified CLI for owtn.tools.

Usage:
    uv run python -m owtn.tools analyze --passage P.txt --caller-model M
    uv run python -m owtn.tools lookup --target austen --n 2
    uv run python -m owtn.tools rebuild-cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lookup_exemplar import lookup_exemplar
from .stylometry import rebuild_cache, stylometry


def main() -> None:
    parser = argparse.ArgumentParser(prog="owtn.tools", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("analyze", help="Run the stylometry tool on a passage")
    p_run.add_argument("--passage", required=True, help="Path to passage text file")
    p_run.add_argument("--caller-model", default=None,
                       help="Short model tag, e.g. sonnet-4-6")
    p_run.add_argument("--neutral", default=None,
                       help="Optional path to a neutral-baseline text file")
    p_run.add_argument("--target-styles", nargs="+", default=None,
                       help="Optional author/tag tokens to compute style distances for")

    p_lookup = sub.add_parser("lookup", help="Retrieve passages from exemplars or baselines")
    p_lookup.add_argument("--target", required=True,
                          help="Author slug, style tag, or entry id")
    p_lookup.add_argument("--n", type=int, default=2,
                          help="How many passages to return (default 2)")
    p_lookup.add_argument("--max-words", type=int, default=400,
                          help="Truncate each passage to N words for compact display")

    sub.add_parser("rebuild-cache", help="Rebuild the stylometric signal cache")

    args = parser.parse_args()

    if args.cmd == "rebuild-cache":
        corpus = rebuild_cache()
        print(f"Cache rebuilt. {len(corpus.entries)} reference entries loaded.")
        return

    if args.cmd == "analyze":
        passage = Path(args.passage).read_text(encoding="utf-8")
        neutral = Path(args.neutral).read_text(encoding="utf-8") if args.neutral else None
        report = stylometry(
            passage,
            caller_model=args.caller_model,
            neutral_baseline=neutral,
            target_styles=args.target_styles,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return

    if args.cmd == "lookup":
        result = lookup_exemplar(args.target, n=args.n, max_words=args.max_words)
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
