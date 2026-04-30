"""Unified CLI for owtn.tools.

Usage:
    uv run python -m owtn.tools analyze --passage P.txt --caller-model M
    uv run python -m owtn.tools lookup --query austen --n 2
    uv run python -m owtn.tools slop --passage P.txt
    uv run python -m owtn.tools style --passage P.txt
    uv run python -m owtn.tools thesaurus --word happy --mode means_like
    uv run python -m owtn.tools rebuild-cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lookup_exemplar import lookup_exemplar
from .slop_score import slop_score
from .stylometry import rebuild_cache, stylometry
from .thesaurus import MODE_TO_PARAM, thesaurus
from .writing_style import writing_style


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

    p_lookup = sub.add_parser(
        "lookup",
        help="Retrieve passages from exemplars or baselines (deterministic key lookup)",
    )
    p_lookup.add_argument("--query", required=True,
                          help="Entry id, author slug, or tag (no NL — this is the deterministic path)")
    p_lookup.add_argument("--n", type=int, default=2,
                          help="How many passages to return (default 2)")
    p_lookup.add_argument("--max-words", type=int, default=400,
                          help="Truncate each passage to N words for compact display")

    p_slop = sub.add_parser("slop", help="Run the slop-score tool on a passage")
    p_slop.add_argument("--passage", required=True, help="Path to passage text file")
    p_slop.add_argument("--compare-to", nargs="+", default=None,
                        help="One or more reference passage files to compare against")

    p_style = sub.add_parser("style", help="Run the writing-style tool on a passage")
    p_style.add_argument("--passage", required=True, help="Path to passage text file")
    p_style.add_argument("--compare-to", nargs="+", default=None,
                        help="One or more reference passage files to compare against")

    p_thes = sub.add_parser("thesaurus", help="Query Datamuse for diction work")
    p_thes.add_argument("--word", required=True, help="Query word")
    p_thes.add_argument("--mode", default="means_like",
                        choices=sorted(MODE_TO_PARAM),
                        help="Lookup mode (default: means_like)")
    p_thes.add_argument("--max", type=int, default=20,
                        help="Max results to return (default 20)")

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
        result = lookup_exemplar(args.query, n=args.n, max_words=args.max_words)
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "slop":
        passage = Path(args.passage).read_text(encoding="utf-8")
        compare_to = ([Path(p).read_text(encoding="utf-8") for p in args.compare_to]
                      if args.compare_to else None)
        report = slop_score(passage, compare_to=compare_to)
        print(json.dumps(report.to_dict(), indent=2))
        return

    if args.cmd == "style":
        passage = Path(args.passage).read_text(encoding="utf-8")
        compare_to = ([Path(p).read_text(encoding="utf-8") for p in args.compare_to]
                      if args.compare_to else None)
        report = writing_style(passage, compare_to=compare_to)
        print(json.dumps(report.to_dict(), indent=2))
        return

    if args.cmd == "thesaurus":
        report = thesaurus(args.word, mode=args.mode, max_results=args.max)
        print(json.dumps(report.to_dict(), indent=2))
        return


if __name__ == "__main__":
    main()
