"""Scalar-mode handoff: per-preset rescore via the configured composition.

Replaces `run_within_concept_tournament` for `scoring_mode == "scalar"`.
Each preset entry's DAG is re-scored by the `handoff_rescore` composition
(typically a persona-ensemble atomic scorer); entries are sorted by
aggregate descending and rank-assigned. Persona-ensemble cost is acceptable
at this call site because volume is low (≤4 presets per concept) and panel
disagreement adds confidence to the top-K selection.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from owtn.evaluation.scalar import build_scorer_from_config
from owtn.evaluation.scalar.renderers import render_stage2_partial
from owtn.evaluation.scalar.types import AggregatedScoreCard, ScoreCard
from owtn.llm.call_logger import llm_context
from owtn.stage_2.tournament import TournamentEntry

logger = logging.getLogger(__name__)


async def rescore_entries_scalar(
    entries: list[TournamentEntry],
    *,
    composition_name: str,
) -> list[TournamentEntry]:
    """Re-score each preset's DAG with the named scalar composition; rank by
    aggregate descending. Mutates entries in place by setting `mcts_reward`
    to the rescored aggregate (so downstream sort_key sees it) and
    `wins`/`dim_wins_total` as proxies of relative ranking.

    Returns entries sorted best-first.
    """
    scorer = build_scorer_from_config(composition_name, render_stage2_partial)

    async def score_one(entry: TournamentEntry) -> tuple[TournamentEntry, ScoreCard | AggregatedScoreCard]:
        prev_ctx = llm_context.get({})
        token = llm_context.set({**prev_ctx, "role": "scalar_handoff_rescore", "preset": entry.preset})
        try:
            card = await scorer.score(entry.dag)
        finally:
            llm_context.reset(token)
        return entry, card

    results = await asyncio.gather(*[score_one(e) for e in entries])

    by_aggregate = sorted(results, key=lambda x: -x[1].aggregate)
    n = len(by_aggregate)
    for rank, (entry, card) in enumerate(by_aggregate):
        entry.mcts_reward = card.aggregate
        entry.wins = n - rank - 1   # top entry has the most "wins" for sort_key compatibility
        entry.dim_wins_total = int(card.aggregate * 100)   # proxy
        logger.info(
            "scalar handoff rescore: preset=%s aggregate=%.3f rank=%d",
            entry.preset, card.aggregate, rank + 1,
        )

    return [entry for entry, _ in by_aggregate]
