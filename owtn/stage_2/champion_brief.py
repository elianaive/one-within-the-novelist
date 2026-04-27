"""Per-tree state for the Stage 2 champion brief.

The brief logic itself lives in `owtn.optimizer.adapters` —
`compute_stage_2_champion_brief` mirrors Stage 1's `compute_stage_1_lineage_brief`.
This module owns only what's Stage-2-specific:

- `TreeBriefState`: a typed accumulator for one MCTS tree's full-panel
  critiques + cached brief. Stage 1 uses Shinka's `private_metrics` dict
  (managed by ShinkaEvolve) for the same purpose, so it doesn't need a
  typed object. Stage 2 lacks that storage layer, so we provide one here.
- `record_full_panel_critique`: append one critique to a tree's history.
- `get_or_compute_brief`: thin wrapper around the optimizer's
  `compute_stage_2_champion_brief` that unpacks/repacks the `TreeBriefState`
  object's fields.

Per `docs/stage-2/mcts.md` §Champion Brief Feedback Loop, the tree itself
is the brief's subject — Stage 2's champion churns too fast for per-champion
accumulation. The tree accumulates critiques continuously across champion
changes; the brief summarizes that history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from owtn.optimizer.adapters import compute_stage_2_champion_brief
from owtn.optimizer.models import LineageBrief


@dataclass
class TreeBriefState:
    """Per-tree accumulator. Owned by the Phase 9 runner — one per
    (concept_id, preset) tree, threaded through forward + backward + Phase 3.

    Fields:
        full_panel_critiques: critique records appended in arrival order.
            Each is a dict matching the shape Stage 1's `MatchCritique` uses
            (keys: self_label, opponent_label, self_was_champion, self_dag,
            opponent_genome, outcome, dim_outcomes, judge_reasonings, timestamp).
        cached_count: count of critiques the cached brief was computed
            against. None = no cache yet.
        cached_brief: the LineageBrief Pydantic instance, cached.
        cached_render: rendered markdown for direct injection into expansion
            prompts. Cached alongside `cached_brief` to avoid re-rendering on
            every read.
    """
    full_panel_critiques: list[dict] = field(default_factory=list)
    cached_count: int | None = None
    cached_brief: LineageBrief | None = None
    cached_render: str | None = None


def record_full_panel_critique(state: TreeBriefState, critique: dict) -> None:
    """Append one full-panel critique to the tree's history.

    Caller (Phase 9 runner) invokes after each promotion gate and each
    within-concept tournament match. The lazy summarizer fires on the next
    `get_or_compute_brief` call subject to the re-summarize cadence.
    """
    state.full_panel_critiques.append(critique)


async def get_or_compute_brief(
    state: TreeBriefState,
    *,
    classifier_model: str,
    re_summarize_every: int = 3,
    force_resummarize: bool = False,
) -> str:
    """Return rendered markdown for injection into the expansion prompt.

    Thin wrapper: delegates the actual cache + summarizer logic to
    `compute_stage_2_champion_brief` in `owtn.optimizer.adapters` (which
    sits alongside Stage 1's parallel adapter). Updates `state` in place
    when the summarizer fires; returns the cached render when cache is fresh.

    `force_resummarize=True` is for champion-promotion events per
    `mcts.md` §Forced re-render.
    """
    rendered, new_cache = await compute_stage_2_champion_brief(
        full_panel_critiques=state.full_panel_critiques,
        cached_count=state.cached_count,
        cached_brief=state.cached_brief,
        classifier_model=classifier_model,
        re_summarize_every=re_summarize_every,
        force_resummarize=force_resummarize,
    )
    if new_cache is not None:
        state.cached_count, state.cached_brief = new_cache
        state.cached_render = rendered
    return rendered
