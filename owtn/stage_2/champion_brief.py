"""Per-tree state for the Stage 2 champion brief.

The brief logic itself lives in `owtn.optimizer.adapters`. This module owns
what's Stage-2-specific: the per-tree accumulator and the small helpers
that record events into it.

Two parallel paths feed the brief, depending on `Stage2Config.scoring_mode`:

- **Pairwise**: `record_full_panel_critique` appends `MatchCritique`-shaped
  dicts after each full-panel gate / within-concept tournament match. The
  `compute_stage_2_champion_brief` adapter reads these.
- **Scalar**: `record_rollout_outcome` appends `ScoredRolloutRecord` rows
  after each rollout (score + reasoning + DAG). Both
  `compute_stage_2_scalar_tree_brief` (whole-tree) and
  `compute_stage_2_scalar_lineage_brief` (per-leaf path) read these.

Per `docs/stage-2/mcts.md` §Champion Brief Feedback Loop, the tree itself
is the brief's subject — Stage 2's champion churns too fast for per-champion
accumulation. The tree accumulates critiques continuously across champion
changes; the brief summarizes that history.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from owtn.optimizer.adapters import (
    compute_stage_2_champion_brief,
    compute_stage_2_scalar_lineage_brief,
    compute_stage_2_scalar_tree_brief,
    lineage_records_for_target,
)
from owtn.optimizer.models import LineageBrief


@dataclass(frozen=True)
class ScoredRolloutRecord:
    """One scalar rollout's outcome, captured for brief-feedback summarization.

    Fields:
        score: aggregate scalar reward in `[scale_min, scale_max]` (typically
            [0, 1] after rubric normalization). The summarizer reads this
            as one signal alongside the reasoning text.
        reasoning: the scalar judge's per-dimension walkthrough. Specific
            named structural concerns here are the strongest signal for
            the brief; generic summary text is weaker.
        dag: `DAG.model_dump()` of the partial that was scored. Kept as
            a dict so the record stays serializable and the summarizer can
            render the structure inline alongside the reasoning.
    """
    score: float
    reasoning: str
    dag: dict


@dataclass
class _LineageCacheEntry:
    """One leaf's cached lineage brief. Keyed in `TreeBriefState.lineage_cache`
    by a stable digest of the target DAG."""
    count: int
    brief: LineageBrief
    rendered: str


@dataclass
class TreeBriefState:
    """Per-tree accumulator. Owned by the runner — one per (concept_id,
    preset) tree, threaded through forward + backward + Phase 3.

    Fields:
        full_panel_critiques: pairwise-mode critique records, appended in
            arrival order. Each is a `MatchCritique`-shaped dict.
        rollout_records: scalar-mode rollout outcomes (score + reasoning + DAG).
            Empty in pairwise mode; populated by `record_rollout_outcome`.
        cached_count: count of records (matching the active-mode list) the
            tree-level cached brief was computed against. None = no cache.
        cached_brief: the tree-level `LineageBrief`, cached.
        cached_render: rendered markdown for the tree-level brief.
        summarize_in_flight: set while a tree-level summarizer call is
            running, so concurrent rollouts skip re-firing it.
        lineage_cache: per-leaf scalar-mode lineage briefs, keyed by the
            target DAG's stable digest. Populated lazily by
            `get_or_compute_scalar_lineage_brief` on the expansion path.
    """
    full_panel_critiques: list[dict] = field(default_factory=list)
    rollout_records: list[ScoredRolloutRecord] = field(default_factory=list)
    cached_count: int | None = None
    cached_brief: LineageBrief | None = None
    cached_render: str | None = None
    summarize_in_flight: bool = False
    lineage_cache: dict[str, _LineageCacheEntry] = field(default_factory=dict)


def record_full_panel_critique(state: TreeBriefState, critique: dict) -> None:
    """Append one full-panel critique to the tree's history.

    Pairwise-mode caller invokes after each promotion gate and each
    within-concept tournament match. The lazy summarizer fires on the next
    `get_or_compute_brief` call subject to the re-summarize cadence.
    """
    state.full_panel_critiques.append(critique)


def record_rollout_outcome(
    state: TreeBriefState, *, score: float, reasoning: str, dag: dict,
) -> None:
    """Append one scalar rollout's outcome to the tree's brief feed.

    Used by `_make_rollout_fn_scalar`. The summarizer pattern-matches across
    many of these to extract what structural moves correlate with reward.
    """
    state.rollout_records.append(
        ScoredRolloutRecord(score=score, reasoning=reasoning, dag=dag)
    )


async def get_or_compute_brief(
    state: TreeBriefState,
    *,
    classifier_model: str,
    re_summarize_every: int = 3,
    force_resummarize: bool = False,
) -> str:
    """Pairwise-mode brief: delegate to `compute_stage_2_champion_brief`.

    Updates `state` in place when the summarizer fires; returns the cached
    render when cache is fresh. `force_resummarize=True` is for
    champion-promotion events per `mcts.md` §Forced re-render.
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


async def get_or_compute_scalar_brief(
    state: TreeBriefState,
    *,
    classifier_model: str,
    re_summarize_every: int = 5,
    force_resummarize: bool = False,
) -> str:
    """Scalar-mode tree brief: delegate to `compute_stage_2_scalar_tree_brief`.

    Mirrors `get_or_compute_brief` but reads from `rollout_records` instead
    of `full_panel_critiques`. The cache slot (`cached_count`, `cached_brief`,
    `cached_render`) is shared — the two modes never interleave on one tree.
    """
    rendered, new_cache = await compute_stage_2_scalar_tree_brief(
        rollout_records=state.rollout_records,
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


def _dag_digest(dag_dict: dict) -> str:
    """Stable short hash of a DAG dict, used as the lineage cache key.

    The DAG is dumped from a Pydantic model — key insertion order is stable —
    so canonical JSON dump + sha256 gives a consistent fingerprint across
    repeat lookups. Only the fields that affect ancestry (nodes, edges,
    motif_threads, story_constraints, character_arcs) feed the digest;
    runtime metadata (preset, target_node_count) is omitted so two leaves
    that differ only in metadata still share a cache entry."""
    keys = ("nodes", "edges", "motif_threads", "story_constraints", "character_arcs")
    payload = {k: dag_dict.get(k) for k in keys}
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


async def get_or_compute_scalar_lineage_brief(
    state: TreeBriefState,
    *,
    target_dag: dict,
    classifier_model: str,
) -> str:
    """Scalar-mode lineage brief: scoped to one MCTS path.

    Filters `state.rollout_records` to those whose DAG is a structural
    ancestor of `target_dag`, then delegates to the lineage adapter. The
    per-leaf cache (`state.lineage_cache`) holds one entry per DAG digest
    so concurrent rollouts on different leaves don't trample each other's
    summarized briefs.
    """
    lineage_records = lineage_records_for_target(state.rollout_records, target_dag)
    if not lineage_records:
        # Cold path: no ancestor records yet — return the seed placeholder
        # without touching the cache or firing an LLM call.
        from owtn.optimizer.adapters import LINEAGE_PATH_SUBJECT
        return LINEAGE_PATH_SUBJECT.seed_placeholder

    digest = _dag_digest(target_dag)
    cached = state.lineage_cache.get(digest)
    cached_count = cached.count if cached is not None else None
    cached_brief = cached.brief if cached is not None else None

    rendered, new_cache = await compute_stage_2_scalar_lineage_brief(
        lineage_records=lineage_records,
        cached_count=cached_count,
        cached_brief=cached_brief,
        classifier_model=classifier_model,
    )
    if new_cache is not None:
        count, brief = new_cache
        state.lineage_cache[digest] = _LineageCacheEntry(
            count=count, brief=brief, rendered=rendered,
        )
    elif cached is not None and cached.count == len(lineage_records):
        # Cache hit — return the previously rendered text rather than
        # re-rendering on top of the cached brief.
        return cached.rendered
    return rendered
