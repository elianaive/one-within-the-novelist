"""Per-concept Stage 2 orchestration.

Runs the full Stage 2 stack for one (Stage1Winner, preset_set) pair:
  seed_root → forward + backward MCTS (per preset, parallel)
            → Phase 3 refinement
            → within-concept tournament across preset winners
            → handoff construction (top-K + archive side-effects)

What lives here:
- `run_concept`: top-level — orchestrates seed_root, all preset trees,
  refinement, tournament, handoff. Per-tree mechanics live in
  `owtn.stage_2.tree_runtime`; per-tree work is fanned out via
  `run_one_preset_tree`.

Why split from `runner.py`: per the plan's cleanliness flag, `runner.py`
must stay under 300 lines. Per-concept logic alone is too much for that
budget — extracting it here keeps the runner focused on the run-level
loop (read Stage 1 → for each concept → aggregate → write manifest).
"""

from __future__ import annotations

import asyncio
import logging

from owtn.llm.call_logger import llm_context
from owtn.models.judge import JudgePersona
from owtn.models.stage_2.config import Stage2Config
from owtn.models.stage_2.handoff import Stage1Winner, Stage2Output
from owtn.stage_2.archive import Stage2Archive
from owtn.stage_2.handoff import build_handoff_for_concept
from owtn.stage_2.operators import seed_root
from owtn.stage_2.tournament import run_within_concept_tournament
from owtn.stage_2.tree_runtime import run_one_preset_tree


logger = logging.getLogger(__name__)


async def run_concept(
    *,
    winner: Stage1Winner,
    config: Stage2Config,
    config_tier: str,
    cheap_judge: JudgePersona,
    full_panel: list[JudgePersona] | None,
    classifier_model: str,
    archive: Stage2Archive,
) -> list[Stage2Output]:
    """Run Stage 2 for one Stage 1 concept. Returns advancing handoff outputs.

    Steps:
    1. Seed: one shared seed DAG per concept (seed_root).
    2. Per preset (parallel): forward → backward → Phase 3 → tournament entry.
    3. Within-concept tournament across all preset entries.
    4. Build handoff (top-K + archive side-effects).
    """
    presets = _select_presets(config, config_tier)
    if not presets:
        logger.warning(
            "concept %s: no presets configured for tier %r — skipping",
            winner.program_id, config_tier,
        )
        return []

    # Tag every LLM call inside this concept's branch with its program_id so
    # logs can be demuxed by concept. asyncio Tasks copy the current context
    # on creation, so this set is isolated to this concept's coroutine tree.
    base_ctx = dict(llm_context.get({}))
    llm_context.set({**base_ctx, "concept_id": winner.program_id})

    logger.info(
        "Stage 2 concept %s (anchor role=%r): %d presets × %d iterations/phase",
        winner.program_id, winner.genome.anchor_scene.role,
        len(presets), config.iterations_per_phase,
    )

    seed_dag = await seed_root(
        winner.genome,
        concept_id=winner.program_id,
        preset=presets[0],  # placeholder — overridden per preset below
        target_node_count=_default_target_node_count(config),  # fallback only
        node_count_targets=config.node_count_targets,  # LLM picks bucket → midpoint
        # Use the heavyweight generator: motif/demand/sizing extraction
        # benefits from generator-tier reasoning. classifier_model is
        # reserved for the brief summarizer's cheap aggregation pass.
        model_name=config.expansion_model,
    )

    preset_tasks = [
        run_one_preset_tree(
            seed_dag=seed_dag.model_copy(update={"preset": p}),
            concept=winner.genome,
            preset=p,
            config=config,
            cheap_judge=cheap_judge,
            full_panel=full_panel,
            classifier_model=classifier_model,
        )
        for p in presets
    ]
    entries = await asyncio.gather(*preset_tasks)

    if len(entries) >= 2:
        ranked = await run_within_concept_tournament(
            entries,
            concept=winner.genome,
            panel=full_panel or [cheap_judge],
        )
    else:
        # Single-preset case: skip the tournament; the lone entry is rank 1.
        ranked = list(entries)
        for entry in ranked:
            entry.wins = 0  # no matches; sort_key still works

    outputs, near_tie = build_handoff_for_concept(
        concept_id=winner.program_id,
        stage_1_forwarded=winner,
        ranked_entries=ranked,
        archive=archive,
        top_k=config.top_k_to_stage_3,
        near_tie_promoted=config.near_tie_promoted,
    )
    if near_tie:
        logger.info("concept %s: near-tie promoted at K boundary", winner.program_id)
    return outputs


def _select_presets(config: Stage2Config, tier: str) -> list[str]:
    if tier == "light":
        return list(config.presets.light)
    if tier == "medium":
        return list(config.presets.medium)
    if tier == "heavy":
        return list(config.presets.heavy)
    raise ValueError(f"unknown config tier: {tier!r}")


_DEFAULT_TARGET_WORD_COUNT = 3000
"""Stub: every concept assumes a 3K-word target prose length.
Real fix is concept-declared `target_word_count` on `ConceptGenome`; until
then, this hardcoded key drives `_default_target_node_count`."""


def _default_target_node_count(config: Stage2Config) -> int:
    """Pick a default target_node_count for seed DAGs.

    Returns the MIDPOINT of the configured `[min, max]` range for the
    default word target. Picking the lower bound under-shoots — the LLM
    can always stop early but can't reasonably extend past target — so
    midpoint is the more permissive default. Until concepts declare
    their own target word count, this drives every concept to the same
    range; see `_DEFAULT_TARGET_WORD_COUNT`.
    """
    target = config.node_count_targets.get(_DEFAULT_TARGET_WORD_COUNT)
    if target is None:
        return 5
    lo, hi = target
    return (lo + hi) // 2
