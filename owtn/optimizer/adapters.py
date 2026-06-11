"""Per-stage adapters for the generic optimizer-state summarizers.

Each stage contributes:
- An input-gathering function (how to read the stage's genome and history)
- Placeholder values (filled into the generic prompt templates)
- An output-render function (formatted text injected into the mutation prompt)

For v1, only Stage 1 is implemented. Split per-stage files only when an
adapter outgrows this shared module.

See `lab/issues/2026-04-22-global-optimizer-state.md`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from owtn.optimizer.lineage_brief import (
    BriefSubject,
    SelfExtractor,
    SelfFormatter,
    get_or_compute_brief,
    render_lineage_brief,
    render_raw_fallback,
    render_scalar_raw_fallback,
    render_scalar_tree_brief,
    summarize_lineage,
    summarize_scalar_tree,
)
from owtn.optimizer.population_brief import (
    PopulationBrief,
    PopulationBriefSummarizer,
    render_exploration_directions,
    render_population_context,
)

logger = logging.getLogger(__name__)

_OPTIMIZER_DIR = Path(__file__).resolve().parent


# Stage 1 fills the generic `lineage_prompt.txt` placeholders with concept-
# genome-specific language. The result is token-equivalent to the pre-refactor
# `owtn/prompts/stage_1/parent_brief.txt`.
_STAGE_1_LINEAGE_PLACEHOLDERS = {
    "domain_hints": "a fiction concept",
    "rubric_summary": "nine resonance dimensions",
    "attractor_examples": (
        "Examples of what belongs here: "
        "\"archive/apparatus framing,\" "
        "\"withholding voice around unnamed grief,\" "
        "\"second-person address to an absent figure,\" "
        "\"italicized final image.\""
    ),
    "subject_noun": "concept",
}


def _load_lineage_prompt_template() -> str:
    return (_OPTIMIZER_DIR / "lineage_prompt.txt").read_text()


def _render_prompt_frame(run_prompt: str) -> str:
    """Frame block injected when the run was given a user prompt; without it
    the summarizer reads prompt-required convergence as drift to escape. See
    `lab/issues/2026-04-30-prompt-aware-optimizer.md`."""
    if not run_prompt or not run_prompt.strip():
        return ""
    return (
        "## Frame: the run is constrained by the user prompt\n\n"
        "The run was given this user prompt:\n\n"
        f"<user_prompt>\n{run_prompt.strip()}\n</user_prompt>\n\n"
        "Treat the prompt as a **non-negotiable frame**, not as drift to "
        "escape. Patterns that recur because the prompt requires them are "
        "**correct convergence on the user's intent**, not population drift, "
        "and must NOT be flagged as drift, attractor signature, or things a "
        "successor should differ on. Drift means narrowing **within** the "
        "prompt's frame — same dramatic structures, imagery palette, "
        "persona-types, voice register, or formal container that the prompt "
        "does not require. Counter-examples and exploration directions you "
        "produce must still satisfy the user's prompt. If a counter-example "
        "would only work by leaving the prompt's frame, it is not a valid "
        "counter-example — find one inside the frame instead."
    )


def _resolve_placeholders(template: str, placeholders: dict[str, str]) -> str:
    out = template
    for key, value in placeholders.items():
        out = out.replace("{" + key + "}", value)
    if placeholders.get("prompt_frame", "") == "":
        # Collapse the placeholder's blank-line pair so an empty frame
        # doesn't leave a triple blank.
        out = out.replace("\n\n\n\n", "\n\n")
    return out


def _stage_1_lineage_system_prompt(run_prompt: str = "") -> str:
    placeholders = {
        **_STAGE_1_LINEAGE_PLACEHOLDERS,
        "prompt_frame": _render_prompt_frame(run_prompt),
    }
    return _resolve_placeholders(_load_lineage_prompt_template(), placeholders)


def _stage_1_format_self(genome: dict) -> str:
    """Render a Stage 1 concept genome into the inline match-block slot.

    Mirrors the pre-refactor `_format_match_block` body: the summarizer sees
    the concept's premise and target_effect (truncated to 400 chars each).
    """
    return (
        f"  premise: {genome.get('premise', '')[:400]}\n"
        f"  target_effect: {genome.get('target_effect', '')[:400]}"
    )


async def compute_stage_1_lineage_brief(
    *,
    self_genome: dict,
    private_metrics: dict,
    classifier_model: str,
    run_prompt: str = "",
) -> tuple[str, dict | None]:
    """Stage 1 entry point: fills Stage 1 placeholders and supplies the
    concept-genome formatter. `run_prompt` (the user-supplied creative
    direction, `Stage1Config.prompt`) renders as a frame block when set;
    without it the summarizer reads prompt-required convergence as drift."""
    return await get_or_compute_brief(
        self_genome=self_genome,
        private_metrics=private_metrics,
        classifier_model=classifier_model,
        system_prompt=_stage_1_lineage_system_prompt(run_prompt),
        format_self=_stage_1_format_self,
    )


# --- Stage 2 champion brief adapter ---------------------------------------
#
# Stage 2 reuses the optimizer's full machinery via the BriefSubject /
# SelfExtractor / SelfFormatter parameterization on the lineage_brief
# functions. The only Stage-2-specific things below are:
# - The TREE_SUBJECT (different wording at the seams).
# - `_stage_2_format_dag_for_match` (DAG rendering instead of premise text).
# - The system prompt (loaded via owtn.prompts.stage_2.registry).
# - The cache shape: caller passes (count, LineageBrief) explicitly rather
#   than via `private_metrics` because Stage 2 has no DB-backed per-program
#   storage. `TreeBriefState` in owtn.stage_2.champion_brief holds it.

TREE_SUBJECT = BriefSubject(
    upper="THIS TREE",
    narrative="this tree",
    block_qualifier="'s structure",
    seed_placeholder="Initial tree — no full-panel critiques accumulated yet.",
)


def _stage_2_load_system_prompt() -> str:
    from owtn.prompts.stage_2.registry import load_champion_brief_system
    return load_champion_brief_system()


def _stage_2_format_dag_for_match(dag_data: dict) -> str:
    """Render a DAG dict (from `DAG.model_dump()`) as a compact match-block snippet.

    Truncated to ~2000 chars per DAG so a 30-event tree stays under context budget.
    Corrupt-cache fallback: emits a one-line note rather than crashing the summarizer.
    """
    # Local imports to keep the optimizer module from pulling stage_2
    # at package-load time (only fires when Stage 2 adapter actually runs).
    from owtn.models.stage_2.dag import DAG
    from owtn.stage_2.rendering import render

    try:
        dag = DAG.model_validate(dag_data)
    except Exception as e:  # noqa: BLE001 — corrupt cache shouldn't crash the summarizer
        logger.warning("Stage 2 brief: could not validate DAG dict: %s", e)
        return f"  (DAG could not be reconstructed: {type(e).__name__})"
    rendered = render(dag)
    if len(rendered) > 2000:
        rendered = rendered[:2000] + "\n  ... (truncated)"
    return rendered


def _stage_2_extract_self(critique: dict) -> dict:
    """Stage 2 critiques carry the tree's iteration-specific DAG inline; the
    lineage's "self" thus differs across critiques (the tree expanded between
    matches). Stage 1's lineage genome is fixed, so its extractor closes over
    a single dict; Stage 2's reads from the critique itself."""
    return critique.get("self_dag", {}) or {}


# Module-level callables so the test mocks have stable references.
_stage_2_extract_self_fn: SelfExtractor = _stage_2_extract_self
_stage_2_format_self_fn: SelfFormatter = _stage_2_format_dag_for_match


def stage_2_render_raw_fallback(critiques: list[dict]) -> str:
    """Last-resort render with tree-flavored wording. Cold start returns the
    seed placeholder; summarizer failure returns the most recent matches."""
    return render_raw_fallback(critiques, TREE_SUBJECT)


async def compute_stage_2_champion_brief(
    *,
    full_panel_critiques: list[dict],
    cached_count: int | None,
    cached_brief: "LineageBrief | None",
    classifier_model: str,
    re_summarize_every: int = 3,
    force_resummarize: bool = False,
) -> tuple[str, "tuple[int, LineageBrief] | None"]:
    """Stage 2 entry point. Mirrors `compute_stage_1_lineage_brief` shape:
    callers supply the critique history + cache, get back rendered text +
    optional new cache payload to write.

    Returns:
        (rendered_text, new_cache_or_None) where new_cache is `(count, brief)`
        when the summarizer fired, or None when the cache was fresh / the
        summarizer failed (raw fallback in `rendered_text`).

    Behavior:
    - Cold start (no critiques): returns the seed placeholder. No LLM call.
    - Cache fresh (count delta < threshold, no force): caller's job to use
      its own cached render — we return None as the new_cache signal.
    - Cache stale or force=True: fires summarizer, returns new render + cache.
    - Summarizer failure: returns raw fallback render, None cache (so caller
      can retry next time).
    """
    if not full_panel_critiques:
        return TREE_SUBJECT.seed_placeholder, None

    if not force_resummarize and cached_count is not None:
        delta = len(full_panel_critiques) - cached_count
        if delta < re_summarize_every and cached_brief is not None:
            # Cache fresh — render from cached brief, signal no-update.
            rendered = render_lineage_brief(
                cached_brief, full_panel_critiques, TREE_SUBJECT
            )
            return rendered, None

    try:
        brief = await summarize_lineage(
            match_critiques=full_panel_critiques,
            classifier_model=classifier_model,
            system_prompt=_stage_2_load_system_prompt(),
            extract_self=_stage_2_extract_self_fn,
            format_self=_stage_2_format_self_fn,
            subject=TREE_SUBJECT,
        )
    except Exception as e:  # noqa: BLE001 — never crash callers on summarizer failure
        logger.warning(
            "Stage 2 champion brief summarizer failed (%s); using raw fallback.", e,
        )
        return stage_2_render_raw_fallback(full_panel_critiques), None

    rendered = render_lineage_brief(brief, full_panel_critiques, TREE_SUBJECT)
    return rendered, (len(full_panel_critiques), brief)


# --- Lineage subject for per-leaf scalar briefs ---------------------------
#
# Distinct from TREE_SUBJECT: when the brief is scoped to one MCTS path
# (only rollouts whose DAG is a structural ancestor of the current target),
# the noun shifts from "the tree" to "this path / trajectory" so the
# summarizer's wording matches the narrower scope.

LINEAGE_PATH_SUBJECT = BriefSubject(
    upper="THIS PATH",
    narrative="this path",
    block_qualifier=" partial",  # "THIS PATH partial:" introduces each ancestor DAG
    seed_placeholder="Initial path — no ancestor rollouts yet.",
)


def lineage_records_for_target(
    rollout_records: list[Any], target_dag: dict,
) -> list[Any]:
    """Filter `rollout_records` to those whose DAG is a structural ancestor
    of `target_dag`.

    A rollout's DAG is an ancestor of `target_dag` iff its node-id set is a
    subset of `target_dag`'s and its edge set is a subset. MCTS only ever
    grows a partial — it never removes nodes or edges within a single
    bidirectional run — so structural prefix === ancestry by construction.

    Returns records sorted ancestor-first (smallest DAG first), so the
    summarizer reads them as a trajectory.
    """
    target_node_ids = {n["id"] for n in target_dag.get("nodes", [])}
    target_edges = frozenset(
        (e["src"], e["dst"], e["type"])
        for e in target_dag.get("edges", [])
    )

    matched = []
    for record in rollout_records:
        dag = record.dag
        node_ids = {n["id"] for n in dag.get("nodes", [])}
        if not node_ids.issubset(target_node_ids):
            continue
        edges = frozenset(
            (e["src"], e["dst"], e["type"])
            for e in dag.get("edges", [])
        )
        if not edges.issubset(target_edges):
            continue
        matched.append(record)

    matched.sort(key=lambda r: (len(r.dag.get("nodes", [])), len(r.dag.get("edges", []))))
    return matched


# --- Stage 2 scalar tree-brief adapter ------------------------------------
#
# Parallel to `compute_stage_2_champion_brief` but consumes scalar rollout
# records instead of pairwise match critiques. Same `LineageBrief` output,
# same `(count, brief)` cache shape, different summarizer prompt + input
# rendering. The pairwise and scalar paths never run on the same tree (the
# mode is a Stage2Config field), so they share `TreeBriefState`'s cache slot.


def _stage_2_load_scalar_system_prompt() -> str:
    from owtn.prompts.stage_2.registry import load_champion_brief_scalar_system
    return load_champion_brief_scalar_system()


async def compute_stage_2_scalar_tree_brief(
    *,
    rollout_records: list[Any],
    cached_count: int | None,
    cached_brief: "LineageBrief | None",
    classifier_model: str,
    re_summarize_every: int = 5,
    force_resummarize: bool = False,
) -> tuple[str, "tuple[int, LineageBrief] | None"]:
    """Stage 2 scalar-mode entry point. Mirrors `compute_stage_2_champion_brief`:
    callers supply records + cache, get back rendered text + optional new
    cache payload.

    `rollout_records` is a list of `ScoredRolloutRecord` (from
    `owtn.stage_2.champion_brief`); typed as `list[Any]` here to keep this
    module from importing the stage_2 package (the stage_2 package depends
    on this one, not the reverse).

    Returns:
        (rendered_text, new_cache_or_None) where new_cache is `(count, brief)`
        when the summarizer fired, or None when the cache was fresh / the
        summarizer failed (raw fallback in `rendered_text`).

    Behavior mirrors `compute_stage_2_champion_brief`:
    - Cold start: seed placeholder, no LLM call.
    - Cache fresh: re-render from cached brief, no new cache payload.
    - Cache stale or `force_resummarize`: fire summarizer, return new
      render + cache.
    - Summarizer failure: raw fallback render, None cache.
    """
    if not rollout_records:
        return TREE_SUBJECT.seed_placeholder, None

    scored = [(r.score, r.reasoning, r.dag) for r in rollout_records]

    if not force_resummarize and cached_count is not None:
        delta = len(rollout_records) - cached_count
        if delta < re_summarize_every and cached_brief is not None:
            rendered = render_scalar_tree_brief(cached_brief, scored, TREE_SUBJECT)
            return rendered, None

    try:
        brief = await summarize_scalar_tree(
            scored_rollouts=scored,
            classifier_model=classifier_model,
            system_prompt=_stage_2_load_scalar_system_prompt(),
            format_dag=_stage_2_format_dag_for_match,
            subject=TREE_SUBJECT,
        )
    except Exception as e:  # noqa: BLE001 — never crash callers on summarizer failure
        logger.warning(
            "Stage 2 scalar tree-brief summarizer failed (%s); using raw fallback.", e,
        )
        return render_scalar_raw_fallback(scored, TREE_SUBJECT), None

    rendered = render_scalar_tree_brief(brief, scored, TREE_SUBJECT)
    return rendered, (len(rollout_records), brief)


# --- Stage 2 scalar lineage-brief adapter ---------------------------------
#
# Phase B of the scalar brief feedback loop. Scope is one MCTS path: only
# the rollouts whose DAG is a structural ancestor of the current expansion
# target. Same `LineageBrief` output shape as the tree adapter; different
# subject wording, different system prompt, and per-leaf cache keyed by the
# ancestor record count for that target.


def _stage_2_load_scalar_lineage_system_prompt() -> str:
    from owtn.prompts.stage_2.registry import load_champion_brief_scalar_lineage_system
    return load_champion_brief_scalar_lineage_system()


async def compute_stage_2_scalar_lineage_brief(
    *,
    lineage_records: list[Any],
    cached_count: int | None,
    cached_brief: "LineageBrief | None",
    classifier_model: str,
) -> tuple[str, "tuple[int, LineageBrief] | None"]:
    """Per-leaf lineage brief: same shape as the tree adapter but scoped to
    a single trajectory.

    Caller pre-filters `lineage_records` (via `lineage_records_for_target`)
    so this function doesn't need to know about the target DAG. Cache is
    one slot per target — caller is responsible for keying it on the
    target's identity.

    Caching behavior is simpler than the tree adapter's: there's no
    `re_summarize_every` cadence — the lineage is small and changes
    infrequently (only when a new beat is committed on this path), so we
    re-summarize whenever the ancestor count grew.

    Returns:
        (rendered_text, new_cache_or_None) — same convention as the tree
        adapter. None for new_cache means cache hit or summarizer failed.
    """
    if not lineage_records:
        return LINEAGE_PATH_SUBJECT.seed_placeholder, None

    scored = [(r.score, r.reasoning, r.dag) for r in lineage_records]

    if cached_count == len(lineage_records) and cached_brief is not None:
        rendered = render_scalar_tree_brief(cached_brief, scored, LINEAGE_PATH_SUBJECT)
        return rendered, None

    try:
        brief = await summarize_scalar_tree(
            scored_rollouts=scored,
            classifier_model=classifier_model,
            system_prompt=_stage_2_load_scalar_lineage_system_prompt(),
            format_dag=_stage_2_format_dag_for_match,
            subject=LINEAGE_PATH_SUBJECT,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Stage 2 scalar lineage-brief summarizer failed (%s); using raw fallback.", e,
        )
        return render_scalar_raw_fallback(scored, LINEAGE_PATH_SUBJECT), None

    rendered = render_scalar_tree_brief(brief, scored, LINEAGE_PATH_SUBJECT)
    return rendered, (len(lineage_records), brief)


# --- Stage 1 population adapter -------------------------------------------

_STAGE_1_POPULATION_PLACEHOLDERS = {
    "stage_name": "concept evolution (Stage 1)",
    "domain_hints": (
        "In this stage, attractors manifest structurally: as shared setting-"
        "types (the kind of world stakes unfold in), shared carrier forms "
        "(the document/record/artifact type the concept imitates), or shared "
        "question-shapes (what kind of question the concept invites readers "
        "to hold). Describe these as shapes, not by naming the specific "
        "content exemplar the run has converged on."
    ),
}


def _stage_1_population_system_prompt(
    judge_names: list[str], run_prompt: str = "",
) -> str:
    template = (_OPTIMIZER_DIR / "population_prompt.txt").read_text()
    placeholders = {
        **_STAGE_1_POPULATION_PLACEHOLDERS,
        "judge_names": ", ".join(judge_names) if judge_names else "(unknown panel)",
        "prompt_frame": _render_prompt_frame(run_prompt),
    }
    return _resolve_placeholders(template, placeholders)


def _stage_1_gather_lineage_briefs(db: Any) -> list[tuple[str, str, dict]]:
    """Read all programs in the run with a cached lineage brief.

    Returns a list of `(program_id, premise_summary, brief_dict)` triples.
    Programs without a cached brief (e.g. seeds that have never been in a
    match) are skipped — they carry no population signal.

    Concurrent-write safety: this runs on the main thread between
    generations; eval workers may have called `set_lineage_brief_threadsafe`
    on the same DB during the previous generation. SQLite is in WAL mode,
    so the read here sees a consistent snapshot — staler reads (a worker's
    in-flight write hasn't committed yet) just mean the next population
    brief picks up that lineage one cycle later, which is fine.
    """
    db.cursor.execute(
        "SELECT id, code, private_metrics FROM programs WHERE correct = 1"
    )
    rows = db.cursor.fetchall()

    entries: list[tuple[str, str, dict]] = []
    for row in rows:
        program_id = row["id"]
        try:
            pm = json.loads(row["private_metrics"] or "{}")
        except json.JSONDecodeError:
            continue
        cache = pm.get("lineage_brief_cache")
        if not cache or "brief" not in cache:
            continue
        try:
            genome = json.loads(row["code"] or "{}")
        except json.JSONDecodeError:
            genome = {}
        premise_summary = (genome.get("premise") or "")[:200]
        entries.append((program_id, premise_summary, cache["brief"]))
    return entries


def _format_stage_1_population_user_msg(
    entries: list[tuple[str, str, dict]],
) -> str:
    blocks = []
    for i, (program_id, premise, brief) in enumerate(entries):
        lines = [
            f"## Lineage {i + 1} — program {program_id[:8]}",
            f"Premise: {premise}",
            "",
            "Established weaknesses:",
            *(f"- {x}" for x in brief.get("established_weaknesses", [])),
            "",
            "Contested strengths:",
            *(f"- {x}" for x in brief.get("contested_strengths", [])),
            "",
            "Attractor signature:",
            *(f"- {x}" for x in brief.get("attractor_signature", [])),
            "",
            "Divergence directions:",
            *(f"- {x}" for x in brief.get("divergence_directions", [])),
        ]
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


async def compute_stage_1_population_brief(
    *,
    db: Any,
    run_brief_model: str,
    judge_names: list[str],
    run_prompt: str = "",
) -> tuple[str, str] | None:
    """Run the Stage 1 population summarizer end-to-end. Reads cached
    `LineageBrief`s from the DB, dispatches the call, and returns two
    markdown blocks `(population_context, exploration_directions)` placed
    at different positions in the mutation prompt by `build_operator_prompt`.

    `run_prompt` (`Stage1Config.prompt`) renders as a frame block; without
    it the summarizer reads prompt-required convergence as drift and produces
    exploration_directions that push mutators away from the user's prompt.

    Returns None when there are no lineage briefs yet or when the summarizer
    fails — callers treat that as "no population signal this generation."
    """
    entries = _stage_1_gather_lineage_briefs(db)
    if not entries:
        return None

    user_msg = _format_stage_1_population_user_msg(entries)
    system_prompt = _stage_1_population_system_prompt(judge_names, run_prompt)
    summarizer = PopulationBriefSummarizer(model_name=run_brief_model)
    try:
        brief = await summarizer.summarize(
            system_prompt=system_prompt, user_msg=user_msg,
        )
    except Exception as e:
        logger.warning("Population brief summarizer failed: %s", e)
        return None
    return render_population_context(brief), render_exploration_directions(brief)
