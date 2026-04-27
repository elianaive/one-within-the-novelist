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
    _format_dim_outcomes,
    get_or_compute_brief,
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
    "output_schema_name": "LineageBrief",
}


def _load_lineage_prompt_template() -> str:
    return (_OPTIMIZER_DIR / "lineage_prompt.txt").read_text()


def _stage_1_lineage_system_prompt() -> str:
    prompt = _load_lineage_prompt_template()
    for key, value in _STAGE_1_LINEAGE_PLACEHOLDERS.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


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
) -> tuple[str, dict | None]:
    """Stage 1 entry point. Drop-in replacement for the old
    `owtn.evaluation.feedback.get_or_compute_brief` call — fills the Stage 1
    placeholders and supplies the concept-genome formatter.
    """
    return await get_or_compute_brief(
        self_genome=self_genome,
        private_metrics=private_metrics,
        classifier_model=classifier_model,
        system_prompt=_stage_1_lineage_system_prompt(),
        format_self=_stage_1_format_self,
    )


# --- Stage 2 champion brief adapter ---------------------------------------
#
# Stage 2 reuses the optimizer's machinery (LineageBrief schema, _format_dim_outcomes,
# render_lineage_brief, summarize_lineage) but supplies its own:
# - System prompt (`owtn/prompts/stage_2/champion_brief.txt`): structure-flavored,
#   names the tree as the subject of evaluation rather than a lineage.
# - format_self: renders DAGs via `owtn.stage_2.rendering.render`. Stage 1 just
#   formats premise/target_effect strings; Stage 2 needs full DAG renderings
#   so judges' reasoning makes sense in context.
# - Match-block wording: "THIS TREE" and "this tree's structure" instead of
#   "THIS LINEAGE" — the system prompt and match blocks must agree on the
#   subject term.
# - Cache shape: caller passes (count, LineageBrief) explicitly rather than
#   reading from a `private_metrics` dict, because Stage 2 has no DB-backed
#   per-program storage. The `TreeBriefState` object in `owtn.stage_2.champion_brief`
#   is what holds these fields per-tree.
#
# What Stage 2 does NOT reuse from Stage 1:
# - `_stage_1_format_self` (concept-specific)
# - The lineage_prompt.txt template (Stage-2 has structure-specific guidance
#   like "stay structural, do not evaluate sketches as if they were prose")
# - `optimizer.lineage_brief.summarize_lineage` itself (its match-block
#   formatter uses LINEAGE wording; ~10 lines of LLM-call boilerplate is
#   duplicated below to keep Stage-2 wording consistent end-to-end)

_STAGE_2_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "stage_2"
_STAGE_2_SEED_PLACEHOLDER = "Initial tree — no full-panel critiques accumulated yet."


def _stage_2_load_system_prompt() -> str:
    return (_STAGE_2_PROMPTS_DIR / "champion_brief.txt").read_text()


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


def _stage_2_format_match_block(index: int, critique: dict) -> str:
    """Tree-flavored match-block formatter. Parallel to
    `lineage_brief._format_match_block` but talks about THIS TREE / structures
    rather than THIS LINEAGE / genomes."""
    self_label = critique["self_label"]
    opponent_label = critique["opponent_label"]
    self_was_champion = critique.get("self_was_champion", False)
    outcome = critique["outcome"]
    self_dag = critique.get("self_dag", {})
    opponent_dag = critique.get("opponent_genome", {})
    judge_reasonings = critique.get("judge_reasonings", [])

    role = "champion (defending)" if self_was_champion else "challenger"
    lines = [
        f"## Match {index + 1} — this tree's structure was the {role}, {outcome}",
        "",
        (
            f"In this match, THIS TREE's structure was labeled '{self_label.upper()}' "
            f"and the opponent was labeled '{opponent_label.upper()}'."
        ),
        "",
        "THIS TREE's structure:",
        _stage_2_format_dag_for_match(self_dag),
        "",
        "OPPONENT structure:",
        _stage_2_format_dag_for_match(opponent_dag),
        "",
        (
            f"Dimension outcomes for THIS TREE: "
            f"{_format_dim_outcomes(critique.get('dim_outcomes', {}))}"
        ),
        "",
        "### Judge reasonings",
    ]
    for rec in judge_reasonings:
        jid = rec.get("judge_id", "?")
        harsh = rec.get("harshness", "?")
        reasoning = rec.get("reasoning", "")
        lines.append(f"\n#### Judge {jid} (harshness={harsh})")
        lines.append(reasoning)
    return "\n".join(lines)


def _stage_2_build_summarizer_user_msg(critiques: list[dict]) -> str:
    return "\n\n---\n\n".join(
        _stage_2_format_match_block(i, c) for i, c in enumerate(critiques)
    )


def stage_2_render_raw_fallback(critiques: list[dict]) -> str:
    """Last-resort render: cold start (zero events) returns the seed
    placeholder; summarizer failure returns the most recent 1-2 matches in
    raw form. Tree-flavored wording so the rendered text matches the brief's
    when both fire on the same tree."""
    if not critiques:
        return _STAGE_2_SEED_PLACEHOLDER
    recent = critiques[-2:]
    blocks = []
    for c in recent:
        outcome = c.get("outcome", "?")
        role = (
            "champion (defending)"
            if c.get("self_was_champion") else "challenger"
        )
        dim_summary = _format_dim_outcomes(c.get("dim_outcomes", {}))
        first_judge = (c.get("judge_reasonings") or [{}])[0]
        reasoning = (first_judge.get("reasoning") or "")[:1500]
        blocks.append(
            f"Prior match (this tree's structure was {role}, {outcome}): "
            f"{dim_summary}\n\n"
            f"Sample reasoning from judge {first_judge.get('judge_id', '?')}:\n"
            f"{reasoning}"
        )
    return "\n\n---\n\n".join(blocks)


async def _stage_2_call_summarizer(
    *,
    user_msg: str,
    classifier_model: str,
) -> "LineageBrief":  # forward-string ref so we don't need to import at top
    """One LLM call → LineageBrief. ~10 lines of `query_async` boilerplate
    duplicated from `optimizer.lineage_brief.summarize_lineage` so we can
    use Stage 2's tree-flavored user message + system prompt instead of
    its LINEAGE-flavored ones. Schema reused directly."""
    from owtn.llm.call_logger import llm_context  # local imports keep test mocks targeted
    from owtn.llm.query import query_async
    from owtn.optimizer.models import LineageBrief

    base_ctx = dict(llm_context.get({}))
    llm_context.set({**base_ctx, "role": "stage_2_brief_summarizer"})
    result = await query_async(
        model_name=classifier_model,
        msg=user_msg,
        system_msg=_stage_2_load_system_prompt(),
        output_model=LineageBrief,
    )
    parsed = result.content
    if not isinstance(parsed, LineageBrief):
        raise RuntimeError(
            f"summarizer returned unexpected content type: {type(parsed).__name__}"
        )
    return parsed


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
    from owtn.optimizer.lineage_brief import render_lineage_brief
    from owtn.optimizer.models import LineageBrief  # noqa: F401 — used in type comments

    if not full_panel_critiques:
        return _STAGE_2_SEED_PLACEHOLDER, None

    if not force_resummarize and cached_count is not None:
        delta = len(full_panel_critiques) - cached_count
        if delta < re_summarize_every and cached_brief is not None:
            # Cache fresh — render from cached brief, signal no-update.
            rendered = _stage_2_render_brief(cached_brief, full_panel_critiques)
            return rendered, None

    # Recompute.
    try:
        brief = await _stage_2_call_summarizer(
            user_msg=_stage_2_build_summarizer_user_msg(full_panel_critiques),
            classifier_model=classifier_model,
        )
    except Exception as e:  # noqa: BLE001 — never crash callers on summarizer failure
        logger.warning(
            "Stage 2 champion brief summarizer failed (%s); using raw fallback.", e,
        )
        return stage_2_render_raw_fallback(full_panel_critiques), None

    rendered = _stage_2_render_brief(brief, full_panel_critiques)
    return rendered, (len(full_panel_critiques), brief)


def _stage_2_render_brief(brief, critiques: list[dict]) -> str:
    """Render a LineageBrief with tree-flavored wording. Reuses the
    optimizer's `render_lineage_brief` and patches "lineage" → "tree" in
    the rendered string — keeps render logic in one place rather than
    forking the renderer."""
    from owtn.optimizer.lineage_brief import render_lineage_brief

    rendered = render_lineage_brief(brief, critiques)
    rendered = rendered.replace("This lineage", "This tree")
    rendered = rendered.replace("this lineage", "this tree")
    return rendered


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


def _stage_1_population_system_prompt(judge_names: list[str]) -> str:
    prompt = (_OPTIMIZER_DIR / "population_prompt.txt").read_text()
    placeholders = {
        **_STAGE_1_POPULATION_PLACEHOLDERS,
        "judge_names": ", ".join(judge_names) if judge_names else "(unknown panel)",
    }
    for key, value in placeholders.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


def _stage_1_gather_lineage_briefs(db: Any) -> list[tuple[str, str, dict]]:
    """Read all programs in the run with a cached lineage brief.

    Returns a list of `(program_id, premise_summary, brief_dict)` triples.
    Programs without a cached brief (e.g. seeds that have never been in a
    match) are skipped — they carry no population signal.
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
) -> tuple[str, str] | None:
    """Run the Stage 1 population summarizer end-to-end.

    Reads all programs' cached `LineageBrief`s from the DB, formats them as
    the summarizer input, fills the Stage 1 placeholders on the generic
    prompt, dispatches the call, and returns two rendered markdown blocks:
    `(population_context, exploration_directions)`. The two blocks are placed
    at different positions in the mutation prompt — see the instruction-
    sandwich wiring in `build_operator_prompt`.

    Returns None if there are no lineage briefs yet (early run) or if the
    summarizer call fails — callers should treat that as "no population
    signal this generation."
    """
    entries = _stage_1_gather_lineage_briefs(db)
    if not entries:
        return None

    user_msg = _format_stage_1_population_user_msg(entries)
    system_prompt = _stage_1_population_system_prompt(judge_names)
    summarizer = PopulationBriefSummarizer(model_name=run_brief_model)
    try:
        brief = await summarizer.summarize(
            system_prompt=system_prompt, user_msg=user_msg,
        )
    except Exception as e:
        logger.warning("Population brief summarizer failed: %s", e)
        return None
    return render_population_context(brief), render_exploration_directions(brief)
