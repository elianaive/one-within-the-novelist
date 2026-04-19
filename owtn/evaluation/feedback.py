"""Lazy summarization of a parent concept's accumulated match critiques.

When a program is selected as a mutation parent, this module produces a
`ParentBrief` — a structured distillation of what reviewers have said about
the concept across every match it has been in. The brief replaces the old
per-match `text_feedback` truncation that was feeding comparative praise of
the champion back into the mutation prompt.

See `lab/issues/2026-04-18-lazy-feedback-summarizer.md`.

Design notes:
- Summarizer model comes from `LLMConfig.classifier_model` (previously dead
  config; finally wired up). Defaults: gpt-4.1-mini in light/dry_run,
  claude-haiku-4-5 in medium. Always a third family relative to the generator
  and judges.
- Cache: keyed on `len(match_critiques)`. Re-compute only when the count has
  grown since the last cached brief.
- Storage: `program.private_metrics["parent_brief_cache"] = {"count": N, "brief": {...}}`.
- Label disambiguation (A/B → YOU/OPPONENT) happens in the prompt, never via
  text substitution. The summarizer is told which label was the concept's own
  label per match.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from owtn.evaluation.models import ParentBrief
from owtn.llm.query import query_async

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "stage_1"

_SEED_PLACEHOLDER = "Initial concept — no prior evaluations yet."


def _load_summarizer_prompt() -> str:
    return (_PROMPTS_DIR / "parent_brief.txt").read_text()


def _format_dim_outcomes(dim_outcomes: dict[str, str]) -> str:
    won = sorted(d for d, v in dim_outcomes.items() if v == "won")
    lost = sorted(d for d, v in dim_outcomes.items() if v == "lost")
    tied = sorted(d for d, v in dim_outcomes.items() if v == "tied")
    parts = []
    if won:
        parts.append(f"won: {', '.join(won)}")
    if lost:
        parts.append(f"lost: {', '.join(lost)}")
    if tied:
        parts.append(f"tied: {', '.join(tied)}")
    return "; ".join(parts) if parts else "(no dimension-level breakdown)"


def _format_match_block(index: int, critique: dict, self_genome: dict) -> str:
    """Render a single match_critique for summarizer input.

    Lays out the concept's role in this match, the opponent, dimension
    outcomes, and every judge's verbatim reasoning. The A/B labels in the
    reasoning text are disambiguated via explicit header lines.
    """
    self_label = critique["self_label"]
    opponent_label = critique["opponent_label"]
    self_was_champion = critique.get("self_was_champion", False)
    outcome = critique["outcome"]
    opponent_genome = critique.get("opponent_genome", {})
    judge_reasonings = critique.get("judge_reasonings", [])

    role = "champion (defending)" if self_was_champion else "challenger"
    lines = [
        f"## Match {index + 1} — this concept was the {role}, {outcome}",
        "",
        (
            f"In this match, THIS CONCEPT was labeled '{self_label.upper()}' "
            f"and the opponent was labeled '{opponent_label.upper()}'."
        ),
        "",
        "THIS CONCEPT:",
        f"  premise: {self_genome.get('premise', '')[:400]}",
        f"  target_effect: {self_genome.get('target_effect', '')[:400]}",
        "",
        "OPPONENT:",
        f"  premise: {opponent_genome.get('premise', '')[:400]}",
        f"  target_effect: {opponent_genome.get('target_effect', '')[:400]}",
        "",
        (
            f"Dimension outcomes for THIS CONCEPT: "
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


def _build_summarizer_user_msg(
    self_genome: dict, match_critiques: list[dict]
) -> str:
    blocks = []
    for i, c in enumerate(match_critiques):
        blocks.append(_format_match_block(i, c, self_genome))
    return "\n\n---\n\n".join(blocks)


async def summarize_parent(
    *,
    self_genome: dict,
    match_critiques: list[dict],
    classifier_model: str,
) -> ParentBrief:
    """Run the lightweight summarizer to produce a ParentBrief.

    Raises: lets query_async exceptions propagate so callers can fall back.
    """
    if not match_critiques:
        raise ValueError("summarize_parent called with no match_critiques")

    system_msg = _load_summarizer_prompt()
    user_msg = _build_summarizer_user_msg(self_genome, match_critiques)

    result = await query_async(
        model_name=classifier_model,
        msg=user_msg,
        system_msg=system_msg,
        output_model=ParentBrief,
    )
    return result.content


def render_parent_brief(
    brief: ParentBrief,
    match_critiques: list[dict],
) -> str:
    """Markdown the mutation prompt will see in place of raw judge reasoning."""
    n = len(match_critiques)
    as_challenger = sum(
        1 for c in match_critiques if not c.get("self_was_champion", False)
    )
    as_defender = n - as_challenger

    def _bullets(items: list[str]) -> str:
        if not items:
            return "- (none identified)"
        return "\n".join(f"- {x}" for x in items)

    return (
        f"This concept has been evaluated in {n} match"
        f"{'es' if n != 1 else ''} "
        f"({as_challenger} as challenger, {as_defender} as defender).\n\n"
        "## Established weaknesses (reviewers' recurring critiques)\n"
        f"{_bullets(brief.established_weaknesses)}\n\n"
        "## Contested strengths (reviewers disagreed)\n"
        f"{_bullets(brief.contested_strengths)}\n\n"
        "## Attractor signature (patterns this concept exhibits)\n"
        f"{_bullets(brief.attractor_signature)}\n\n"
        "## Divergence directions (what a successor should try differently)\n"
        f"{_bullets(brief.divergence_directions)}"
    )


def render_raw_fallback(match_critiques: list[dict]) -> str:
    """Last-resort renderer when the summarizer fails entirely.

    Dumps the most recent 1-2 matches in a minimally-processed form so the
    mutator still has some signal. Loses structure but preserves content.
    """
    if not match_critiques:
        return _SEED_PLACEHOLDER
    recent = match_critiques[-2:]
    blocks = []
    for c in recent:
        outcome = c.get("outcome", "?")
        role = (
            "champion (defending)"
            if c.get("self_was_champion")
            else "challenger"
        )
        dim_summary = _format_dim_outcomes(c.get("dim_outcomes", {}))
        first_judge = (c.get("judge_reasonings") or [{}])[0]
        reasoning = (first_judge.get("reasoning") or "")[:1500]
        blocks.append(
            f"Prior match ({role}, {outcome}): {dim_summary}\n\n"
            f"Sample reasoning from judge {first_judge.get('judge_id', '?')}:\n"
            f"{reasoning}"
        )
    return "\n\n---\n\n".join(blocks)


def _cache_is_fresh(private_metrics: dict, current_count: int) -> bool:
    cache = private_metrics.get("parent_brief_cache") or {}
    return bool(cache) and cache.get("count") == current_count


async def get_or_compute_brief(
    *,
    self_genome: dict,
    private_metrics: dict,
    classifier_model: str,
) -> tuple[str, dict | None]:
    """Return (rendered_markdown, new_cache_payload_or_None).

    - If no match_critiques: returns (SEED_PLACEHOLDER, None).
    - If cache is fresh: returns cached render, None (nothing to persist).
    - If cache is stale or missing: recomputes, returns render + new cache
      payload (caller writes to DB).
    - If summarizer LLM fails: returns raw fallback render, None.
    """
    critiques = private_metrics.get("match_critiques") or []
    if not critiques:
        return _SEED_PLACEHOLDER, None

    if _cache_is_fresh(private_metrics, len(critiques)):
        cached = private_metrics["parent_brief_cache"]
        try:
            brief = ParentBrief.model_validate(cached["brief"])
            return render_parent_brief(brief, critiques), None
        except Exception:
            logger.warning("Corrupt parent_brief_cache; recomputing.")

    try:
        brief = await summarize_parent(
            self_genome=self_genome,
            match_critiques=critiques,
            classifier_model=classifier_model,
        )
    except Exception as e:
        logger.warning(
            "Summarizer failed (%s); falling back to raw critique render.", e
        )
        return render_raw_fallback(critiques), None

    payload = {"count": len(critiques), "brief": brief.model_dump()}
    return render_parent_brief(brief, critiques), payload
