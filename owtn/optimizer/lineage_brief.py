"""Lazy summarization of one lineage's accumulated match critiques.

At parent-selection time, all of a program's `match_critiques` are fed to
this module, which produces a `LineageBrief` — a structured distillation of
what reviewers have said about the lineage across every match. Formerly
`owtn/evaluation/feedback.py`'s `ParentBrief`; stage-agnostic now, with
Stage-1-specific genome formatting lifted into `owtn/optimizer/adapters.py`.

See `lab/issues/2026-04-24-refactor-feedback-to-optimizer-module.md` and
`lab/issues/closed/2026-04-18-lazy-feedback-summarizer.md`.

Design notes:
- Summarizer model comes from `LLMConfig.classifier_model`. Defaults:
  gpt-4.1-mini (light/dry_run), claude-haiku-4-5 (medium). Always a third
  family relative to the generator and judges.
- Cache: keyed on `len(match_critiques)`. Re-compute only when the count has
  grown since the last cached brief.
- Storage: `program.private_metrics["lineage_brief_cache"] = {"count": N, "brief": {...}}`.
- Label disambiguation (A/B → THIS LINEAGE / OPPONENT) happens in the
  prompt, never via text substitution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from owtn.llm.query import query_async
from owtn.optimizer.models import LineageBrief

logger = logging.getLogger(__name__)


# Match a Setup line: 'FP1 = "..." (corresponds to concept labeled A)'.
# Tolerant of whitespace and smart quotes.
_FP_SETUP_RE = re.compile(
    r'FP([12])\s*=\s*["“][^"”]*["”]\s*\(\s*corresponds\s+to\s+concept\s+labeled\s+([AB])',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BriefSubject:
    """Stage-specific wording for the brief's subject.

    Stage 1's subject is "the lineage" (the parent genome and its descendants);
    Stage 2's subject is "the tree" (the MCTS tree, whose self-DAG changes
    across iterations). Same brief schema, same render skeleton, different
    nouns at the seams.
    """
    upper: str          # capitalized form: "THIS LINEAGE" / "THIS TREE"
    narrative: str      # lowercase form: "this lineage" / "this tree"
    block_qualifier: str = ""  # appended to upper when naming the artifact:
                               # "" for Stage 1 (lineage IS the genome);
                               # "'s structure" for Stage 2 (DAG is the
                               # structure of the tree, not the tree itself).
    seed_placeholder: str = "Initial — no prior evaluations yet."


LINEAGE_SUBJECT = BriefSubject(
    upper="THIS LINEAGE",
    narrative="this lineage",
    block_qualifier="",
    seed_placeholder="Initial lineage — no prior evaluations yet.",
)


def _resolve_fp_to_subject(reasoning: str, self_label: str, subject_upper: str) -> str:
    """Substitute FP1/FP2 with the subject's uppercase form / OPPONENT.

    Empirically required: gpt-4.1-mini summarizer fails to chain Setup line
    → label → subject attribution and misattributes opponent traits to the
    subject (observed in early Stage 1 runs). Pre-substituting FP1/FP2 at
    render time replaces the chain with a direct lookup the summarizer
    can't get wrong.

    If the Setup line is missing or malformed, leaves the reasoning
    unchanged so the summarizer can still try the prompt-level fallback.
    """
    fp_to_label: dict[str, str] = {}
    for m in _FP_SETUP_RE.finditer(reasoning):
        fp_to_label[f"FP{m.group(1)}"] = m.group(2).lower()
        if len(fp_to_label) == 2:
            break
    if "FP1" not in fp_to_label or "FP2" not in fp_to_label:
        return reasoning
    self_lower = self_label.lower()
    fp_to_role = {
        fp: (subject_upper if lbl == self_lower else "OPPONENT")
        for fp, lbl in fp_to_label.items()
    }
    out = reasoning
    for fp, role in fp_to_role.items():
        out = re.sub(rf"\b{fp}\b", role, out)
    return out


# A SelfFormatter turns a genome/DAG dict into a short text block that
# appears inside a match header. Stage-specific — supplied by the per-stage
# adapter.
SelfFormatter = Callable[[dict], str]

# A SelfExtractor pulls the subject's own artifact (genome/DAG) from a
# critique dict. Stage 1: ignores the critique and returns a fixed lineage
# genome closed over at call time. Stage 2: reads `critique["self_dag"]`
# (the tree's iteration-specific DAG, which changes across critiques).
SelfExtractor = Callable[[dict], dict]


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


def _format_match_block(
    index: int,
    critique: dict,
    extract_self: SelfExtractor,
    format_self: SelfFormatter,
    subject: BriefSubject,
) -> str:
    """Render a single match_critique for summarizer input.

    Lays out the subject's role in this match, the opponent, dimension
    outcomes, and every judge's verbatim reasoning. The A/B labels in the
    reasoning text are disambiguated via explicit header lines.
    """
    self_label = critique["self_label"]
    opponent_label = critique["opponent_label"]
    self_was_champion = critique.get("self_was_champion", False)
    outcome = critique["outcome"]
    self_data = extract_self(critique)
    opponent_genome = critique.get("opponent_genome", {})
    judge_reasonings = critique.get("judge_reasonings", [])

    role = "champion (defending)" if self_was_champion else "challenger"
    subject_block_label = f"{subject.upper}{subject.block_qualifier}"
    lines = [
        f"## Match {index + 1} — {subject.narrative}{subject.block_qualifier} was the {role}, {outcome}",
        "",
        (
            f"In this match, {subject_block_label} was labeled "
            f"'{self_label.upper()}' "
            f"and the opponent was labeled '{opponent_label.upper()}'."
        ),
        "",
        f"{subject_block_label}:",
        format_self(self_data),
        "",
        "OPPONENT:",
        format_self(opponent_genome),
        "",
        (
            f"Dimension outcomes for {subject.upper}: "
            f"{_format_dim_outcomes(critique.get('dim_outcomes', {}))}"
        ),
        "",
        "### Judge reasonings",
    ]
    for rec in judge_reasonings:
        jid = rec.get("judge_id", "?")
        harsh = rec.get("harshness", "?")
        reasoning = _resolve_fp_to_subject(
            rec.get("reasoning", ""), self_label, subject.upper
        )
        lines.append(f"\n#### Judge {jid} (harshness={harsh})")
        lines.append(reasoning)
    return "\n".join(lines)


def _build_summarizer_user_msg(
    match_critiques: list[dict],
    extract_self: SelfExtractor,
    format_self: SelfFormatter,
    subject: BriefSubject,
) -> str:
    blocks = []
    for i, c in enumerate(match_critiques):
        blocks.append(_format_match_block(i, c, extract_self, format_self, subject))
    return "\n\n---\n\n".join(blocks)


async def summarize_lineage(
    *,
    match_critiques: list[dict],
    classifier_model: str,
    system_prompt: str,
    extract_self: SelfExtractor,
    format_self: SelfFormatter,
    subject: BriefSubject = LINEAGE_SUBJECT,
) -> LineageBrief:
    """Run the lightweight summarizer to produce a LineageBrief.

    Raises: lets query_async exceptions propagate so callers can fall back.
    """
    if not match_critiques:
        raise ValueError("summarize_lineage called with no match_critiques")

    user_msg = _build_summarizer_user_msg(
        match_critiques, extract_self, format_self, subject
    )

    result = await query_async(
        model_name=classifier_model,
        msg=user_msg,
        system_msg=system_prompt,
        output_model=LineageBrief,
    )
    parsed = result.content
    if not isinstance(parsed, LineageBrief):
        # Defensive: provider-wrapper drift could give us a raw string
        # despite output_model=LineageBrief. Raise so caller falls back to
        # the raw critique render rather than silently passing junk along.
        raise RuntimeError(
            f"summarizer returned unexpected content type: {type(parsed).__name__}"
        )
    return parsed


def render_lineage_brief(
    brief: LineageBrief,
    match_critiques: list[dict],
    subject: BriefSubject = LINEAGE_SUBJECT,
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

    # Capitalize narrative form for the opening sentence ("This lineage..."
    # / "This tree...").
    cap_narrative = subject.narrative[0].upper() + subject.narrative[1:]
    return (
        f"{cap_narrative} has been evaluated in {n} match"
        f"{'es' if n != 1 else ''} "
        f"({as_challenger} as challenger, {as_defender} as defender).\n\n"
        "## Established weaknesses (reviewers' recurring critiques)\n"
        f"{_bullets(brief.established_weaknesses)}\n\n"
        "## Contested strengths (reviewers disagreed)\n"
        f"{_bullets(brief.contested_strengths)}\n\n"
        f"## Attractor signature (patterns {subject.narrative} exhibits)\n"
        f"{_bullets(brief.attractor_signature)}\n\n"
        "## Divergence directions (what a successor should try differently)\n"
        f"{_bullets(brief.divergence_directions)}"
    )


def render_raw_fallback(
    match_critiques: list[dict],
    subject: BriefSubject = LINEAGE_SUBJECT,
) -> str:
    """Last-resort renderer when the summarizer fails entirely.

    Dumps the most recent 1-2 matches in a minimally-processed form so the
    mutator still has some signal. Loses structure but preserves content.
    """
    if not match_critiques:
        return subject.seed_placeholder
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
    cache = private_metrics.get("lineage_brief_cache") or {}
    return bool(cache) and cache.get("count") == current_count


async def get_or_compute_brief(
    *,
    self_genome: dict,
    private_metrics: dict,
    classifier_model: str,
    system_prompt: str,
    format_self: SelfFormatter,
    subject: BriefSubject = LINEAGE_SUBJECT,
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
        return subject.seed_placeholder, None

    extract_self: SelfExtractor = lambda _critique: self_genome  # noqa: E731

    if _cache_is_fresh(private_metrics, len(critiques)):
        cached = private_metrics["lineage_brief_cache"]
        try:
            brief = LineageBrief.model_validate(cached["brief"])
            return render_lineage_brief(brief, critiques, subject), None
        except Exception:
            logger.warning("Corrupt lineage_brief_cache; recomputing.")

    try:
        brief = await summarize_lineage(
            match_critiques=critiques,
            classifier_model=classifier_model,
            system_prompt=system_prompt,
            extract_self=extract_self,
            format_self=format_self,
            subject=subject,
        )
    except Exception as e:
        logger.warning(
            "Summarizer failed (%s); falling back to raw critique render.", e
        )
        return render_raw_fallback(critiques, subject), None

    payload = {"count": len(critiques), "brief": brief.model_dump()}
    return render_lineage_brief(brief, critiques, subject), payload
