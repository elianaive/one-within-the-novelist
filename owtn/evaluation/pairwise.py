"""Per-criteria pairwise comparison of two concept genomes.

Each judge compares on all 9 dimensions independently and marks each
dim-vote with a magnitude (narrow/clear/decisive). Each judge sees both
orderings (A-first, B-first) to mitigate position bias; a dim vote counts
only if both orderings agree on side, and the resolved magnitude is the
min of the two. Overall winner = weighted sum of (dim_weight × mean
magnitude on winning side) with asymmetric tiebreaker on close contests.
See `lab/issues/2026-04-22-pairwise-win-margin.md`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter

from datetime import datetime, timezone

from owtn.evaluation.models import (
    DIMENSION_NAMES,
    JudgeReasoningRecord,
    MatchCritique,
    PairwiseJudgment,
    PairwiseResult,
    encode_vote,
    parse_vote,
)
from owtn.evaluation.prompts import build_pairwise_system, build_pairwise_user
from owtn.llm.call_logger import llm_context
from owtn.llm.providers import PROVIDERS
from owtn.llm.providers.model_resolver import resolve_model_backend
from owtn.llm.query import query_async
from owtn.models.judge import JudgePersona, load_panel
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_1.config import StageConfig

logger = logging.getLogger(__name__)

# Verbose reasoning judges (qwen3.5-27b, GLM-4.7) routinely exhausted the
# previous 6144-token cap — Stage 1 logs show ~85% truncation rate for the
# wales (qwen) judge. The budget covers BOTH invisible reasoning tokens
# (low-effort ≈ 2k for these models) AND visible output (~4-5k for the
# 8-dim per-sub-criterion structured analysis). 12288 gives the reasoning-
# spiral room to land plus a safety buffer; non-verbose models pay nothing
# extra because this is a ceiling, not a target.
_JUDGE_MAX_OUTPUT_TOKENS = 12288

# Min mean-magnitude differential to break a tied-count dim (e.g. 2-2 on a
# 4-judge panel). Matches the smallest gap between adjacent magnitude buckets
# {0.5 narrow, 0.75 clear, 1.0 decisive}, so any single-bucket shift between
# sides breaks the tie; same-bucket splits (e.g. 2 clear + 2 clear) stay tied.
# Only applies when a_count == b_count > 0 (genuine split with magnitude on
# both sides); 0-0-all-tie cases are unchanged. See
# lab/issues/2026-04-24-aggregate-magnitude-tiebreaker.md.
_MAGNITUDE_TIEBREAKER_THRESHOLD = 0.25


def _build_judge_kwargs(judge: JudgePersona) -> tuple[str, dict]:
    """Return ``(model_name, kwargs)`` for a judge call.

    Delegates to the model's Provider for the heavy lifting (per-provider
    rewriting of reasoning effort, temperature forcing under
    think_temp_fixed, top_p/k filtering, OpenRouter extra_body workaround,
    DeepSeek thinking toggle). Judge-specific behavior is just two things:

    - `max_tokens`: only set for OpenAI/OpenRouter/Azure judges. Non-reasoning
      OpenRouter upstreams (e.g. kimi-k2-0905 via :nitro) were observed
      truncating mid-JSON — this is a ceiling override. Other providers
      have generous defaults; we omit max_tokens there.

    Note: don't re-add `extra_body.provider.sort=throughput` with
    `require_parameters=true` to curb OpenRouter routing variance — that
    combination forces Kimi onto a backend that leaks `[EOS]` markers.
    """
    model_name = judge.model[0]
    resolved = resolve_model_backend(model_name)
    provider = PROVIDERS[resolved.provider]

    requested: dict = {
        "reasoning_effort": judge.reasoning_effort,
        "thinking_tokens": judge.thinking_tokens,
        "temperature": judge.temperature,
        "top_p": judge.top_p,
        "top_k": judge.top_k,
        "min_p": judge.min_p,
    }
    if resolved.provider in ("openai", "openrouter", "azure_openai"):
        requested["max_tokens"] = _JUDGE_MAX_OUTPUT_TOKENS

    kwargs = provider.build_call_kwargs(api_model=resolved.api_model_name, requested=requested)
    return model_name, kwargs


async def _judge_one_ordering(
    judge: JudgePersona,
    genome_a: ConceptGenome,
    genome_b: ConceptGenome,
) -> tuple[PairwiseJudgment, float]:
    """Run one judge on one ordering. Returns (judgment, cost)."""
    # Set inside the coroutine so each task writes its own judge_id into its
    # own Task-local context. Setting in the caller before asyncio.gather
    # makes every task see the last loop iteration's value.
    llm_context.set({"role": "pairwise_judge", "judge_id": judge.id})
    system_msg = build_pairwise_system(judge)
    user_msg = build_pairwise_user(genome_a, genome_b)
    model_name, judge_kwargs = _build_judge_kwargs(judge)

    # The judge's entire system message is stable across all comparisons by
    # this judge — pass it as system_prefix so Anthropic caches it (and
    # OpenAI/DeepSeek auto-match the prefix for their own discounts).
    result = await query_async(
        model_name=model_name,
        msg=user_msg,
        system_msg="",
        system_prefix=system_msg,
        output_model=PairwiseJudgment,
        **judge_kwargs,
    )
    return result.content, result.cost


def _flip_votes(judgment: PairwiseJudgment) -> dict[str, str]:
    """Flip a/b in votes (for the reversed ordering). Magnitude and tie are
    preserved — a judge who votes 'a_decisive' when A is labeled A should
    flip to 'b_decisive' when A is labeled B."""
    flipped = {}
    for dim, vote in judgment.votes().items():
        if vote == "tie":
            flipped[dim] = "tie"
        elif vote.startswith("a_"):
            flipped[dim] = "b_" + vote.split("_", 1)[1]
        else:  # b_*
            flipped[dim] = "a_" + vote.split("_", 1)[1]
    return flipped


def _classify_resolution(fwd: str, rev: str) -> str:
    """Label the (fwd, rev) combination for diagnostic logging.

    - confident-{a,b}: both orderings agree on side AND magnitude
    - agreed-{a,b}-min-{narrow,clear,decisive}: both orderings agree on side
      but magnitudes differ — resolved magnitude is the min (the judge
      compressed their own confidence between orderings)
    - confident-tie: judge declared tie in both orderings
    - resolution-tie: orderings disagree on winner (position bias caught)
    - soft-tie: one ordering picked a winner, the other declared tie
    """
    fwd_side, fwd_mag = parse_vote(fwd)
    rev_side, rev_mag = parse_vote(rev)
    if fwd_side == rev_side:
        if fwd_side == "tie":
            return "confident-tie"
        if fwd_mag == rev_mag:
            return f"confident-{fwd_side}"
        min_label = encode_vote(fwd_side, min(fwd_mag, rev_mag)).split("_", 1)[1]
        return f"agreed-{fwd_side}-min-{min_label}"
    if fwd_side == "tie" or rev_side == "tie":
        return "soft-tie"
    return "resolution-tie"


def _resolve_votes(
    forward_votes: dict[str, str],
    reverse_votes: dict[str, str],
) -> dict[str, str]:
    """Resolve per-dimension votes across two orderings.

    Conservative dual-ordering extended to magnitudes: a side wins only if
    both orderings agree on side, and the resolved magnitude is the min of
    the two ordering magnitudes. A judge who waffles between 'decisive' and
    'narrow' is expressing less confidence than either ordering alone; we
    record the minimum. Side-disagreement or either-ordering tie resolves
    to 'tie'. See `lab/issues/2026-04-22-pairwise-win-margin.md` and
    `lab/issues/2026-04-18-reintroduce-harshness-pairwise.md`.
    """
    resolved = {}
    for dim in DIMENSION_NAMES:
        fwd = forward_votes.get(dim, "tie")
        rev = reverse_votes.get(dim, "tie")
        fwd_side, fwd_mag = parse_vote(fwd)
        rev_side, rev_mag = parse_vote(rev)

        if fwd_side == rev_side and fwd_side != "tie":
            resolved[dim] = encode_vote(fwd_side, min(fwd_mag, rev_mag))
        else:
            # Either both tie, or sides disagree, or one side tied.
            resolved[dim] = "tie"
    return resolved


def _classify_votes(
    forward_votes: dict[str, str],
    reverse_votes: dict[str, str],
) -> dict[str, str]:
    """Per-dimension classification of the (fwd, rev) combination."""
    return {
        dim: _classify_resolution(
            forward_votes.get(dim, "tie"),
            reverse_votes.get(dim, "tie"),
        )
        for dim in DIMENSION_NAMES
    }


def _aggregate(
    all_resolved: list[dict[str, str]],
    dim_weights: dict[str, float],
) -> tuple[dict[str, str], int, int, int, float, float, float]:
    """Aggregate resolved votes across all judges.

    Per dimension: majority of non-tie votes picks the winning side (magnitude
    does not affect side majority — one narrow win still counts as one win).
    Weighted contribution uses the MEAN magnitude among judges voting the
    winning side, multiplied by the dim weight. Mean (not sum) keeps each
    dim's contribution bounded by its weight regardless of panel size:
    "weight × judge confidence" is the headline interpretation.

    **Tied-count magnitude tiebreaker.** When `a_count == b_count > 0` (e.g.
    a 2-2 split on a 4-judge panel), the dim is won by whichever side has
    the higher mean magnitude if the gap is ≥ `_MAGNITUDE_TIEBREAKER_THRESHOLD`.
    Otherwise the dim stays tied. 0-0 all-tie cases (no magnitude signal) are
    unchanged. See `lab/issues/2026-04-24-aggregate-magnitude-tiebreaker.md`.

    Integer counts (`a_wins`, `b_wins`, `ties`) are preserved for display
    and match-history; weighted totals drive winner selection via
    `_select_winner`.

    Returns: (dim_winners, a_wins, b_wins, ties, a_weighted, b_weighted, tie_weighted)
    """
    dim_winners = {}
    a_wins = b_wins = tie_count = 0
    a_weighted = b_weighted = tie_weighted = 0.0

    for dim in DIMENSION_NAMES:
        votes = [r[dim] for r in all_resolved]
        parsed = [parse_vote(v) for v in votes]
        sides = [s for s, _ in parsed]
        counts = Counter(sides)
        weight = dim_weights[dim]
        a_count = counts.get("a", 0)
        b_count = counts.get("b", 0)

        if a_count > b_count:
            dim_winners[dim] = "a"
            a_wins += 1
            a_mags = [m for s, m in parsed if s == "a"]
            a_weighted += weight * (sum(a_mags) / len(a_mags))
        elif b_count > a_count:
            dim_winners[dim] = "b"
            b_wins += 1
            b_mags = [m for s, m in parsed if s == "b"]
            b_weighted += weight * (sum(b_mags) / len(b_mags))
        elif a_count > 0:
            # Tied count, magnitude signal on both sides (e.g. 2-2 on 4 judges).
            a_mags = [m for s, m in parsed if s == "a"]
            b_mags = [m for s, m in parsed if s == "b"]
            a_mean = sum(a_mags) / len(a_mags)
            b_mean = sum(b_mags) / len(b_mags)
            if a_mean - b_mean >= _MAGNITUDE_TIEBREAKER_THRESHOLD:
                dim_winners[dim] = "a"
                a_wins += 1
                a_weighted += weight * a_mean
            elif b_mean - a_mean >= _MAGNITUDE_TIEBREAKER_THRESHOLD:
                dim_winners[dim] = "b"
                b_wins += 1
                b_weighted += weight * b_mean
            else:
                dim_winners[dim] = "tie"
                tie_count += 1
                tie_weighted += weight
        else:
            # a_count == b_count == 0: all judges voted tie, no magnitude
            # signal to break with.
            dim_winners[dim] = "tie"
            tie_count += 1
            tie_weighted += weight

    return dim_winners, a_wins, b_wins, tie_count, a_weighted, b_weighted, tie_weighted


# Short-label map for log summaries: maps magnitude value → compact suffix
# (1=narrow, 2=clear, 3=decisive) so "indelibility=A3" reads in a log line.
_MAG_SHORT_LABEL: dict[float, str] = {0.5: "1", 0.75: "2", 1.0: "3"}


def _format_dim_summary(
    dim_winners: dict[str, str],
    all_resolved: list[dict[str, str]],
) -> str:
    """Per-dim log summary with magnitude suffix on winning sides.

    Format: "indelibility=A3  grip=B1  novelty==". A/B suffixes carry the
    winning side's mean-magnitude bucket (1=narrow, 2=clear, 3=decisive);
    '==' marks ties. Magnitudes are bucketed to the nearest canonical value
    so mean values (e.g. a 3-judge panel with narrow/clear/decisive on one
    dim has mean 0.75 → "2") stay readable.
    """
    parts = []
    for dim, winner in dim_winners.items():
        if winner == "tie":
            parts.append(f"{dim}==")
            continue
        mags = [
            parse_vote(r[dim])[1]
            for r in all_resolved
            if parse_vote(r[dim])[0] == winner
        ]
        if not mags:
            parts.append(f"{dim}={winner.upper()}?")
            continue
        mean_mag = sum(mags) / len(mags)
        bucket = min(_MAG_SHORT_LABEL, key=lambda v: abs(v - mean_mag))
        parts.append(f"{dim}={winner.upper()}{_MAG_SHORT_LABEL[bucket]}")
    return "  ".join(parts)


def _select_winner(
    a_weighted: float,
    b_weighted: float,
    dim_winners: dict[str, str],
    champion_label: str,
    tiebreaker_threshold: float,
    tiebreaker_dims: list[str],
) -> tuple[str, str | None]:
    """Apply Option E: weighted aggregate + asymmetric tiebreaker.

    When |a_weighted - b_weighted| > tiebreaker_threshold, whoever has the
    higher weighted total wins (tiebreaker_used=None).

    Otherwise (close contest): walk tiebreaker_dims in order; first dim whose
    winner is 'a' or 'b' wins. If all tiebreaker dims are tied, champion wins
    (incumbent advantage).

    Using `<=` for threshold activation: we'd rather over-invoke the tiebreaker
    than under-invoke it, since the tiebreaker reduces to incumbent when
    top dims are tied.

    Returns: (winner_label, tiebreaker_used).
    """
    gap = abs(a_weighted - b_weighted)
    if gap > tiebreaker_threshold:
        return ("a" if a_weighted > b_weighted else "b"), None

    for dim in tiebreaker_dims:
        w = dim_winners.get(dim)
        if w in ("a", "b"):
            return w, dim
    return champion_label, "incumbent"


def _invert_outcome(outcome: str) -> str:
    """Flip a per-concept outcome from one side to the other."""
    if outcome == "won":
        return "lost"
    if outcome == "lost":
        return "won"
    return "tied"


def _build_match_critiques(
    *,
    genome_a: ConceptGenome,
    genome_b: ConceptGenome,
    champion_label: str,
    winner: str,
    dim_winners: dict[str, str],
    panel: list[JudgePersona],
    fwd_judgments: list[PairwiseJudgment],
) -> dict[str, MatchCritique]:
    """Build two per-concept critique records (keyed by 'a' / 'b') for this match.

    Reasoning text is stored verbatim (references concepts as 'A' / 'B').
    Label disambiguation is the summarizer's job at prompt time; we never
    text-substitute into judge reasoning here.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Forward-ordering reasonings from each judge that did not error.
    reasonings = [
        JudgeReasoningRecord(
            judge_id=judge.id,
            harshness=judge.harshness,
            reasoning=judgment.reasoning,
        )
        for judge, judgment in zip(panel, fwd_judgments)
    ]

    # Outcome from A's perspective.
    a_outcome: str
    if winner == "a":
        a_outcome = "won"
    elif winner == "b":
        a_outcome = "lost"
    else:
        a_outcome = "tied"
    b_outcome = _invert_outcome(a_outcome)

    # Per-dimension outcomes from A's perspective (resolved "a" → won, etc.).
    def _dim_outcomes_from(label: str) -> dict[str, str]:
        result = {}
        for dim, w in dim_winners.items():
            if w == "tie":
                result[dim] = "tied"
            elif w == label:
                result[dim] = "won"
            else:
                result[dim] = "lost"
        return result

    critique_a = MatchCritique(
        self_label="a",
        opponent_label="b",
        self_was_champion=(champion_label == "a"),
        opponent_genome=genome_b.model_dump(),
        outcome=a_outcome,
        dim_outcomes=_dim_outcomes_from("a"),
        judge_reasonings=reasonings,
        timestamp=timestamp,
    )
    critique_b = MatchCritique(
        self_label="b",
        opponent_label="a",
        self_was_champion=(champion_label == "b"),
        opponent_genome=genome_a.model_dump(),
        outcome=b_outcome,
        dim_outcomes=_dim_outcomes_from("b"),
        judge_reasonings=reasonings,
        timestamp=timestamp,
    )
    return {"a": critique_a, "b": critique_b}


def _format_feedback(
    dim_winners: dict[str, str],
    a_wins: int,
    b_wins: int,
    winner: str,
    reasoning_texts: list[str],
    champion_label: str,
) -> str:
    """Format pairwise result as mutation feedback.

    The mutation model sees which dimensions the champion won and lost,
    with judge reasoning for each. Header uses actual `winner` (not the
    integer dim-count) — under weighted selection, dim-counts can disagree
    with the real winner and the mutation model must not receive
    contradictory signals.

    champion_label indicates which label ('a' or 'b') is the incumbent
    champion. The challenger is the other label. Feedback is phrased from
    the challenger's perspective.
    """
    challenger_label = "b" if champion_label == "a" else "a"
    loser_label = "b" if champion_label == "a" else "a"

    won = [d for d, w in dim_winners.items() if w == champion_label]
    lost = [d for d, w in dim_winners.items() if w == loser_label]
    tied = [d for d, w in dim_winners.items() if w == "tie"]

    challenger_won = winner == challenger_label
    # Dim-count display shows challenger's count first (a_wins is challenger's
    # integer dim-wins in the standard runner call site).
    lines = [
        f"Pairwise result: {'Won' if challenger_won else 'Lost'} "
        f"({a_wins}-{b_wins})"
    ]
    if lost:
        lines.append(f"\nDimensions to improve (lost to champion): {', '.join(lost)}")
    if won:
        lines.append(f"Dimensions to preserve (beat champion): {', '.join(won)}")
    if tied:
        lines.append(f"Tied dimensions: {', '.join(tied)}")

    # Include first judge's reasoning as detailed feedback
    if reasoning_texts:
        lines.append(f"\nJudge reasoning:\n{reasoning_texts[0]}")

    return "\n".join(lines)


async def compare(
    genome_a: ConceptGenome,
    genome_b: ConceptGenome,
    config: StageConfig,
    champion_label: str = "a",
) -> PairwiseResult:
    """Compare two concepts across all dimensions with position bias mitigation.

    Args:
        genome_a: First concept (labeled "a" in prompts).
        genome_b: Second concept (labeled "b" in prompts).
        config: Stage config for judge panel loading.
        champion_label: Which label ("a" or "b") is the current champion.
            Used for formatting feedback only — doesn't affect the comparison.

    Returns:
        PairwiseResult with overall winner, per-dimension wins, and feedback.
    """
    panel = load_panel(config.judges.judges_dir, config.judges.panel)
    t0 = time.perf_counter()

    # Run all judges in parallel, each in both orderings.
    # _judge_one_ordering sets llm_context inside its own Task's context.
    tasks = []
    for judge in panel:
        tasks.append(_judge_one_ordering(judge, genome_a, genome_b))  # A-first
        tasks.append(_judge_one_ordering(judge, genome_b, genome_a))  # B-first

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results: pairs of (forward, reverse) per judge.
    # Skip judges whose calls failed.
    all_resolved = []
    judgments_data = []
    reasoning_texts = []
    surviving_panel: list[JudgePersona] = []
    surviving_fwd_judgments: list[PairwiseJudgment] = []
    total_cost = 0.0

    for i, judge in enumerate(panel):
        fwd_result = results[i * 2]
        rev_result = results[i * 2 + 1]

        if isinstance(fwd_result, Exception) or isinstance(rev_result, Exception):
            err = fwd_result if isinstance(fwd_result, Exception) else rev_result
            logger.warning("Judge %s failed, skipping: %s", judge.id, err)
            continue

        fwd_judgment, fwd_cost = fwd_result
        rev_judgment, rev_cost = rev_result
        total_cost += fwd_cost + rev_cost

        forward_votes = fwd_judgment.votes()
        reverse_votes = _flip_votes(rev_judgment)  # flip because labels were swapped
        resolved = _resolve_votes(forward_votes, reverse_votes)
        classifications = _classify_votes(forward_votes, reverse_votes)
        all_resolved.append(resolved)

        reasoning_texts.append(fwd_judgment.reasoning)
        surviving_panel.append(judge)
        surviving_fwd_judgments.append(fwd_judgment)
        judgments_data.append({
            "judge_id": judge.id,
            "model_used": judge.model[0],
            "harshness": judge.harshness,
            "forward_votes": forward_votes,
            "reverse_votes": reverse_votes,
            "resolved_votes": resolved,
            "classifications": classifications,
            "cost": fwd_cost + rev_cost,
        })

    if not all_resolved:
        logger.error("All judges failed — champion retains by default.")
        return PairwiseResult(
            winner=champion_label,
            dimension_wins={d: "tie" for d in DIMENSION_NAMES},
            a_wins=0, b_wins=0, ties=len(DIMENSION_NAMES),
            feedback="All judges failed — no comparison possible.",
        )

    pw_cfg = config.evaluation.pairwise
    (
        dim_winners,
        a_wins,
        b_wins,
        tie_count,
        a_weighted,
        b_weighted,
        tie_weighted,
    ) = _aggregate(all_resolved, pw_cfg.dim_weights)

    # Selection: weighted dim-votes + asymmetric tiebreaker (Option E).
    # See lab/issues/2026-04-21-rubric-reweighting.md.
    winner, tiebreaker_used = _select_winner(
        a_weighted=a_weighted,
        b_weighted=b_weighted,
        dim_winners=dim_winners,
        champion_label=champion_label,
        tiebreaker_threshold=pw_cfg.tiebreaker_threshold,
        tiebreaker_dims=pw_cfg.tiebreaker_dims,
    )

    feedback = _format_feedback(
        dim_winners, a_wins, b_wins, winner, reasoning_texts, champion_label,
    )

    critiques_by_label = _build_match_critiques(
        genome_a=genome_a,
        genome_b=genome_b,
        champion_label=champion_label,
        winner=winner,
        dim_winners=dim_winners,
        panel=surviving_panel,
        fwd_judgments=surviving_fwd_judgments,
    )

    dt = time.perf_counter() - t0

    # Log per-dimension breakdown with magnitude suffix (1=narrow, 2=clear,
    # 3=decisive) derived from the mean magnitude on the winning side.
    dim_summary = _format_dim_summary(dim_winners, all_resolved)
    logger.info(
        "Pairwise: %s wins %d-%d-%d (weighted %.2f-%.2f, tiebreaker=%s) "
        "(%.1fs $%.4f)  [%s]",
        winner.upper(), a_wins, b_wins, tie_count,
        a_weighted, b_weighted, tiebreaker_used or "none",
        dt, total_cost, dim_summary,
    )

    return PairwiseResult(
        winner=winner,
        dimension_wins=dim_winners,
        a_wins=a_wins,
        b_wins=b_wins,
        ties=tie_count,
        a_weighted=a_weighted,
        b_weighted=b_weighted,
        tie_weighted=tie_weighted,
        tiebreaker_used=tiebreaker_used,
        judgments=judgments_data,
        feedback=feedback,
        critiques_by_label=critiques_by_label,
    )
