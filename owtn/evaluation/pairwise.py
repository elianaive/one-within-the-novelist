"""Per-criteria pairwise comparison of two concept genomes.

Each judge compares on all 9 dimensions independently. Each judge sees
both orderings (A-first, B-first) to mitigate position bias. A dimension
vote counts only if the judge picks the same winner in both orderings;
otherwise it's a tie. Overall winner = most dimension-wins across all judges.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter

from owtn.evaluation.models import (
    DIMENSION_NAMES,
    PairwiseJudgment,
    PairwiseResult,
)
from owtn.evaluation.prompts import build_pairwise_system, build_pairwise_user
from owtn.llm.call_logger import llm_context
from owtn.llm.query import query_async
from owtn.models.judge import JudgePersona, load_panel
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_1.config import StageConfig

logger = logging.getLogger(__name__)


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
    model_name = judge.model[0]

    # The judge's entire system message is stable across all comparisons by
    # this judge — pass it as system_prefix so Anthropic caches it (and
    # OpenAI/DeepSeek auto-match the prefix for their own discounts).
    result = await query_async(
        model_name=model_name,
        msg=user_msg,
        system_msg="",
        system_prefix=system_msg,
        output_model=PairwiseJudgment,
    )
    return result.content, result.cost


def _flip_votes(judgment: PairwiseJudgment) -> dict[str, str]:
    """Flip a/b in votes (for the reversed ordering)."""
    flipped = {}
    for dim, winner in judgment.votes().items():
        if winner == "a":
            flipped[dim] = "b"
        elif winner == "b":
            flipped[dim] = "a"
        else:
            flipped[dim] = winner
    return flipped


def _resolve_votes(
    forward_votes: dict[str, str],
    reverse_votes: dict[str, str],
) -> dict[str, str]:
    """Resolve per-dimension votes across two orderings.

    Same winner in both orderings → that winner.
    Different winners → tie (position bias detected).
    """
    resolved = {}
    for dim in DIMENSION_NAMES:
        fwd = forward_votes.get(dim, "tie")
        rev = reverse_votes.get(dim, "tie")
        if fwd == rev:
            resolved[dim] = fwd
        else:
            resolved[dim] = "tie"
    return resolved


def _aggregate(
    all_resolved: list[dict[str, str]],
) -> tuple[dict[str, str], int, int, int]:
    """Aggregate resolved votes across all judges.

    Per dimension: majority of non-tie votes wins.
    Returns: (dim_winners, a_total, b_total, tie_total)
    """
    dim_winners = {}
    a_total = 0
    b_total = 0
    tie_total = 0

    for dim in DIMENSION_NAMES:
        votes = [r[dim] for r in all_resolved]
        counts = Counter(votes)
        a_count = counts.get("a", 0)
        b_count = counts.get("b", 0)

        if a_count > b_count:
            dim_winners[dim] = "a"
            a_total += 1
        elif b_count > a_count:
            dim_winners[dim] = "b"
            b_total += 1
        else:
            dim_winners[dim] = "tie"
            tie_total += 1

    return dim_winners, a_total, b_total, tie_total


def _format_feedback(
    dim_winners: dict[str, str],
    a_total: int,
    b_total: int,
    reasoning_texts: list[str],
    champion_label: str,
) -> str:
    """Format pairwise result as mutation feedback.

    The mutation model sees which dimensions the champion won and lost,
    with judge reasoning for each.
    """
    loser_label = "b" if champion_label == "a" else "a"

    won = [d for d, w in dim_winners.items() if w == champion_label]
    lost = [d for d, w in dim_winners.items() if w == loser_label]
    tied = [d for d, w in dim_winners.items() if w == "tie"]

    lines = [f"Pairwise result: {'Won' if a_total > b_total else 'Lost'} ({a_total}-{b_total})"]
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
        all_resolved.append(resolved)

        reasoning_texts.append(fwd_judgment.reasoning)
        judgments_data.append({
            "judge_id": judge.id,
            "model_used": judge.model[0],
            "forward_votes": forward_votes,
            "reverse_votes": reverse_votes,
            "resolved_votes": resolved,
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

    dim_winners, a_total, b_total, tie_total = _aggregate(all_resolved)

    # Overall winner: most dimension-wins. Ties favor champion.
    if a_total > b_total:
        winner = "a"
    elif b_total > a_total:
        winner = "b"
    else:
        winner = champion_label  # incumbent advantage

    feedback = _format_feedback(
        dim_winners, a_total, b_total, reasoning_texts, champion_label,
    )

    dt = time.perf_counter() - t0

    # Log per-dimension breakdown.
    dim_summary = "  ".join(
        f"{d}={'A' if w == 'a' else 'B' if w == 'b' else '='}"
        for d, w in dim_winners.items()
    )
    logger.info(
        "Pairwise: %s wins %d-%d-%d (%.1fs $%.4f)  [%s]",
        winner.upper(), a_total, b_total, tie_total, dt, total_cost, dim_summary,
    )

    return PairwiseResult(
        winner=winner,
        dimension_wins=dim_winners,
        a_wins=a_total,
        b_wins=b_total,
        ties=tie_total,
        judgments=judgments_data,
        feedback=feedback,
    )
