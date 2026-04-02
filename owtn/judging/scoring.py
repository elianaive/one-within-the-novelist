"""Scoring math for the judging system.

Hölder mean (soft minimum) for within-judge aggregation.
Selection score with diversity bonus for cross-judge aggregation.
"""

from dataclasses import dataclass

import numpy as np


def holder_mean(scores: list[float], p: float = 0.4) -> float:
    """Compute Hölder mean (generalized/power mean) of scores.

    Acts as a soft minimum: weaknesses drag the score down more than
    strengths compensate. A story scoring [5,5,5,5,1] gets ~2.6 while
    [3,3,3,3,3] gets 3.0.

    Args:
        scores: Dimension scores (0-5 scale).
        p: Penalty parameter. Lower = harsher on weak dimensions.
           p=1.0 is arithmetic mean, p→-inf is strict minimum.
    """
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)
    arr = np.maximum(arr, 1e-6)
    return float(np.power(np.mean(np.power(arr, p)), 1.0 / p))


def selection_score(
    judge_mean: float,
    judge_std: float,
    diversity_weight: float = 0.15,
    std_threshold: float = 0.8,
) -> float:
    """Compute selection score with diversity bonus.

    Stories with high mean AND high variance get a bonus — this protects
    bold, polarizing work from being ground down to consensus mediocrity.

    The bonus only activates when std exceeds threshold.
    """
    diversity_bonus = max(0.0, judge_std - std_threshold) * diversity_weight
    return judge_mean + diversity_bonus


@dataclass
class AggregateResult:
    combined_score: float  # selection_score: mean + diversity bonus (parent selection, advancement)
    holder_score: float    # raw mean of judge Hölder scores (MAP-Elites cell replacement)
    judge_mean: float
    judge_std: float
    diversity_bonus: float


def aggregate_judge_scores(
    judge_holder_scores: list[float],
    diversity_weight: float = 0.15,
    std_threshold: float = 0.8,
) -> AggregateResult:
    """Aggregate per-judge Hölder scores into a selection decision.

    Args:
        judge_holder_scores: Each judge's Hölder mean (one float per judge).
        diversity_weight: Weight of the disagreement bonus.
        std_threshold: Minimum std to activate bonus.
    """
    if not judge_holder_scores:
        return AggregateResult(0.0, 0.0, 0.0, 0.0, 0.0)

    arr = np.array(judge_holder_scores, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    bonus = max(0.0, std - std_threshold) * diversity_weight
    combined = mean + bonus

    return AggregateResult(
        combined_score=combined,
        holder_score=mean,
        judge_mean=mean,
        judge_std=std,
        diversity_bonus=bonus,
    )
