"""Anchor-based scale calibration + bootstrap CI.

Each call site (Stage-1 challenger / Stage-2 partial / archive) fits its
own calibrator from its own anchor set. Cross-stage anchor reuse is unsafe
because score distributions differ across the artifact types.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from owtn.evaluation.scalar.scorer import Scorer
from owtn.evaluation.scalar.types import AggregatedScoreCard, Rubric, ScoreCard


@dataclass
class AnchorCalibrator:
    """Piecewise-linear monotone transform raw_aggregate -> calibrated [0, 1].

    Knots = (raw, expected) pairs from scoring known anchor artifacts at fit
    time. Transform via numpy.interp (clamped at endpoints).
    """
    raw_anchor_aggregates: list[float]
    expected_normalized: list[float]   # same length, sorted parallel to raws

    @classmethod
    async def fit(
        cls,
        anchors: list[tuple[Any, float]],   # (artifact, expected_normalized_in_[0,1])
        scorer: Scorer,
    ) -> "AnchorCalibrator":
        scored = await asyncio.gather(*[scorer.score(a) for (a, _) in anchors])
        pairs = sorted(
            ((sc.aggregate, exp) for sc, (_, exp) in zip(scored, anchors)),
            key=lambda x: x[0],
        )
        return cls(
            raw_anchor_aggregates=[r for r, _ in pairs],
            expected_normalized=[e for _, e in pairs],
        )

    def calibrate(self, raw: float) -> float:
        if not self.raw_anchor_aggregates:
            return raw
        return float(np.interp(raw, self.raw_anchor_aggregates, self.expected_normalized))


def bootstrap_ci(
    score_card: ScoreCard | AggregatedScoreCard,
    rubric: Rubric,
    n: int = 500,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap CI over dimensional aggregation.

    Resamples (dim_score, weight) tuples with replacement, recomputes the
    aggregate. Characterizes how much the score depends on the dim weighting
    / dim selection. For ensembles, also includes member dim_scores in the
    resampling pool.
    """
    rng = rng or np.random.default_rng(0xC0DA)

    if isinstance(score_card, AggregatedScoreCard):
        rows = [
            (m.dim_scores[d.name], d.weight)
            for m in score_card.members
            for d in rubric.dims
            if d.name in m.dim_scores
        ]
    else:
        rows = [
            (score_card.dim_scores[d.name], d.weight)
            for d in rubric.dims
            if d.name in score_card.dim_scores
        ]
    if not rows:
        return (0.0, 0.0)

    arr = np.array(rows)
    aggregates = []
    for _ in range(n):
        idx = rng.integers(0, len(arr), size=len(arr))
        sample = arr[idx]
        s, w = sample[:, 0], sample[:, 1]
        wmean = (s * w).sum() / w.sum() if w.sum() > 0 else 0.0
        aggregates.append(wmean / rubric.scale_max)

    aggregates = np.array(aggregates)
    return (
        float(np.quantile(aggregates, alpha / 2)),
        float(np.quantile(aggregates, 1 - alpha / 2)),
    )
