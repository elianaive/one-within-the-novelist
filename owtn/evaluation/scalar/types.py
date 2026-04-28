"""Core data types for scalar scoring.

`Rubric` and `Dim` are loaded from YAML; `ScoreCard` and `AggregatedScoreCard`
are the runtime outputs of `Scorer.score()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Dim:
    """One scoring dimension: name, prose definition, polarity, and weight.

    Polarity "negative" inverts the raw score at aggregation (so all dims
    have uniform "higher is better" semantics in the aggregate). Weight is
    applied at the aggregation step.
    """
    name: str
    description: str
    polarity: Literal["positive", "negative"] = "positive"
    weight: float = 1.0


@dataclass(frozen=True)
class Rubric:
    """A scoring rubric — ordered tuple of dims plus scale parameters."""
    dims: tuple[Dim, ...]
    scale_min: int = 0
    scale_max: int = 20
    scale_anchors: str = (
        "0-3 catastrophic failure on this dimension; "
        "4-7 thin or generic, no real engagement; "
        "8-11 competent but unremarkable; "
        "12-15 clearly working, would notice in reading; "
        "16-19 exemplary, the dimension is a strength; "
        "20 reserved for masterpiece-level execution."
    )

    @property
    def dim_names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dims)

    def get_dim(self, name: str) -> Dim:
        for d in self.dims:
            if d.name == name:
                return d
        raise KeyError(f"dim {name!r} not in rubric (have: {self.dim_names})")


@dataclass
class ScoreCard:
    """One judge's score on one artifact across all rubric dims.

    `dim_scores` are post-polarity-inversion (uniform higher-is-better).
    `aggregate` is weighted-mean-on-[0,1]: `sum(weight * score) / (sum_weight * scale_max)`.
    """
    dim_scores: dict[str, float]
    aggregate: float
    n_calls: int                     # 1 for single-call; len(dims) for atomic
    judge_label: str                 # provenance: judge_model::persona_id::mode
    raw_responses: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedScoreCard:
    """Ensemble result — wraps N member ScoreCards with aggregated dim/agg."""
    members: list[ScoreCard]
    dim_scores: dict[str, float]     # mean (or median) across members
    aggregate: float
    n_calls: int                     # sum across members
    cost_usd: float                  # sum across members
    judge_label: str                 # ensemble label
    raw_responses: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def card_to_dict(card: ScoreCard | AggregatedScoreCard) -> dict[str, Any]:
    """Serialize a card for JSONL persistence (excludes raw_responses to
    keep logs compact; raw_responses are written separately when needed)."""
    if isinstance(card, AggregatedScoreCard):
        return {
            "kind": "aggregated",
            "dim_scores": card.dim_scores,
            "aggregate": card.aggregate,
            "n_calls": card.n_calls,
            "cost_usd": card.cost_usd,
            "judge_label": card.judge_label,
            "members": [card_to_dict(m) for m in card.members],
            "metadata": card.metadata,
        }
    return {
        "kind": "single",
        "dim_scores": card.dim_scores,
        "aggregate": card.aggregate,
        "n_calls": card.n_calls,
        "cost_usd": card.cost_usd,
        "judge_label": card.judge_label,
        "metadata": card.metadata,
    }
