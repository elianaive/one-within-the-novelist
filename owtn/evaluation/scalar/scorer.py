"""Scorer protocol and concrete implementations.

Composition over inheritance. Three primitives:

  - SingleCallScorer:  one LLM call returns scores for all rubric dims at once
  - AtomicPerDimScorer: one LLM call per dim (parallel via asyncio.gather)
  - EnsembleScorer:     N base scorers, aggregated mean/median per dim

S1 vs SP is the persona axis (None vs full persona system msg). single vs
atomic is the rubric structure axis. Configurations are pure compositions:

  S1_single  = SingleCallScorer(persona=None)
  S1_atomic  = AtomicPerDimScorer(persona=None)
  SP_single  = EnsembleScorer([SingleCallScorer(persona=p) for p in panel])
  SP_atomic  = EnsembleScorer([AtomicPerDimScorer(persona=p) for p in panel])
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, Field

from owtn.evaluation.scalar.types import (
    AggregatedScoreCard,
    Dim,
    Rubric,
    ScoreCard,
)
from owtn.llm.call_logger import llm_context
from owtn.llm.query import query_async

logger = logging.getLogger(__name__)


# ============================================================
# Aggregation helpers
# ============================================================


def _aggregate(dim_scores: dict[str, float], rubric: Rubric) -> float:
    """Weighted mean of dim_scores normalized to [0, 1]."""
    total_w = 0.0
    total = 0.0
    for d in rubric.dims:
        if d.name not in dim_scores:
            continue
        total += d.weight * dim_scores[d.name]
        total_w += d.weight
    if total_w == 0:
        return 0.0
    return (total / total_w) / rubric.scale_max


def _polarity_correct(raw: dict[str, int], rubric: Rubric) -> dict[str, float]:
    """Invert negative-polarity dims so the aggregate is uniformly higher-is-better."""
    out: dict[str, float] = {}
    for d in rubric.dims:
        if d.name not in raw:
            continue
        s = float(raw[d.name])
        if d.polarity == "negative":
            s = rubric.scale_max - s
        out[d.name] = s
    return out


# ============================================================
# Pydantic schemas for structured LLM output
# ============================================================


class _AllDimsResponse(BaseModel):
    reasoning: str = Field(..., description="Per-dimension reasoning.")
    scores: dict[str, int] = Field(..., description="Map of dim_name -> integer score.")


class _OneDimResponse(BaseModel):
    reasoning: str = Field(..., description="Sub-criteria walkthrough.")
    score: int = Field(..., ge=0, le=20)


# ============================================================
# Prompt assembly
# ============================================================


def _build_system_msg(
    rubric: Rubric,
    persona_system_msg: str | None,
    atomic_dim: Dim | None,
) -> str:
    """Identity + rubric block + scale + output format.

    `persona_system_msg`: when None, generic neutral judge identity.
    `atomic_dim`: when set, the rubric block contains only that dim.
    """
    if persona_system_msg is None:
        identity = (
            "You are an expert literary critic, reading short story concepts "
            "(or DAG-structured story plans) with rigor. You apply the rubric "
            "below and emit structured scores."
        )
    else:
        identity = persona_system_msg

    if atomic_dim is None:
        rubric_block = "\n\n".join(
            f"## {d.name.upper()}\n{d.description}"
            + ("\n[NEGATIVE criterion — higher raw score = MORE of this flaw, inverted at aggregation.]"
               if d.polarity == "negative" else "")
            for d in rubric.dims
        )
    else:
        rubric_block = (
            f"## {atomic_dim.name.upper()}\n{atomic_dim.description}"
            + ("\n[NEGATIVE criterion — higher = MORE of this flaw.]"
               if atomic_dim.polarity == "negative" else "")
        )

    scale = (
        f"Score each dimension on a {rubric.scale_min}-{rubric.scale_max} integer scale. "
        f"{rubric.scale_anchors}"
    )

    if atomic_dim is None:
        format_block = (
            "Output a JSON object with two fields:\n"
            "  - reasoning: string, with one block per dimension. For each dim, "
            "walk the sub-criteria and cite specific concept material before scoring.\n"
            f"  - scores: object mapping dim_name to integer 0-{rubric.scale_max}.\n"
            f"  Required keys: {list(rubric.dim_names)}\n"
        )
    else:
        format_block = (
            "Output a JSON object with two fields:\n"
            "  - reasoning: walk this dim's sub-criteria, cite specific material, then conclude.\n"
            f"  - score: integer 0-{rubric.scale_max}.\n"
        )

    return (
        f"{identity}\n\n# RUBRIC\n\n{rubric_block}\n\n"
        f"# SCALE\n{scale}\n\n# OUTPUT\n{format_block}"
    )


def _build_user_msg(rendered: str, rubric: Rubric, atomic_dim: Dim | None) -> str:
    if atomic_dim is None:
        return f"# ARTIFACT TO SCORE\n\n{rendered}\n\nScore on all {len(rubric.dims)} dimensions."
    return f"# ARTIFACT TO SCORE\n\n{rendered}\n\nScore only the {atomic_dim.name.upper()} dimension."


# ============================================================
# Scorer protocol + implementations
# ============================================================


class Scorer(Protocol):
    rubric: Rubric

    async def score(self, artifact: Any) -> ScoreCard | AggregatedScoreCard:
        ...


@dataclass
class SingleCallScorer:
    """One LLM call returns scores for all rubric dims at once.

    Cheaper but susceptible to criterion conflation (Autorubric's finding —
    multi-dim single-call collapses distinct signals across dims).
    """
    rubric: Rubric
    judge_model: str
    artifact_renderer: Callable[[Any], str]
    persona_system_msg: str | None = None
    persona_label: str = "neutral"
    sampling_kwargs: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})

    async def score(self, artifact: Any) -> ScoreCard:
        rendered = self.artifact_renderer(artifact)
        system_msg = _build_system_msg(self.rubric, self.persona_system_msg, atomic_dim=None)
        user_msg = _build_user_msg(rendered, self.rubric, atomic_dim=None)

        prev_ctx = llm_context.get({})
        token = llm_context.set({**prev_ctx, "role": "scalar_score_single", "persona_id": self.persona_label})
        try:
            result = await query_async(
                model_name=self.judge_model,
                msg=user_msg,
                system_msg="",
                system_prefix=system_msg,
                output_model=_AllDimsResponse,
                **self.sampling_kwargs,
            )
        finally:
            llm_context.reset(token)
        parsed: _AllDimsResponse = result.content

        # Default missing dims to scale midpoint — rare in practice but the
        # scorer must not crash when a model omits one in its JSON output.
        missing = [d for d in self.rubric.dim_names if d not in parsed.scores]
        if missing:
            logger.warning("missing dims %s; defaulting to midpoint", missing)
            for d in missing:
                parsed.scores[d] = self.rubric.scale_max // 2

        clipped = {
            d: max(self.rubric.scale_min, min(self.rubric.scale_max, int(v)))
            for d, v in parsed.scores.items()
            if d in self.rubric.dim_names
        }
        polarity_corrected = _polarity_correct(clipped, self.rubric)
        return ScoreCard(
            dim_scores=polarity_corrected,
            aggregate=_aggregate(polarity_corrected, self.rubric),
            n_calls=1,
            judge_label=f"{self.judge_model}::{self.persona_label}::single",
            raw_responses=[parsed.reasoning],
            cost_usd=result.cost,
            metadata={"raw_scores": clipped},
        )


@dataclass
class AtomicPerDimScorer:
    """One LLM call per rubric dim, fanned out via asyncio.gather.

    Higher cost (one call per dim) but no criterion conflation. Per-dim
    independent calls also decorrelate dim votes — a coherence-strong
    artifact won't drag up its novelty score, etc.
    """
    rubric: Rubric
    judge_model: str
    artifact_renderer: Callable[[Any], str]
    persona_system_msg: str | None = None
    persona_label: str = "neutral"
    sampling_kwargs: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})

    async def score(self, artifact: Any) -> ScoreCard:
        rendered = self.artifact_renderer(artifact)

        async def score_one(dim: Dim) -> tuple[str, int, str, float]:
            system_msg = _build_system_msg(self.rubric, self.persona_system_msg, atomic_dim=dim)
            user_msg = _build_user_msg(rendered, self.rubric, atomic_dim=dim)
            prev_ctx = llm_context.get({})
            token = llm_context.set({
                **prev_ctx,
                "role": "scalar_score_atomic",
                "dim_name": dim.name,
                "persona_id": self.persona_label,
            })
            try:
                result = await query_async(
                    model_name=self.judge_model,
                    msg=user_msg,
                    system_msg="",
                    system_prefix=system_msg,
                    output_model=_OneDimResponse,
                    **self.sampling_kwargs,
                )
            finally:
                llm_context.reset(token)
            parsed: _OneDimResponse = result.content
            score = max(self.rubric.scale_min, min(self.rubric.scale_max, int(parsed.score)))
            return dim.name, score, parsed.reasoning, result.cost

        results = await asyncio.gather(*[score_one(d) for d in self.rubric.dims])
        raw = {name: score for (name, score, _, _) in results}
        polarity_corrected = _polarity_correct(raw, self.rubric)
        return ScoreCard(
            dim_scores=polarity_corrected,
            aggregate=_aggregate(polarity_corrected, self.rubric),
            n_calls=len(self.rubric.dims),
            judge_label=f"{self.judge_model}::{self.persona_label}::atomic",
            raw_responses=[f"[{name}] {r}" for (name, _, r, _) in results],
            cost_usd=sum(c for *_, c in results),
            metadata={"raw_scores": raw},
        )


@dataclass
class EnsembleScorer:
    """N base scorers run in parallel; per-dim aggregation via mean or median.

    Persona-ensemble cost (typically 4×) is justified at low-volume call
    sites (handoff selection, top-K re-score) where panel disagreement
    itself is signal. For high-volume call sites (rollout reward) the
    single-judge atomic configuration discriminates sufficiently at lower cost.
    """
    base_scorers: list[Scorer]
    aggregation: Literal["mean", "median"] = "mean"
    label: str = "ensemble"

    @property
    def rubric(self) -> Rubric:
        return self.base_scorers[0].rubric

    async def score(self, artifact: Any) -> AggregatedScoreCard:
        members = list(await asyncio.gather(*[s.score(artifact) for s in self.base_scorers]))
        agg_fn = statistics.mean if self.aggregation == "mean" else statistics.median

        dim_scores: dict[str, float] = {}
        for d in self.rubric.dims:
            vals = [m.dim_scores[d.name] for m in members if d.name in m.dim_scores]
            if vals:
                dim_scores[d.name] = agg_fn(vals)

        return AggregatedScoreCard(
            members=members,
            dim_scores=dim_scores,
            aggregate=_aggregate(dim_scores, self.rubric),
            n_calls=sum(m.n_calls for m in members),
            cost_usd=sum(m.cost_usd for m in members),
            judge_label=self.label,
            raw_responses=[r for m in members for r in m.raw_responses],
            metadata={"aggregation": self.aggregation},
        )
