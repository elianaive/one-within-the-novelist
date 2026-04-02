from __future__ import annotations

from pydantic import BaseModel, Field


# Dimension field names — canonical order for to_list() and averaging.
DIMENSION_NAMES = [
    "originality",
    "transportation_potential",
    "narrative_tension",
    "thematic_resonance",
    "scope_calibration",
    "anti_cliche",
    "concept_coherence",
    "generative_fertility",
    "over_explanation_resistance",
]


class JudgeScores(BaseModel):
    """Structured output from a single judge evaluation.

    Field order matters: instructor fills `reasoning` first, forcing CoT
    before score assignment. Score field names match judge_user.txt keys.
    """

    reasoning: str = Field(
        description="Step-by-step evaluation reasoning for each dimension. "
        "Complete all reasoning before assigning scores."
    )
    originality: float = Field(ge=0, le=5)
    transportation_potential: float = Field(ge=0, le=5)
    narrative_tension: float = Field(ge=0, le=5)
    thematic_resonance: float = Field(ge=0, le=5)
    scope_calibration: float = Field(ge=0, le=5)
    anti_cliche: float = Field(ge=0, le=5)
    concept_coherence: float = Field(ge=0, le=5)
    generative_fertility: float = Field(ge=0, le=5)
    over_explanation_resistance: float = Field(ge=0, le=5)

    def to_list(self) -> list[float]:
        """All 9 scores in canonical order (for holder_mean)."""
        return [getattr(self, name) for name in DIMENSION_NAMES]

    def to_dict(self) -> dict[str, float]:
        """Dimension scores as {name: score}, excluding reasoning."""
        return {name: getattr(self, name) for name in DIMENSION_NAMES}


class JudgeEvaluation(BaseModel):
    """Complete result from one judge: scores + reasoning + metadata."""

    judge_id: str
    scores: JudgeScores
    holder_score: float
    model_used: str
    cost: float


class EvaluationResult(BaseModel):
    """Complete evaluation output for one concept. Written as metrics.json."""

    correct: bool
    error: str | None = None
    combined_score: float = 0.0
    holder_score: float = 0.0
    public_metrics: dict = Field(default_factory=dict)
    private_metrics: dict = Field(default_factory=dict)
    text_feedback: str = ""
