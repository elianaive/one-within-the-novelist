from __future__ import annotations

from pydantic import BaseModel, Field


# Dimension field names — canonical order for to_list() and averaging.
DIMENSION_NAMES = [
    "novelty",
    "grip",
    "tension_architecture",
    "emotional_depth",
    "thematic_resonance",
    "concept_coherence",
    "generative_fertility",
    "scope_calibration",
    "indelibility",
]


class JudgeScores(BaseModel):
    """Structured output from a single judge evaluation.

    Field order matters: instructor fills `reasoning` first, forcing CoT
    before score assignment. Score field names match rubric_anchors.txt.
    """

    reasoning: str = Field(
        description="For each dimension, answer the sub-criteria questions "
        "citing specific evidence, then assign the score."
    )
    novelty: float = Field(ge=0, le=5)
    grip: float = Field(ge=0, le=5)
    tension_architecture: float = Field(ge=0, le=5)
    emotional_depth: float = Field(ge=0, le=5)
    thematic_resonance: float = Field(ge=0, le=5)
    concept_coherence: float = Field(ge=0, le=5)
    generative_fertility: float = Field(ge=0, le=5)
    scope_calibration: float = Field(ge=0, le=5)
    indelibility: float = Field(ge=0, le=5)

    def to_list(self) -> list[float]:
        """All 9 scores in canonical order."""
        return [getattr(self, name) for name in DIMENSION_NAMES]

    def to_dict(self) -> dict[str, float]:
        """Dimension scores as {name: score}, excluding reasoning."""
        return {name: getattr(self, name) for name in DIMENSION_NAMES}


class EvaluationResult(BaseModel):
    """Complete evaluation output for one concept. Written as metrics.json."""

    correct: bool
    error: str | None = None
    combined_score: float = 0.0
    holder_score: float = 0.0
    public_metrics: dict = Field(default_factory=dict)
    private_metrics: dict = Field(default_factory=dict)
    text_feedback: str = ""


# --- Pairwise comparison models ---


class PairwiseJudgment(BaseModel):
    """Structured output from one judge comparing two concepts.

    The judge evaluates each dimension independently: for each dimension,
    which concept is stronger, and why? Output is a list of per-dimension
    verdicts followed by the dimension-level votes.
    """

    reasoning: str = Field(
        description="For each dimension, compare the two concepts using "
        "the sub-criteria as a lens. State which concept is stronger and why."
    )
    novelty: str = Field(description="Winner for novelty: 'a' or 'b'")
    grip: str = Field(description="Winner for grip: 'a' or 'b'")
    tension_architecture: str = Field(description="Winner for tension_architecture: 'a' or 'b'")
    emotional_depth: str = Field(description="Winner for emotional_depth: 'a' or 'b'")
    thematic_resonance: str = Field(description="Winner for thematic_resonance: 'a' or 'b'")
    concept_coherence: str = Field(description="Winner for concept_coherence: 'a' or 'b'")
    generative_fertility: str = Field(description="Winner for generative_fertility: 'a' or 'b'")
    scope_calibration: str = Field(description="Winner for scope_calibration: 'a' or 'b'")
    indelibility: str = Field(description="Winner for indelibility: 'a' or 'b'")

    def votes(self) -> dict[str, str]:
        """Per-dimension votes as {name: 'a'/'b'}."""
        return {name: getattr(self, name) for name in DIMENSION_NAMES}


class PairwiseResult(BaseModel):
    """Aggregated pairwise comparison result across all judges and orderings."""

    winner: str  # "a" or "b"
    dimension_wins: dict[str, str]  # {dim_name: "a"/"b"/"tie"} majority per dim
    a_wins: int  # total dimension-wins for concept a
    b_wins: int  # total dimension-wins for concept b
    ties: int
    judgments: list[dict] = Field(default_factory=list)  # raw judge data
    feedback: str = ""  # formatted for mutation model

    @property
    def a_score(self) -> float:
        """Win percentage for concept a (ties count as 0.5)."""
        total = self.a_wins + self.b_wins + self.ties
        return (self.a_wins + 0.5 * self.ties) / total if total else 0.0

    @property
    def b_score(self) -> float:
        """Win percentage for concept b (ties count as 0.5)."""
        total = self.a_wins + self.b_wins + self.ties
        return (self.b_wins + 0.5 * self.ties) / total if total else 0.0
