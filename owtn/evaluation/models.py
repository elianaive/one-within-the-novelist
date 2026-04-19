from __future__ import annotations

from typing import Literal

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


Vote = Literal["a", "b", "tie"]


class PairwiseJudgment(BaseModel):
    """Structured output from one judge comparing two concepts.

    The judge evaluates each dimension independently: for each dimension,
    which concept is stronger, or declare tie if neither has a dimension-
    specific advantage. Output is reasoning followed by the dimension votes.
    """

    reasoning: str = Field(
        description="For each dimension, compare the two concepts using "
        "the sub-criteria as a lens. State which concept is stronger and why, "
        "or declare tie if neither has a dimension-specific advantage."
    )
    novelty: Vote = Field(description="Winner for novelty: 'a', 'b', or 'tie'")
    grip: Vote = Field(description="Winner for grip: 'a', 'b', or 'tie'")
    tension_architecture: Vote = Field(description="Winner for tension_architecture: 'a', 'b', or 'tie'")
    emotional_depth: Vote = Field(description="Winner for emotional_depth: 'a', 'b', or 'tie'")
    thematic_resonance: Vote = Field(description="Winner for thematic_resonance: 'a', 'b', or 'tie'")
    concept_coherence: Vote = Field(description="Winner for concept_coherence: 'a', 'b', or 'tie'")
    generative_fertility: Vote = Field(description="Winner for generative_fertility: 'a', 'b', or 'tie'")
    scope_calibration: Vote = Field(description="Winner for scope_calibration: 'a', 'b', or 'tie'")
    indelibility: Vote = Field(description="Winner for indelibility: 'a', 'b', or 'tie'")

    def votes(self) -> dict[str, str]:
        """Per-dimension votes as {name: 'a'|'b'|'tie'}."""
        return {name: getattr(self, name) for name in DIMENSION_NAMES}


class JudgeReasoningRecord(BaseModel):
    """One judge's forward-ordering reasoning for a single match, stored verbatim.

    The reasoning text references concepts by their match labels ('A' / 'B').
    Label disambiguation happens at summarizer-prompt time, not here.
    """

    judge_id: str
    harshness: str
    reasoning: str


class MatchCritique(BaseModel):
    """One concept's perspective on a single pairwise match.

    Stored per-concept in accumulating lists. At parent-selection time, all of
    a concept's match critiques are fed to the summarizer which produces a
    ParentBrief. See `lab/issues/2026-04-18-lazy-feedback-summarizer.md`.
    """

    # Label disambiguation for the summarizer: in the reasoning text below,
    # `self_label` refers to the concept owning this critique, and
    # `opponent_label` refers to the other concept.
    self_label: Literal["a", "b"]
    opponent_label: Literal["a", "b"]
    self_was_champion: bool
    # Full genome of the opponent concept — enables the summarizer to cite
    # specific elements ("you lost to a concept built around X").
    opponent_genome: dict
    # This concept's result. Match-level outcome: won / lost / tied (champion
    # retention on tie is enforced upstream; here we report raw dim totals).
    outcome: Literal["won", "lost", "tied"]
    dim_outcomes: dict[str, Literal["won", "lost", "tied"]]
    judge_reasonings: list[JudgeReasoningRecord]
    timestamp: str


class ParentBrief(BaseModel):
    """Structured critique of a concept, distilled from its accumulated
    `match_critiques`. Produced by a lightweight summarizer LLM at parent-
    selection time; rendered into the mutation prompt. See
    `lab/issues/2026-04-18-lazy-feedback-summarizer.md`.
    """

    established_weaknesses: list[str] = Field(
        description="Critiques of THIS CONCEPT that recurred across multiple "
        "matches or were voiced by multiple judges. Most robust signal."
    )
    contested_strengths: list[str] = Field(
        description="Points where judges disagreed on THIS CONCEPT's merits. "
        "Exploration zones — a successor could double down or diverge."
    )
    attractor_signature: list[str] = Field(
        description="Specific patterns, motifs, or structural choices THIS "
        "CONCEPT uses that have drawn critical attention. Used as an avoid-"
        "list for successor concepts."
    )
    divergence_directions: list[str] = Field(
        description="Concrete, prescriptive suggestions a successor concept "
        "could take to escape the established weaknesses. Extract from the "
        "judges' reasoning; do not invent."
    )


class PairwiseResult(BaseModel):
    """Aggregated pairwise comparison result across all judges and orderings."""

    winner: str  # "a" or "b"
    dimension_wins: dict[str, str]  # {dim_name: "a"/"b"/"tie"} majority per dim
    a_wins: int  # total dimension-wins for concept a
    b_wins: int  # total dimension-wins for concept b
    ties: int
    judgments: list[dict] = Field(default_factory=list)  # raw judge data
    feedback: str = ""  # formatted for mutation model (legacy; kept for logging)
    critiques_by_label: dict[str, MatchCritique] = Field(default_factory=dict)

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
