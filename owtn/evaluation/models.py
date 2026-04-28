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

    Field order matters: native structured output (Anthropic tool use,
    OpenAI Responses API, Gemini response_schema) fills `reasoning` first,
    forcing CoT before score assignment. Score field names match
    rubric_anchors.txt.
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


# Per-dimension votes carry a side ("a"/"b") and a magnitude
# (narrow/clear/decisive), or "tie". See
# `lab/issues/2026-04-22-pairwise-win-margin.md` for rationale — magnitudes
# multiply into the weighted aggregation, sharpening discrimination at match
# boundaries while preserving Option-E weighted-dim + tiebreaker selection.
Vote = Literal[
    "a_narrow", "a_clear", "a_decisive",
    "tie",
    "b_narrow", "b_clear", "b_decisive",
]

# Magnitude labels → aggregation multipliers. Bounded above at 1.0 so a
# per-dim contribution never exceeds today's (pre-magnitude) dim weight; mean
# magnitude on the winning side multiplies the dim weight in `_aggregate`.
MAGNITUDE_VALUE: dict[str, float] = {
    "narrow": 0.5,
    "clear": 0.75,
    "decisive": 1.0,
}
_VALUE_TO_LABEL: dict[float, str] = {v: k for k, v in MAGNITUDE_VALUE.items()}

# Legacy mapping for binary votes stored in older match_history JSON. A saved
# "a" vote predates magnitude support and is inferred as "clear" (middle
# magnitude — a deliberate, non-decisive pick).
_LEGACY_VOTE_MAP: dict[str, str] = {
    "a": "a_clear",
    "b": "b_clear",
    "tie": "tie",
}


def parse_vote(raw: str) -> tuple[str, float]:
    """Decode a vote into (side, magnitude).

    Returns:
        ("a"|"b"|"tie", magnitude ∈ {0.5, 0.75, 1.0} or 0.0 for tie).

    Accepts both magnitude-encoded votes ("a_narrow", "b_decisive", "tie") and
    legacy binary votes ("a", "b", "tie") for backward compat when replaying
    match_history entries written before magnitudes existed.
    """
    if raw in _LEGACY_VOTE_MAP:
        raw = _LEGACY_VOTE_MAP[raw]
    if raw == "tie":
        return ("tie", 0.0)
    side, level = raw.split("_", 1)
    return (side, MAGNITUDE_VALUE[level])


def encode_vote(side: str, magnitude: float) -> str:
    """Inverse of parse_vote. Round-trips through `_resolve_votes` so
    downstream code sees a single uniform string type."""
    if side == "tie" or magnitude == 0.0:
        return "tie"
    return f"{side}_{_VALUE_TO_LABEL[magnitude]}"


# Judge-facing description for each dim field — enumerates the 7 allowed
# values and glosses each magnitude so structured-output parsers emit the
# right tokens. The full magnitude rubric lives in `rubric_anchors.txt`.
_VOTE_FIELD_DESCRIPTION = (
    "Winner + magnitude: 'a_narrow' / 'a_clear' / 'a_decisive' (A wins at that"
    " magnitude), 'b_narrow' / 'b_clear' / 'b_decisive' (B wins), or 'tie'."
    " narrow = visible edge but small or one sub-criterion only;"
    " clear = preponderance of sub-criteria favor one concept;"
    " decisive = the two concepts are not in the same class on this dimension."
)


class PairwiseJudgment(BaseModel):
    """Structured output from one judge comparing two concepts.

    The judge evaluates each dimension independently: for each dimension,
    which concept is stronger AND by how much, or declare tie if neither has a
    dimension-specific advantage. Output is reasoning followed by the
    dimension votes.
    """

    reasoning: str = Field(
        description="Structured per-dimension analysis following the exact "
        "format laid out in the user-message TASK block. Each dimension "
        "block must include fingerprints, independent strengths, "
        "sub-criterion analysis, initial edge, adversarial check, "
        "reversal check, final winner, and mapping. Concepts are referred "
        "to by content fingerprints (FP1/FP2) inside the reasoning, never "
        "by their A/B labels."
    )
    novelty: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    grip: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    tension_architecture: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    emotional_depth: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    thematic_resonance: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    concept_coherence: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    generative_fertility: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    scope_calibration: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    indelibility: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)

    def votes(self) -> dict[str, str]:
        """Per-dimension votes as {name: magnitude-encoded string}."""
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
    a concept's match critiques are fed to the lineage summarizer which
    produces a `LineageBrief` (see `owtn/optimizer/`). Historical context:
    `lab/issues/closed/2026-04-18-lazy-feedback-summarizer.md`.
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
    # Stage 2 only: the challenger's own DAG, included so the tree-level
    # summarizer can cite the structural choice that just lost ("your
    # cross-edge made indelibility narrow"). Stage 1 leaves this None
    # because the lineage's self-genome is already implicit in the parent.
    self_dag: dict | None = None
    # This concept's result. Match-level outcome: won / lost / tied (champion
    # retention on tie is enforced upstream; here we report raw dim totals).
    outcome: Literal["won", "lost", "tied"]
    dim_outcomes: dict[str, Literal["won", "lost", "tied"]]
    judge_reasonings: list[JudgeReasoningRecord]
    timestamp: str


class PairwiseResult(BaseModel):
    """Aggregated pairwise comparison result across all judges and orderings.

    Winner-selection uses weighted dim-votes (Option E:
    lab/issues/2026-04-21-rubric-reweighting.md). `a_wins`/`b_wins`/`ties` are
    integer dim-counts preserved for display and match-history; the actual
    selection decision is driven by `a_weighted`/`b_weighted` plus the
    asymmetric tiebreaker, and the resulting signal flows through
    `a_weighted_score` → `combined_score` for shinka parent-selection.
    """

    winner: str  # "a" or "b"
    dimension_wins: dict[str, str]  # {dim_name: "a"/"b"/"tie"} majority per dim
    a_wins: int  # total dimension-wins for concept a (integer count, display)
    b_wins: int  # total dimension-wins for concept b (integer count, display)
    ties: int
    a_weighted: float = 0.0    # weighted total that drove winner selection
    b_weighted: float = 0.0
    tie_weighted: float = 0.0  # mass of tied-dim weights (for weighted_score denom)
    tiebreaker_used: str | None = None  # dim name, "incumbent", or None
    judgments: list[dict] = Field(default_factory=list)  # raw judge data
    feedback: str = ""  # formatted for mutation model (legacy; kept for logging)
    critiques_by_label: dict[str, MatchCritique] = Field(default_factory=dict)

    @property
    def a_score(self) -> float:
        """Integer dim-count win fraction (ties count as 0.5). Display/legacy."""
        total = self.a_wins + self.b_wins + self.ties
        return (self.a_wins + 0.5 * self.ties) / total if total else 0.0

    @property
    def b_score(self) -> float:
        """Integer dim-count win fraction (ties count as 0.5). Display/legacy."""
        total = self.a_wins + self.b_wins + self.ties
        return (self.b_wins + 0.5 * self.ties) / total if total else 0.0

    @property
    def a_weighted_score(self) -> float:
        """Weighted win fraction. THIS is the selection signal; feeds shinka's
        combined_score. Matches the winner-selection basis (weighted totals)."""
        total = self.a_weighted + self.b_weighted + self.tie_weighted
        return (self.a_weighted + 0.5 * self.tie_weighted) / total if total else 0.0

    @property
    def b_weighted_score(self) -> float:
        total = self.a_weighted + self.b_weighted + self.tie_weighted
        return (self.b_weighted + 0.5 * self.tie_weighted) / total if total else 0.0


# --- Stage 2 pairwise judgment ---


# Stage 2's 8 dimensions per `docs/stage-2/rubric-anchors.md`. Different from
# Stage 1's 9. The rubric anchor files at
# `owtn/prompts/stage_2/rubric_anchors/<dim>.txt` are concatenated in this
# order in the judge system message.
STAGE_2_DIMENSION_NAMES = [
    "edge_logic",
    "motivational_coherence",
    "tension_information_arch",
    "post_dictability",
    "arc_integrity_ending",
    "structural_coherence",
    "beat_quality",
    "concept_fidelity_thematic",
]


class Stage2PairwiseJudgment(BaseModel):
    """Structured output from one judge comparing two structures (DAGs).

    Mirrors `PairwiseJudgment` (Stage 1) but with the 8 Stage 2 dimensions
    instead of Stage 1's 9. The aggregation machinery in
    `owtn/evaluation/stage_2.py` consumes this via the `votes()` method;
    field names match the rubric anchors at
    `owtn/prompts/stage_2/rubric_anchors/<dim>.txt`.
    """

    reasoning: str = Field(
        description="Structured per-dimension analysis. For each of the 8 "
        "Stage 2 dimensions, produce a block: DIMENSION_NAME header, one line "
        "per sub-criterion in the form '(a) [sub-criterion name] — A: "
        "[concrete observation citing specific node ids or edges]. B: "
        "[concrete observation].', addressing every sub-criterion the rubric "
        "lists for that dimension, followed by a 'Verdict: [a_narrow|a_clear|"
        "a_decisive|b_narrow|b_clear|b_decisive|tie] — [rationale naming the "
        "decisive sub-criterion]' line. Reach each verdict FROM the "
        "sub-criteria, not alongside them."
    )
    edge_logic: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    motivational_coherence: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    tension_information_arch: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    post_dictability: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    arc_integrity_ending: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    structural_coherence: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    beat_quality: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)
    concept_fidelity_thematic: Vote = Field(description=_VOTE_FIELD_DESCRIPTION)

    def votes(self) -> dict[str, str]:
        """Per-dimension votes as {name: magnitude-encoded string}."""
        return {name: getattr(self, name) for name in STAGE_2_DIMENSION_NAMES}
