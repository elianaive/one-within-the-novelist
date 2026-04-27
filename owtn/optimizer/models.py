from __future__ import annotations

from pydantic import BaseModel, Field


class LineageBrief(BaseModel):
    """Structured critique of one lineage (a single program's match history),
    distilled from its accumulated pairwise-match critiques. Produced by a
    lightweight summarizer LLM and cached on the program's `private_metrics`;
    rendered into the mutation prompt at parent-selection time.

    Stage-agnostic: the same four-field shape applies across stages
    (concepts, DAGs, voice specs, prose). Stage-specific genome formatting
    lives in the per-stage adapter, not here.
    """

    established_weaknesses: list[str] = Field(
        description="Critiques of THIS LINEAGE that recurred across multiple "
        "matches or were voiced by multiple judges. Most robust signal."
    )
    contested_strengths: list[str] = Field(
        description="Points where judges disagreed on THIS LINEAGE's merits. "
        "Exploration zones — a successor could double down or diverge."
    )
    attractor_signature: list[str] = Field(
        description="Specific patterns, motifs, or structural choices THIS "
        "LINEAGE uses that have drawn critical attention. Used as an avoid-"
        "list for successors."
    )
    divergence_directions: list[str] = Field(
        description="Concrete, prescriptive suggestions a successor could "
        "take to escape the established weaknesses. Extract from the "
        "judges' reasoning; do not invent."
    )
