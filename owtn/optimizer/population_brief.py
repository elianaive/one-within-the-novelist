"""Generic population-level summarizer.

Consumes already-distilled per-lineage `LineageBrief` briefs across the whole
run and produces a `PopulationBrief` — run-wide attractor shapes, per-judge
taste signals, drift diagnosis, and positively framed exploration directions.
Injected into mutation prompts alongside the lineage brief.

Framing choices are load-bearing: exploration_directions is positively framed
("find Y instead") rather than prohibitive ("do not produce X"), and attractor
descriptions deliberately avoid naming specific exemplars — both follow from
research showing that negative instructions prime the forbidden pattern and
that naming attractors causes priming failure. See
`lab/deep-research/runs/20260424_031609-avoidance-instruction-compliance/`.

See `lab/issues/2026-04-22-global-optimizer-state.md`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from owtn.llm.query import query_async

logger = logging.getLogger(__name__)

_OPTIMIZER_DIR = Path(__file__).resolve().parent


class PopulationBrief(BaseModel):
    """Run-wide signal distilled from per-lineage briefs across all lineages.

    Paired with (not a replacement for) `LineageBrief`: the lineage brief
    addresses one parent's attractor, the population brief addresses
    cross-lineage attractor shapes and per-judge taste the whole panel has
    been expressing. See parent issue for design rationale.
    """

    population_attractors: list[str] = Field(
        description="Abstract structural SHAPES of recurring patterns across "
        "multiple lineages — describe the structural function (e.g. 'single-"
        "system paradox where stakes derive entirely from one mechanism's "
        "properties') rather than naming the specific content exemplar (e.g. "
        "NOT 'transformer inference'). Naming the specific exemplar primes "
        "successors toward it; describing the shape does not."
    )
    per_judge_signals: list[str] = Field(
        description="Per-judge consistent rewards/penalties across matches, "
        "with judge attribution preserved (e.g. 'Gwern rewards lineages where "
        "X'). Captures panel taste the mutator should write toward."
    )
    population_drift: list[str] = Field(
        description="Diagnosis of what overall shape the population is "
        "collapsing toward right now. Not prescriptive — just the shape."
    )
    exploration_directions: list[str] = Field(
        description="MAX 3 items. Each is a POSITIVE direction + concrete "
        "counter-example. Format: '[Positive framing of what to explore — "
        "stated as what TO do, not what to avoid]. [One-sentence concrete "
        "counter-example concept that exemplifies this direction.]'. Ground "
        "each item in one specific population_attractor or population_drift "
        "observation above. Never phrase as 'do not' or 'avoid'."
    )


def _load_population_prompt_template() -> str:
    return (_OPTIMIZER_DIR / "population_prompt.txt").read_text()


class PopulationBriefSummarizer:
    """Stage-agnostic driver for the population-level summarizer.

    The caller (a per-stage adapter) fills the prompt placeholders and
    formats the list of per-lineage briefs into the user message. This class
    just dispatches the LLM call with the right output model.
    """

    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name

    async def summarize(
        self, *, system_prompt: str, user_msg: str
    ) -> PopulationBrief:
        result = await query_async(
            model_name=self.model_name,
            msg=user_msg,
            system_msg=system_prompt,
            output_model=PopulationBrief,
        )
        return result.content


def _bullets(items: list[str]) -> str:
    if not items:
        return "- (none identified)"
    return "\n".join(f"- {x}" for x in items)


def render_population_context(brief: PopulationBrief) -> str:
    """Diagnostic context block — goes in the middle of the mutation prompt.

    Describes what the run has been producing (attractor shapes, per-judge
    taste, drift) without naming specific content exemplars that would prime
    successors. The actionable "what to do" lives in `exploration_directions`
    and is rendered separately for the instruction-sandwich placement.
    """
    return (
        "## Attractor shapes (structural patterns, not named content)\n"
        f"{_bullets(brief.population_attractors)}\n\n"
        "## Per-judge signal (what each panel member has been rewarding/penalizing)\n"
        f"{_bullets(brief.per_judge_signals)}\n\n"
        "## Population drift (what shape the population is collapsing toward)\n"
        f"{_bullets(brief.population_drift)}"
    )


def render_exploration_directions(brief: PopulationBrief) -> str:
    """Actionable block — goes at both the start AND end of the user message
    (instruction sandwich). Positively framed; each item is a concrete
    counter-example to a specific population attractor or drift observation.
    """
    return (
        "## Exploration directions for this mutation\n"
        f"{_bullets(brief.exploration_directions)}"
    )
