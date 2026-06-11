"""Tier 3: concept-demand fidelity verdict.

Per `docs/stage-2/evaluation.md` §Tier 3, every preset's final terminal DAG
is checked against its `concept_demands` list — predicates the structure
must realize that escape the 8-dimension pairwise rubric (reader-address,
form-as-device, structural rhyme, deliberate irresolution). One classifier-
model LLM call per terminal verdicts each demand as
`satisfied | partial | failed` with a one-sentence rationale.

Used by:
- `owtn.stage_2.orchestration.run_concept` — fires once per `TournamentEntry`
  after MCTS produces it; populates `entry.concept_demand_failed` and
  `entry.concept_demand_verdicts` so the within-concept ranker can demote
  failed entries below all-satisfied entries (both pairwise and scalar
  paths apply this priority).
- The handoff manifest — verdicts persist on `Stage2Output.concept_demand_results`
  so Stage 3 can compensate for `partial` verdicts at voice / prose time.

Failure handling. Empty `concept_demands` → skip without warning (the common
case; most concepts' mechanisms are fully schema-expressible). Classifier
model unset → skip with warning. LLM call raises or returns mismatched-
length verdicts → log, treat as `failed=False` (don't punish the DAG for
an extraction failure that wasn't its fault).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from owtn.llm.call_logger import llm_context
from owtn.llm.query import query_async
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG
from owtn.models.stage_2.handoff import ConceptDemandVerdict
from owtn.prompts.stage_2.registry import build_tier3_prompt
from owtn.stage_2.rendering import render


logger = logging.getLogger(__name__)


class _Tier3LLMResponse(BaseModel):
    """Structured-output schema for the LLM call."""
    verdicts: list[ConceptDemandVerdict] = Field(default_factory=list)


@dataclass
class Tier3Result:
    """Outcome of one Tier 3 evaluation.

    Fields:
        verdicts: per-demand verdicts in input order. Empty when skipped.
        failed: True iff at least one verdict is `failed`. Drives the
            tournament priority gate.
        skipped_reason: present when Tier 3 was skipped (no demands, no
            classifier model, or LLM failure). None when verdicts were
            actually obtained.
        cost: total LLM cost.
    """
    verdicts: list[ConceptDemandVerdict] = field(default_factory=list)
    failed: bool = False
    skipped_reason: str | None = None
    cost: float = 0.0


async def evaluate_concept_demands(
    dag: DAG,
    *,
    concept: ConceptGenome,
    classifier_model: str | None,
) -> Tier3Result:
    """Verdict each demand on `dag.concept_demands` against the rendered DAG.

    Returns `Tier3Result` with `failed=True` iff any verdict is `failed`.
    Empty demand list short-circuits to a clean skip (no LLM call). Missing
    classifier model logs a warning and returns a clean skip.

    LLM call is single-shot, structured output. On any failure (provider
    error, mismatched verdict count, malformed shape), logs and returns
    a skipped result with `failed=False` — extraction failures must not
    punish the DAG.
    """
    demands = list(dag.concept_demands or [])
    if not demands:
        return Tier3Result(skipped_reason="no_concept_demands")
    if classifier_model is None:
        logger.warning(
            "Tier 3 skipped: classifier_model not configured (concept has %d demand(s))",
            len(demands),
        )
        return Tier3Result(skipped_reason="no_classifier_model")

    system_msg, user_msg = build_tier3_prompt(
        concept,
        render(dag),
        demands,
    )

    base_ctx = dict(llm_context.get({}))
    llm_context.set({**base_ctx, "role": "stage2_tier3"})
    try:
        result = await query_async(
            model_name=classifier_model,
            msg=user_msg,
            system_msg=system_msg,
            output_model=_Tier3LLMResponse,
        )
    except Exception as e:  # noqa: BLE001 — never crash a preset's handoff
        logger.warning(
            "Tier 3 LLM call failed (%s: %s); skipping demand verdicts",
            type(e).__name__, e,
        )
        return Tier3Result(skipped_reason=f"llm_error:{type(e).__name__}")

    response = result.content
    if not isinstance(response, _Tier3LLMResponse):
        logger.warning(
            "Tier 3 returned unexpected content type %s; skipping demand verdicts",
            type(response).__name__,
        )
        return Tier3Result(skipped_reason="malformed_response", cost=result.cost)

    if len(response.verdicts) != len(demands):
        logger.warning(
            "Tier 3 verdict count mismatch (got %d, expected %d); skipping",
            len(response.verdicts), len(demands),
        )
        return Tier3Result(skipped_reason="verdict_count_mismatch", cost=result.cost)

    failed = any(v.verdict == "failed" for v in response.verdicts)
    logger.info(
        "Tier 3: %d demand(s), failed=%s, verdicts=%s",
        len(demands), failed,
        ", ".join(f"{v.verdict}" for v in response.verdicts),
    )
    return Tier3Result(
        verdicts=list(response.verdicts),
        failed=failed,
        skipped_reason=None,
        cost=result.cost,
    )
