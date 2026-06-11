"""Pre-stage filter — audience framing + expert needs.

Two cheap haiku-class structured-output calls fire at session setup,
in parallel. Their outputs populate the writer agent's system prompt
(audience) and instantiate the dynamic `domain_expert` critic factory
(experts; consumed when that factory lands). Mirrors `casting.py`'s
classifier shape — read the YAML inputs, call the LLM, parse the typed
output, return.
"""

from __future__ import annotations

import asyncio
import logging

from owtn.llm.call_logger import llm_context
from owtn.llm.kwargs import sample_model_kwargs
from owtn.llm.query import query_async
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_3 import VoiceGenome
from owtn.models.stage_4 import (
    AudienceFraming,
    ExpertNeedsList,
    Stage4FilterConfig,
)
from owtn.prompts.stage_4 import _load


logger = logging.getLogger(__name__)


# ─── Prompt rendering ────────────────────────────────────────────────────


def _concept_block(concept: ConceptGenome) -> str:
    parts = [f"Premise: {concept.premise.strip()}"]
    if concept.target_effect:
        parts.append(f"Target effect: {concept.target_effect.strip()}")
    if concept.thematic_engine:
        parts.append(f"Thematic engine: {concept.thematic_engine.strip()}")
    if concept.character_seeds:
        parts.append("")
        parts.append("Characters:")
        for cs in concept.character_seeds:
            label = cs.label if hasattr(cs, "label") else cs.get("label", "")
            sketch = cs.sketch if hasattr(cs, "sketch") else cs.get("sketch", "")
            parts.append(f"- {label}: {sketch}")
    if concept.setting_seeds:
        parts.append("")
        parts.append(f"Setting: {concept.setting_seeds.strip()}")
    if concept.constraints:
        parts.append("")
        parts.append("Constraints:")
        for c in concept.constraints:
            parts.append(f"- {c}")
    return "\n".join(parts)


def _voice_block(voice: VoiceGenome) -> str:
    parts = [
        f"POV: {voice.pov}, tense: {voice.tense}",
        "",
        "Description:",
        voice.description.strip(),
        "",
        "Diction:",
        voice.diction.strip(),
    ]
    if voice.positive_constraints:
        parts.append("")
        parts.append("Positive constraints:")
        for c in voice.positive_constraints:
            parts.append(f"- {c}")
    return "\n".join(parts)


def _build_audience_prompt(concept: ConceptGenome, voice: VoiceGenome) -> str:
    return _load("filter_audience.txt").replace(
        "{CONCEPT}", _concept_block(concept),
    ).replace(
        "{VOICE}", _voice_block(voice),
    )


def _build_experts_prompt(
    concept: ConceptGenome, voice: VoiceGenome, dag_rendering: str,
) -> str:
    return _load("filter_experts.txt").replace(
        "{CONCEPT}", _concept_block(concept),
    ).replace(
        "{DAG}", dag_rendering.strip(),
    ).replace(
        "{VOICE}", _voice_block(voice),
    )


def _reasoning_kwargs(model_name: str, effort: str) -> dict:
    if effort == "disabled":
        return {}
    out = sample_model_kwargs(
        model_names=[model_name],
        reasoning_efforts=[effort],
        temperatures=[0.6],
        max_tokens=[8192],
    )
    out.pop("model_name", None)
    return out


# ─── Sub-calls ───────────────────────────────────────────────────────────


async def _classify_audience(
    concept: ConceptGenome,
    voice: VoiceGenome,
    cfg: Stage4FilterConfig,
) -> AudienceFraming | None:
    user_msg = _build_audience_prompt(concept, voice)
    base_ctx = dict(llm_context.get({}))
    token = llm_context.set({**base_ctx, "role": "stage_4_filter_audience"})
    try:
        result = await query_async(
            model_name=cfg.audience_model,
            msg=user_msg,
            system_msg=(
                "You are a literary editor positioning a short story for "
                "publication. Your job is to identify the implied audience "
                "the work assumes."
            ),
            output_model=AudienceFraming,
            **_reasoning_kwargs(cfg.audience_model, cfg.reasoning_effort),
        )
    except Exception as e:
        logger.warning("filter_audience failed (%s: %s)", type(e).__name__, e)
        return None
    finally:
        llm_context.reset(token)
    parsed = result.content
    if not isinstance(parsed, AudienceFraming):
        logger.warning("filter_audience returned %s, expected AudienceFraming", type(parsed).__name__)
        return None
    return parsed


async def _classify_experts(
    concept: ConceptGenome,
    voice: VoiceGenome,
    dag_rendering: str,
    cfg: Stage4FilterConfig,
) -> ExpertNeedsList | None:
    user_msg = _build_experts_prompt(concept, voice, dag_rendering)
    base_ctx = dict(llm_context.get({}))
    token = llm_context.set({**base_ctx, "role": "stage_4_filter_experts"})
    try:
        result = await query_async(
            model_name=cfg.experts_model,
            msg=user_msg,
            system_msg=(
                "You are a developmental editor reading a short story to "
                "identify whether it demands specific domain expertise to "
                "land convincingly."
            ),
            output_model=ExpertNeedsList,
            **_reasoning_kwargs(cfg.experts_model, cfg.reasoning_effort),
        )
    except Exception as e:
        logger.warning("filter_experts failed (%s: %s)", type(e).__name__, e)
        return None
    finally:
        llm_context.reset(token)
    parsed = result.content
    if not isinstance(parsed, ExpertNeedsList):
        logger.warning("filter_experts returned %s, expected ExpertNeedsList", type(parsed).__name__)
        return None
    return parsed


# ─── Public entry ────────────────────────────────────────────────────────


async def run_stage_4_filter(
    *,
    concept: ConceptGenome,
    voice_genome: VoiceGenome,
    dag_rendering: str,
    config: Stage4FilterConfig | None = None,
) -> tuple[AudienceFraming | None, ExpertNeedsList]:
    """Two parallel haiku calls — one audience, one expert-needs.

    Returns the AudienceFraming (None if the call failed; caller decides
    whether to proceed) and an ExpertNeedsList (always non-None — empty
    `experts` when no domain expertise is demanded; defaults to empty
    on call failure so the session can continue without expert critics).
    """
    cfg = config or Stage4FilterConfig()
    audience, experts = await asyncio.gather(
        _classify_audience(concept, voice_genome, cfg),
        _classify_experts(concept, voice_genome, dag_rendering, cfg),
    )
    if experts is None:
        experts = ExpertNeedsList()
    return audience, experts
