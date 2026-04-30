"""Adjacent-scene test bench — picker + neutral-voice drafter.

Per (concept, structure) pair, the bench is generated once at session
start: a smart-LLM-with-reasoning call picks three AU scenes that place
distinct demands on voice for *this* story, and a follow-up call drafts
each in flat default register. The drafts are the baseline that the four
voice agents transform in Phase 1; the articulated demands travel with
each scene so judges and the diversity monitor can inspect the bench's
coverage post-hoc.

Design rationale: `lab/issues/2026-04-28-adjacent-scene-picker.md` and
`docs/stage-3/overview.md` §"Adjacent Scenes".
"""

from __future__ import annotations

import asyncio
import json
import logging

from pydantic import BaseModel, Field, model_validator

from owtn.llm.call_logger import llm_context
from owtn.llm.query import query_async
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.prompts.stage_3 import (
    build_adjacent_scene_drafter_prompt,
    build_adjacent_scene_picker_prompt,
)


logger = logging.getLogger(__name__)


DEFAULT_PICKER_MODEL = "claude-sonnet-4-6"
"""Default picker model — needs reasoning effort to evaluate cross-demand
distinctiveness, not just generate three tonally-varied scenes."""

DEFAULT_DRAFTER_MODEL = "claude-sonnet-4-6"
"""Default drafter model. Same family as picker for consistency; the
drafter's job is denotative and doesn't itself need reasoning effort."""

DRAFT_WORD_CAP = 400
"""Soft cap on neutral-voice draft length (overview specifies ~150-300
words; cap allows headroom without inviting paragraph-long drafts)."""


class AdjacentScenePick(BaseModel):
    """One AU scene the picker proposed.

    `demand` is what voice work this scene stresses, articulated by the
    picker in its own taxonomy. `why_distinct` is the cross-demand
    justification — one sentence on what this scene tests that the other
    two don't.
    """
    scene_id: str = Field(min_length=2, max_length=80)
    synopsis: str = Field(min_length=20)
    demand: str = Field(min_length=10)
    why_distinct: str = Field(min_length=10)


def _coerce_stringified_list(data, list_field: str):
    # Sonnet 4.6 occasionally serializes list-of-objects fields as a
    # JSON-encoded string under forced-tool-use; recover it so a stochastic
    # blip doesn't burn an Anthropic call. Handles both the list-only case
    # and the case where the model concatenates everything into one string.
    # `strict=False` accepts unescaped newlines inside string values — the
    # model's stringified payload commonly contains literal newlines.
    if not isinstance(data, dict):
        return data
    val = data.get(list_field)
    if isinstance(val, str):
        try:
            parsed = json.loads(val, strict=False)
        except json.JSONDecodeError:
            return data
        if isinstance(parsed, list):
            return {**data, list_field: parsed}
        if isinstance(parsed, dict):
            return {**parsed, **{k: v for k, v in data.items() if k != list_field}}
    return data


class AdjacentScenePickerOutput(BaseModel):
    """Structured output from the picker call."""
    scenes: list[AdjacentScenePick] = Field(min_length=3, max_length=3)
    bench_rationale: str = Field(min_length=20)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        return _coerce_stringified_list(data, "scenes")


class AdjacentSceneDraft(BaseModel):
    """A picked scene plus its neutral-voice draft."""
    scene_id: str
    synopsis: str
    demand: str
    why_distinct: str
    neutral_draft: str


class AdjacentSceneBench(BaseModel):
    """The full test bench shared across voice agents.

    Cached at session start; `neutral_draft` is the baseline against which
    each voice agent's transformation is measured.
    """
    drafts: list[AdjacentSceneDraft] = Field(min_length=3, max_length=3)
    bench_rationale: str
    picker_model: str
    drafter_model: str


PICKER_MAX_ATTEMPTS = 3
"""Retry cap for the picker call. Structured-output failures are
occasionally one-shot stochastic errors (model emits a JSON-looking
string in a list field); rebuilding the prompt unchanged on a fresh
call usually succeeds."""


async def _pick_scenes(
    concept: ConceptGenome,
    dag_rendering: str,
    *,
    model_name: str,
    **llm_kwargs,
) -> AdjacentScenePickerOutput | None:
    system_msg, user_msg = build_adjacent_scene_picker_prompt(concept, dag_rendering)

    base_ctx = dict(llm_context.get({}))
    llm_context.set({**base_ctx, "role": "stage_3_adjacent_scene_picker"})

    last_error: str = ""
    for attempt in range(1, PICKER_MAX_ATTEMPTS + 1):
        try:
            result = await query_async(
                model_name=model_name,
                msg=user_msg,
                system_msg=system_msg,
                output_model=AdjacentScenePickerOutput,
                **llm_kwargs,
            )
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            logger.warning(
                "adjacent-scene picker attempt %d/%d failed (%s)",
                attempt, PICKER_MAX_ATTEMPTS, last_error,
            )
            continue

        parsed = result.content
        if not isinstance(parsed, AdjacentScenePickerOutput):
            last_error = f"unexpected content type: {type(parsed).__name__}"
            logger.warning(
                "adjacent-scene picker attempt %d/%d returned %s",
                attempt, PICKER_MAX_ATTEMPTS, last_error,
            )
            continue

        seen: set[str] = set()
        duplicate = False
        for pick in parsed.scenes:
            if pick.scene_id in seen:
                logger.warning(
                    "adjacent-scene picker attempt %d/%d duplicate scene_id %r",
                    attempt, PICKER_MAX_ATTEMPTS, pick.scene_id,
                )
                duplicate = True
                break
            seen.add(pick.scene_id)
        if duplicate:
            last_error = "duplicate scene_id"
            continue

        if attempt > 1:
            logger.info("adjacent-scene picker recovered on attempt %d", attempt)
        return parsed

    logger.warning(
        "adjacent-scene picker exhausted %d attempts; no bench produced. last error: %s",
        PICKER_MAX_ATTEMPTS, last_error,
    )
    return None


async def _draft_one(
    concept: ConceptGenome,
    pick: AdjacentScenePick,
    *,
    model_name: str,
    **llm_kwargs,
) -> str | None:
    system_msg, user_msg = build_adjacent_scene_drafter_prompt(
        concept,
        scene_id=pick.scene_id,
        synopsis=pick.synopsis,
    )

    base_ctx = dict(llm_context.get({}))
    llm_context.set({
        **base_ctx,
        "role": "stage_3_adjacent_scene_drafter",
        "scene_id": pick.scene_id,
    })
    try:
        result = await query_async(
            model_name=model_name,
            msg=user_msg,
            system_msg=system_msg,
            **llm_kwargs,
        )
    except Exception as e:
        logger.warning(
            "neutral-voice drafter failed for scene %r (%s: %s).",
            pick.scene_id, type(e).__name__, e,
        )
        return None

    text = (result.content or "").strip()
    if not text:
        logger.warning("drafter returned empty content for scene %r.", pick.scene_id)
        return None
    return text


async def generate_adjacent_scenes(
    concept: ConceptGenome,
    dag_rendering: str,
    *,
    picker_model: str = DEFAULT_PICKER_MODEL,
    drafter_model: str = DEFAULT_DRAFTER_MODEL,
    picker_kwargs: dict | None = None,
    drafter_kwargs: dict | None = None,
) -> AdjacentSceneBench | None:
    """Generate the three-scene test bench for one (concept, structure) pair.

    Two-call flow: picker proposes 3 scenes + articulated demands, then
    drafter renders each in neutral voice (concurrently, one call per scene
    so register-bleed between drafts is avoided).

    Returns None on any picker or drafter failure; orchestrator treats
    this as "bench unavailable" and the caller decides whether to retry
    or skip the (concept, structure) pair.
    """
    picker_kwargs = picker_kwargs or {}
    drafter_kwargs = drafter_kwargs or {}
    # The picker is a 3-pass reasoning task (brainstorm, pick, voice-latitude
    # check). The anthropic provider translates reasoning_effort → thinking
    # and switches forced tool use to tool_choice=auto so thinking coexists
    # with structured output.
    picker_kwargs.setdefault("reasoning_effort", "medium")

    picker_output = await _pick_scenes(
        concept,
        dag_rendering,
        model_name=picker_model,
        **picker_kwargs,
    )
    if picker_output is None:
        return None

    draft_tasks = [
        _draft_one(concept, pick, model_name=drafter_model, **drafter_kwargs)
        for pick in picker_output.scenes
    ]
    drafts = await asyncio.gather(*draft_tasks)

    if any(d is None for d in drafts):
        logger.warning("at least one draft failed; bench is incomplete.")
        return None

    return AdjacentSceneBench(
        drafts=[
            AdjacentSceneDraft(
                scene_id=pick.scene_id,
                synopsis=pick.synopsis,
                demand=pick.demand,
                why_distinct=pick.why_distinct,
                neutral_draft=draft,
            )
            for pick, draft in zip(picker_output.scenes, drafts)
        ],
        bench_rationale=picker_output.bench_rationale,
        picker_model=picker_model,
        drafter_model=drafter_model,
    )
