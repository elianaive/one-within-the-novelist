"""Voice-panel casting classifier — three-call shape with vocabulary-driven filter.

Stage 1 (`classify_concept_features`) — concept-side classification against
the closed `casting-vocabulary.yaml`. The classifier produces a TRUE/FALSE
judgment per vocabulary tag with reasons. No persona reasoning here.

Deterministic intersection — for each persona, intersect its `starved_by`
tags with the TRUE concept-features. Non-empty intersection ⇒ starved.

Stage 2a (`argue_each_engaged`) — per-persona argument. Builds a `case_for`
/ `risks` / `cast_role` record for each engaged persona, before any
ranking. Brainstorm-before-commit per `docs/prompting-guide.md`.

Stage 2b (`select_from_arguments`) — top-4 selection. Follows up Stage 2a
in additive context (msg_history carries the prior arguments).

Public entry: `cast_voice_panel` composes all three. Failure of any LLM
stage returns None; the orchestrator decides bench-unavailable policy.

Design rationale: `lab/issues/2026-04-28-voice-panel-casting-classifier.md`.
"""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from owtn.llm.call_logger import llm_context
from owtn.llm.kwargs import sample_model_kwargs
from owtn.llm.query import query_async
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.prompts.stage_3 import (
    build_casting_argue_prompt,
    build_casting_classify_prompt,
    build_casting_select_user_msg,
    load_casting_system,
)

from .adjacent_scenes import _coerce_stringified_list
from .personas import (
    VoicePersona,
    load_casting_vocabulary,
    load_persona_pool,
    validate_pool_against_vocabulary,
)


logger = logging.getLogger(__name__)


DEFAULT_CASTER_MODEL = "claude-sonnet-4-6"
"""Per the casting decision: 'the casting call is important enough to use
a strong model.' Same default as the adjacent-scene picker; pilot may
push higher (claude-opus-4-7) if classifications drift."""

DEFAULT_REASONING_EFFORT = "medium"
"""Reasoning effort for all three casting calls. The work is selection-
under-constraint: classify 28 concept-feature tags (Stage 1), build
per-persona case + risk + cast_role (Stage 2a), pick top-N with
schema-nameable orthogonality and failure-mode checks (Stage 2b). All
three benefit from explicit reasoning. Override per-call via the
`*_kwargs` arguments to `cast_voice_panel`."""

DEFAULT_PANEL_SIZE = 4
"""Cast size — how many personas develop voice proposals per session.
Configurable via `cast_voice_panel(panel_size=...)`. Pool-signal thresholds
scale with this value: INSUFFICIENT < panel_size, NARROW < 2*panel_size,
HEALTHY otherwise."""


def _reasoning_kwargs(model_name: str, effort: str) -> dict:
    """Provider-translated kwargs for a given reasoning effort.

    Wraps `sample_model_kwargs` for deterministic single-call usage —
    the function does no sampling on single-element inputs but applies
    the provider-correct translation (Anthropic `thinking` dict + temp
    coercion to 1.0; DeepSeek `extra_body` + `reasoning_effort`; OpenAI
    `reasoning` dict; Google `thinking_budget`).

    Returns {} when effort is "disabled". The model_name slot is dropped
    from the output since the caller passes it separately to query_async.
    """
    if effort == "disabled":
        return {}
    out = sample_model_kwargs(
        model_names=[model_name],
        reasoning_efforts=[effort],
        temperatures=[1.0],
        max_tokens=[16384],
    )
    out.pop("model_name", None)
    return out


# ─── Stage 1 schema ──────────────────────────────────────────────────────


class ConceptFeatureClassification(BaseModel):
    """One vocabulary tag's TRUE/FALSE decision against the concept."""
    tag: str
    is_true: bool
    reason: str = Field(min_length=10)


class CastingClassifyOutput(BaseModel):
    """Stage 1 LLM structured output."""
    classifications: list[ConceptFeatureClassification] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        return _coerce_stringified_list(data, "classifications")


class StarvationRecord(BaseModel):
    """Computed (post-Stage-1) record naming which tags fire for a persona.

    Empty `firing_tags` means the persona is engaged. Non-empty means
    starved, with the per-tag reasons taken from the persona's
    `starved_by` field (since the persona authored those reasons).
    """
    persona_id: str
    firing_tags: list[str] = Field(default_factory=list)
    starvation_reasons: list[str] = Field(default_factory=list)

    @property
    def engaged(self) -> bool:
        return not self.firing_tags


# ─── Stage 2a schema ─────────────────────────────────────────────────────


class PersonaArgument(BaseModel):
    """One engaged persona's case for casting, written before ranking."""
    persona_id: str
    case_for: str = Field(min_length=20)
    manifestability: str = Field(min_length=20)
    risks: str = Field(min_length=10)
    cast_role: str = Field(min_length=10)


class CastingArgueOutput(BaseModel):
    """Stage 2a LLM structured output."""
    arguments: list[PersonaArgument] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        return _coerce_stringified_list(data, "arguments")


# ─── Stage 2b schema ─────────────────────────────────────────────────────


class CastingChoice(BaseModel):
    """One member of the cast — an engaged persona selected for the panel.

    The cast is a SET; iteration order in `cast` is the order the model
    surfaced them, not a quality ranking. Casting picks the panel for
    orthogonal coverage; voice quality is decided downstream by the
    panel's own cross-critique and Borda.
    """
    persona_id: str
    affordance: str = Field(min_length=10)
    coverage: str = Field(min_length=10)


class CastingSelectOutput(BaseModel):
    """Stage 2b LLM structured output.

    No schema cap on `cast` length — `panel_size` is per-call config and
    is enforced at runtime. Exclusion of engaged-but-not-cast personas
    is computed deterministically downstream, not produced by the model.
    """
    cast: list[CastingChoice]
    coverage_rationale: str = Field(min_length=20)
    style_hint_treatment: str = Field(min_length=10)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        return _coerce_stringified_list(data, "cast")


# ─── Public schema ───────────────────────────────────────────────────────


class PoolSignal(str, Enum):
    """Pool-health signal for the orchestrator to consume.

    The casting research proposed ≥13/16 starved → preflight gate fires;
    that gate was rejected as a separate step (per casting-architecture
    issue), but the underlying signal still matters here.
    """
    HEALTHY = "healthy"           # ≥ 8 engaged
    NARROW = "narrow"             # 4-7 engaged
    INSUFFICIENT = "insufficient" # < 4 engaged


class CastingOutput(BaseModel):
    """Full casting result — cast + per-stage trace + pool signal.

    `excluded_engaged` is computed deterministically as the engaged
    persona ids minus those in `cast`.
    """
    cast: list[CastingChoice]
    panel_size: int
    coverage_rationale: str
    style_hint_treatment: str
    classifications: list[ConceptFeatureClassification] = Field(default_factory=list)
    starvation_records: list[StarvationRecord] = Field(default_factory=list)
    arguments: list[PersonaArgument] = Field(default_factory=list)
    excluded_engaged: list[str] = Field(default_factory=list)
    pool_signal: PoolSignal


# ─── Stage 1 ─────────────────────────────────────────────────────────────


async def classify_concept_features(
    concept: ConceptGenome,
    dag_rendering: str,
    vocabulary: dict[str, str],
    *,
    model_name: str = DEFAULT_CASTER_MODEL,
    **llm_kwargs,
) -> list[ConceptFeatureClassification] | None:
    """Classify the concept against the closed vocabulary.

    Returns one classification per vocabulary tag, or None if the call
    or parse fails. The result is concept-side: persona engagement is
    computed downstream by deterministic intersection.
    """
    if not vocabulary:
        logger.warning("classify_concept_features called with empty vocabulary")
        return None

    system_msg, user_msg = build_casting_classify_prompt(
        concept, dag_rendering, vocabulary,
    )

    base_ctx = dict(llm_context.get({}))
    llm_context.set({**base_ctx, "role": "stage_3_casting_classify"})

    effort = llm_kwargs.pop("reasoning_effort", DEFAULT_REASONING_EFFORT)
    for k, v in _reasoning_kwargs(model_name, effort).items():
        llm_kwargs.setdefault(k, v)
    try:
        result = await query_async(
            model_name=model_name,
            msg=user_msg,
            system_msg=system_msg,
            output_model=CastingClassifyOutput,
            **llm_kwargs,
        )
    except Exception as e:
        logger.warning(
            "casting classify failed (%s: %s)", type(e).__name__, e,
        )
        return None

    parsed = result.content
    if not isinstance(parsed, CastingClassifyOutput):
        logger.warning(
            "casting classify returned unexpected content type: %s",
            type(parsed).__name__,
        )
        return None

    vocab_set = set(vocabulary.keys())
    received = {c.tag for c in parsed.classifications}
    extra = received - vocab_set
    if extra:
        logger.warning("casting classify produced unknown tags: %s", sorted(extra))
    missing = vocab_set - received
    if missing:
        logger.warning("casting classify missing tags: %s", sorted(missing))

    return [c for c in parsed.classifications if c.tag in vocab_set]


def compute_starvation_records(
    pool: list[VoicePersona],
    classifications: list[ConceptFeatureClassification],
) -> list[StarvationRecord]:
    """Deterministic intersection of persona starved_by × concept TRUE tags.

    A persona is starved when its `starved_by` set contains any tag whose
    classification is TRUE. The per-tag reasons come from the persona's
    own `starved_by` entries (not from the classifier's per-tag reasons —
    the persona authored the foreclosure semantics).
    """
    true_tags = {c.tag for c in classifications if c.is_true}
    records: list[StarvationRecord] = []
    for persona in pool:
        firing: list[str] = []
        reasons: list[str] = []
        for sb in persona.starved_by:
            if sb.tag in true_tags:
                firing.append(sb.tag)
                reasons.append(sb.reason)
        records.append(StarvationRecord(
            persona_id=persona.id,
            firing_tags=firing,
            starvation_reasons=reasons,
        ))
    return records


# ─── Stage 2a ────────────────────────────────────────────────────────────


async def argue_each_engaged(
    concept: ConceptGenome,
    dag_rendering: str,
    engaged_personas: list[VoicePersona],
    *,
    model_name: str = DEFAULT_CASTER_MODEL,
    **llm_kwargs,
) -> tuple[CastingArgueOutput, list[dict]] | None:
    """Build per-persona arguments for casting, before any ranking.

    Single LLM call producing `case_for` / `risks` / `cast_role` records
    per engaged persona. Returns the structured output AND the LLM's
    `new_msg_history` so Stage 2b can follow up in additive context.
    """
    if not engaged_personas:
        logger.warning("argue_each_engaged called with empty engaged list")
        return None

    system_msg, user_msg = build_casting_argue_prompt(
        concept, dag_rendering, engaged_personas,
    )

    base_ctx = dict(llm_context.get({}))
    llm_context.set({**base_ctx, "role": "stage_3_casting_argue"})

    effort = llm_kwargs.pop("reasoning_effort", DEFAULT_REASONING_EFFORT)
    for k, v in _reasoning_kwargs(model_name, effort).items():
        llm_kwargs.setdefault(k, v)
    try:
        result = await query_async(
            model_name=model_name,
            msg=user_msg,
            system_msg=system_msg,
            output_model=CastingArgueOutput,
            **llm_kwargs,
        )
    except Exception as e:
        logger.warning(
            "casting argue failed (%s: %s)", type(e).__name__, e,
        )
        return None

    parsed = result.content
    if not isinstance(parsed, CastingArgueOutput):
        logger.warning(
            "casting argue returned unexpected content type: %s",
            type(parsed).__name__,
        )
        return None

    engaged_ids = {p.id for p in engaged_personas}
    received_ids = {a.persona_id for a in parsed.arguments}
    missing = engaged_ids - received_ids
    if missing:
        logger.warning("casting argue missing arguments for: %s", sorted(missing))
    extra = received_ids - engaged_ids
    if extra:
        logger.warning("casting argue argued unknown persona ids: %s", sorted(extra))

    return parsed, result.new_msg_history


# ─── Stage 2b ────────────────────────────────────────────────────────────


async def select_from_arguments(
    arguments: CastingArgueOutput,
    argue_msg_history: list[dict],
    engaged_ids: set[str],
    panel_size: int,
    *,
    model_name: str = DEFAULT_CASTER_MODEL,
    **llm_kwargs,
) -> CastingSelectOutput | None:
    """Rank top-`panel_size` from per-persona arguments in additive context.

    The 2a call's `new_msg_history` is passed back as `msg_history`, so
    the model sees its own arguments when selecting. Cast-size, rank-range,
    and engaged-subset membership are validated at runtime (not in schema).
    """
    if not arguments.arguments:
        logger.warning("select_from_arguments called with empty arguments")
        return None

    user_msg = build_casting_select_user_msg(panel_size)

    effort = llm_kwargs.pop("reasoning_effort", DEFAULT_REASONING_EFFORT)
    for k, v in _reasoning_kwargs(model_name, effort).items():
        llm_kwargs.setdefault(k, v)

    base_ctx = dict(llm_context.get({}))
    role_token = llm_context.set({**base_ctx, "role": "stage_3_casting_select"})
    try:
        # Same system_msg as Stage 2a so the casting role context is
        # present (providers send `system` separately from message history;
        # `new_msg_history` carries only user/assistant turns). On Anthropic
        # this also lets the cache prefix from Stage 2a hit.
        result = await query_async(
            model_name=model_name,
            msg=user_msg,
            system_msg=load_casting_system(),
            msg_history=argue_msg_history,
            output_model=CastingSelectOutput,
            **llm_kwargs,
        )
    except Exception as e:
        logger.warning(
            "casting select failed (%s: %s)", type(e).__name__, e,
        )
        return None
    finally:
        llm_context.reset(role_token)

    parsed = result.content
    if not isinstance(parsed, CastingSelectOutput):
        logger.warning(
            "casting select returned unexpected content type: %s",
            type(parsed).__name__,
        )
        return None

    cast_ids = [c.persona_id for c in parsed.cast]
    if len(cast_ids) != len(set(cast_ids)):
        logger.warning("casting select produced duplicate persona ids: %s", cast_ids)
        return None
    if len(parsed.cast) > panel_size:
        logger.warning(
            "casting select returned %d picks; panel_size is %d",
            len(parsed.cast), panel_size,
        )
        return None
    unknown = [pid for pid in cast_ids if pid not in engaged_ids]
    if unknown:
        logger.warning(
            "casting select picked persona(s) outside engaged subset: %s", unknown,
        )
        return None

    return parsed


# ─── Public entry ────────────────────────────────────────────────────────


def _classify_pool_signal(n_engaged: int, panel_size: int) -> PoolSignal:
    """Pool-health signal scaled to panel size.

    INSUFFICIENT if engaged < panel_size (cast cannot fill).
    NARROW if engaged < 2 * panel_size (cast possible but low search slack).
    HEALTHY otherwise.
    """
    if n_engaged < panel_size:
        return PoolSignal.INSUFFICIENT
    if n_engaged < 2 * panel_size:
        return PoolSignal.NARROW
    return PoolSignal.HEALTHY


async def cast_voice_panel(
    concept: ConceptGenome,
    dag_rendering: str,
    *,
    panel_size: int = DEFAULT_PANEL_SIZE,
    pool: list[VoicePersona] | None = None,
    pool_dir: str | None = None,
    vocabulary: dict[str, str] | None = None,
    model_name: str = DEFAULT_CASTER_MODEL,
    classify_kwargs: dict | None = None,
    argue_kwargs: dict | None = None,
    select_kwargs: dict | None = None,
) -> CastingOutput | None:
    """Three-call cast for one (concept, structure) pair.

    Runs concept-side classification (Stage 1), deterministic intersection
    with each persona's starved_by tags, per-persona arguments (Stage 2a),
    and top-`panel_size` selection (Stage 2b). The model produces only the
    cast; engaged-but-not-cast persona ids are computed deterministically.

    Returns None when any LLM stage fails or when pool/vocabulary are
    empty. Returns a `CastingOutput` with `pool_signal=INSUFFICIENT` and
    a short or empty cast when fewer than `panel_size` personas engage.
    """
    classify_kwargs = classify_kwargs or {}
    argue_kwargs = argue_kwargs or {}
    select_kwargs = select_kwargs or {}

    if panel_size < 1:
        raise ValueError(f"panel_size must be >= 1; got {panel_size}")

    if pool is None:
        pool = load_persona_pool(pool_dir)
    if not pool:
        logger.warning("cast_voice_panel: empty persona pool; cannot cast")
        return None

    if vocabulary is None:
        vocabulary = load_casting_vocabulary()
    if not vocabulary:
        logger.warning("cast_voice_panel: empty casting vocabulary; cannot cast")
        return None

    vocab_errors = validate_pool_against_vocabulary(pool, vocabulary)
    if vocab_errors:
        for err in vocab_errors:
            logger.warning("pool/vocabulary mismatch: %s", err)

    classifications = await classify_concept_features(
        concept, dag_rendering, vocabulary,
        model_name=model_name, **classify_kwargs,
    )
    if classifications is None:
        return None

    starvation_records = compute_starvation_records(pool, classifications)
    by_id = {p.id: p for p in pool}
    engaged_personas = [
        by_id[r.persona_id] for r in starvation_records if r.engaged
    ]
    pool_signal = _classify_pool_signal(len(engaged_personas), panel_size)

    if not engaged_personas:
        logger.warning("cast_voice_panel: zero personas engaged; bailing")
        return CastingOutput(
            cast=[],
            panel_size=panel_size,
            coverage_rationale="No personas engaged; voice search cannot proceed.",
            style_hint_treatment="Not evaluated; pool exhausted by starvation filter.",
            classifications=classifications,
            starvation_records=starvation_records,
            pool_signal=pool_signal,
        )

    argue_result = await argue_each_engaged(
        concept, dag_rendering, engaged_personas,
        model_name=model_name, **argue_kwargs,
    )
    if argue_result is None:
        return None
    arguments, argue_msg_history = argue_result

    engaged_ids = {p.id for p in engaged_personas}
    select_result = await select_from_arguments(
        arguments, argue_msg_history, engaged_ids, panel_size,
        model_name=model_name, **select_kwargs,
    )
    if select_result is None:
        return None

    cast_ids = {c.persona_id for c in select_result.cast}
    excluded_engaged = sorted(engaged_ids - cast_ids)

    return CastingOutput(
        cast=select_result.cast,
        panel_size=panel_size,
        coverage_rationale=select_result.coverage_rationale,
        style_hint_treatment=select_result.style_hint_treatment,
        classifications=classifications,
        starvation_records=starvation_records,
        arguments=list(arguments.arguments),
        excluded_engaged=excluded_engaged,
        pool_signal=pool_signal,
    )
