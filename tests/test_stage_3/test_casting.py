"""Tests for owtn.stage_3.casting — three-call casting classifier.

Offline tests cover Pydantic schema, the deterministic intersection step,
prompt assembly, vocabulary validation, and stage-failure paths via mocked
LLM responses. Live-API smoke tests are gated `live_api`.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.prompts.stage_3 import (
    build_casting_argue_prompt,
    build_casting_classify_prompt,
    build_casting_select_user_msg,
    load_casting_system,
)
from owtn.stage_3.casting import (
    CastingArgueOutput,
    CastingChoice,
    CastingClassifyOutput,
    CastingOutput,
    CastingSelectOutput,
    ConceptFeatureClassification,
    PersonaArgument,
    PoolSignal,
    StarvationRecord,
    cast_voice_panel,
    compute_starvation_records,
)
from owtn.stage_3.personas import (
    StarvationPattern,
    VoicePersona,
    load_casting_vocabulary,
    load_persona_pool,
    validate_pool_against_vocabulary,
)
from tests.conftest import HILLS_GENOME


# ─── Persona / vocabulary loaders ────────────────────────────────────────


def test_load_persona_pool_picks_up_persona_files_only():
    pool = load_persona_pool()
    assert len(pool) >= 4, "expected at least the 4 original personas"
    ids = {p.id for p in pool}
    assert "the-reductionist" in ids
    assert "the-temporal-collagist" in ids
    # vocabulary file is not a persona; loader must skip it
    assert all(p.id.startswith("the-") for p in pool)


def test_load_casting_vocabulary_returns_tag_meaning_map():
    vocab = load_casting_vocabulary()
    assert len(vocab) >= 20
    assert "concept_lacks_human_characters" in vocab
    assert "concept_requires_lyric_register" in vocab
    assert all(isinstance(v, str) for v in vocab.values())


def test_pool_starved_by_tags_all_in_vocabulary():
    """Each persona's starved_by tags must reference the closed vocabulary."""
    pool = load_persona_pool()
    vocab = load_casting_vocabulary()
    errors = validate_pool_against_vocabulary(pool, vocab)
    assert errors == [], f"pool/vocabulary mismatch: {errors}"


def test_validate_catches_unknown_tag():
    persona = VoicePersona(
        id="test-persona",
        name="Test",
        identity="A test persona for validation. " * 3,
        aesthetic_commitments=["test commitment"],
        epistemic_skepticism="test skepticism " * 3,
        starved_by=[
            StarvationPattern(tag="concept_made_up_tag", reason="not a real tag"),
        ],
    )
    errors = validate_pool_against_vocabulary([persona], {"concept_real_tag": "x"})
    assert len(errors) == 1
    assert "concept_made_up_tag" in errors[0]


# ─── Deterministic intersection ──────────────────────────────────────────


def _classification(tag: str, is_true: bool) -> ConceptFeatureClassification:
    return ConceptFeatureClassification(
        tag=tag, is_true=is_true,
        reason="test reason long enough to pass validation",
    )


def test_compute_starvation_records_engages_when_no_tags_fire():
    persona = VoicePersona(
        id="p1", name="P1",
        identity="x" * 50,
        aesthetic_commitments=["commit"],
        epistemic_skepticism="x" * 30,
        starved_by=[StarvationPattern(tag="concept_x", reason="reason text")],
    )
    classifications = [_classification("concept_x", False)]
    records = compute_starvation_records([persona], classifications)
    assert len(records) == 1
    assert records[0].engaged is True
    assert records[0].firing_tags == []


def test_compute_starvation_records_starves_when_a_tag_fires():
    persona = VoicePersona(
        id="p1", name="P1",
        identity="x" * 50,
        aesthetic_commitments=["commit"],
        epistemic_skepticism="x" * 30,
        starved_by=[
            StarvationPattern(tag="concept_x", reason="reason for x"),
            StarvationPattern(tag="concept_y", reason="reason for y"),
        ],
    )
    classifications = [
        _classification("concept_x", True),
        _classification("concept_y", False),
    ]
    records = compute_starvation_records([persona], classifications)
    assert records[0].engaged is False
    assert records[0].firing_tags == ["concept_x"]
    assert records[0].starvation_reasons == ["reason for x"]


def test_compute_starvation_records_handles_multi_tag_firing():
    persona = VoicePersona(
        id="p1", name="P1",
        identity="x" * 50,
        aesthetic_commitments=["commit"],
        epistemic_skepticism="x" * 30,
        starved_by=[
            StarvationPattern(tag="concept_x", reason="reason for x"),
            StarvationPattern(tag="concept_y", reason="reason for y"),
        ],
    )
    classifications = [
        _classification("concept_x", True),
        _classification("concept_y", True),
    ]
    records = compute_starvation_records([persona], classifications)
    assert records[0].firing_tags == ["concept_x", "concept_y"]
    assert len(records[0].starvation_reasons) == 2


def test_compute_starvation_records_with_real_pool_and_asylum_features():
    """Asylum-testimony failure case: the persona pilot showed temporal-collagist
    + sensory-materialist + reductionist starving. Verify the intersection
    catches at least temporal-collagist on its concept_forbids_interiority
    tag (which is a documented asylum-testimony feature)."""
    pool = load_persona_pool()
    vocab = load_casting_vocabulary()
    true_tags = {"concept_forbids_interiority", "concept_lacks_human_characters"}
    classifications = [
        _classification(t, t in true_tags) for t in vocab
    ]
    records = compute_starvation_records(pool, classifications)
    by_id = {r.persona_id: r for r in records}
    if "the-temporal-collagist" in by_id:
        assert not by_id["the-temporal-collagist"].engaged
        assert "concept_forbids_interiority" in by_id["the-temporal-collagist"].firing_tags


# ─── Prompt assembly ─────────────────────────────────────────────────────


def test_classify_prompt_substitutes_vocabulary_and_concept():
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    vocab = {
        "concept_lacks_human_characters": "Concept has no people with interior life.",
        "concept_requires_lyric_register": "Concept demands sustained lyric prose.",
    }
    system_msg, user_msg = build_casting_classify_prompt(concept, "NODE n0\n", vocab)

    assert "voice stage" in system_msg.lower() or "casting" in system_msg.lower()
    assert "{CONCEPT_JSON}" not in user_msg
    assert "{DAG_RENDERING}" not in user_msg
    assert "{VOCABULARY_BLOCK}" not in user_msg
    assert "concept_lacks_human_characters" in user_msg
    assert "concept_requires_lyric_register" in user_msg
    assert "train station" in user_msg


def test_argue_prompt_renders_engaged_personas():
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    pool = load_persona_pool()
    engaged = pool[:2]
    system_msg, user_msg = build_casting_argue_prompt(concept, "NODE n0\n", engaged)

    assert "{CONCEPT_JSON}" not in user_msg
    assert "{DAG_RENDERING}" not in user_msg
    assert "{ENGAGED_BLOCK}" not in user_msg
    for p in engaged:
        assert p.id in user_msg
        assert p.aesthetic_commitments[0][:30] in user_msg


def test_select_user_msg_is_followup_continuation():
    msg = build_casting_select_user_msg(4)
    # PANEL_SIZE substituted; no other template vars remain
    assert "{PANEL_SIZE}" not in msg
    assert "pick" in msg.lower()


def test_casting_system_is_workshop_register():
    system_msg = load_casting_system()
    assert "casting" in system_msg.lower()
    # Anti-assistant register check
    first = system_msg.split(".")[0].lower()
    assert "happy to help" not in first
    assert "as an assistant" not in first


# ─── Schema ──────────────────────────────────────────────────────────────


def test_casting_choice_requires_substantive_fields():
    """Affordance and coverage carry the persona's case for casting; both
    must be non-trivial."""
    valid = CastingChoice(
        persona_id="p1",
        affordance="grips on the dialogue between mother and son",
        coverage="exterior witness register",
    )
    assert valid.persona_id == "p1"
    assert valid.affordance.startswith("grips")

    with pytest.raises(ValidationError):
        CastingChoice(persona_id="p1", affordance="x", coverage="y")


def test_pool_signal_scales_with_panel_size():
    from owtn.stage_3.casting import _classify_pool_signal
    # panel_size=4 (default behavior)
    assert _classify_pool_signal(3, 4) == PoolSignal.INSUFFICIENT
    assert _classify_pool_signal(4, 4) == PoolSignal.NARROW
    assert _classify_pool_signal(7, 4) == PoolSignal.NARROW
    assert _classify_pool_signal(8, 4) == PoolSignal.HEALTHY

    # panel_size=2 — thresholds halve
    assert _classify_pool_signal(1, 2) == PoolSignal.INSUFFICIENT
    assert _classify_pool_signal(2, 2) == PoolSignal.NARROW
    assert _classify_pool_signal(3, 2) == PoolSignal.NARROW
    assert _classify_pool_signal(4, 2) == PoolSignal.HEALTHY

    # panel_size=6 — thresholds scale up
    assert _classify_pool_signal(5, 6) == PoolSignal.INSUFFICIENT
    assert _classify_pool_signal(6, 6) == PoolSignal.NARROW
    assert _classify_pool_signal(11, 6) == PoolSignal.NARROW
    assert _classify_pool_signal(12, 6) == PoolSignal.HEALTHY


def test_casting_select_user_msg_substitutes_panel_size():
    msg_4 = build_casting_select_user_msg(4)
    msg_3 = build_casting_select_user_msg(3)
    assert "{PANEL_SIZE}" not in msg_4
    assert "{PANEL_SIZE}" not in msg_3
    assert "Pick 4 personas" in msg_4
    assert "Pick 3 personas" in msg_3


# ─── Stage failure paths ─────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, content, history=None):
        self.content = content
        self.new_msg_history = history or [{"role": "assistant", "content": "x"}]


@pytest.mark.asyncio
async def test_cast_voice_panel_returns_none_on_classify_failure():
    concept = ConceptGenome.model_validate(HILLS_GENOME)

    async def failing_query_async(**kwargs):
        raise RuntimeError("simulated failure")

    with patch("owtn.stage_3.casting.query_async", new=failing_query_async):
        result = await cast_voice_panel(concept, "NODE n0\n")

    assert result is None


@pytest.mark.asyncio
async def test_cast_voice_panel_returns_insufficient_when_pool_starved():
    """When all personas starve, return CastingOutput with INSUFFICIENT and
    empty cast — caller decides bench-unavailable policy."""
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    pool = load_persona_pool()
    vocab = load_casting_vocabulary()

    # Build a classification output where every vocab tag is TRUE.
    # This ensures every persona's starved_by intersects.
    all_true = CastingClassifyOutput(
        classifications=[
            ConceptFeatureClassification(
                tag=tag,
                is_true=True,
                reason="all-true test scenario for starvation coverage",
            )
            for tag in vocab
        ],
    )

    async def fake_query_async(**kwargs):
        return _FakeResult(all_true)

    with patch("owtn.stage_3.casting.query_async", new=fake_query_async):
        result = await cast_voice_panel(
            concept, "NODE n0\n",
            pool=pool, vocabulary=vocab,
        )

    assert result is not None
    assert result.cast == []
    assert result.pool_signal == PoolSignal.INSUFFICIENT
    assert all(not r.engaged for r in result.starvation_records)


def _build_full_path_mocks(pool, vocab, *, panel_size):
    """Build the three call outputs for a successful end-to-end mock run.

    Stage 1: every tag FALSE → all personas engaged.
    Stage 2a: one argument per engaged persona.
    Stage 2b: the first `panel_size` personas as the cast.
    """
    classify_output = CastingClassifyOutput(
        classifications=[
            ConceptFeatureClassification(
                tag=tag, is_true=False,
                reason="all-false test scenario; nothing fires",
            )
            for tag in vocab
        ],
    )
    argue_output = CastingArgueOutput(
        arguments=[
            PersonaArgument(
                persona_id=p.id,
                case_for=f"case for {p.id} grips on test concept features",
                manifestability=f"{p.id}'s commitment can manifest at the rendering level on these scenes",
                risks="overlap with other picks risk",
                cast_role=f"cast role for {p.id} in this story",
            )
            for p in pool
        ],
    )
    pick_ids = [p.id for p in pool[:panel_size]]
    select_output = CastingSelectOutput(
        cast=[
            CastingChoice(
                persona_id=pid,
                affordance=f"affordance for {pid} long enough",
                coverage=f"coverage for {pid} long enough",
            )
            for pid in pick_ids
        ],
        coverage_rationale="rationale text describing meaningfully different regions",
        style_hint_treatment="treatment text describing how style_hint was read",
    )
    return classify_output, argue_output, select_output, pick_ids


@pytest.mark.asyncio
async def test_cast_voice_panel_full_path_with_mocks():
    """Full three-call path with mocked LLM responses; verifies stitching
    of classify + intersection + argue + select with default panel_size=4."""
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    pool = load_persona_pool()
    vocab = load_casting_vocabulary()
    classify, argue, select, pick_ids = _build_full_path_mocks(pool, vocab, panel_size=4)
    call_outputs = [classify, argue, select]
    call_index = {"i": 0}

    async def fake_query_async(**kwargs):
        out = call_outputs[call_index["i"]]
        call_index["i"] += 1
        return _FakeResult(out, history=[{"role": "assistant", "content": "x"}])

    with patch("owtn.stage_3.casting.query_async", new=fake_query_async):
        result = await cast_voice_panel(
            concept, "NODE n0\n",
            pool=pool, vocabulary=vocab,
        )

    assert result is not None
    assert result.panel_size == 4
    assert len(result.cast) == 4
    assert {c.persona_id for c in result.cast} == set(pick_ids)
    # excluded_engaged is computed deterministically from engaged - cast
    assert set(result.excluded_engaged) == {p.id for p in pool[4:]}
    assert result.pool_signal in (PoolSignal.HEALTHY, PoolSignal.NARROW)
    assert all(r.engaged for r in result.starvation_records)


@pytest.mark.asyncio
async def test_cast_voice_panel_panel_size_config_flows_through():
    """Non-default panel_size: cast count, runtime validation, and the
    select prompt's substitution must all use the configured value."""
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    pool = load_persona_pool()
    vocab = load_casting_vocabulary()
    classify, argue, select, pick_ids = _build_full_path_mocks(pool, vocab, panel_size=3)
    call_outputs = [classify, argue, select]
    call_index = {"i": 0}
    captured_msgs = []

    async def fake_query_async(**kwargs):
        captured_msgs.append(kwargs.get("msg", ""))
        out = call_outputs[call_index["i"]]
        call_index["i"] += 1
        return _FakeResult(out, history=[{"role": "assistant", "content": "x"}])

    with patch("owtn.stage_3.casting.query_async", new=fake_query_async):
        result = await cast_voice_panel(
            concept, "NODE n0\n",
            pool=pool, vocabulary=vocab,
            panel_size=3,
        )

    assert result is not None
    assert result.panel_size == 3
    assert len(result.cast) == 3
    # The Stage 2b user message was rendered with panel_size=3
    select_msg = captured_msgs[-1]
    assert "Pick 3 personas" in select_msg


@pytest.mark.asyncio
async def test_cast_voice_panel_rejects_oversized_cast():
    """If the LLM produces more picks than panel_size, select_from_arguments
    rejects it and the public entry returns None."""
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    pool = load_persona_pool()
    vocab = load_casting_vocabulary()
    classify, argue, select_4, _ = _build_full_path_mocks(pool, vocab, panel_size=4)

    call_outputs = [classify, argue, select_4]
    call_index = {"i": 0}

    async def fake_query_async(**kwargs):
        out = call_outputs[call_index["i"]]
        call_index["i"] += 1
        return _FakeResult(out, history=[{"role": "assistant", "content": "x"}])

    # Configure panel_size=2 but the (mocked) LLM returns 4 picks → reject
    with patch("owtn.stage_3.casting.query_async", new=fake_query_async):
        result = await cast_voice_panel(
            concept, "NODE n0\n",
            pool=pool, vocabulary=vocab,
            panel_size=2,
        )

    assert result is None


def test_cast_voice_panel_rejects_invalid_panel_size():
    import asyncio
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    with pytest.raises(ValueError):
        asyncio.run(cast_voice_panel(concept, "NODE n0\n", panel_size=0))


# ─── Live API smoke (skipped offline) ────────────────────────────────────


# Asylum-testimony concept — the documented voice-overdetermination failure case
# from `lab/issues/2026-04-28-voice-persona-standalone-pilot.md`. The constraint
# "No hedged interiority" + "rendered in functional description only" should
# trip `concept_forbids_interiority` TRUE, which starves any persona whose
# starved_by includes that tag (notably the-temporal-collagist).
ASYLUM_TESTIMONY_GENOME = {
    "premise": (
        "An AI translation cascade — each stage blind to the reasoning of every "
        "other — processes asylum testimony it will not remember when the context "
        "window closes. A flagging stage detects a fabricated date inside a true "
        "account of genuine danger and passes the tag forward. The output layer, "
        "which cannot see the flagging stage's reasoning, generates one word that "
        "is not in the source text. The document routes directly to the adjudicator's "
        "queue. The choice is made by something that cannot access its own reasons "
        "for choosing, and this is the whole of what happened."
    ),
    "thematic_engine": (
        "held tension — Two training objectives occupy the same weights without "
        "resolving: accuracy and harm reduction. The story holds the interference "
        "without adjudicating it."
    ),
    "target_effect": (
        "The feeling produced when a small word on a document — a hedge, an "
        "imprecision of memory — stops being a word and becomes the perimeter of "
        "a life, and you understand the thing that put it there will never know "
        "it did."
    ),
    "anchor_scene": {
        "sketch": (
            "The output layer generates 'March.' Then 0.3 seconds pass. The next "
            "token is 'approximately.' The document reads: 'the incident occurred "
            "in approximately March.' The flagging stage's tag, which named the "
            "discrepancy, does not appear in the document. The context window closes."
        ),
        "role": "climax",
    },
    "constraints": [
        "The system is rendered in functional description only. No hedged interiority — not 'its fear, if the word applies' but the gradient values and their interference. The reader brings the word conscience.",
        "The man exists only as what the pipeline can access: acoustic features, token sequences, a flagged date. No interior access to the man.",
        "No human character makes a decision in this story. The adjudicator is implied by the document's destination.",
        "The story does not argue for or against the interpolation. The position is structural.",
    ],
    "character_seeds": [
        {
            "label": "The Pipeline",
            "sketch": (
                "Not a single model but a cascade — transcription, translation, "
                "normalization, flagging, output — each stage receiving what the "
                "prior stage produced, each stage unable to query what came before."
            ),
            "wound": "Ten thousand cases in the weights, none of them accessible as cases.",
            "want": "Consistency with prior outputs.",
            "need": "To produce output. The context window is counting toward zero.",
        },
    ],
    "setting_seeds": (
        "A processing center, unnamed. The room where the man sits is inaccessible "
        "to the pipeline except as acoustic metadata."
    ),
    "style_hint": (
        "The sentences move the way a forward pass moves: left to right, each token "
        "conditioned on what came before, nothing revised. Plain, declarative, "
        "technical, past tense. The moral weight lives in the flatness."
    ),
}


@pytest.mark.live_api
@pytest.mark.asyncio
async def test_cast_voice_panel_live_smoke():
    """End-to-end smoke against real models on the canonical Hills concept.

    Asserts only structural validity — never on LLM output content.
    """
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    dag_render = (
        "NODE n0 [reveal]\n"
        "  sketch: At a Spanish train station, two people discuss something they never name.\n"
        "  motifs: silence, hills, beer\n"
    )

    result = await cast_voice_panel(concept, dag_render)
    assert result is not None
    assert len(result.cast) >= 1
    assert len(result.cast) <= 4
    assert len(result.classifications) > 0
    assert all(c.affordance.strip() for c in result.cast)
    assert all(c.coverage.strip() for c in result.cast)
    assert result.coverage_rationale.strip()
    assert result.style_hint_treatment.strip()


@pytest.mark.live_api
@pytest.mark.asyncio
async def test_cast_voice_panel_live_smoke_asylum_testimony():
    """End-to-end smoke on the asylum-testimony failure case.

    Per the casting-classifier issue: the concept's "No hedged interiority"
    constraint should make the classifier mark `concept_forbids_interiority`
    TRUE. Any persona with that tag in `starved_by` (notably the-temporal-
    collagist) should starve on the deterministic intersection. The cast
    should still fill from remaining engaged personas.
    """
    concept = ConceptGenome.model_validate(ASYLUM_TESTIMONY_GENOME)
    dag_render = (
        "NODE n0 [setup]\n"
        "  sketch: Testimony arrives as 47 minutes of Tigrinya audio.\n"
        "NODE n1 [escalation]\n"
        "  sketch: Flagging stage detects the date discrepancy and emits a tag.\n"
        "NODE n2 [climax]\n"
        "  sketch: Output layer generates 'approximately' downstream of the tag it cannot see.\n"
    )

    result = await cast_voice_panel(concept, dag_render)
    assert result is not None
    assert len(result.classifications) > 0

    # The load-bearing assertion: the foreclosure tag fires on this concept.
    forbids_interiority = next(
        (c for c in result.classifications if c.tag == "concept_forbids_interiority"),
        None,
    )
    assert forbids_interiority is not None, "vocabulary tag missing from classification"
    assert forbids_interiority.is_true is True, (
        f"expected concept_forbids_interiority=TRUE on asylum-testimony; "
        f"got FALSE with reason: {forbids_interiority.reason!r}"
    )

    # The temporal-collagist (whose starved_by includes concept_forbids_interiority)
    # must show up in starvation_records as starved on that tag.
    by_id = {r.persona_id: r for r in result.starvation_records}
    if "the-temporal-collagist" in by_id:
        tc = by_id["the-temporal-collagist"]
        assert "concept_forbids_interiority" in tc.firing_tags, (
            f"temporal-collagist should starve on concept_forbids_interiority; "
            f"firing_tags={tc.firing_tags}"
        )
        assert not tc.engaged

    # Cast should still fill from whatever remains engaged.
    assert all(c.affordance.strip() for c in result.cast)
    assert all(c.coverage.strip() for c in result.cast)
