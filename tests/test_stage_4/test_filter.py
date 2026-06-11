"""Tests for owtn.stage_4.filter — pre-stage classification."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_3 import (
    ConsciousnessRendering,
    Craft,
    DialogicMode,
    ImpliedAuthor,
    Rendering,
    VoiceGenome,
)
from owtn.models.stage_4 import (
    AudienceFraming,
    ExpertNeed,
    ExpertNeedsList,
    Stage4FilterConfig,
)
from owtn.stage_4.filter import run_stage_4_filter

from tests.conftest import HILLS_GENOME


class _FakeResult:
    def __init__(self, content):
        self.content = content
        self.cost = 0.0
        self.new_msg_history: list = []


def _voice() -> VoiceGenome:
    from tests.conftest import _signature_risk_for_test
    body = "He looked at her. She did not look back. " * 8
    return VoiceGenome(
        pov="third", tense="past",
        consciousness_rendering=ConsciousnessRendering(mode="external_focalization", fid_depth="none"),
        implied_author=ImpliedAuthor(stance_toward_characters="elegiac", moral_temperature="cool"),
        dialogic_mode=DialogicMode(type="monologic"),
        craft=Craft(sentence_rhythm="varied", crowding_leaping="leaping"),
        description="Spare third-person past, elegiac without emotional gloss.",
        diction="Plain Anglo-Saxon nouns; no abstractions.",
        positive_constraints=["Render scene exits with stripped declarative."],
        prohibitions=[],
        signature_risk=_signature_risk_for_test(),
        renderings=[
            Rendering(scene_id="ab", text=body),
            Rendering(scene_id="bc", text=body),
            Rendering(scene_id="cd", text=body),
        ],
        pair_id="c_test", agent_id="test",
    )


def _audience() -> AudienceFraming:
    return AudienceFraming(
        description=(
            "A reader of literary short fiction in the Hemingway / Carver / "
            "Munro tradition. Comfortable with prose that refuses to interpret "
            "for them and endings that decline closure. Reads slowly when the "
            "writing earns it."
        ),
        recognizes=["free indirect discourse without flagging", "endings that decline to interpret"],
        tolerates=["ambiguity at the level of plot, not just theme"],
    )


@pytest.mark.asyncio
async def test_run_filter_returns_audience_and_experts_in_parallel():
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    voice = _voice()
    captured: dict = {}

    async def fake_query(**kwargs):
        output_model = kwargs["output_model"]
        captured.setdefault(output_model.__name__, []).append(kwargs)
        if output_model is AudienceFraming:
            return _FakeResult(_audience())
        if output_model is ExpertNeedsList:
            return _FakeResult(ExpertNeedsList(experts=[]))
        raise AssertionError(f"unexpected output_model: {output_model}")

    with patch("owtn.stage_4.filter.query_async", new=fake_query):
        audience, experts = await run_stage_4_filter(
            concept=concept, voice_genome=voice, dag_rendering="NODE n0\n  sketch: train station\n",
        )

    assert isinstance(audience, AudienceFraming)
    assert "Hemingway" in audience.description
    assert experts.experts == []
    assert "AudienceFraming" in captured
    assert "ExpertNeedsList" in captured


@pytest.mark.asyncio
async def test_filter_returns_empty_experts_on_dispatch_failure():
    """If the experts call fails, the filter falls back to an empty list
    so the session can still proceed without domain critics."""
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    voice = _voice()

    async def fake_query(**kwargs):
        output_model = kwargs["output_model"]
        if output_model is AudienceFraming:
            return _FakeResult(_audience())
        # Experts call simulates a provider error
        raise RuntimeError("experts provider outage")

    with patch("owtn.stage_4.filter.query_async", new=fake_query):
        audience, experts = await run_stage_4_filter(
            concept=concept, voice_genome=voice, dag_rendering="x" * 50,
        )

    assert audience is not None
    assert isinstance(experts, ExpertNeedsList)
    assert experts.experts == []


@pytest.mark.asyncio
async def test_filter_returns_none_audience_on_dispatch_failure():
    """If the audience call fails, audience is None — caller decides
    whether to proceed without it."""
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    voice = _voice()

    async def fake_query(**kwargs):
        output_model = kwargs["output_model"]
        if output_model is ExpertNeedsList:
            return _FakeResult(ExpertNeedsList())
        raise RuntimeError("audience provider outage")

    with patch("owtn.stage_4.filter.query_async", new=fake_query):
        audience, experts = await run_stage_4_filter(
            concept=concept, voice_genome=voice, dag_rendering="x" * 50,
        )

    assert audience is None
    assert experts.experts == []


@pytest.mark.asyncio
async def test_filter_extracts_concept_and_voice_into_prompts():
    """The prompts must surface the concept's premise and the voice spec
    so the haiku call has something to read against."""
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    voice = _voice()
    seen_msgs: list[str] = []

    async def fake_query(**kwargs):
        seen_msgs.append(kwargs["msg"])
        output_model = kwargs["output_model"]
        if output_model is AudienceFraming:
            return _FakeResult(_audience())
        return _FakeResult(ExpertNeedsList())

    with patch("owtn.stage_4.filter.query_async", new=fake_query):
        await run_stage_4_filter(
            concept=concept, voice_genome=voice, dag_rendering="NODE n0\n  sketch: train station\n",
        )

    # Both calls should reference the premise; the experts call should
    # additionally reference the structural plan.
    audience_msg = next(m for m in seen_msgs if "implied audience" in m.lower())
    experts_msg = next(m for m in seen_msgs if "domain expertise" in m.lower())
    assert "train station" in audience_msg
    assert "train station" in experts_msg
    assert "Structural plan" in experts_msg


@pytest.mark.asyncio
async def test_filter_with_experts_populated():
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    voice = _voice()
    expert = ExpertNeed(
        domain="quantum optics",
        expertise_focus=["entanglement protocols", "measurement back-action"],
        persona_hint="A postdoc in atomic physics who reviews fiction manuscripts as a side gig.",
        web_search_recommended=True,
    )

    async def fake_query(**kwargs):
        if kwargs["output_model"] is AudienceFraming:
            return _FakeResult(_audience())
        return _FakeResult(ExpertNeedsList(experts=[expert]))

    with patch("owtn.stage_4.filter.query_async", new=fake_query):
        audience, experts = await run_stage_4_filter(
            concept=concept, voice_genome=voice, dag_rendering="x" * 50,
        )

    assert len(experts.experts) == 1
    assert experts.experts[0].domain == "quantum optics"
    assert experts.experts[0].web_search_recommended is True
