"""Tests for owtn.stage_3.adjacent_scenes — picker + neutral-voice drafter.

Offline tests cover Pydantic schema, prompt assembly, and the picker's
duplicate-scene-id rejection. The end-to-end live-API smoke test is
marked `live_api` so the offline suite stays fast.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.prompts.stage_3 import (
    build_adjacent_scene_drafter_prompt,
    build_adjacent_scene_picker_prompt,
    load_base_system,
)
from owtn.stage_3.adjacent_scenes import (
    AdjacentSceneBench,
    AdjacentSceneDraft,
    AdjacentScenePick,
    AdjacentScenePickerOutput,
    _pick_scenes,
    generate_adjacent_scenes,
)
from tests.conftest import HILLS_GENOME


# ─── Schema ──────────────────────────────────────────────────────────────


def test_adjacent_scene_pick_requires_all_fields():
    pick = AdjacentScenePick(
        scene_id="morning-kitchen",
        synopsis="She makes coffee while he reads. Neither speaks for a long time.",
        demand="Render slow domestic time without filler.",
        why_distinct="Tests stillness; the other two test motion and pressure.",
    )
    assert pick.scene_id == "morning-kitchen"

    with pytest.raises(ValidationError):
        AdjacentScenePick(scene_id="x", synopsis="too short", demand="d", why_distinct="w")


def test_picker_output_requires_exactly_three_scenes():
    base_pick = dict(
        scene_id="s",
        synopsis="A scene where things happen and people speak.",
        demand="Test demand text long enough.",
        why_distinct="Distinct in this way from the others.",
    )
    with pytest.raises(ValidationError):
        AdjacentScenePickerOutput(
            scenes=[AdjacentScenePick(**{**base_pick, "scene_id": "a"})],
            bench_rationale="rationale text long enough to pass validation",
        )

    with pytest.raises(ValidationError):
        AdjacentScenePickerOutput(
            scenes=[
                AdjacentScenePick(**{**base_pick, "scene_id": f"s{i}"})
                for i in range(4)
            ],
            bench_rationale="rationale text long enough to pass validation",
        )


def test_bench_serialization_roundtrip():
    bench = AdjacentSceneBench(
        drafts=[
            AdjacentSceneDraft(
                scene_id=f"scene-{i}",
                synopsis="A scene where things happen and people speak.",
                demand="Test demand text long enough.",
                why_distinct="Distinct in this way from the others.",
                neutral_draft="The man walked into the room. He looked at her.",
            )
            for i in range(3)
        ],
        bench_rationale="rationale text long enough",
        picker_model="claude-sonnet-4-6",
        drafter_model="claude-sonnet-4-6",
    )
    dumped = bench.model_dump()
    restored = AdjacentSceneBench.model_validate(dumped)
    assert restored.drafts[0].scene_id == "scene-0"
    assert restored.picker_model == "claude-sonnet-4-6"


# ─── Prompt assembly ─────────────────────────────────────────────────────


def test_picker_prompt_substitutes_concept_and_dag():
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    dag_render = "NODE n0 [climax]\n  sketch: She tells him she is leaving.\n"
    system_msg, user_msg = build_adjacent_scene_picker_prompt(concept, dag_render)

    assert "voice stage" in system_msg.lower()
    assert "{CONCEPT_JSON}" not in user_msg
    assert "{DAG_RENDERING}" not in user_msg
    assert "train station" in user_msg
    assert "She tells him she is leaving." in user_msg
    assert "scene_id" in user_msg
    assert "demand" in user_msg
    assert "why_distinct" in user_msg


def test_drafter_prompt_substitutes_scene_fields():
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    system_msg, user_msg = build_adjacent_scene_drafter_prompt(
        concept,
        scene_id="morning-kitchen",
        synopsis="She makes coffee while he reads.",
    )

    assert "{SCENE_ID}" not in user_msg
    assert "{SCENE_SYNOPSIS}" not in user_msg
    assert "{CONCEPT_CONTEXT}" not in user_msg
    assert "morning-kitchen" in user_msg
    assert "She makes coffee while he reads." in user_msg
    assert "neutral" in user_msg.lower()


def test_drafter_concept_context_omits_goal_fields():
    """Per prompting-guide §'Goals Cannot Ride in Plaintext': target_effect,
    thematic_engine, and style_hint must NOT be passed verbatim into a
    prose-generating prompt."""
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    _, user_msg = build_adjacent_scene_drafter_prompt(
        concept,
        scene_id="morning-kitchen",
        synopsis="She makes coffee while he reads.",
    )

    # World/character grounding is preserved
    assert "train station" in user_msg.lower() or "spain" in user_msg.lower()
    assert "the man" in user_msg.lower() or "the woman" in user_msg.lower()
    # Goal fields and stylistic hints are stripped — they would prime the
    # drafter toward themed/stylized prose
    assert concept.target_effect not in user_msg
    if concept.thematic_engine:
        assert concept.thematic_engine not in user_msg
    if concept.style_hint:
        assert concept.style_hint not in user_msg


def test_base_system_workshop_frame():
    text = load_base_system().lower()
    assert "working-mode" in text or "voice stage" in text
    # Anti-assistant-register tokens may appear inside quoted anti-examples
    # (the base_system explicitly lists them as patterns to AVOID), so we
    # only assert they don't lead the prose.
    first_sentence = text.split(".")[0]
    assert "happy to help" not in first_sentence


# ─── Picker behavior ─────────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, content):
        self.content = content


def _stub_picker_output(scene_ids: list[str]) -> AdjacentScenePickerOutput:
    return AdjacentScenePickerOutput(
        scenes=[
            AdjacentScenePick(
                scene_id=sid,
                synopsis=f"Synopsis for {sid} — stuff happens here.",
                demand=f"Demand placed by {sid}, articulated.",
                why_distinct=f"What {sid} tests that the others don't.",
            )
            for sid in scene_ids
        ],
        bench_rationale="Three demands cover stillness, pressure, and disclosure.",
    )


@pytest.mark.asyncio
async def test_picker_rejects_duplicate_scene_ids():
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    dag_render = "NODE n0\n"

    duplicate_output = _stub_picker_output(["alpha", "alpha", "beta"])

    async def fake_query_async(**kwargs):
        return _FakeResult(duplicate_output)

    with patch("owtn.stage_3.adjacent_scenes.query_async", new=fake_query_async):
        result = await _pick_scenes(concept, dag_render, model_name="test-model")

    assert result is None


@pytest.mark.asyncio
async def test_picker_accepts_distinct_scene_ids():
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    dag_render = "NODE n0\n"

    distinct_output = _stub_picker_output(["alpha", "beta", "gamma"])

    async def fake_query_async(**kwargs):
        return _FakeResult(distinct_output)

    with patch("owtn.stage_3.adjacent_scenes.query_async", new=fake_query_async):
        result = await _pick_scenes(concept, dag_render, model_name="test-model")

    assert result is not None
    assert [p.scene_id for p in result.scenes] == ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
async def test_generate_returns_none_when_picker_fails():
    concept = ConceptGenome.model_validate(HILLS_GENOME)

    async def failing_query_async(**kwargs):
        raise RuntimeError("simulated picker failure")

    with patch("owtn.stage_3.adjacent_scenes.query_async", new=failing_query_async):
        bench = await generate_adjacent_scenes(concept, "NODE n0\n")

    assert bench is None


@pytest.mark.asyncio
async def test_generate_returns_bench_on_full_success():
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    distinct_output = _stub_picker_output(["alpha", "beta", "gamma"])

    drafts = [
        "Draft for scene alpha. The man walked into the room.",
        "Draft for scene beta. She sat down at the kitchen table.",
        "Draft for scene gamma. The phone rang twice and stopped.",
    ]
    call_count = {"n": 0}

    async def fake_query_async(**kwargs):
        if "output_model" in kwargs and kwargs["output_model"] is AdjacentScenePickerOutput:
            return _FakeResult(distinct_output)
        idx = call_count["n"]
        call_count["n"] += 1
        return _FakeResult(drafts[idx])

    with patch("owtn.stage_3.adjacent_scenes.query_async", new=fake_query_async):
        bench = await generate_adjacent_scenes(
            concept,
            "NODE n0\n",
            picker_model="picker-model",
            drafter_model="drafter-model",
        )

    assert bench is not None
    assert len(bench.drafts) == 3
    assert {d.scene_id for d in bench.drafts} == {"alpha", "beta", "gamma"}
    assert all(d.neutral_draft.startswith("Draft for scene") for d in bench.drafts)
    assert bench.picker_model == "picker-model"
    assert bench.drafter_model == "drafter-model"


# ─── Live API smoke (skipped by default) ─────────────────────────────────


@pytest.mark.live_api
@pytest.mark.asyncio
async def test_generate_adjacent_scenes_live():
    """End-to-end smoke test against the real picker + drafter models.

    Asserts only structural validity — never assert on LLM output content.
    """
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    dag_render = (
        "NODE n0 [reveal]\n"
        "  sketch: At a Spanish train station, two people discuss something they never name.\n"
        "  motifs: silence, hills, beer\n"
    )

    bench = await generate_adjacent_scenes(concept, dag_render)
    assert bench is not None
    assert len(bench.drafts) == 3
    assert len({d.scene_id for d in bench.drafts}) == 3
    assert all(d.neutral_draft.strip() for d in bench.drafts)
    assert all(d.demand.strip() and d.why_distinct.strip() for d in bench.drafts)
    assert bench.bench_rationale.strip()
