"""Tests for owtn.stage_3.session — composition + end-to-end mock + live smoke."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_3 import (
    BordaRanking,
    ConsciousnessRendering,
    Craft,
    CritiqueBody,
    CritiqueSet,
    DialogicMode,
    ImpliedAuthor,
    Rendering,
    VoiceGenomeBody,
)
from owtn.stage_3.adjacent_scenes import AdjacentSceneBench, AdjacentSceneDraft
from owtn.stage_3.personas import load_persona_pool
from owtn.stage_3.session import (
    build_voice_agent,
    render_persona_system_prompt,
    run_voice_session,
)
from tests.conftest import HILLS_GENOME


# ─── render_persona_system_prompt ────────────────────────────────────────


def test_persona_system_prompt_contains_key_components():
    pool = load_persona_pool()
    persona = next(p for p in pool if p.id == "the-reductionist")
    prompt = render_persona_system_prompt(persona)

    assert "The Reductionist" in prompt
    assert "Aesthetic commitments" in prompt
    assert "Compression because" in prompt  # commitment 1
    assert "Aversions" in prompt
    assert "Adverbs modifying said-tags" in prompt  # aversion
    assert "Epistemic skepticism" in prompt
    assert "first version" in prompt.lower()  # epistemic skepticism content
    # Workshop frame, not assistant register
    assert "happy to help" not in prompt.lower()
    assert "as a creative writing assistant" not in prompt.lower()


def test_persona_system_prompt_loads_exemplar_passages():
    pool = load_persona_pool()
    persona = next(p for p in pool if p.id == "the-reductionist")
    prompt = render_persona_system_prompt(persona)

    # Persona must declare at least one exemplar and that exemplar's id +
    # its prose must reach the prompt; the loader looks up the .txt file.
    assert persona.exemplars, "reductionist should declare exemplars in YAML"
    first_id = persona.exemplars[0].id
    assert first_id in prompt, f"exemplar id {first_id!r} missing from prompt"
    assert "What this exemplifies" in prompt


# ─── build_voice_agent ───────────────────────────────────────────────────


def test_build_voice_agent_constructs_orchestration_agent():
    pool = load_persona_pool()
    persona = next(p for p in pool if p.id == "the-reductionist")
    agent = build_voice_agent(persona)

    assert agent.id == "the-reductionist"
    assert agent.model == "deepseek-v4-pro"  # per persona YAML default
    assert "render_adjacent_scene" in agent.tools
    assert "stylometry" in agent.tools
    assert agent.sampler["temperature"] == persona.temperature
    assert "The Reductionist" in agent.system_prompt


# ─── End-to-end mocked run_voice_session ─────────────────────────────────


def _bench() -> AdjacentSceneBench:
    return AdjacentSceneBench(
        drafts=[
            AdjacentSceneDraft(
                scene_id=f"scene-{i}",
                synopsis="A scene where things happen and people speak.",
                demand="Test demand text long enough.",
                why_distinct="Distinct in this way from the others.",
                neutral_draft=(
                    "She poured the coffee. He turned a page. "
                    "The kitchen was cold. Outside a car passed."
                ),
            )
            for i in range(3)
        ],
        bench_rationale="rationale long enough to pass validation",
        picker_model="claude-sonnet-4-6",
        drafter_model="claude-sonnet-4-6",
    )


def _voice_body() -> VoiceGenomeBody:
    text = (
        "She set the cup down and did not look up. "
        "He waited a long time before he spoke. The kitchen was cold."
    )
    return VoiceGenomeBody(
        pov="third",
        tense="past",
        consciousness_rendering=ConsciousnessRendering(
            mode="narrated_monologue", fid_depth="shallow",
        ),
        implied_author=ImpliedAuthor(
            stance_toward_characters="elegiac", moral_temperature="cool",
        ),
        dialogic_mode=DialogicMode(type="monologic"),
        craft=Craft(sentence_rhythm="varied", crowding_leaping="leaping"),
        description=(
            "Voice that holds back where most prose would explain, "
            "trusting the reader to assemble feeling from gesture."
        ),
        diction="Plain declarative; no figurative ornament.",
        positive_constraints=["Render emotion through the body's small refusals."],
        renderings=[
            Rendering(scene_id=f"scene-{i}", text=text) for i in range(3)
        ],
    )


class _FakeResult:
    def __init__(self, content, history=None, cost=0.0):
        self.content = content
        self.new_msg_history = history or []
        self.cost = cost


@pytest.mark.asyncio
async def test_run_voice_session_end_to_end_mocked(tmp_path: Path):
    pool = load_persona_pool()
    cast = [
        next(p for p in pool if p.id == "the-reductionist"),
        next(p for p in pool if p.id == "the-temporal-collagist"),
    ]
    bench = _bench()
    concept = ConceptGenome.model_validate(HILLS_GENOME)

    async def fake_explore(**kwargs):
        # Phase 1 + Phase 4 explore loops commit via dispatch, simulating the
        # agent's finalize_voice_genome tool call landing during exploration.
        sysprompt = kwargs.get("system_msg", "")
        if "Reductionist" in sysprompt:
            agent_id = "the-reductionist"
        else:
            agent_id = "the-temporal-collagist"
        await kwargs["dispatch"]("finalize_voice_genome", _voice_body().model_dump())
        return _FakeResult("done", history=[], cost=0.01)

    async def fake_query(**kwargs):
        # Identify the agent from the system_prompt; both prompts contain
        # the persona display name.
        sysprompt = kwargs.get("system_msg", "")
        if "Reductionist" in sysprompt:
            agent_id = "the-reductionist"
        else:
            agent_id = "the-temporal-collagist"

        output_model = kwargs.get("output_model")
        if output_model is not None and output_model.__name__ == "CritiqueSet":
            others = ["the-reductionist", "the-temporal-collagist"]
            others.remove(agent_id)
            cs = CritiqueSet(
                critiques=[
                    CritiqueBody(
                        target_id=t,
                        strengths=["one specific strength", "two specific strength"],
                        concern=f"specific concern about {t} for testing purposes",
                    )
                    for t in others
                ],
            )
            return _FakeResult(cs, cost=0.002)
        if output_model is not None and output_model.__name__ == "BordaRanking":
            others = ["the-reductionist", "the-temporal-collagist"]
            others.remove(agent_id)
            return _FakeResult(BordaRanking(ranking=others), cost=0.001)
        # Fallback
        return _FakeResult("plain text", cost=0.0)

    with patch("owtn.stage_3.phases.query_async_with_tools", new=fake_explore), \
         patch("owtn.stage_3.phases.query_async", new=fake_query):
        result = await run_voice_session(
            cast=cast,
            bench=bench,
            concept=concept,
            dag_rendering="NODE n0 [reveal]\n  sketch: train station\n",
            pair_id="c_test_struct_0",
            session_log_dir=tmp_path / "session",
        )

    assert result.pair_id == "c_test_struct_0"
    assert result.winner.agent_id in {"the-reductionist", "the-temporal-collagist"}
    assert len(result.proposals) == 2
    # Each agent received 1 critique (from the other), so 2 total
    assert len(result.critiques) == 2
    assert set(result.borda_points.keys()) == {
        "the-reductionist", "the-temporal-collagist",
    }
    # Cost accumulated across all phase calls
    assert result.cost_usd > 0

    # Log tree was written
    assert (tmp_path / "session" / "session.yaml").exists()
    assert (tmp_path / "session" / "phases").is_dir()


# ─── Live-API smoke ──────────────────────────────────────────────────────


@pytest.mark.live_api
@pytest.mark.asyncio
async def test_run_voice_session_live_smoke_hills(tmp_path: Path):
    """End-to-end live smoke on HILLS_GENOME with two voice agents.

    Asserts only structural validity. Costs ~$0.50-$1.50 per run.
    """
    pool = load_persona_pool()
    cast = [
        next(p for p in pool if p.id == "the-reductionist"),
        next(p for p in pool if p.id == "the-temporal-collagist"),
    ]
    bench = _bench()
    concept = ConceptGenome.model_validate(HILLS_GENOME)

    result = await run_voice_session(
        cast=cast,
        bench=bench,
        concept=concept,
        dag_rendering=(
            "NODE n0 [reveal]\n"
            "  sketch: At a Spanish train station, two people discuss "
            "something they never name.\n"
            "  motifs: silence, hills, beer\n"
        ),
        pair_id="c_hills_struct_0",
        session_log_dir=tmp_path / "session",
    )

    assert result.winner is not None
    assert result.winner.agent_id in {"the-reductionist", "the-temporal-collagist"}
    assert len(result.proposals) == 2
    assert {r.scene_id for r in result.winner.renderings} == {
        "scene-0", "scene-1", "scene-2",
    }
    # Borda totals must sum correctly: with N=2, each ranker awards 0 points
    # (rank-1 gets N-2 = 0). So all points are zero — degenerate but valid.
    # With N>=3 the assertion would be points sum == (N-1)*(N-2).
    assert result.cost_usd > 0
