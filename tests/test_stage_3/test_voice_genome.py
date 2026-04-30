"""Tests for owtn.models.stage_3.voice_genome — schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from owtn.models.stage_3 import (
    ConsciousnessRendering,
    Craft,
    Critique,
    DialogicMode,
    ImpliedAuthor,
    Rendering,
    VoiceGenome,
    VoiceMode,
    VoiceSessionResult,
)


def _minimal_genome(**overrides) -> dict:
    """Smallest valid genome fields. Tests override what they care about."""
    base = dict(
        pair_id="c_test_struct_0",
        agent_id="the-reductionist",
        pov="third",
        tense="past",
        consciousness_rendering=ConsciousnessRendering(
            mode="narrated_monologue", fid_depth="shallow",
        ),
        implied_author=ImpliedAuthor(
            stance_toward_characters="compassionate", moral_temperature="warm",
        ),
        dialogic_mode=DialogicMode(type="monologic"),
        craft=Craft(sentence_rhythm="varied", crowding_leaping="balanced"),
        description=(
            "A voice that holds back where most prose would explain, "
            "trusting the reader to assemble feeling from gesture."
        ),
        diction="Plain declarative; no figurative ornament.",
        positive_constraints=[
            "Render emotion through the body's small refusals; no said-tag adverbs.",
        ],
        prohibitions=[],
        renderings=[
            Rendering(
                scene_id=f"scene-{i}",
                text=(
                    "She set the cup down and did not look up. "
                    "He waited a long time before he spoke. "
                    "The kitchen was cold. The clock on the stove read four."
                ),
            )
            for i in range(3)
        ],
    )
    base.update(overrides)
    return base


# ─── VoiceGenome ─────────────────────────────────────────────────────────


def test_voice_genome_minimal_validates():
    g = VoiceGenome(**_minimal_genome())
    assert g.pov == "third"
    assert len(g.renderings) == 3
    assert g.voice_modes == []
    assert g.voice_dynamics == ""


def test_voice_genome_requires_three_renderings():
    short = _minimal_genome(renderings=[
        Rendering(scene_id="aa", text="x" * 100),
        Rendering(scene_id="bb", text="x" * 100),
    ])
    with pytest.raises(ValidationError):
        VoiceGenome(**short)

    long = _minimal_genome(renderings=[
        Rendering(scene_id=f"s{i}", text="x" * 100) for i in range(4)
    ])
    with pytest.raises(ValidationError):
        VoiceGenome(**long)


def test_voice_genome_rejects_invalid_pov():
    with pytest.raises(ValidationError):
        VoiceGenome(**_minimal_genome(pov="omniscient"))


def test_voice_genome_rejects_invalid_tense():
    with pytest.raises(ValidationError):
        VoiceGenome(**_minimal_genome(tense="future"))


def test_voice_genome_requires_at_least_one_positive_constraint():
    with pytest.raises(ValidationError):
        VoiceGenome(**_minimal_genome(positive_constraints=[]))


def test_voice_genome_prohibitions_default_empty():
    g = VoiceGenome(**_minimal_genome())
    assert g.prohibitions == []


def test_voice_genome_supports_voice_modes():
    g = VoiceGenome(**_minimal_genome(
        voice_modes=[
            VoiceMode(
                name="memory_register",
                condition="passages of remembered past time",
                overrides={"pov": "first", "tense": "present"},
            ),
        ],
    ))
    assert len(g.voice_modes) == 1
    assert g.voice_modes[0].overrides["pov"] == "first"


# ─── Lit-theory sub-models ───────────────────────────────────────────────


def test_consciousness_rendering_enums():
    cr = ConsciousnessRendering(mode="narrated_monologue", fid_depth="deep")
    assert cr.fid_depth == "deep"

    with pytest.raises(ValidationError):
        ConsciousnessRendering(mode="freestyle", fid_depth="deep")
    with pytest.raises(ValidationError):
        ConsciousnessRendering(mode="narrated_monologue", fid_depth="ultra")


def test_implied_author_enums():
    ia = ImpliedAuthor(stance_toward_characters="elegiac", moral_temperature="cool")
    assert ia.stance_toward_characters == "elegiac"

    with pytest.raises(ValidationError):
        ImpliedAuthor(stance_toward_characters="loving", moral_temperature="warm")


def test_craft_enums():
    c = Craft(sentence_rhythm="staccato", crowding_leaping="leaping")
    assert c.crowding_leaping == "leaping"

    with pytest.raises(ValidationError):
        Craft(sentence_rhythm="meandering", crowding_leaping="balanced")


def test_dialogic_mode_enum():
    assert DialogicMode(type="heteroglossic").type == "heteroglossic"

    with pytest.raises(ValidationError):
        DialogicMode(type="dialogue-heavy")


# ─── Renderings ──────────────────────────────────────────────────────────


def test_rendering_requires_substantial_text():
    Rendering(scene_id="ok", text="x" * 100)
    with pytest.raises(ValidationError):
        Rendering(scene_id="ok", text="too short")


# ─── Critique ────────────────────────────────────────────────────────────


def test_critique_requires_exactly_two_strengths():
    Critique(
        critic_id="the-reductionist",
        target_id="the-temporal-collagist",
        strengths=["one strength here", "second strength here"],
        concern="The voice loses force in the second rendering.",
    )

    with pytest.raises(ValidationError):
        Critique(
            critic_id="a", target_id="b",
            strengths=["only one"],
            concern="x" * 30,
        )

    with pytest.raises(ValidationError):
        Critique(
            critic_id="a", target_id="b",
            strengths=["one", "two", "three"],
            concern="x" * 30,
        )


# ─── VoiceSessionResult ──────────────────────────────────────────────────


def test_voice_session_result_carries_winner_and_metadata():
    g = VoiceGenome(**_minimal_genome())
    result = VoiceSessionResult(
        pair_id="c_test_struct_0",
        winner=g,
        proposals=[g],
        critiques=[],
        borda_points={"the-reductionist": 6},
        cost_usd=1.23,
        session_log_dir="/tmp/run/stage_3/by_pair/c_test_struct_0",
    )
    assert result.winner.agent_id == "the-reductionist"
    assert result.borda_points["the-reductionist"] == 6
