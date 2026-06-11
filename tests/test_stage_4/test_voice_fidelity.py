"""voice_fidelity tests — promotion + tool-using critic dispatch."""

from __future__ import annotations

from pathlib import Path
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
from owtn.models.stage_4 import CriticPersona, CriticReportBody, Issue, Severity
from owtn.orchestration import ToolContext
from owtn.stage_3.personas import VoicePersona
from owtn.stage_4 import CriticRegistry, dispatch_critic
from owtn.stage_4.critics import (
    CRITIC_TOOL_REGISTRY,
    FINALIZE_CRITIC_REPORT,
    _CRITIC_REPORT_SLOT,
    _finalize_critic_report_handler,
)
from owtn.stage_4.voice_fidelity import (
    VOICE_FIDELITY_FOCUS_AREAS,
    VOICE_FIDELITY_ID,
    VOICE_FIDELITY_MECHANISM,
    VOICE_FIDELITY_TOOLS,
    promote_voice_persona_to_critic,
)

from tests.conftest import HILLS_GENOME


# ─── Helpers ─────────────────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, content, history=None, cost: float = 0.0):
        self.content = content
        self.new_msg_history = history or []
        self.cost = cost


def _stage_3_persona() -> VoicePersona:
    """A Stage 3 VoicePersona stand-in for the promotion tests. Matches
    the v3 pool's shape (identity + commitments + skepticism + model)."""
    return VoicePersona(
        id="the-test-stage3-persona",
        name="The Test Persona",
        identity=(
            "Spent fourteen years as an insurance claims adjuster before "
            "leaving the trade. The job taught the lesson the persona "
            "has never forgotten: emotion in a claim narrative gets the "
            "claim denied. Now reads voraciously and thinks about prose "
            "as an extension of that discipline."
        ),
        aesthetic_commitments=["Compression because the reader's attention is sacred."],
        epistemic_skepticism=(
            "When the first version of a sentence arrives, look at it as if "
            "it's already been printed in someone else's book."
        ),
    )


def _voice_genome() -> VoiceGenome:
    from tests.conftest import _signature_risk_for_test
    body = "He looked at her. She did not look back. " * 8
    return VoiceGenome(
        pov="third", tense="past",
        consciousness_rendering=ConsciousnessRendering(mode="external_focalization", fid_depth="none"),
        implied_author=ImpliedAuthor(stance_toward_characters="elegiac", moral_temperature="cool"),
        dialogic_mode=DialogicMode(type="monologic"),
        craft=Craft(sentence_rhythm="varied", crowding_leaping="leaping"),
        description="Spare third-person past, elegiac without emotional gloss.",
        diction="Plain Anglo-Saxon nouns; no abstractions of feeling.",
        positive_constraints=["Render scene exits with a stripped declarative."],
        prohibitions=[],
        signature_risk=_signature_risk_for_test(),
        renderings=[
            Rendering(scene_id="ab", text=body),
            Rendering(scene_id="bc", text=body),
            Rendering(scene_id="cd", text=body),
        ],
        pair_id="c_test", agent_id="the-test-stage3-persona",
    )


def _state(tmp_path: Path, *, registry: CriticRegistry) -> dict:
    state: dict = {
        "run_dir": str(tmp_path),
        "story_path": str(tmp_path / "story.md"),
        "concept": ConceptGenome.model_validate(HILLS_GENOME),
        "voice_genome": _voice_genome(),
        "dag_rendering": "NODE n0\n  sketch: train station\n",
        "critic_registry": registry,
        "phase_3_revise": {
            "cycles": [
                {"cycle": 0, "critic_calls": [], "plan": None, "completed": False},
            ],
        },
    }
    (tmp_path / "story.md").write_text("## opening\n\nThe room was quiet.\n")
    return state


# ─── Promotion ───────────────────────────────────────────────────────────


def test_promotion_carries_stage3_identity():
    promoted = promote_voice_persona_to_critic(_stage_3_persona())
    assert promoted.id == VOICE_FIDELITY_ID
    assert promoted.tier == "tier_a"
    assert promoted.persona is True
    assert "insurance claims adjuster" in promoted.identity
    assert promoted.name == "The Test Persona"


def test_promotion_uses_fixed_template_for_voice_fidelity_work():
    promoted = promote_voice_persona_to_critic(_stage_3_persona())
    assert promoted.mechanism == VOICE_FIDELITY_MECHANISM
    assert promoted.focus_areas == list(VOICE_FIDELITY_FOCUS_AREAS)
    assert promoted.tools == list(VOICE_FIDELITY_TOOLS)


def test_promotion_passes_model_through():
    promoted = promote_voice_persona_to_critic(
        _stage_3_persona(), model="claude-sonnet-4-6", reasoning_effort="high",
    )
    assert promoted.model == "claude-sonnet-4-6"
    assert promoted.reasoning_effort == "high"


def test_promoted_persona_is_tool_using():
    promoted = promote_voice_persona_to_critic(_stage_3_persona())
    assert promoted.is_tool_using is True


# ─── finalize_critic_report handler ──────────────────────────────────────


@pytest.mark.asyncio
async def test_finalize_handler_rejects_outside_dispatch_context():
    """The slot ContextVar is only set during a tool-using dispatch.
    Calling the handler outside one returns an actionable error."""
    ctx = ToolContext(session_id="s", phase_id="p", agent_id="a", state_view={})
    result = await _finalize_critic_report_handler({"issues": []}, ctx)
    assert result.startswith("ERROR")
    assert "not in a critic dispatch context" in result


@pytest.mark.asyncio
async def test_finalize_handler_stashes_body_in_slot():
    ctx = ToolContext(session_id="s", phase_id="p", agent_id="a", state_view={})
    slot: dict = {"body": None}
    token = _CRITIC_REPORT_SLOT.set(slot)
    try:
        body_args = {
            "not_load_bearing": False,
            "issues": [
                {"severity": "moderate", "observation": "first observation for testing purposes"},
                {"severity": "minor", "observation": "second observation for testing purposes"},
            ],
        }
        result = await _finalize_critic_report_handler(body_args, ctx)
    finally:
        _CRITIC_REPORT_SLOT.reset(token)
    assert "Critic report committed" in result
    assert isinstance(slot["body"], CriticReportBody)
    assert len(slot["body"].issues) == 2


@pytest.mark.asyncio
async def test_finalize_handler_rejects_double_commit():
    ctx = ToolContext(session_id="s", phase_id="p", agent_id="a", state_view={})
    slot: dict = {"body": None}
    token = _CRITIC_REPORT_SLOT.set(slot)
    try:
        body_args = {"issues": [{"severity": "minor", "observation": "first observation here for testing"}]}
        first = await _finalize_critic_report_handler(body_args, ctx)
        assert "Critic report committed" in first
        second = await _finalize_critic_report_handler(body_args, ctx)
        assert second.startswith("ERROR")
        assert "already committed" in second
    finally:
        _CRITIC_REPORT_SLOT.reset(token)


# ─── Tool-using dispatch ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_tool_using_runs_explore_then_commit(tmp_path: Path):
    """Tool-using dispatch: the model calls metric tools, then commits
    via finalize_critic_report. dispatch_critic returns a CriticReport
    with the right critic_id and cycle attached."""
    promoted = promote_voice_persona_to_critic(_stage_3_persona())
    # Need all four Tier A or registry must support voice_fidelity at least
    others = [
        CriticPersona(
            id=cid, tier="tier_a", persona=False,
            mechanism="x" * 100, focus_areas=[f"x"],
            model="deepseek-v4-pro",
        )
        for cid in ("payload_enactment", "continuity", "motif_fidelity")
    ]
    registry = CriticRegistry([promoted, *others])
    state = _state(tmp_path, registry=registry)

    captured: dict = {}

    async def fake_query_with_tools(**kwargs):
        captured["tools_seen"] = [t["name"] for t in kwargs["tools"]]
        captured["model"] = kwargs["model_name"]
        # Simulate the agent calling a metric tool then finalizing.
        dispatch = kwargs["dispatch"]
        await dispatch("read_file", {"path": "story.md"})
        await dispatch("finalize_critic_report", {
            "not_load_bearing": False,
            "issues": [
                {
                    "severity": "moderate",
                    "observation": "Voice loosens in the closing scene; rhythm shifts to default.",
                    "scene_id": "closing",
                },
            ],
        })
        return _FakeResult("done", history=[], cost=0.05)

    with patch("owtn.stage_4.critics.query_async_with_tools", new=fake_query_with_tools):
        report = await dispatch_critic(
            VOICE_FIDELITY_ID, state_view=state, cycle=2,
        )

    assert report.critic_id == VOICE_FIDELITY_ID
    assert report.cycle == 2  # force-corrected
    assert len(report.issues) == 1
    assert report.issues[0].severity == Severity.MODERATE
    # Persona's tools surfaced to the LLM along with finalize_critic_report
    assert "stylometry" in captured["tools_seen"]
    assert "slop_score" in captured["tools_seen"]
    assert "writing_style" in captured["tools_seen"]
    assert "read_file" in captured["tools_seen"]
    assert "finalize_critic_report" in captured["tools_seen"]


@pytest.mark.asyncio
async def test_dispatch_tool_using_raises_when_no_finalize(tmp_path: Path):
    """If the explore loop returns without the critic calling
    finalize_critic_report, the dispatcher raises so the orchestrator
    can convert it to a tool-result ERROR."""
    promoted = promote_voice_persona_to_critic(_stage_3_persona())
    registry = CriticRegistry([promoted])
    state = _state(tmp_path, registry=registry)

    async def fake_query_with_tools(**kwargs):
        # Don't dispatch finalize at all.
        await kwargs["dispatch"]("read_file", {"path": "story.md"})
        return _FakeResult("done", history=[], cost=0.01)

    with patch("owtn.stage_4.critics.query_async_with_tools", new=fake_query_with_tools):
        with pytest.raises(RuntimeError, match="did not call finalize_critic_report"):
            await dispatch_critic(VOICE_FIDELITY_ID, state_view=state, cycle=0)


@pytest.mark.asyncio
async def test_tool_using_critic_only_exposes_persona_declared_tools(tmp_path: Path):
    """If the persona declares tools=['stylometry'], the dispatch only
    exposes stylometry + finalize_critic_report — not the full
    CRITIC_TOOL_REGISTRY."""
    promoted = promote_voice_persona_to_critic(_stage_3_persona())
    promoted = promoted.model_copy(update={"tools": ["stylometry"]})
    registry = CriticRegistry([promoted])
    state = _state(tmp_path, registry=registry)

    captured: dict = {}

    async def fake_query_with_tools(**kwargs):
        captured["tools"] = sorted(t["name"] for t in kwargs["tools"])
        await kwargs["dispatch"]("finalize_critic_report", {"issues": []})
        return _FakeResult("done", history=[], cost=0.0)

    with patch("owtn.stage_4.critics.query_async_with_tools", new=fake_query_with_tools):
        await dispatch_critic(VOICE_FIDELITY_ID, state_view=state, cycle=0)

    assert captured["tools"] == ["finalize_critic_report", "stylometry"]


# ─── Session-time swap ───────────────────────────────────────────────────


def test_with_replaced_swaps_voice_fidelity():
    stub = CriticPersona(
        id="voice_fidelity", tier="tier_a", persona=False,
        mechanism="x" * 100, focus_areas=["one"],
        model="deepseek-v4-pro",
    )
    other = CriticPersona(
        id="continuity", tier="tier_a", persona=False,
        mechanism="x" * 100, focus_areas=["one"],
        model="deepseek-v4-pro",
    )
    registry = CriticRegistry([stub, other])
    promoted = promote_voice_persona_to_critic(_stage_3_persona())
    new_registry = registry.with_replaced(promoted)
    assert new_registry.get("voice_fidelity").is_tool_using is True
    assert new_registry.get("continuity") is other
    assert sorted(new_registry.ids()) == ["continuity", "voice_fidelity"]
