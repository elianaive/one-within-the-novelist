"""End-to-end mocked Stage 4 session.

Validates the orchestration spine: filter → PreThink → DownDraft → Revise
(with cycles, prelaunch, plateau wiring) → Stage4SessionResult. Real LLM
calls and real prose quality are NOT validated — that's the live-API
test. This test verifies that the wiring runs end-to-end without
crashing and produces the expected file layout + result shape.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG
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
    CriticReport,
    ExpertNeedsList,
    Issue,
    Severity,
    Stage4Config,
    Stage4SessionResult,
)
from owtn.models.stage_4 import CriticPersona
from owtn.stage_4 import CriticRegistry, run_stage_4_session

from tests.conftest import HILLS_GENOME


# ─── Fakes ───────────────────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, content, history=None, cost: float = 0.0):
        self.content = content
        self.new_msg_history = history or []
        self.cost = cost


def _voice() -> VoiceGenome:
    from tests.conftest import _signature_risk_for_test
    body = "He looked at her. She did not look back. " * 8
    return VoiceGenome(
        pov="third", tense="past",
        consciousness_rendering=ConsciousnessRendering(mode="external_focalization", fid_depth="none"),
        implied_author=ImpliedAuthor(stance_toward_characters="elegiac", moral_temperature="cool"),
        dialogic_mode=DialogicMode(type="monologic"),
        craft=Craft(sentence_rhythm="varied", crowding_leaping="leaping"),
        description="Spare third-person past, elegiac without emotional gloss; dialogue carries weight.",
        diction="Plain Anglo-Saxon nouns; no abstractions of feeling.",
        positive_constraints=["Render scene exits with a stripped declarative."],
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
            "A reader of literary short fiction comfortable with prose "
            "that refuses to interpret for them. Reads slowly when the "
            "writing earns it."
        ),
        recognizes=["free indirect discourse without flagging"],
        tolerates=["ambiguity at the level of plot, not just theme"],
    )


def _synthetic_story(dag: DAG) -> str:
    """Story file content with prose for every scene id, in topological
    order — what a successful DownDraft would produce."""
    topo = dag._check_acyclic_and_topo()
    ordered_ids = sorted(topo.keys(), key=lambda nid: topo[nid])
    parts = []
    for sid in ordered_ids:
        parts.append(
            f"## {sid}\n\nPlaceholder prose for scene {sid}. "
            f"This is a synthetic body the test puts in place of the "
            f"real prose the writer would produce.\n"
        )
    return "\n".join(parts) + "\n"


def _synthetic_prethink(dag: DAG) -> str:
    """Pre-think file with a planning paragraph per scene."""
    topo = dag._check_acyclic_and_topo()
    ordered_ids = sorted(topo.keys(), key=lambda nid: topo[nid])
    parts = []
    for sid in ordered_ids:
        parts.append(f"## {sid}\n\nSynthetic per-scene plan for {sid}.\n")
    return "\n".join(parts) + "\n"


def _criteria_direct_tier_a_registry() -> CriticRegistry:
    """All four Tier A critics as criteria-direct stubs — no tool-using
    path. Useful for tests of the orchestration spine that don't exercise
    voice_fidelity's metric-tool subagent loop (that's tested separately
    in test_voice_fidelity.py)."""
    return CriticRegistry([
        CriticPersona(
            id=cid, tier="tier_a", persona=False,
            mechanism="x" * 100, focus_areas=[f"focus on {cid}"],
            model="deepseek-v4-pro",
        )
        for cid in ("voice_fidelity", "payload_enactment", "continuity", "motif_fidelity")
    ])


# ─── Mocked end-to-end ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_stage_4_session_end_to_end_mocked(tmp_path: Path, canonical_lottery: DAG):
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    voice = _voice()

    phases_seen: list[str] = []
    cycle_idx_seen: list[int] = []
    story_text = _synthetic_story(canonical_lottery)
    prethink_text = _synthetic_prethink(canonical_lottery)

    async def fake_explore(**kwargs):
        """Simulate the writer agent's tool-use loop. Phase routing via
        the available finalize tool — each phase exposes exactly one
        finalize_* tool, so we route on its presence."""
        tool_names = {t["name"] for t in kwargs.get("tools", [])}
        dispatch = kwargs["dispatch"]
        history: list = []

        if "finalize_pre_think" in tool_names:
            phases_seen.append("phase_1")
            r = await dispatch("write_file", {"path": "pre_think.md", "content": prethink_text})
            assert "Wrote" in r, r
            r = await dispatch("finalize_pre_think", {})
            assert "Pre-think committed" in r, r
            return _FakeResult("done", history=history, cost=0.05)

        if "finalize_down_draft" in tool_names:
            phases_seen.append("phase_2")
            r = await dispatch("write_file", {"path": "story.md", "content": story_text})
            assert "Wrote" in r, r
            r = await dispatch("finalize_down_draft", {})
            assert "Down draft committed" in r, r
            return _FakeResult("done", history=history, cost=0.10)

        if "finalize_critique_plan" in tool_names:
            phases_seen.append("phase_3a")
            for cid in ("voice_fidelity", "payload_enactment", "continuity", "motif_fidelity"):
                r = await dispatch("call_critic", {"critic_id": cid})
                assert r.startswith("{"), f"call_critic({cid}) failed: {r}"
            r = await dispatch("finalize_critique_plan", {
                "plan_summary": "Tighten the disclosure beat and sharpen voice in the closing scenes.",
                "intended_revisions": ["Strip emotional gloss from the closing scene."],
            })
            assert "Critique plan committed" in r, r
            return _FakeResult("done", history=history, cost=0.20)

        if "finalize_cycle" in tool_names:
            # Sub-phase B: revise. Cycle 0 finalize_cycle → orchestrator
            # runs plateau (insufficient-history → continue) and begins
            # cycle 1. Cycle 1 sub-phase B calls finalize_stage_4 (now
            # one complete cycle exists, so the gate allows it).
            phases_seen.append("phase_3b")
            cycle = len([p for p in phases_seen if p == "phase_3b"]) - 1
            cycle_idx_seen.append(cycle)
            r = await dispatch("edit_file", {
                "path": "story.md",
                "find": "synthetic body",
                "replace": f"revised body cycle {cycle}",
                "replace_all": True,
            })
            assert "Edited" in r or r.startswith("ERROR: `find` not found"), r
            if cycle == 0:
                r = await dispatch("finalize_cycle", {})
                assert "Revision pass complete" in r, r
            else:
                r = await dispatch("finalize_stage_4", {})
                assert "Stage 4 complete" in r, r
            return _FakeResult("done", history=history, cost=0.15)

        raise AssertionError(f"unrecognized phase: tools={sorted(tool_names)}")

    async def fake_query(**kwargs):
        """LLM calls for the filter and the critics (single-turn structured
        output). All return canned content."""
        output_model = kwargs.get("output_model")
        if output_model is AudienceFraming:
            return _FakeResult(_audience())
        if output_model is ExpertNeedsList:
            return _FakeResult(ExpertNeedsList())
        if output_model is CriticReport:
            return _FakeResult(CriticReport(
                critic_id="placeholder", cycle=0,
                issues=[
                    Issue(severity=Severity.MODERATE,
                          observation="placeholder observation from the mocked critic call for testing"),
                ],
            ))
        raise AssertionError(f"unexpected output_model: {output_model}")

    cfg = Stage4Config()
    cfg.revise.cycle_cap = 3  # tighten test loop bound
    cfg.revise.gather_max_iters = 5
    cfg.revise.revise_max_iters = 5

    with patch("owtn.stage_4.filter.query_async", new=fake_query), \
         patch("owtn.stage_4.critics.query_async", new=fake_query), \
         patch("owtn.stage_4._explore.query_async_with_tools", new=fake_explore):
        result = await run_stage_4_session(
            concept=concept,
            dag=canonical_lottery,
            voice_genome=voice,
            tuple_id="c_test_struct_0_voice_R",
            run_dir=tmp_path / "run",
            config=cfg,
            critic_registry=_criteria_direct_tier_a_registry(),
        )

    assert isinstance(result, Stage4SessionResult)
    assert result.tuple_id == "c_test_struct_0_voice_R"
    # All phases ran in order
    assert phases_seen[:3] == ["phase_1", "phase_2", "phase_3a"]
    assert "phase_3b" in phases_seen
    # Manuscript and pre-think were written to disk
    manuscript = Path(result.manuscript_path)
    pre_think = Path(result.pre_think_path)
    assert manuscript.exists()
    assert pre_think.exists()
    # The synthetic edits ran across both cycles — manuscript reflects revision
    text = manuscript.read_text()
    assert "revised body cycle 0" in text or "revised body cycle 1" in text
    # Cycle bookkeeping: cycle 0 completed via finalize_cycle; cycle 1
    # ended via finalize_stage_4 (which doesn't mark its own cycle completed).
    assert result.cycles_completed == 1
    assert result.exit_reason == "finalize_stage_4"


@pytest.mark.asyncio
async def test_run_stage_4_session_filter_skipped_when_inputs_provided(tmp_path: Path, canonical_lottery: DAG):
    """When audience_framing + expert_needs are passed in, the filter
    should not be called at all."""
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    voice = _voice()
    audience = _audience()
    experts = ExpertNeedsList()
    story_text = _synthetic_story(canonical_lottery)
    prethink_text = _synthetic_prethink(canonical_lottery)

    filter_call_count = 0
    nonlocal_state: dict = {}

    async def fake_explore(**kwargs):
        tool_names = {t["name"] for t in kwargs.get("tools", [])}
        dispatch = kwargs["dispatch"]
        if "finalize_pre_think" in tool_names:
            await dispatch("write_file", {"path": "pre_think.md", "content": prethink_text})
            await dispatch("finalize_pre_think", {})
        elif "finalize_down_draft" in tool_names:
            await dispatch("write_file", {"path": "story.md", "content": story_text})
            await dispatch("finalize_down_draft", {})
        elif "finalize_critique_plan" in tool_names:
            for cid in ("voice_fidelity", "payload_enactment", "continuity", "motif_fidelity"):
                await dispatch("call_critic", {"critic_id": cid})
            await dispatch("finalize_critique_plan", {
                "plan_summary": "Quick test plan summary text for this cycle.",
                "intended_revisions": [],
            })
        elif "finalize_cycle" in tool_names:
            # Two-step: cycle 0 finalize_cycle, cycle 1 finalize_stage_4.
            cycles_so_far_record = nonlocal_state.get("phase_3b_count", 0)
            nonlocal_state["phase_3b_count"] = cycles_so_far_record + 1
            if cycles_so_far_record == 0:
                await dispatch("finalize_cycle", {})
            else:
                await dispatch("finalize_stage_4", {})
        else:
            raise AssertionError(f"unrecognized: {sorted(tool_names)}")
        return _FakeResult("done", cost=0.01)

    async def fake_query(**kwargs):
        nonlocal filter_call_count
        output_model = kwargs.get("output_model")
        if output_model in (AudienceFraming, ExpertNeedsList):
            filter_call_count += 1
            raise AssertionError("filter should not have been called")
        if output_model is CriticReport:
            return _FakeResult(CriticReport(critic_id="placeholder", cycle=0, issues=[]))
        raise AssertionError(f"unexpected: {output_model}")

    cfg = Stage4Config()
    cfg.revise.cycle_cap = 2
    cfg.revise.gather_max_iters = 5
    cfg.revise.revise_max_iters = 5

    with patch("owtn.stage_4.filter.query_async", new=fake_query), \
         patch("owtn.stage_4.critics.query_async", new=fake_query), \
         patch("owtn.stage_4._explore.query_async_with_tools", new=fake_explore):
        result = await run_stage_4_session(
            concept=concept,
            dag=canonical_lottery,
            voice_genome=voice,
            tuple_id="c_skip_filter",
            run_dir=tmp_path / "run",
            config=cfg,
            audience_framing=audience,
            expert_needs=experts,
            critic_registry=_criteria_direct_tier_a_registry(),
        )

    assert filter_call_count == 0
    assert result.exit_reason == "finalize_stage_4"
