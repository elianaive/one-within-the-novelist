"""Tests for owtn.stage_4.critics — registry, dispatch, concurrency."""

from __future__ import annotations

import asyncio
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
from owtn.models.stage_4 import CriticPersona, CriticReport, Issue, Severity
from owtn.orchestration import ToolContext
from owtn.stage_4.critics import (
    CriticRegistry,
    TIER_A_CRITICS,
    _call_critic_handler,
    dispatch_critic,
    prelaunch_critics,
)
from owtn.stage_4.personas import load_critic_pool

from tests.conftest import HILLS_GENOME


# ─── Helpers ─────────────────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, content, cost: float = 0.0):
        self.content = content
        self.new_msg_history: list = []
        self.cost = cost


def _voice_genome() -> VoiceGenome:
    """Minimal valid VoiceGenome — three renderings of >80 chars each."""
    from tests.conftest import _signature_risk_for_test
    body = "He looked at her. She did not look back. " * 8  # ~320 chars
    return VoiceGenome(
        pov="third",
        tense="past",
        consciousness_rendering=ConsciousnessRendering(
            mode="external_focalization", fid_depth="none",
        ),
        implied_author=ImpliedAuthor(
            stance_toward_characters="elegiac", moral_temperature="cool",
        ),
        dialogic_mode=DialogicMode(type="monologic"),
        craft=Craft(sentence_rhythm="varied", crowding_leaping="leaping"),
        description="Spare third-person past, elegiac without emotional gloss, dialogue carrying load.",
        diction="Plain Anglo-Saxon nouns; no abstractions of feeling.",
        positive_constraints=["Render scene exits with a stripped declarative."],
        prohibitions=["No abstract emotion words."],
        signature_risk=_signature_risk_for_test(),
        renderings=[
            Rendering(scene_id="opening", text=body),
            Rendering(scene_id="middle", text=body),
            Rendering(scene_id="closing", text=body),
        ],
        pair_id="c_test",
        agent_id="the-test-agent",
    )


def _continuity_persona() -> CriticPersona:
    return next(p for p in load_critic_pool() if p.id == "continuity")


def _payload_enactment_persona() -> CriticPersona:
    return next(p for p in load_critic_pool() if p.id == "payload_enactment")


def _state(tmp_path: Path, *, with_cycle: bool = True, registry: CriticRegistry | None = None) -> dict:
    """Assemble a state.payload with the registry, work context, and one
    active Phase 3 cycle."""
    if registry is None:
        registry = CriticRegistry([_continuity_persona(), _payload_enactment_persona()])
    state: dict = {
        "run_dir": str(tmp_path),
        "story_path": str(tmp_path / "story.md"),
        "concept": ConceptGenome.model_validate(HILLS_GENOME),
        "voice_genome": _voice_genome(),
        "dag_rendering": "NODE n0 [reveal]\n  sketch: train station\n",
        "critic_registry": registry,
    }
    (tmp_path / "story.md").write_text("## opening\n\nThe room was quiet. He spoke first.\n")
    if with_cycle:
        state["phase_3_revise"] = {
            "cycles": [
                {"cycle": 0, "critic_calls": [], "plan": None, "completed": False},
            ],
        }
    return state


def _ctx(state: dict, *, phase_id: str = "phase_3a_gather") -> ToolContext:
    return ToolContext(
        session_id="sess_test", phase_id=phase_id,
        agent_id="stage_4_agent", state_view=state,
    )


def _fake_report(critic_id: str, cycle: int, *, issues: int = 1) -> CriticReport:
    return CriticReport(
        critic_id=critic_id,
        cycle=cycle,
        issues=[
            Issue(severity=Severity.MODERATE, observation=f"observation #{i} from {critic_id} for testing")
            for i in range(issues)
        ],
    )


# ─── CriticRegistry ──────────────────────────────────────────────────────


def test_registry_basic_lookup():
    reg = CriticRegistry([_continuity_persona(), _payload_enactment_persona()])
    assert "continuity" in reg
    assert "voice_fidelity" not in reg
    assert reg.get("continuity").id == "continuity"
    assert len(reg) == 2


def test_registry_rejects_duplicate_ids():
    p = _continuity_persona()
    with pytest.raises(ValueError, match="duplicate critic id"):
        CriticRegistry([p, p])


def test_registry_unknown_id_raises():
    reg = CriticRegistry([_continuity_persona()])
    with pytest.raises(KeyError) as exc:
        reg.get("nonexistent")
    assert "available" in str(exc.value)


def test_registry_tier_partitioning():
    """Two starter critics — both Tier A — should appear in tier_a_ids
    and not in tier_b_ids."""
    reg = CriticRegistry([_continuity_persona(), _payload_enactment_persona()])
    assert reg.tier_a_ids() == ["continuity", "payload_enactment"]
    assert reg.tier_b_ids() == []


def test_registered_critics_match_tier_a_constant():
    """The two committed critics must be in TIER_A_CRITICS — sanity check
    that the persona-tier flag stays in sync with the constant."""
    reg = CriticRegistry([_continuity_persona(), _payload_enactment_persona()])
    for cid in reg.tier_a_ids():
        assert cid in TIER_A_CRITICS


def test_render_critic_list_surfaces_domain_experts():
    """Domain-expert critics created by the filter must appear in the
    revise_gather critic list with their domain descriptor — otherwise
    the writer can't see them and won't call them. Regression for the
    deodand-pilot bug where two ExpertNeed specs were instantiated but
    invisible to the writer."""
    from owtn.models.stage_4 import ExpertNeed, ExpertNeedsList
    from owtn.stage_4.domain_expert import instantiate_domain_experts
    from owtn.stage_4.revise import _render_critic_list

    needs = ExpertNeedsList(experts=[
        ExpertNeed(
            domain="transformer architecture / NLP engineering",
            persona_hint="ML researcher who has trained models like the one in the story",
            expertise_focus=["model dimensionality conventions", "tokenizer vocabularies"],
            web_search_recommended=True,
        ),
    ])
    experts = instantiate_domain_experts(needs)
    reg = CriticRegistry([_continuity_persona(), _payload_enactment_persona(), *experts])

    lines = _render_critic_list(reg)
    domain_lines = [line for line in lines if line.startswith("- domain_expert_")]
    assert len(domain_lines) == 1
    assert "transformer architecture" in domain_lines[0]


# ─── dispatch_critic ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_critic_returns_report_and_force_corrects_identity(tmp_path: Path):
    state = _state(tmp_path)

    async def fake_query(**kwargs):
        # The model returns a report with WRONG critic_id and cycle —
        # dispatch_critic must overwrite both post-parse.
        return _FakeResult(_fake_report("HALLUCINATED", 99))

    with patch("owtn.stage_4.critics.query_async", new=fake_query):
        report = await dispatch_critic("continuity", state_view=state, cycle=3)

    assert report.critic_id == "continuity"
    assert report.cycle == 3
    assert len(report.issues) == 1


@pytest.mark.asyncio
async def test_dispatch_critic_unknown_id_raises(tmp_path: Path):
    state = _state(tmp_path)
    with pytest.raises(KeyError):
        await dispatch_critic("nonexistent_critic", state_view=state, cycle=0)


@pytest.mark.asyncio
async def test_tool_using_critic_nudge_finalizes_when_explore_hits_max_iters(tmp_path: Path):
    """When a tool-using critic exhausts its explore budget without
    calling finalize_critic_report, the nudge fallback should re-prompt
    with only that tool available, and the critic should commit. This
    is the same nudge-restriction pattern from Stage 4's PreThink and
    DownDraft phases.

    Repro from `2026-04-30-stage-4-domain-expert-critic-nudge.md`: the
    medieval-deodand expert in cycle 2 ran 10 iterations of web_search
    without finalizing and crashed; with the nudge, that call would
    have committed its partial report instead.
    """
    from owtn.stage_4.critics import (
        FINALIZE_CRITIC_REPORT,
        _CRITIC_REPORT_SLOT,
        _dispatch_tool_using,
    )
    from owtn.models.stage_4 import CriticReportBody

    persona = CriticPersona(
        id="domain_expert_test",
        tier="domain",
        persona=False,
        mechanism="x" * 60,
        focus_areas=["test"],
        model="deepseek-v4-pro",
        tools=["read_file"],  # plus implicit FINALIZE_CRITIC_REPORT
    )

    call_count = {"n": 0}

    async def fake_with_tools(*, model_name, msg, system_msg, tools, dispatch, max_iters, msg_history=None, **kw):
        call_count["n"] += 1
        from owtn.llm.result import QueryResult
        if call_count["n"] == 1:
            # Explore call: simulates the model burning iters without finalizing.
            # No tool dispatch happens — return without setting slot.body.
            return QueryResult(
                content="",
                msg=msg,
                system_msg=system_msg,
                new_msg_history=[],
                model_name=model_name,
                kwargs={},
                input_tokens=0,
                output_tokens=0,
                thinking_tokens=0,
                input_cost=0.0,
                output_cost=0.0,
                cost=0.0,
                cache_read_tokens=0,
            )
        # Nudge call: tool surface should be just finalize_critic_report.
        assert len(tools) == 1, f"nudge should restrict to finalize only; got {[t['name'] for t in tools]}"
        assert tools[0]["name"] == FINALIZE_CRITIC_REPORT.name
        # Simulate the critic firing finalize during the nudge.
        body = CriticReportBody(
            focus="committed under nudge",
            issues=[Issue(severity=Severity.MODERATE, observation="partial finding from the nudge fallback")],
        )
        slot = _CRITIC_REPORT_SLOT.get()
        if slot is not None:
            slot["body"] = body
        return QueryResult(
            content="",
            msg=msg,
            system_msg=system_msg,
            new_msg_history=[],
            model_name=model_name,
            kwargs={},
            input_tokens=0,
            output_tokens=0,
            thinking_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            cost=0.0,
            cache_read_tokens=0,
        )

    with patch("owtn.stage_4.critics.query_async_with_tools", new=fake_with_tools):
        body, msg_history = await _dispatch_tool_using(
            persona,
            system_msg="critic system",
            user_msg="critic user",
            state_payload={},
            max_iters=10,
        )

    assert call_count["n"] == 2, "explore + nudge = 2 calls"
    assert body.focus == "committed under nudge"
    assert len(body.issues) == 1


@pytest.mark.asyncio
async def test_tool_using_critic_skips_nudge_when_explore_commits(tmp_path: Path):
    """If the explore loop commits cleanly, the nudge does not fire."""
    from owtn.stage_4.critics import _CRITIC_REPORT_SLOT, _dispatch_tool_using
    from owtn.models.stage_4 import CriticReportBody

    persona = CriticPersona(
        id="continuity_test",
        tier="tier_a",
        persona=False,
        mechanism="x" * 60,
        focus_areas=["test"],
        model="deepseek-v4-pro",
        tools=["read_file"],
    )

    call_count = {"n": 0}

    async def fake_with_tools(*, model_name, msg, system_msg, tools, dispatch, max_iters, msg_history=None, **kw):
        from owtn.llm.result import QueryResult
        call_count["n"] += 1
        # Commit on the explore call.
        body = CriticReportBody(
            focus="explore committed",
            issues=[Issue(severity=Severity.MINOR, observation="explore-time observation")],
        )
        slot = _CRITIC_REPORT_SLOT.get()
        if slot is not None:
            slot["body"] = body
        return QueryResult(
            content="", msg=msg, system_msg=system_msg, new_msg_history=[],
            model_name=model_name, kwargs={}, input_tokens=0, output_tokens=0,
            thinking_tokens=0, input_cost=0.0, output_cost=0.0, cost=0.0, cache_read_tokens=0,
        )

    with patch("owtn.stage_4.critics.query_async_with_tools", new=fake_with_tools):
        body, _ = await _dispatch_tool_using(
            persona,
            system_msg="s", user_msg="u", state_payload={}, max_iters=10,
        )

    assert call_count["n"] == 1, "no nudge needed when explore commits"
    assert body.focus == "explore committed"


@pytest.mark.asyncio
async def test_tool_using_critic_raises_when_nudge_also_fails(tmp_path: Path):
    """If even the nudge can't elicit a commit, dispatch raises."""
    from owtn.stage_4.critics import _dispatch_tool_using

    persona = CriticPersona(
        id="domain_expert_stuck",
        tier="domain",
        persona=False,
        mechanism="x" * 60,
        focus_areas=["test"],
        model="deepseek-v4-pro",
        tools=["read_file"],
    )

    async def fake_with_tools(*, model_name, msg, system_msg, tools, dispatch, max_iters, msg_history=None, **kw):
        from owtn.llm.result import QueryResult
        # Never commit. Two consecutive empty runs.
        return QueryResult(
            content="", msg=msg, system_msg=system_msg, new_msg_history=[],
            model_name=model_name, kwargs={}, input_tokens=0, output_tokens=0,
            thinking_tokens=0, input_cost=0.0, output_cost=0.0, cost=0.0, cache_read_tokens=0,
        )

    with patch("owtn.stage_4.critics.query_async_with_tools", new=fake_with_tools):
        with pytest.raises(RuntimeError, match="did not call finalize_critic_report"):
            await _dispatch_tool_using(
                persona,
                system_msg="s", user_msg="u", state_payload={}, max_iters=10,
            )


@pytest.mark.asyncio
async def test_dispatch_critic_passes_focus_to_prompt(tmp_path: Path):
    state = _state(tmp_path)
    captured: dict = {}

    async def fake_query(**kwargs):
        captured.update(kwargs)
        return _FakeResult(_fake_report("continuity", 0))

    with patch("owtn.stage_4.critics.query_async", new=fake_query):
        await dispatch_critic(
            "continuity", state_view=state, cycle=0, focus="scene weights-freeze",
        )

    user_msg = captured["msg"]
    assert "Focus: scene weights-freeze" in user_msg


# ─── prelaunch_critics ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prelaunch_critics_creates_tasks(tmp_path: Path):
    state = _state(tmp_path)

    async def fake_query(**kwargs):
        return _FakeResult(_fake_report("continuity", 0))

    with patch("owtn.stage_4.critics.query_async", new=fake_query):
        await prelaunch_critics(["continuity", "payload_enactment"], state_payload=state, cycle=0)
        prelaunched = state["phase_3_revise"]["cycles"][0]["prelaunched"]
        assert set(prelaunched.keys()) == {"continuity", "payload_enactment"}
        assert all(isinstance(t, asyncio.Task) for t in prelaunched.values())
        # Drain so the event loop doesn't warn about un-awaited tasks
        await asyncio.gather(*prelaunched.values())


@pytest.mark.asyncio
async def test_prelaunch_critics_idempotent(tmp_path: Path):
    state = _state(tmp_path)

    async def fake_query(**kwargs):
        return _FakeResult(_fake_report("continuity", 0))

    with patch("owtn.stage_4.critics.query_async", new=fake_query):
        await prelaunch_critics(["continuity"], state_payload=state, cycle=0)
        first_task = state["phase_3_revise"]["cycles"][0]["prelaunched"]["continuity"]
        await prelaunch_critics(["continuity"], state_payload=state, cycle=0)
        # Same Task object — not relaunched.
        assert state["phase_3_revise"]["cycles"][0]["prelaunched"]["continuity"] is first_task
        await first_task


@pytest.mark.asyncio
async def test_prelaunch_critics_skips_unknown(tmp_path: Path):
    state = _state(tmp_path)

    async def fake_query(**kwargs):
        return _FakeResult(_fake_report("continuity", 0))

    with patch("owtn.stage_4.critics.query_async", new=fake_query):
        await prelaunch_critics(
            ["continuity", "nonexistent_critic"], state_payload=state, cycle=0,
        )
        prelaunched = state["phase_3_revise"]["cycles"][0]["prelaunched"]
        assert "continuity" in prelaunched
        assert "nonexistent_critic" not in prelaunched
        await prelaunched["continuity"]


# ─── _call_critic_handler ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_critic_handler_errors_when_no_active_cycle(tmp_path: Path):
    state = _state(tmp_path, with_cycle=False)
    result = await _call_critic_handler({"critic_id": "continuity"}, _ctx(state))
    assert result.startswith("ERROR")
    assert "no active Phase 3 cycle" in result


@pytest.mark.asyncio
async def test_call_critic_handler_errors_when_registry_not_set(tmp_path: Path):
    state = _state(tmp_path)
    del state["critic_registry"]
    result = await _call_critic_handler({"critic_id": "continuity"}, _ctx(state))
    assert result.startswith("ERROR")
    assert "critic_registry" in result


@pytest.mark.asyncio
async def test_call_critic_handler_errors_on_unknown_id(tmp_path: Path):
    state = _state(tmp_path)
    result = await _call_critic_handler({"critic_id": "fake"}, _ctx(state))
    assert result.startswith("ERROR")
    assert "unknown critic_id" in result
    assert "continuity" in result  # lists what's available


@pytest.mark.asyncio
async def test_call_critic_handler_errors_on_empty_id(tmp_path: Path):
    state = _state(tmp_path)
    result = await _call_critic_handler({}, _ctx(state))
    assert result.startswith("ERROR")
    assert "non-empty `critic_id`" in result


@pytest.mark.asyncio
async def test_call_critic_handler_runs_fresh_when_no_prelaunch(tmp_path: Path):
    """Cycles 2+ Tier B is on-demand — no prelaunched task; handler runs
    fresh and tracks the call."""
    state = _state(tmp_path)

    async def fake_query(**kwargs):
        return _FakeResult(_fake_report("continuity", 0, issues=2))

    with patch("owtn.stage_4.critics.query_async", new=fake_query):
        result = await _call_critic_handler({"critic_id": "continuity"}, _ctx(state))

    assert result.startswith("{")  # JSON CriticReport
    assert "continuity" in state["phase_3_revise"]["cycles"][0]["critic_calls"]


@pytest.mark.asyncio
async def test_call_critic_handler_awaits_prelaunch_when_no_focus(tmp_path: Path):
    state = _state(tmp_path)
    call_count = 0

    async def fake_query(**kwargs):
        nonlocal call_count
        call_count += 1
        return _FakeResult(_fake_report("continuity", 0))

    with patch("owtn.stage_4.critics.query_async", new=fake_query):
        await prelaunch_critics(["continuity"], state_payload=state, cycle=0)
        result = await _call_critic_handler({"critic_id": "continuity"}, _ctx(state))

    assert result.startswith("{")
    # Only one LLM call — the handler awaited the prelaunched task,
    # didn't run a fresh dispatch.
    assert call_count == 1
    assert "continuity" in state["phase_3_revise"]["cycles"][0]["critic_calls"]
    # Prelaunched entry consumed by the handler so it's not awaited twice.
    assert "continuity" not in state["phase_3_revise"]["cycles"][0]["prelaunched"]


@pytest.mark.asyncio
async def test_call_critic_handler_cancels_prelaunch_when_focus_given(tmp_path: Path):
    """A focused call should cancel the unfocused pre-launch and run
    fresh. Verifies the cancel-on-divergent-focus semantics."""
    state = _state(tmp_path)
    captured_focuses: list[str | None] = []

    # Use an asyncio.Event so the prelaunched task is still in-flight when
    # the handler fires — proves cancellation happens, not just
    # already-completed.
    pre_started = asyncio.Event()
    pre_release = asyncio.Event()

    async def fake_query(**kwargs):
        # The user-msg encodes whether `focus` was passed; capture for assertion.
        msg = kwargs.get("msg", "")
        captured_focuses.append("Focus:" in msg and msg.split("Focus: ", 1)[1].split("\n", 1)[0] or None)
        # If this is the unfocused prelaunch, hold here so cancel hits a
        # live coroutine.
        if "Focus:" not in msg:
            pre_started.set()
            await pre_release.wait()  # never set — task gets cancelled instead
        return _FakeResult(_fake_report("continuity", 0))

    with patch("owtn.stage_4.critics.query_async", new=fake_query):
        await prelaunch_critics(["continuity"], state_payload=state, cycle=0)
        # Wait for the prelaunched coroutine to be in-flight.
        await asyncio.wait_for(pre_started.wait(), timeout=1.0)
        result = await _call_critic_handler(
            {"critic_id": "continuity", "focus": "the disclosure beat"}, _ctx(state),
        )

    assert result.startswith("{")
    # Two query attempts: one prelaunched (cancelled mid-flight), one fresh with focus.
    assert len(captured_focuses) == 2
    # The fresh call carried the focus.
    assert any("disclosure" in f for f in captured_focuses if f)


@pytest.mark.asyncio
async def test_call_critic_handler_dedup_critic_calls(tmp_path: Path):
    """Calling the same critic twice in one cycle should only append once
    to critic_calls — Tier A enforcement reads this list as a set."""
    state = _state(tmp_path)

    async def fake_query(**kwargs):
        return _FakeResult(_fake_report("continuity", 0))

    with patch("owtn.stage_4.critics.query_async", new=fake_query):
        await _call_critic_handler({"critic_id": "continuity"}, _ctx(state))
        await _call_critic_handler({"critic_id": "continuity"}, _ctx(state))

    calls = state["phase_3_revise"]["cycles"][0]["critic_calls"]
    assert calls == ["continuity"]


@pytest.mark.asyncio
async def test_call_critic_handler_returns_error_on_dispatch_failure(tmp_path: Path):
    state = _state(tmp_path)

    async def fake_query(**kwargs):
        raise RuntimeError("simulated provider outage")

    with patch("owtn.stage_4.critics.query_async", new=fake_query):
        result = await _call_critic_handler({"critic_id": "continuity"}, _ctx(state))

    assert result.startswith("ERROR")
    assert "RuntimeError" in result
    # Failed call should NOT have been tracked.
    assert state["phase_3_revise"]["cycles"][0]["critic_calls"] == []


# ─── End-to-end with finalize_critique_plan ──────────────────────────────


@pytest.mark.asyncio
async def test_call_critic_then_finalize_critique_plan_flow(tmp_path: Path):
    """Sub-phase A flow: agent fires the four Tier A critics via the
    handler (which tracks them), then commits the plan. Tier A
    enforcement gate passes once all four have fired."""
    from owtn.stage_4.tools import _finalize_critique_plan_handler

    # Build a registry containing all four Tier A critics. We only have
    # YAMLs for two of them; create stub personas for the other two so
    # the registry recognizes their ids.
    real = [_continuity_persona(), _payload_enactment_persona()]
    stub_voice = CriticPersona(
        id="voice_fidelity", tier="tier_a", persona=True,
        mechanism="x" * 100, focus_areas=["voice match"],
        model="deepseek-v4-pro",
    )
    stub_motif = CriticPersona(
        id="motif_fidelity", tier="tier_a", persona=False,
        mechanism="x" * 100, focus_areas=["motif fidelity"],
        model="deepseek-v4-pro",
    )
    registry = CriticRegistry([*real, stub_voice, stub_motif])
    state = _state(tmp_path, registry=registry)

    async def fake_query(**kwargs):
        return _FakeResult(_fake_report("placeholder", 0))

    with patch("owtn.stage_4.critics.query_async", new=fake_query):
        for cid in TIER_A_CRITICS:
            res = await _call_critic_handler({"critic_id": cid}, _ctx(state))
            assert res.startswith("{"), f"call_critic for {cid} failed: {res}"

        plan_result = await _finalize_critique_plan_handler(
            {
                "plan_summary": "Tighten voice in scenes 2-3; sharpen the disclosure beat in 4.",
                "intended_revisions": ["Strip emotional gloss from scene 2 opening."],
            },
            _ctx(state),
        )

    assert "Critique plan committed" in plan_result
    plan = state["phase_3_revise"]["cycles"][0]["plan"]
    assert plan["plan_summary"].startswith("Tighten voice")
