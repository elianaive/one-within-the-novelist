"""Tests for owtn.stage_3.phases — phase implementations with mocked LLMs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from owtn.models.stage_3 import (
    BordaRanking,
    ConsciousnessRendering,
    Craft,
    Critique,
    CritiqueBody,
    CritiqueSet,
    DialogicMode,
    ImpliedAuthor,
    Rendering,
    VoiceGenome,
    VoiceGenomeBody,
)
from owtn.orchestration import Agent, SessionState, ToolRegistry
from owtn.stage_3.phases import (
    BordaPhase,
    PrivateBriefPhase,
    RevealCritiquePhase,
    RevisePhase,
)
from owtn.stage_3.tools import ALL_VOICE_TOOLS, VOICE_PHASE_ALLOW


# ─── Fixtures ────────────────────────────────────────────────────────────


def _bench_payload() -> dict:
    return {
        "drafts": [
            {
                "scene_id": "morning-kitchen",
                "synopsis": "She makes coffee while he reads.",
                "demand": "Render slow domestic time without filler.",
                "why_distinct": "Tests stillness; others test motion.",
                "neutral_draft": "She poured the coffee. " * 10,
            },
            {
                "scene_id": "argument-at-door",
                "synopsis": "They disagree about whether to leave.",
                "demand": "Hold pressure inside ordinary speech.",
                "why_distinct": "Tests motion under speech.",
                "neutral_draft": "He said no. She did not move. " * 10,
            },
            {
                "scene_id": "after-the-news",
                "synopsis": "They sit with what was said.",
                "demand": "Render aftermath without resolution.",
                "why_distinct": "Tests stillness after rupture.",
                "neutral_draft": "The room held. The clock kept time. " * 10,
            },
        ],
        "bench_rationale": "rationale long enough to pass validation",
        "picker_model": "claude-sonnet-4-6",
        "drafter_model": "claude-sonnet-4-6",
    }


def _agent(agent_id: str, model: str = "deepseek-v4-pro") -> Agent:
    return Agent(
        id=agent_id,
        system_prompt=f"system prompt for {agent_id}",
        model=model,
        sampler={"temperature": 1.2},
        tools=frozenset({
            "render_adjacent_scene", "think", "note_to_self", "lookup_reference",
            "thesaurus", "stylometry", "slop_score", "writing_style",
            "finalize_voice_genome",
        }),
    )


def _voice_body(agent_id: str) -> VoiceGenomeBody:
    """A valid VoiceGenomeBody whose renderings match the bench's scene_ids."""
    text = (
        "She set the cup down and did not look up. "
        "He waited a long time before he spoke. "
        "The kitchen was cold. The clock on the stove read four."
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
            f"Voice for {agent_id}: holds back where most prose explains, "
            "trusting the reader to assemble feeling from gesture."
        ),
        diction="Plain declarative; no figurative ornament.",
        positive_constraints=[
            "Render emotion through the body's small refusals.",
        ],
        renderings=[
            Rendering(scene_id="morning-kitchen", text=text),
            Rendering(scene_id="argument-at-door", text=text),
            Rendering(scene_id="after-the-news", text=text),
        ],
    )


class _FakeResult:
    def __init__(self, content, history=None, cost=0.0):
        self.content = content
        self.new_msg_history = history or []
        self.cost = cost


def _make_state(agents: list[Agent], pair_id: str = "c_test_struct_0") -> SessionState:
    state = SessionState.new(agents, pair_id=pair_id, session_id="sess_test")
    from tests.conftest import HILLS_GENOME
    from owtn.models.stage_1.concept_genome import ConceptGenome
    state.payload["concept"] = ConceptGenome.model_validate(HILLS_GENOME)
    state.payload["dag_rendering"] = "NODE n0 [reveal]\n  sketch: train station\n"
    state.payload["adjacent_scene_bench"] = _bench_payload()
    state.payload["agent_models"] = {a.id: a.model for a in agents}
    return state


# ─── PrivateBriefPhase ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_private_brief_phase_runs_all_agents_in_parallel():
    agents = [_agent("the-reductionist"), _agent("the-temporal-collagist")]
    state = _make_state(agents)
    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)

    async def fake_explore(**kwargs):
        # Simulate the agent calling finalize_voice_genome during explore by
        # invoking dispatch directly — this writes to _pending_commits.
        agent_id = kwargs["system_msg"].split()[-1]
        body = _voice_body(agent_id)
        await kwargs["dispatch"]("finalize_voice_genome", body.model_dump())
        return _FakeResult(
            "exploration done",
            history=[
                {"role": "user", "content": kwargs["msg"]},
                {"role": "assistant", "content": "ok"},
            ],
            cost=0.01,
        )

    with patch("owtn.stage_3.phases.query_async_with_tools", new=fake_explore):
        phase = PrivateBriefPhase()
        new_state = await phase.run(agents, state, registry)

    assert "phase_1_private_brief" in new_state.payload
    proposals = new_state.payload["phase_1_private_brief"]
    assert set(proposals.keys()) == {"the-reductionist", "the-temporal-collagist"}
    for agent_id, genome in proposals.items():
        assert genome.agent_id == agent_id
        assert genome.pair_id == "c_test_struct_0"
        assert {r.scene_id for r in genome.renderings} == {
            "morning-kitchen", "argument-at-door", "after-the-news",
        }
    # Cost accumulated from explore × both agents (no fallback fired)
    assert new_state.cost_usd == pytest.approx(0.01 * 2)


@pytest.mark.asyncio
async def test_private_brief_rejects_renderings_with_wrong_scene_ids():
    """Agent that produces renderings with scene_ids not in the bench
    should fail the post-commit validation."""
    agents = [_agent("the-reductionist")]
    state = _make_state(agents)
    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)

    bad_body = _voice_body("the-reductionist")
    bad_body.renderings[0] = Rendering(
        scene_id="not-a-real-scene", text="x" * 100,
    )

    async def fake_explore(**kwargs):
        await kwargs["dispatch"]("finalize_voice_genome", bad_body.model_dump())
        return _FakeResult("done", history=[], cost=0.0)

    with patch("owtn.stage_3.phases.query_async_with_tools", new=fake_explore):
        phase = PrivateBriefPhase()
        with pytest.raises(RuntimeError, match="do not match bench"):
            await phase.run(agents, state, registry)


@pytest.mark.asyncio
async def test_private_brief_requires_concept_and_bench_in_payload():
    agents = [_agent("the-reductionist")]
    state = SessionState.new(agents, pair_id="x")
    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)

    phase = PrivateBriefPhase()
    with pytest.raises(ValueError, match="state.payload"):
        await phase.run(agents, state, registry)


@pytest.mark.asyncio
async def test_private_brief_nudge_fires_when_explore_did_not_commit():
    """When the explore loop ends without a finalize_voice_genome call, a
    tool-use nudge runs with msg_history threaded through and tools
    restricted to finalize_voice_genome only."""
    agents = [_agent("the-reductionist")]
    state = _make_state(agents)
    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)

    explore_history = [
        {"role": "user", "content": "develop voice"},
        {"role": "assistant", "content": "I'll start by reading drafts."},
        {"role": "user", "content": "tool result"},
        {"role": "assistant", "content": "Still iterating."},
    ]

    captured = {}
    call_count = {"n": 0}

    async def fake_query_with_tools(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Explore call: returns history without committing
            return _FakeResult("done", history=list(explore_history), cost=0.01)
        # Nudge call: capture, then commit via dispatch
        captured["msg_history"] = kwargs.get("msg_history")
        captured["tools"] = kwargs.get("tools")
        captured["msg"] = kwargs.get("msg")
        await kwargs["dispatch"]("finalize_voice_genome", _voice_body("the-reductionist").model_dump())
        return _FakeResult("committed", history=list(explore_history) + [{"role": "user", "content": "x"}], cost=0.005)

    with patch("owtn.stage_3.phases.query_async_with_tools", new=fake_query_with_tools):
        phase = PrivateBriefPhase()
        await phase.run(agents, state, registry)

    assert call_count["n"] == 2
    assert captured["msg_history"] == explore_history
    assert [t["name"] for t in captured["tools"]] == ["finalize_voice_genome"]
    assert "finalize_voice_genome" in captured["msg"]


# ─── RevealCritiquePhase ─────────────────────────────────────────────────


def _make_voice_genome(agent_id: str, pair_id: str = "c_test_struct_0") -> VoiceGenome:
    body = _voice_body(agent_id)
    return VoiceGenome(**body.model_dump(), pair_id=pair_id, agent_id=agent_id)


@pytest.mark.asyncio
async def test_reveal_critique_runs_for_all_agents():
    agents = [
        _agent("the-reductionist"),
        _agent("the-temporal-collagist"),
        _agent("the-sensory-materialist"),
    ]
    state = _make_state(agents)
    state.payload["phase_1_private_brief"] = {
        a.id: _make_voice_genome(a.id) for a in agents
    }

    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)

    async def fake_query(**kwargs):
        # Identify the critic by the system_msg suffix; produce a CritiqueSet
        # targeting the OTHER two agents.
        critic = kwargs["system_msg"].split()[-1]
        targets = [a.id for a in agents if a.id != critic]
        cs = CritiqueSet(
            critiques=[
                CritiqueBody(
                    target_id=t,
                    strengths=["concrete strength one for review", "specific strength two"],
                    concern=f"concern about {t} from {critic}'s perspective for testing",
                )
                for t in targets
            ],
        )
        return _FakeResult(cs, cost=0.002)

    with patch("owtn.stage_3.phases.query_async", new=fake_query):
        phase = RevealCritiquePhase()
        new_state = await phase.run(agents, state, registry)

    critiques = new_state.payload["phase_3_reveal_critique"]
    assert set(critiques.keys()) == {a.id for a in agents}
    for critic_id, lst in critiques.items():
        assert len(lst) == 2  # critiques the other two
        assert all(c.critic_id == critic_id for c in lst)
        assert critic_id not in {c.target_id for c in lst}


@pytest.mark.asyncio
async def test_reveal_critique_rejects_missing_target():
    agents = [_agent("the-reductionist"), _agent("the-temporal-collagist")]
    state = _make_state(agents)
    state.payload["phase_1_private_brief"] = {
        a.id: _make_voice_genome(a.id) for a in agents
    }
    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)

    async def fake_query(**kwargs):
        # Return empty target list — should fail target coverage check
        cs = CritiqueSet(critiques=[
            CritiqueBody(
                target_id="nonexistent-agent",
                strengths=["one", "two"],
                concern="concern text long enough to pass minimum",
            ),
        ])
        return _FakeResult(cs, cost=0.0)

    with patch("owtn.stage_3.phases.query_async", new=fake_query):
        phase = RevealCritiquePhase()
        with pytest.raises(RuntimeError, match="target coverage"):
            await phase.run(agents, state, registry)


# ─── RevisePhase ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_revise_phase_routes_critiques_to_their_targets():
    agents = [_agent("the-reductionist"), _agent("the-temporal-collagist")]
    state = _make_state(agents)
    state.payload["phase_1_private_brief"] = {
        a.id: _make_voice_genome(a.id) for a in agents
    }
    state.payload["phase_3_reveal_critique"] = {
        "the-reductionist": [
            Critique(
                critic_id="the-reductionist",
                target_id="the-temporal-collagist",
                strengths=["one strength", "two strength"],
                concern="concern about temporal-collagist's voice consistency",
            ),
        ],
        "the-temporal-collagist": [
            Critique(
                critic_id="the-temporal-collagist",
                target_id="the-reductionist",
                strengths=["one strength", "two strength"],
                concern="concern about reductionist's compression breaking continuity",
            ),
        ],
    }
    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)

    captured_msgs: dict[str, str] = {}

    async def fake_explore(**kwargs):
        agent_id = kwargs["system_msg"].split()[-1]
        # Phase 4 sends the revise prompt as msg on the explore call;
        # the nudge fallback would send a different msg. Only capture the
        # first call per agent (the explore prompt).
        if agent_id not in captured_msgs:
            captured_msgs[agent_id] = kwargs["msg"]
        body = _voice_body(agent_id)
        await kwargs["dispatch"]("finalize_voice_genome", body.model_dump())
        return _FakeResult("done", history=[], cost=0.01)

    with patch("owtn.stage_3.phases.query_async_with_tools", new=fake_explore):
        phase = RevisePhase()
        new_state = await phase.run(agents, state, registry)

    # Reductionist should have received the critique FROM temporal-collagist
    assert "compression breaking" in captured_msgs["the-reductionist"]
    # And vice versa
    assert "voice consistency" in captured_msgs["the-temporal-collagist"]

    revised = new_state.payload["phase_4_revise"]
    assert set(revised.keys()) == {"the-reductionist", "the-temporal-collagist"}
    for aid, g in revised.items():
        assert g.agent_id == aid
        assert g.pair_id == "c_test_struct_0"


# ─── BordaPhase ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_borda_phase_aggregates_rankings():
    agents = [_agent(f"a{i}") for i in range(3)]
    state = _make_state(agents)
    state.payload["phase_4_revise"] = {a.id: _make_voice_genome(a.id) for a in agents}
    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)

    # Each agent ranks the others; pick simple deterministic orderings.
    expected_rankings = {
        "a0": ["a1", "a2"],  # a1 first
        "a1": ["a2", "a0"],  # a2 first
        "a2": ["a1", "a0"],  # a1 first
    }
    # Borda points: a0 gets 0+0=0, a1 gets 1+1=2, a2 gets 1+0=1

    async def fake_query(**kwargs):
        # Identify the agent by system_msg suffix
        ranker = kwargs["system_msg"].split()[-1]
        return _FakeResult(BordaRanking(ranking=expected_rankings[ranker]), cost=0.001)

    with patch("owtn.stage_3.phases.query_async", new=fake_query):
        phase = BordaPhase()
        new_state = await phase.run(agents, state, registry)

    out = new_state.payload["phase_5_borda"]
    assert out["rankings"] == expected_rankings
    assert out["points"] == {"a0": 0, "a1": 2, "a2": 1}


def test_summarize_tool_calls_walks_history():
    """Helper that surfaces tool calls + their parameter keys for inspection."""
    from owtn.stage_3.phases import _summarize_tool_calls

    history = [
        {"role": "user", "content": "develop a voice"},
        {
            "role": "assistant",
            "content": "I'll start by reading scenes.",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {
                        "name": "render_adjacent_scene",
                        "arguments": '{"scene_id": "morning-platform"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "tc_1", "content": "{...draft...}"},
        {
            "role": "assistant",
            "content": "Now metric check.",
            "tool_calls": [
                {
                    "id": "tc_2",
                    "type": "function",
                    "function": {
                        "name": "stylometry",
                        "arguments": '{"passage": "...", "scene_id": "morning-platform"}',
                    },
                },
                {
                    "id": "tc_3",
                    "type": "function",
                    "function": {
                        "name": "slop_score",
                        "arguments": '{"passage": "..."}',
                    },
                },
            ],
        },
    ]

    summary = _summarize_tool_calls(history)
    assert len(summary) == 3
    assert summary[0] == {"tool": "render_adjacent_scene", "args_keys": ["scene_id"]}
    assert summary[1] == {"tool": "stylometry", "args_keys": ["passage", "scene_id"]}
    assert summary[2] == {"tool": "slop_score", "args_keys": ["passage"]}


def test_summarize_tool_calls_handles_unparseable_args():
    from owtn.stage_3.phases import _summarize_tool_calls

    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "thesaurus", "arguments": "garbage{"},
                },
            ],
        },
    ]
    summary = _summarize_tool_calls(history)
    assert summary == [{"tool": "thesaurus", "args_keys": ["<unparsed>"]}]


def test_summarize_tool_calls_empty_history_returns_empty():
    from owtn.stage_3.phases import _summarize_tool_calls
    assert _summarize_tool_calls([]) == []
    assert _summarize_tool_calls([{"role": "user", "content": "hi"}]) == []


def test_format_transcript_renders_full_chain():
    """The transcript renderer should turn an msg_history into a readable
    chat log: system prompt + numbered turns + tool calls + tool results."""
    from owtn.stage_3.phases import _format_transcript

    history = [
        {"role": "user", "content": "develop a voice for this story"},
        {
            "role": "assistant",
            "content": "Reading the bench first.",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {
                        "name": "render_adjacent_scene",
                        "arguments": '{"scene_id": "morning-platform"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_1",
            "content": '{"scene_id": "morning-platform", "neutral_draft": "She sat on the bench."}',
        },
        {
            "role": "assistant",
            "content": "Now I have a sense of the rhythm.",
        },
    ]

    md = _format_transcript(
        title="Phase 1 (private brief) — the-reductionist",
        system_msg="You are: The Reductionist.",
        msg_history=history,
        final_output='{"pov": "third", "tense": "past"}',
    )

    # Headings present
    assert "# Phase 1 (private brief) — the-reductionist" in md
    assert "## System" in md
    assert "## Turn 1 — User" in md
    assert "## Turn 2 — Assistant" in md
    assert "### → tool call: `render_adjacent_scene`" in md
    assert "### ← tool result (`tc_1`)" in md
    assert "## Turn 3 — Assistant" in md
    # Final committed artifact appears as a separate section — not a turn.
    # Adding it as "Turn N+1" would fabricate a turn for tool_call commits
    # where the finalize_voice_genome call is already in msg_history.
    assert "## Final committed artifact" in md

    # Content present
    assert "develop a voice" in md
    assert "morning-platform" in md
    assert "She sat on the bench" in md
    assert '"pov": "third"' in md


def test_format_transcript_handles_empty_history():
    from owtn.stage_3.phases import _format_transcript
    md = _format_transcript(
        title="empty",
        system_msg="sys",
        msg_history=[],
    )
    assert "# empty" in md
    assert "## System" in md
    assert "sys" in md


def test_format_transcript_surfaces_reasoning_content():
    """When reasoning is enabled, deepseek's response includes
    `reasoning_content` on assistant turns. The transcript should show it."""
    from owtn.stage_3.phases import _format_transcript

    history = [
        {"role": "user", "content": "rank these"},
        {
            "role": "assistant",
            "content": "the-reductionist is strongest",
            "reasoning_content": "Walking through each proposal: R has the cleanest rupture lines.",
        },
    ]
    md = _format_transcript(
        title="phase 5 borda",
        system_msg="You are: The Reductionist.",
        msg_history=history,
    )
    assert "**reasoning:**" in md
    assert "Walking through each proposal" in md


def test_analytical_sampler_enables_reasoning():
    """Phase 3 + Phase 5 should use reasoning_effort='medium' (analytical
    work). Phase 1 + Phase 4 commits should NOT (prose detachment cost)."""
    from owtn.stage_3.phases import _analytical_sampler, _commit_sampler

    persona = {"temperature": 1.2, "top_p": None}
    a = _analytical_sampler(persona)
    assert a["reasoning_effort"] == "medium"
    assert a["temperature"] == 0.6  # commit-temp override still applies

    c = _commit_sampler(persona)
    assert "reasoning_effort" not in c  # prose commits keep reasoning OFF
    assert c["temperature"] == 0.6
    # iter5 deodand showed whitespace-truncation when the default cap was
    # too low for dense renderings; explicit cap fixes that.
    assert c["max_tokens"] == 16384


@pytest.mark.asyncio
async def test_borda_phase_rejects_self_in_ranking():
    agents = [_agent("a0"), _agent("a1")]
    state = _make_state(agents)
    state.payload["phase_4_revise"] = {a.id: _make_voice_genome(a.id) for a in agents}
    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)

    async def fake_query(**kwargs):
        ranker = kwargs["system_msg"].split()[-1]
        # Self-include — should raise
        return _FakeResult(BordaRanking(ranking=[ranker, "a1" if ranker == "a0" else "a0"]),
                           cost=0.0)

    with patch("owtn.stage_3.phases.query_async", new=fake_query):
        phase = BordaPhase()
        with pytest.raises(RuntimeError, match="ranking mismatch|self appears"):
            await phase.run(agents, state, registry)
