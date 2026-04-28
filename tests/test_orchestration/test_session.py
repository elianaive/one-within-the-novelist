"""End-to-end session run with a toy 2-phase 2-agent fixture.

No LLM calls — phases are stubs that mutate `state.payload` directly.
Validates: phase sequencing, state accumulation, ContextVar propagation
into `llm_context`, log-tree shape under `session_log_dir`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from owtn.llm.call_logger import llm_context
from owtn.orchestration import (
    Agent,
    SessionState,
    ToolRegistry,
    push_llm_context,
    run_session,
    session_log_dir,
)


# ─── Toy phases ──────────────────────────────────────────────────────────


@dataclass
class _TallyPhase:
    """Each agent contributes a 'note' under payload[phase.name][agent.id].

    Demonstrates the parallel-fan-out pattern: per-agent coroutine sets
    `agent_id` in llm_context inside its task, so concurrent agents get
    correctly tagged log entries.
    """
    name: str
    contribution_prefix: str

    async def run(self, agents, state, registry):
        bucket: dict[str, str] = {}
        seen_contexts: list[dict] = []

        async def _one(agent: Agent):
            with push_llm_context(agent_id=agent.id):
                seen_contexts.append(dict(llm_context.get({})))
                await asyncio.sleep(0)
                bucket[agent.id] = f"{self.contribution_prefix}-{agent.id}"

        await asyncio.gather(*(_one(a) for a in agents))
        state.payload[self.name] = bucket
        state.payload.setdefault("_seen_contexts", []).extend(seen_contexts)
        state.cost_usd += 0.001 * len(agents)
        return state


@dataclass
class _AggregatePhase:
    """Reads previous phase's bucket and concatenates it."""
    name: str
    source_phase: str

    async def run(self, agents, state, registry):
        source = state.payload.get(self.source_phase, {})
        state.payload[self.name] = {"summary": "+".join(sorted(source.values()))}
        state.cost_usd += 0.0005
        return state


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def cast() -> list[Agent]:
    return [
        Agent(id="alice", system_prompt="x", model="m1"),
        Agent(id="bob", system_prompt="y", model="m2"),
    ]


@pytest.fixture
def empty_registry() -> ToolRegistry:
    return ToolRegistry([])


# ─── Tests ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_runs_phases_in_order(cast, empty_registry):
    state = SessionState.new(cast, pair_id="pair_test")
    phases = [
        _TallyPhase(name="phase_1_tally", contribution_prefix="hi"),
        _AggregatePhase(name="phase_2_aggregate", source_phase="phase_1_tally"),
    ]
    state = await run_session(state, phases, empty_registry)

    assert state.phases_completed == ["phase_1_tally", "phase_2_aggregate"]
    assert state.payload["phase_1_tally"] == {
        "alice": "hi-alice",
        "bob": "hi-bob",
    }
    assert state.payload["phase_2_aggregate"]["summary"] == "hi-alice+hi-bob"
    assert state.cost_usd == pytest.approx(0.001 * 2 + 0.0005)


@pytest.mark.asyncio
async def test_per_agent_context_propagates_in_parallel(cast, empty_registry):
    state = SessionState.new(cast, pair_id="pair_test")
    phases = [_TallyPhase(name="phase_1_tally", contribution_prefix="x")]
    state = await run_session(state, phases, empty_registry)

    seen = state.payload["_seen_contexts"]
    by_agent = {ctx["agent_id"]: ctx for ctx in seen}
    assert set(by_agent) == {"alice", "bob"}
    for agent_id, ctx in by_agent.items():
        assert ctx["session_id"] == state.session_id
        assert ctx["phase_id"] == "phase_1_tally"
        assert ctx["pair_id"] == "pair_test"


@pytest.mark.asyncio
async def test_session_writes_log_tree(cast, empty_registry, tmp_path):
    token = session_log_dir.set(str(tmp_path))
    try:
        state = SessionState.new(cast, pair_id="pair_test")
        phases = [
            _TallyPhase(name="phase_1_tally", contribution_prefix="hi"),
            _AggregatePhase(name="phase_2_aggregate", source_phase="phase_1_tally"),
        ]
        await run_session(state, phases, empty_registry)
    finally:
        session_log_dir.reset(token)

    manifest_path = tmp_path / "session.yaml"
    assert manifest_path.exists()
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["pair_id"] == "pair_test"
    assert manifest["phases_completed"] == ["phase_1_tally", "phase_2_aggregate"]
    assert sorted(c["id"] for c in manifest["cast"]) == ["alice", "bob"]

    phase_1_path = tmp_path / "phases" / "phase_1_tally.yaml"
    assert phase_1_path.exists()
    p1 = yaml.safe_load(phase_1_path.read_text())
    assert p1["phase"] == "phase_1_tally"
    assert p1["outputs"]["agents"] == ["alice", "bob"]
    assert p1["cost_delta_usd"] == pytest.approx(0.002)

    phase_2_path = tmp_path / "phases" / "phase_2_aggregate.yaml"
    assert phase_2_path.exists()


@pytest.mark.asyncio
async def test_no_log_tree_when_session_log_dir_unset(cast, empty_registry, tmp_path):
    state = SessionState.new(cast)
    phases = [_TallyPhase(name="phase_1_tally", contribution_prefix="x")]
    await run_session(state, phases, ToolRegistry([]))
    assert not (tmp_path / "session.yaml").exists()


@pytest.mark.asyncio
async def test_phase_exception_aborts_session(cast, empty_registry):
    @dataclass
    class _ExplodingPhase:
        name: str = "phase_boom"

        async def run(self, agents, state, registry):
            raise RuntimeError("intentional")

    state = SessionState.new(cast)
    phases = [_ExplodingPhase()]
    with pytest.raises(RuntimeError, match="intentional"):
        await run_session(state, phases, empty_registry)
    assert state.phases_completed == []


@pytest.mark.asyncio
async def test_session_id_auto_generated_when_omitted(cast, empty_registry):
    state = SessionState.new(cast)
    assert state.session_id.startswith("sess_")
    assert len(state.session_id) > len("sess_")


# ─── Tool dispatch integration ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_phase_can_dispatch_tools_via_registry(cast):
    """Wires ToolRegistry into a Phase: the phase resolves a tool by name
    and gets back the handler's response. Demonstrates the contract a
    voice phase will use when running tool-using agent calls."""
    from owtn.orchestration import ToolContext, ToolRegistry, ToolSpec

    async def note_handler(params, ctx):
        return f"noted by {ctx.agent_id}: {params.get('text', '')}"

    registry = ToolRegistry(
        [ToolSpec(
            name="note",
            description="record a private note",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=note_handler,
        )],
        per_phase_allow={"phase_1_brief": frozenset({"note"})},
    )

    @dataclass
    class _NotingPhase:
        name: str = "phase_1_brief"

        async def run(self, agents, state, registry):
            results: dict[str, str] = {}
            for agent in agents:
                with push_llm_context(agent_id=agent.id):
                    ctx = ToolContext(
                        session_id=state.session_id,
                        phase_id=self.name,
                        agent_id=agent.id,
                        state_view=state.payload,
                    )
                    results[agent.id] = await registry.dispatch(
                        "note", {"text": f"{agent.id}-was-here"}, ctx,
                    )
            state.payload[self.name] = results
            return state

    state = SessionState.new(cast)
    state = await run_session(state, [_NotingPhase()], registry)

    notes = state.payload["phase_1_brief"]
    assert notes["alice"] == "noted by alice: alice-was-here"
    assert notes["bob"] == "noted by bob: bob-was-here"
