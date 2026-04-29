"""ToolRegistry — schema selection, allowlist intersection, dispatch."""

from __future__ import annotations

import pytest

from owtn.orchestration import ToolContext, ToolRegistry, ToolSpec


async def _echo_handler(params, ctx):
    return f"agent={ctx.agent_id} phase={ctx.phase_id} got={params!r}"


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        handler=_echo_handler,
    )


def test_registry_rejects_duplicate_specs():
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([_spec("a"), _spec("a")])


def test_registry_rejects_unknown_tool_in_allowlist():
    with pytest.raises(ValueError, match="unknown"):
        ToolRegistry([_spec("a")], per_phase_allow={"phase_1": frozenset({"b"})})


def test_schemas_for_intersects_agent_and_phase():
    reg = ToolRegistry(
        [_spec("ask"), _spec("look"), _spec("note")],
        per_phase_allow={
            "phase_1": frozenset({"ask", "look", "note"}),
            "phase_5": frozenset(),
        },
    )
    schemas = reg.schemas_for(frozenset({"ask", "note"}), "phase_1")
    assert sorted(s["name"] for s in schemas) == ["ask", "note"]


def test_schemas_empty_when_phase_not_in_allowlist():
    reg = ToolRegistry([_spec("ask")], per_phase_allow={"phase_1": frozenset({"ask"})})
    assert reg.schemas_for(frozenset({"ask"}), "phase_unknown") == []


def test_schemas_empty_for_borda_style_phase():
    reg = ToolRegistry(
        [_spec("ask")],
        per_phase_allow={"phase_5_borda": frozenset()},
    )
    assert reg.schemas_for(frozenset({"ask"}), "phase_5_borda") == []


@pytest.mark.asyncio
async def test_dispatch_routes_to_handler():
    reg = ToolRegistry([_spec("echo")])
    ctx = ToolContext(
        session_id="s1", phase_id="p1", agent_id="alice", state_view={},
    )
    out = await reg.dispatch("echo", {"x": "hi"}, ctx)
    assert "agent=alice" in out
    assert "phase=p1" in out


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_raises():
    reg = ToolRegistry([_spec("a")])
    ctx = ToolContext(
        session_id="s1", phase_id="p1", agent_id="alice", state_view={},
    )
    with pytest.raises(KeyError, match="unknown tool"):
        await reg.dispatch("nope", {}, ctx)
