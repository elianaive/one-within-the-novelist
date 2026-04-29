"""Agent dataclass — id/model required, tools default to empty frozenset."""

from __future__ import annotations

import pytest

from owtn.orchestration import Agent


def test_agent_basic():
    a = Agent(
        id="alice",
        system_prompt="be helpful",
        model="claude-sonnet-4-6",
        sampler={"temperature": 0.7},
        tools=frozenset({"thesaurus"}),
    )
    assert a.id == "alice"
    assert a.tools == frozenset({"thesaurus"})
    assert a.metadata == {}


def test_agent_defaults():
    a = Agent(id="bob", system_prompt="x", model="m")
    assert a.tools == frozenset()
    assert a.sampler == {}


def test_agent_rejects_empty_id():
    with pytest.raises(ValueError, match="id"):
        Agent(id="", system_prompt="x", model="m")


def test_agent_rejects_empty_model():
    with pytest.raises(ValueError, match="model"):
        Agent(id="alice", system_prompt="x", model="")


def test_agent_is_frozen():
    a = Agent(id="alice", system_prompt="x", model="m")
    with pytest.raises(Exception):
        a.id = "bob"  # type: ignore[misc]
