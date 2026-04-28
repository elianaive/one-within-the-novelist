"""Tool registry, dispatch, and per-(agent, phase) allowlist.

A `ToolSpec` declares a callable an agent can invoke during a phase. The
registry intersects each agent's permitted tools with the phase's
allowlist to determine what schemas to expose for any given (agent, phase)
combination. Dispatch routes a tool-call back to the spec's handler with
a `ToolContext` carrying session/phase/agent identity and a read-only
view of session state.

Voice-specific tool implementations (`ask_judge`, `lookup_reference`,
`thesaurus`, `render_adjacent_scene`, `note_to_self`) live in
`owtn/stage_3/tools.py`; this module is stage-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Passed to a tool handler at dispatch.

    Gives handlers identity context (for logging) and a read-only view of
    session state (for tools like `render_adjacent_scene` that need the
    cached test bench). Handlers must not mutate `state_view`.
    """
    session_id: str
    phase_id: str
    agent_id: str
    state_view: Mapping[str, Any]


ToolHandler = Callable[[Mapping[str, Any], ToolContext], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool an agent can invoke.

    `parameters` is a JSON schema (dict) — passed to the LLM as the tool's
    function-call shape. `handler` is async; receives validated params and
    a ToolContext, returns a string result that gets appended as a
    tool_result message in the next LLM call.
    """
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler


class ToolRegistry:
    """Holds the spec set + per-phase allowlist and dispatches calls.

    Allowlist semantics: for an LLM call inside (agent, phase), expose only
    tools in `agent.tools ∩ per_phase_allow[phase_name]`. A phase absent
    from `per_phase_allow` exposes nothing — explicit allow only.
    """

    def __init__(
        self,
        specs: list[ToolSpec],
        per_phase_allow: Mapping[str, frozenset[str]] = (),
    ):
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ValueError(f"duplicate tool spec: {spec.name!r}")
            self._specs[spec.name] = spec
        self._per_phase_allow: dict[str, frozenset[str]] = {
            phase: frozenset(names) for phase, names in dict(per_phase_allow).items()
        }
        unknown = {
            name
            for names in self._per_phase_allow.values()
            for name in names
            if name not in self._specs
        }
        if unknown:
            raise ValueError(f"per_phase_allow references unknown tools: {sorted(unknown)}")

    def names(self) -> frozenset[str]:
        return frozenset(self._specs)

    def schemas_for(self, agent_tools: frozenset[str], phase_name: str) -> list[dict]:
        """Tool schemas to pass to a tool-use loop for one (agent, phase) call.

        Returns the neutral shape `{name, description, parameters}`. Each
        provider's `query_async_with_tools` translates this into its native
        tool-call format (Anthropic `input_schema`, OpenAI/DeepSeek `function`
        wrapper). Returns an empty list when the phase is not in the
        allowlist or when the agent has no allowed tools.
        """
        phase_allow = self._per_phase_allow.get(phase_name, frozenset())
        allowed = agent_tools & phase_allow
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": dict(spec.parameters),
            }
            for spec_name in sorted(allowed)
            if (spec := self._specs.get(spec_name))
        ]

    async def dispatch(
        self,
        name: str,
        params: Mapping[str, Any],
        ctx: ToolContext,
    ) -> str:
        """Run a tool by name. Raises KeyError if name is not registered."""
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"unknown tool: {name!r}")
        return await spec.handler(params, ctx)
