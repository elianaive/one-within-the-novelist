"""The Agent abstraction.

An Agent is one participant in a multi-agent session: a system-prompt +
model + sampler + tool allowlist, with an id used for routing and tracing.
Stage-agnostic — voice agents (Stage 3) and prose agents (Stage 4) both
construct Agents from their own persona schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Agent:
    id: str
    system_prompt: str
    model: str
    sampler: Mapping[str, Any] = field(default_factory=dict)
    tools: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Agent.id must be non-empty")
        if not self.model:
            raise ValueError("Agent.model must be non-empty")
