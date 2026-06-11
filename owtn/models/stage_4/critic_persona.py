"""Stage 4 critic-persona schema.

A `CriticPersona` is the YAML-loaded definition of one critic. Each one
checks a specific cognitive/literary mechanism (Transportation, Suspense,
Causal Inevitability, etc.) and returns observations — never rewrites.

Persona vs criteria-direct: per CritiCS Fig 2, persona-driven critics
catch coherence issues at 59/49/56 vs non-persona 37/42/38 on
Interesting/Coherence/Creative. Most critics get personas; a few
(continuity, motif_fidelity, flow_check) have criteria direct enough
that a persona adds noise rather than insight.

Personas are *ordinary-specific* — material work backgrounds (the
developmental-editor; the entomologist-who-reads-fiction-for-pattern) —
not famous-author archetypes, which collapse to caricature (CoMPosT
EMNLP 2023; Mikros 2025).

Tier classification informs orchestrator behavior:
- `tier_a` — fidelity check; mandatory in every cycle; required before
  finalize_critique_plan accepts the agent's commit
- `tier_b` — resonance check; agent-judgment after cycle 1's full sweep
- `domain` — runtime-instantiated `domain_expert`; not loaded from YAML
  (`tier='domain'` reserved; this module never returns one)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


CriticTier = Literal["tier_a", "tier_b", "domain"]
ReasoningEffort = Literal["disabled", "low", "medium", "high"]


class CriticExemplar(BaseModel):
    """Pointer to a passage where this critic's dimension operates well.

    Resolved against `data/voice-references/passages/` like Stage 3's
    `Exemplar`. Loaded at session-start into the system prompt as
    few-shot anchoring per `feedback_few_shot_exemplars_for_voice.md`.
    """
    id: str = Field(min_length=2)
    note: str = Field(min_length=10)


class CriticPersona(BaseModel):
    """One Stage 4 critic, parsed from `configs/stage_4/critics/<id>.yaml`.

    Free-form prose fields (`mechanism`, `identity`) carry the literary
    and cognitive-science grounding; structured fields (`focus_areas`,
    `severity_calibration`) keep the critic's report shape consistent
    across runs.
    """
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=2)
    name: str = ""
    tier: CriticTier
    persona: bool = True

    mechanism: str = Field(min_length=40)
    identity: str = ""
    focus_areas: list[str] = Field(min_length=1)
    exemplars: list[CriticExemplar] = Field(default_factory=list)
    severity_calibration: dict[str, str] = Field(default_factory=dict)

    model: str
    reasoning_effort: ReasoningEffort = "medium"

    tools: list[str] = Field(default_factory=list)

    @property
    def is_tier_a(self) -> bool:
        return self.tier == "tier_a"

    @property
    def is_tier_b(self) -> bool:
        return self.tier == "tier_b"

    @property
    def is_tool_using(self) -> bool:
        """A non-empty tools list means dispatch runs an explore-then-commit
        loop (`finalize_critic_report` ends it). Empty tools means
        single-turn structured output."""
        return len(self.tools) > 0
