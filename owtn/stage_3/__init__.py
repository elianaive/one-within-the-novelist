"""Stage 3 — voice-session machinery.

Modules:
- `adjacent_scenes` — per-(concept, structure) test bench (picker + drafter)
- `personas` — voice-agent persona Pydantic + pool loader
- `casting` — three-call casting classifier (filter + argue + select)
- `phases` — the 5-phase voice-session phase implementations
- `voting` — Borda no-self-vote aggregation
- `tools` — voice-specific ToolSpec set + per-phase allowlist
- `session` — `run_voice_session` end-to-end entry

Cross-stage tools (stylometry, lookup_exemplar, slop) live in `owtn.tools`,
NOT here. Stage-3-specific orchestration belongs here.
"""

from .adjacent_scenes import (
    AdjacentSceneBench,
    AdjacentSceneDraft,
    AdjacentScenePick,
    generate_adjacent_scenes,
)
from .casting import (
    CastingChoice,
    CastingOutput,
    ConceptFeatureClassification,
    PersonaArgument,
    PoolSignal,
    StarvationRecord,
    cast_voice_panel,
)
from .personas import (
    VoicePersona,
    load_casting_vocabulary,
    load_persona_pool,
    validate_pool_against_vocabulary,
)
from .phases import (
    BordaPhase,
    PrivateBriefPhase,
    RevealCritiquePhase,
    RevisePhase,
)
from .session import build_voice_agent, render_persona_system_prompt, run_voice_session
from .tools import ALL_VOICE_TOOLS, VOICE_PHASE_ALLOW
from .voting import borda_no_self_vote

__all__ = [
    "ALL_VOICE_TOOLS",
    "AdjacentSceneBench",
    "AdjacentSceneDraft",
    "AdjacentScenePick",
    "BordaPhase",
    "CastingChoice",
    "CastingOutput",
    "ConceptFeatureClassification",
    "PersonaArgument",
    "PoolSignal",
    "PrivateBriefPhase",
    "RevealCritiquePhase",
    "RevisePhase",
    "StarvationRecord",
    "VOICE_PHASE_ALLOW",
    "VoicePersona",
    "borda_no_self_vote",
    "build_voice_agent",
    "cast_voice_panel",
    "generate_adjacent_scenes",
    "load_casting_vocabulary",
    "load_persona_pool",
    "render_persona_system_prompt",
    "run_voice_session",
    "validate_pool_against_vocabulary",
]
