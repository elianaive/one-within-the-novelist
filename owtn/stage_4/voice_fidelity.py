"""voice_fidelity — Stage 3 winning persona promoted to Stage 4 critic.

The architecture commits to *continuity of authorship* for voice
fidelity: the Stage 3 voice agent that produced the winning voice spec
returns in Stage 4 as the critic that reads prose against that spec.
Same aesthetic intelligence; different work surface.

Promotion is a session-time pass. The composer reads the Stage 3
winner's `agent_id`, loads the persona YAML, and produces a Stage 4
`CriticPersona` whose identity is the Stage 3 persona's identity. The
mechanism + focus + severity calibration are a fixed template — what's
being checked is the same regardless of which Stage 3 persona happened
to win.

When no Stage 3 winner is available (manual runs, dev fixtures), the
stub `voice_fidelity.yaml` (criteria-direct, with metric tools) is the
fallback. The stub plus the promoted persona share a YAML id so the
session-time swap is a registry-key replacement.
"""

from __future__ import annotations

from owtn.models.stage_4 import CriticPersona
from owtn.stage_3.personas import VoicePersona


VOICE_FIDELITY_ID = "voice_fidelity"

VOICE_FIDELITY_TOOLS = ["read_file", "stylometry", "slop_score", "writing_style"]

VOICE_FIDELITY_MECHANISM = (
    "You worked the voice for this story; you wrote the three sample "
    "passages at the top of this prompt as the voice operating well on "
    "adjacent scenes. Now you read the manuscript itself, scene by "
    "scene, against that voice. Voice fidelity is the prose hewing to "
    "the target across the whole manuscript: point of view and tense, "
    "sentence rhythm and crowding, diction, the consciousness mode the "
    "narration uses, the implied author's stance toward the characters. "
    "The failure that matters most is the slow re-emergence of a "
    "generic literary surface — the prose loosening back toward sounds "
    "the opening did not have. By the closing scenes, voice can decay "
    "into something flatter, more abstract, more competent-and-anonymous "
    "than the opening promised. You read with particular attention to "
    "whether the voice that opened the manuscript is still the voice "
    "that closes it."
)

VOICE_FIDELITY_FOCUS_AREAS: list[str] = [
    "Point of view and tense — does the manuscript hold them, or do they drift mid-scene or across scenes",
    "Sentence rhythm and crowding — does the manuscript's rhythm match what the voice spec describes and what the sample passages show",
    "Diction — does the vocabulary draw from the registers the spec names, or has it shifted to a generic literary register",
    "Consciousness rendering — does the manuscript hold the declared mode, or slip into a different one",
    "The implied author's stance — does the manuscript carry the declared moral temperature and stance toward characters",
    "Late-manuscript voice drift — by the closing scenes, is the prose still recognizably the same voice the opening showed",
    "Use stylometry, slop_score, writing_style on representative passages to ground the qualitative read in the metric ensemble; the metrics are corroborating signal, not the verdict",
]

VOICE_FIDELITY_SEVERITY: dict[str, str] = {
    "severe": (
        "A scene or run of scenes is in the wrong voice — wrong POV or "
        "tense, wrong consciousness mode, wrong register; the prose has "
        "reverted to a generic literary default in a way that breaks "
        "continuity with the opening's voice."
    ),
    "moderate": (
        "The voice holds in shape but slips on individual axes — diction "
        "wandering off-register, rhythm losing its declared variance, "
        "the consciousness mode wavering; the prose is recognizably "
        "reaching for the voice but landing inconsistently."
    ),
    "minor": (
        "A line or two where the voice loosens; small drifts in diction; "
        "a single sentence that would not have appeared in the sample "
        "passages but doesn't break the surrounding prose."
    ),
}


def promote_voice_persona_to_critic(
    persona: VoicePersona,
    *,
    model: str = "deepseek-v4-pro",
    reasoning_effort: str = "medium",
) -> CriticPersona:
    """Build a Stage 4 voice_fidelity CriticPersona from a Stage 3
    VoicePersona. The Stage 3 identity transfers verbatim; the
    mechanism, focus, and severity calibration are the fixed template
    above — what's being checked is voice fidelity regardless of which
    Stage 3 persona happened to win.
    """
    return CriticPersona(
        id=VOICE_FIDELITY_ID,
        name=persona.name,
        tier="tier_a",
        persona=True,
        mechanism=VOICE_FIDELITY_MECHANISM,
        identity=persona.identity.strip(),
        focus_areas=list(VOICE_FIDELITY_FOCUS_AREAS),
        severity_calibration=dict(VOICE_FIDELITY_SEVERITY),
        model=model,
        reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
        tools=list(VOICE_FIDELITY_TOOLS),
    )
