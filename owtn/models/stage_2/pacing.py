"""Pacing preset constants.

Four presets parameterize MCTS expansion (Phase 5+) by injecting different
pacing hints into the expansion prompt. Per docs/stage-2/overview.md
§Pacing Presets, v1 reads only `name`, `expansion_hint`, and `character`
numerically — the descriptive params are forward-compat metadata for v1.5
parametric priors if semantic hints prove insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


IntensityVariance = Literal["tight", "wide"]


@dataclass(frozen=True)
class PresetSpec:
    name: str
    character: str             # one-line summary of the preset's philosophy
    expansion_hint: str        # 1-2 sentence hint substituted into expansion prompts
    min_rest_beats: int        # descriptive in v1; numeric only if v1.5 restores parametric priors
    max_flat_beats: int
    intensity_variance: IntensityVariance
    recovery_required: bool


_PRESETS: dict[str, PresetSpec] = {
    "cassandra_ish": PresetSpec(
        name="cassandra_ish",
        character="Escalating peaks with guaranteed relief; high regularity",
        expansion_hint=(
            "Favor actions that establish rise-then-relief patterns. After each "
            "high-tension beat, the next move should offer breathing room before "
            "the story escalates again. Escalating peaks, guaranteed recovery."
        ),
        min_rest_beats=1,
        max_flat_beats=4,
        intensity_variance="tight",
        recovery_required=True,
    ),
    "phoebe_ish": PresetSpec(
        name="phoebe_ish",
        character="Long recoveries; softer peaks; \"benevolent\" alternation",
        expansion_hint=(
            "Favor long recoveries and softer peaks. After high-tension beats, "
            "insert at least one contemplative beat before escalating. Prefer "
            "accumulated dread and slow intensification to sharp spikes."
        ),
        min_rest_beats=3,
        max_flat_beats=6,
        intensity_variance="tight",
        recovery_required=True,
    ),
    "randy_ish": PresetSpec(
        name="randy_ish",
        character="Stochastic, no forced recovery, high variance in peak intensity",
        expansion_hint=(
            "Tolerate variance. High-tension peaks may follow each other without "
            "recovery; low-intensity stretches can be long. Avoid a forced "
            "escalation cadence; let the story's logic set the rhythm."
        ),
        min_rest_beats=0,
        max_flat_beats=8,
        intensity_variance="wide",
        recovery_required=False,
    ),
    "winston_ish": PresetSpec(
        name="winston_ish",
        character="Discrete numbered waves; each climax followed by a clear reward beat",
        expansion_hint=(
            "Structure tension as discrete waves. Each complication should resolve "
            "with an explicit reward beat before the next complication begins. "
            "Numbered stakes rather than continuous escalation."
        ),
        min_rest_beats=2,
        max_flat_beats=3,
        intensity_variance="tight",
        recovery_required=True,
    ),
}


PRESET_NAMES: tuple[str, ...] = tuple(_PRESETS.keys())


def get_preset(name: str) -> PresetSpec:
    """Look up a preset by name. Raises KeyError with the known names."""
    try:
        return _PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown preset {name!r}; known: {sorted(_PRESETS)}"
        ) from None
