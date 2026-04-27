from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from owtn.models.stage_1.classification import ConstraintDensity


class CharacterSeed(BaseModel):
    label: str
    sketch: str
    wound: str | None = None
    fear: str | None = None
    lie: str | None = None
    want: str | None = None
    need: str | None = None


class AnchorScene(BaseModel):
    """The single scene the concept hinges on — the moment the story lives or dies.

    Required on every genome: "does this concept have a landing?" is the
    selection pressure. Three roles, all content-load (the weight is in what
    happens, not how the sentence is written) — prose-load roles were dropped
    because they require sentence-level craft LLM execution can't reliably
    deliver. See lab/issues/2026-04-22-anchor-scene-in-stage-1-genome.md.
    """
    sketch: str = Field(min_length=40)
    role: Literal["climax", "reveal", "pivot"]


class ConceptGenome(BaseModel):
    premise: str = Field(min_length=20)
    thematic_engine: str | None = None
    target_effect: str = Field(min_length=15)
    anchor_scene: AnchorScene
    character_seeds: list[CharacterSeed] | None = None
    setting_seeds: str | None = None
    constraints: list[str] | None = None
    style_hint: str | None = None

    def classify_constraint_density(self) -> ConstraintDensity:
        """Rule-based classification — no LLM call needed."""
        constraints = self.constraints or []
        count = sum(1 for c in constraints if isinstance(c, str) and c.strip())
        if count == 0:
            return ConstraintDensity.UNCONSTRAINED
        elif count <= 2:
            return ConstraintDensity.MODERATE
        else:
            return ConstraintDensity.HEAVY

    def to_prompt_fields(self) -> dict[str, str]:
        """Template vars for judge_user.txt placeholders."""
        def _format_characters(seeds: list[CharacterSeed] | None) -> str:
            if not seeds:
                return ""
            parts = []
            for cs in seeds:
                lines = [f"{cs.label}: {cs.sketch}"]
                for attr in ("wound", "fear", "lie", "want", "need"):
                    val = getattr(cs, attr)
                    if val:
                        lines.append(f"  {attr}: {val}")
                parts.append("\n".join(lines))
            return "\n".join(parts)

        def _format_constraints(constraints: list[str] | None) -> str:
            if not constraints:
                return ""
            return "\n".join(f"- {c}" for c in constraints if c.strip())

        return {
            "premise": self.premise,
            "thematic_engine": self.thematic_engine or "",
            "target_effect": self.target_effect,
            "anchor_sketch": self.anchor_scene.sketch,
            "anchor_role": self.anchor_scene.role,
            "character_seeds": _format_characters(self.character_seeds),
            "setting_seeds": self.setting_seeds or "",
            "constraints": _format_constraints(self.constraints),
            "style_hint": self.style_hint or "",
        }

    @classmethod
    def from_code_string(cls, code: str) -> ConceptGenome:
        """Parse from JSON string (ShinkaEvolve Program.code format)."""
        return cls.model_validate(json.loads(code))
