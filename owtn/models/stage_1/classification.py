from enum import StrEnum

from pydantic import BaseModel


class ConceptType(StrEnum):
    THOUGHT_EXPERIMENT = "thought_experiment"
    SITUATION_WITH_REVEAL = "situation_with_reveal"
    VOICE_CONSTRAINT = "voice_constraint"
    CHARACTER_COLLISION = "character_collision"
    ATMOSPHERIC_ASSOCIATIVE = "atmospheric_associative"
    CONSTRAINT_DRIVEN = "constraint_driven"


class ArcShape(StrEnum):
    RISE = "rise"
    FALL = "fall"
    FALL_RISE = "fall_rise"
    RISE_FALL = "rise_fall"
    RISE_FALL_RISE = "rise_fall_rise"
    FALL_RISE_FALL = "fall_rise_fall"


class ConstraintDensity(StrEnum):
    UNCONSTRAINED = "unconstrained"
    MODERATE = "moderate"
    HEAVY = "heavy"


class TonalRegister(StrEnum):
    COMEDIC = "comedic"
    TRAGIC = "tragic"
    IRONIC = "ironic"
    EARNEST = "earnest"
    SURREAL = "surreal"
    MATTER_OF_FACT = "matter_of_fact"


class ThematicDomain(StrEnum):
    INTERPERSONAL = "interpersonal"
    SOCIETAL = "societal"
    PHILOSOPHICAL = "philosophical"
    EXISTENTIAL = "existential"
    MUNDANE_ELEVATED = "mundane_elevated"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClassificationResult(BaseModel):
    concept_type: ConceptType
    concept_type_confidence: Confidence
    arc_shape: ArcShape
    arc_shape_confidence: Confidence
    tonal_register: TonalRegister
    tonal_register_confidence: Confidence
    thematic_domain: ThematicDomain
    thematic_domain_confidence: Confidence
    constraint_density: ConstraintDensity

    def cell_key(self) -> tuple[ConceptType, ArcShape]:
        """MAP-Elites cell coordinates (2D grid, 36 cells)."""
        return (self.concept_type, self.arc_shape)
