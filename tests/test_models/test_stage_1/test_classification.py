from owtn.models.stage_1.classification import (
    ArcShape,
    ClassificationResult,
    Confidence,
    ConceptType,
    ConstraintDensity,
    ThematicDomain,
    TonalRegister,
)


class TestEnums:
    def test_concept_type_count(self):
        assert len(ConceptType) == 6

    def test_arc_shape_count(self):
        assert len(ArcShape) == 6

    def test_constraint_density_count(self):
        assert len(ConstraintDensity) == 3

    def test_tonal_register_count(self):
        assert len(TonalRegister) == 6

    def test_thematic_domain_count(self):
        assert len(ThematicDomain) == 5

    def test_str_enum_values(self):
        assert ConceptType.THOUGHT_EXPERIMENT == "thought_experiment"
        assert ArcShape.FALL_RISE == "fall_rise"
        assert TonalRegister.MATTER_OF_FACT == "matter_of_fact"
        assert ThematicDomain.MUNDANE_ELEVATED == "mundane_elevated"


class TestClassificationResult:
    def test_parse_and_cell_key(self):
        result = ClassificationResult(
            concept_type=ConceptType.THOUGHT_EXPERIMENT,
            concept_type_confidence=Confidence.HIGH,
            arc_shape=ArcShape.FALL,
            arc_shape_confidence=Confidence.MEDIUM,
            tonal_register=TonalRegister.IRONIC,
            tonal_register_confidence=Confidence.LOW,
            thematic_domain=ThematicDomain.PHILOSOPHICAL,
            thematic_domain_confidence=Confidence.HIGH,
            constraint_density=ConstraintDensity.UNCONSTRAINED,
        )
        key = result.cell_key()
        assert key == (
            ConceptType.THOUGHT_EXPERIMENT,
            ArcShape.FALL,
        )

    def test_cell_key_is_2_tuple(self):
        result = ClassificationResult(
            concept_type="character_collision",
            concept_type_confidence="high",
            arc_shape="rise_fall",
            arc_shape_confidence="medium",
            tonal_register="tragic",
            tonal_register_confidence="high",
            thematic_domain="interpersonal",
            thematic_domain_confidence="medium",
            constraint_density="moderate",
        )
        key = result.cell_key()
        assert len(key) == 2
        assert key[0] == ConceptType.CHARACTER_COLLISION
        assert key[1] == ArcShape.RISE_FALL

    def test_from_dict(self):
        data = {
            "concept_type": "atmospheric_associative",
            "concept_type_confidence": "low",
            "arc_shape": "rise_fall_rise",
            "arc_shape_confidence": "low",
            "tonal_register": "surreal",
            "tonal_register_confidence": "medium",
            "thematic_domain": "existential",
            "thematic_domain_confidence": "high",
            "constraint_density": "heavy",
        }
        result = ClassificationResult.model_validate(data)
        assert result.concept_type == ConceptType.ATMOSPHERIC_ASSOCIATIVE
        assert result.constraint_density == ConstraintDensity.HEAVY
