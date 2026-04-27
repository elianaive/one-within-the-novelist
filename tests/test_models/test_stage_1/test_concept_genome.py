import json

import pytest
from pydantic import ValidationError

from owtn.models.stage_1.classification import ConstraintDensity
from owtn.models.stage_1.concept_genome import AnchorScene, ConceptGenome

from tests.conftest import HILLS_GENOME


def _anchor():
    """A valid AnchorScene payload for tests that don't care about its contents."""
    return {
        "sketch": "The keeper finds his own signature in the previous keeper's log — the light has been signaling him across years.",
        "role": "reveal",
    }


class TestParsing:
    def test_full_genome(self):
        genome = ConceptGenome.model_validate(HILLS_GENOME)
        assert genome.premise.startswith("Two people")
        assert genome.anchor_scene.role == "reveal"
        assert len(genome.character_seeds) == 2
        assert genome.character_seeds[0].label == "the man"
        assert genome.character_seeds[0].wound is None
        assert genome.character_seeds[0].want is not None

    def test_minimal_genome(self):
        genome = ConceptGenome(
            premise="A lighthouse keeper discovers the light has been signaling someone.",
            anchor_scene=_anchor(),
            target_effect="Creeping dread and the vertigo of complicity.",
        )
        assert genome.character_seeds is None
        assert genome.constraints is None
        assert genome.anchor_scene.role == "reveal"

    def test_from_code_string(self):
        code = json.dumps(HILLS_GENOME)
        genome = ConceptGenome.from_code_string(code)
        assert genome.thematic_engine.startswith("held tension")

    def test_round_trip(self):
        genome = ConceptGenome.model_validate(HILLS_GENOME)
        serialized = genome.model_dump_json()
        restored = ConceptGenome.model_validate_json(serialized)
        assert restored == genome


class TestValidation:
    def test_missing_premise(self):
        with pytest.raises(ValidationError):
            ConceptGenome(
                premise=None,
                anchor_scene=_anchor(),
                target_effect="Something unsettling.",
            )

    def test_premise_too_short(self):
        with pytest.raises(ValidationError):
            ConceptGenome(
                premise="Too short.",
                anchor_scene=_anchor(),
                target_effect="Something unsettling and heavy.",
            )

    def test_target_effect_too_short(self):
        with pytest.raises(ValidationError):
            ConceptGenome(
                premise="A lighthouse keeper discovers the light has been signaling someone.",
                anchor_scene=_anchor(),
                target_effect="Bad.",
            )

    def test_character_seed_missing_label(self):
        with pytest.raises(ValidationError):
            ConceptGenome(
                premise="A lighthouse keeper discovers the light has been signaling someone.",
                anchor_scene=_anchor(),
                target_effect="Creeping dread and complicity.",
                character_seeds=[{"sketch": "Nervous, watchful."}],
            )


class TestAnchorScene:
    def test_anchor_required(self):
        with pytest.raises(ValidationError):
            ConceptGenome(
                premise="A lighthouse keeper discovers the light has been signaling someone.",
                target_effect="Creeping dread and the vertigo of complicity.",
            )

    def test_sketch_too_short(self):
        with pytest.raises(ValidationError):
            AnchorScene(sketch="The keeper sees.", role="reveal")

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            AnchorScene(
                sketch="The keeper finds his own signature in the previous keeper's log, spanning years.",
                role="constraint-moment",
            )

    def test_all_valid_roles(self):
        sketch = "The keeper finds his own signature in the previous keeper's log, spanning years."
        for role in ("climax", "reveal", "pivot"):
            scene = AnchorScene(sketch=sketch, role=role)
            assert scene.role == role


class TestConstraintDensity:
    def test_unconstrained_none(self):
        genome = ConceptGenome(
            premise="A lighthouse keeper discovers the light has been signaling someone.",
            anchor_scene=_anchor(),
            target_effect="Creeping dread and the vertigo of complicity.",
        )
        assert genome.classify_constraint_density() == ConstraintDensity.UNCONSTRAINED

    def test_unconstrained_empty(self):
        genome = ConceptGenome(
            premise="A lighthouse keeper discovers the light has been signaling someone.",
            anchor_scene=_anchor(),
            target_effect="Creeping dread and the vertigo of complicity.",
            constraints=[],
        )
        assert genome.classify_constraint_density() == ConstraintDensity.UNCONSTRAINED

    def test_moderate(self):
        genome = ConceptGenome(
            premise="A lighthouse keeper discovers the light has been signaling someone.",
            anchor_scene=_anchor(),
            target_effect="Creeping dread and the vertigo of complicity.",
            constraints=["No dialogue."],
        )
        assert genome.classify_constraint_density() == ConstraintDensity.MODERATE

    def test_heavy(self):
        genome = ConceptGenome.model_validate(HILLS_GENOME)
        assert genome.classify_constraint_density() == ConstraintDensity.HEAVY

    def test_whitespace_only_skipped(self):
        genome = ConceptGenome(
            premise="A lighthouse keeper discovers the light has been signaling someone.",
            anchor_scene=_anchor(),
            target_effect="Creeping dread and the vertigo of complicity.",
            constraints=["  ", "", "\t"],
        )
        assert genome.classify_constraint_density() == ConstraintDensity.UNCONSTRAINED

    def test_whitespace_mixed_with_real(self):
        genome = ConceptGenome(
            premise="A lighthouse keeper discovers the light has been signaling someone.",
            anchor_scene=_anchor(),
            target_effect="Creeping dread and the vertigo of complicity.",
            constraints=["No dialogue.", "  ", ""],
        )
        assert genome.classify_constraint_density() == ConstraintDensity.MODERATE


class TestPromptFields:
    def test_full_genome_fields(self):
        genome = ConceptGenome.model_validate(HILLS_GENOME)
        fields = genome.to_prompt_fields()
        assert fields["premise"] == genome.premise
        assert fields["anchor_sketch"] == genome.anchor_scene.sketch
        assert fields["anchor_role"] == genome.anchor_scene.role
        assert fields["target_effect"] == genome.target_effect
        assert "the man" in fields["character_seeds"]
        assert "want:" in fields["character_seeds"]
        assert fields["thematic_engine"].startswith("held tension")
        assert "- The word" in fields["constraints"]
        assert fields["style_hint"].startswith("Spare")

    def test_minimal_genome_fields(self):
        genome = ConceptGenome(
            premise="A lighthouse keeper discovers the light has been signaling someone.",
            anchor_scene=_anchor(),
            target_effect="Creeping dread and the vertigo of complicity.",
        )
        fields = genome.to_prompt_fields()
        assert fields["character_seeds"] == ""
        assert fields["setting_seeds"] == ""
        assert fields["constraints"] == ""
