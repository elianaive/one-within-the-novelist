"""Tests for Gate 1 trivial-content detection."""

from owtn.evaluation.stage_1 import _is_trivial
from owtn.models.stage_1.concept_genome import ConceptGenome

from tests.conftest import HILLS_GENOME, MINIMAL_GENOME

_ANCHOR = {
    "sketch": "The keeper finds his own signature in the previous keeper's log — the light has been signaling him across years.",
    "role": "reveal",
}


class TestGate1Validation:
    def test_valid_genome_passes(self):
        genome = ConceptGenome.model_validate(HILLS_GENOME)
        assert _is_trivial(genome) is None

    def test_minimal_genome_passes(self):
        genome = ConceptGenome.model_validate(MINIMAL_GENOME)
        assert _is_trivial(genome) is None

    def test_todo_premise(self):
        genome = ConceptGenome(
            premise="TODO write a real premise here later",
            anchor_scene=_ANCHOR,
            target_effect="Something unsettling and existential.",
        )
        assert _is_trivial(genome) is not None

    def test_lorem_ipsum(self):
        genome = ConceptGenome(
            premise="Lorem ipsum dolor sit amet, this is a test premise.",
            anchor_scene=_ANCHOR,
            target_effect="Something unsettling and existential.",
        )
        assert _is_trivial(genome) is not None

    def test_placeholder_premise(self):
        genome = ConceptGenome(
            premise="Insert your creative premise here please.",
            anchor_scene=_ANCHOR,
            target_effect="Something unsettling and existential.",
        )
        assert _is_trivial(genome) is not None

    def test_generic_premise(self):
        genome = ConceptGenome(
            premise="A person faces a challenge.",
            anchor_scene=_ANCHOR,
            target_effect="Something unsettling and existential.",
        )
        assert _is_trivial(genome) is not None

    def test_meta_commentary_premise(self):
        genome = ConceptGenome(
            premise="This is a story about loss and redemption.",
            anchor_scene=_ANCHOR,
            target_effect="Something unsettling and existential.",
        )
        assert _is_trivial(genome) is not None

    def test_trivial_target_effect(self):
        genome = ConceptGenome(
            premise="A lighthouse keeper discovers the light has been signaling someone.",
            anchor_scene=_ANCHOR,
            target_effect="Write a story about this premise.",
        )
        assert _is_trivial(genome) is not None

    def test_trivial_anchor_sketch(self):
        genome = ConceptGenome(
            premise="A lighthouse keeper discovers the light has been signaling someone.",
            anchor_scene={
                "sketch": "TODO write the anchor scene here later, it will be good",
                "role": "reveal",
            },
            target_effect="Creeping dread and the vertigo of complicity.",
        )
        assert _is_trivial(genome) is not None
