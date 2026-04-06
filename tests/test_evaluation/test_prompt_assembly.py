"""Tests for pairwise prompt construction."""

import pytest

from owtn.evaluation.prompts import build_pairwise_system, build_pairwise_user
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome

from tests.conftest import HILLS_GENOME


class TestPairwisePromptAssembly:
    @pytest.fixture
    def judge(self):
        return JudgePersona(
            id="test-judge",
            name="Test Judge",
            identity="A test judge persona.",
            values=["Grip", "Surprise"],
            exemplars=["Example story reference."],
            harshness="moderate",
            priority="primary",
            model=["gpt-4o"],
        )

    def test_pairwise_system_all_placeholders_filled(self, judge):
        prompt = build_pairwise_system(judge)
        assert "{" not in prompt or "{{" in prompt
        assert judge.name in prompt
        assert "Grip" in prompt

    def test_pairwise_system_contains_rubric_anchors(self, judge):
        prompt = build_pairwise_system(judge)
        assert "NOVELTY" in prompt
        assert "INDELIBILITY" in prompt

    def test_pairwise_user_has_both_concepts(self):
        genome_a = ConceptGenome.model_validate(HILLS_GENOME)
        genome_b = ConceptGenome.model_validate(HILLS_GENOME)
        prompt = build_pairwise_user(genome_a, genome_b)
        assert "CONCEPT A:" in prompt
        assert "CONCEPT B:" in prompt
        assert "Two people at a train station" in prompt
