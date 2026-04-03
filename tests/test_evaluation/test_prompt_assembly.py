"""Tests for judge prompt construction."""

import pytest

from owtn.evaluation.prompts import build_judge_system, build_judge_user
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome

from tests.conftest import HILLS_GENOME


class TestPromptAssembly:
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

    def test_judge_system_all_placeholders_filled(self, judge):
        prompt = build_judge_system(judge)
        assert "{" not in prompt or "{{" in prompt
        assert judge.name in prompt
        assert "Grip" in prompt
        assert "use your best judgment" in prompt

    def test_judge_system_contains_rubric_anchors(self, judge):
        prompt = build_judge_system(judge)
        assert "ORIGINALITY" in prompt
        assert "OVER-EXPLANATION RESISTANCE" in prompt

    def test_judge_user_has_genome_fields(self):
        genome = ConceptGenome.model_validate(HILLS_GENOME)
        prompt = build_judge_user(genome)
        assert "Two people at a train station" in prompt
        assert "autonomy vs. obligation" in prompt
        assert "the man" in prompt
