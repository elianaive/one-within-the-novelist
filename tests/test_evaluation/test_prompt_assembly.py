"""Tests for pairwise prompt construction."""

import pytest

from owtn.evaluation.prompts import build_pairwise_system, build_pairwise_user
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome

from tests.conftest import HILLS_GENOME


def _judge(harshness: str = "standard") -> JudgePersona:
    return JudgePersona(
        id="test-judge",
        name="Test Judge",
        identity="A test judge persona.",
        values=["Grip", "Surprise"],
        exemplars=["Example story reference."],
        lean_in_signals=["Test lean-in signal"],
        harshness=harshness,
        priority="primary",
        model=["gpt-4o"],
    )


class TestPairwisePromptAssembly:
    @pytest.fixture
    def judge(self):
        return _judge()

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


class TestHarshnessCalibrationRendering:
    """The CALIBRATION block must render a band label and band-specific body."""

    def test_advancing_body(self):
        prompt = build_pairwise_system(_judge("advancing"))
        assert "CALIBRATION: advancing" in prompt
        assert "default is forward motion" in prompt

    def test_standard_body(self):
        prompt = build_pairwise_system(_judge("standard"))
        assert "CALIBRATION: standard" in prompt
        assert "clearly stronger on the sub-criteria" in prompt

    def test_demanding_body(self):
        prompt = build_pairwise_system(_judge("demanding"))
        assert "CALIBRATION: demanding" in prompt
        assert "multiple sub-criteria" in prompt

    def test_failing_unless_exceptional_body(self):
        prompt = build_pairwise_system(_judge("failing_unless_exceptional"))
        assert "CALIBRATION: failing_unless_exceptional" in prompt
        assert "exceptionally" in prompt

    def test_minimum_evidence_tie_phrasing_present(self):
        """Lazy-default guardrail must be rendered regardless of band."""
        prompt = build_pairwise_system(_judge("standard"))
        assert "dimension-specific advantage" in prompt
        assert "not merely because both have merit" in prompt
