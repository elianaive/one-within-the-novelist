"""Validate that metrics.json output matches ShinkaEvolve's async_runner expectations.

ShinkaEvolve reads metrics.json via:
    metrics_val.get("public_metrics", {})
    metrics_val.get("private_metrics", {})
    metrics_val.get("combined_score", 0.0)
    metrics_val.get("text_feedback", "")

And correct.json via:
    results.get("correct", {}).get("correct", False)

This test catches field-name drift between the two systems.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from owtn.evaluation.models import DIMENSION_NAMES, JudgeScores
from owtn.evaluation.stage_1 import evaluate


SAMPLE_GENOME = {
    "premise": "Two people at a train station discuss something they never name.",
    "target_effect": "The weight of what remains unsaid — dread, helplessness.",
    "character_seeds": [
        {"label": "the man", "sketch": "Confident on the surface."},
        {"label": "the woman", "sketch": "Deflecting with imagery."},
    ],
    "thematic_tension": "autonomy vs. obligation",
    "constraints": [
        "The word 'abortion' never appears.",
        "No interiority — only dialogue and physical action.",
        "Single scene, near-real-time.",
    ],
}

MOCK_SCORES = JudgeScores(
    reasoning="Solid concept.",
    originality=4.0,
    transportation_potential=3.5,
    narrative_tension=4.0,
    thematic_resonance=3.0,
    scope_calibration=4.5,
    anti_cliche=3.5,
    concept_coherence=4.0,
    generative_fertility=3.0,
    over_explanation_resistance=4.0,
)

MOCK_CLASSIFICATION = json.dumps({
    "concept_type": "voice_constraint",
    "concept_type_confidence": "high",
    "arc_shape": "fall",
    "arc_shape_confidence": "medium",
    "tonal_register": "matter_of_fact",
    "tonal_register_confidence": "high",
    "thematic_domain": "interpersonal",
    "thematic_domain_confidence": "high",
})


class _FakeResult:
    def __init__(self, content, model_name="gpt-4o", cost=0.005):
        self.content = content
        self.model_name = model_name
        self.cost = cost


class TestShinkaContract:
    """Verify serialized JSON matches the keys ShinkaEvolve reads."""

    @pytest.fixture
    def genome_file(self, tmp_path):
        p = tmp_path / "concept.json"
        p.write_text(json.dumps(SAMPLE_GENOME))
        return p

    @pytest.fixture
    def results_dir(self, tmp_path):
        return tmp_path / "results"

    @pytest.mark.asyncio
    async def test_metrics_json_field_names(self, genome_file, results_dir):
        """The keys in metrics.json must match what async_runner.py reads."""

        async def mock_query(model_name, msg, system_msg, output_model=None, **kw):
            if output_model is JudgeScores:
                return _FakeResult(MOCK_SCORES)
            return _FakeResult(MOCK_CLASSIFICATION, model_name="claude-haiku-4-5-20251001", cost=0.001)

        with patch("owtn.evaluation.stage_1.query_async", side_effect=mock_query):
            await evaluate(str(genome_file), str(results_dir), "configs/stage_1_default.yaml")

        metrics = json.loads((results_dir / "metrics.json").read_text())

        # Top-level keys that async_runner.py reads.
        assert "combined_score" in metrics
        assert "holder_score" in metrics
        assert "public_metrics" in metrics
        assert "private_metrics" in metrics
        assert "text_feedback" in metrics

        # These old key names must NOT appear (would cause silent data loss).
        assert "public" not in metrics or metrics.get("public") == metrics.get("public_metrics")
        assert "private" not in metrics or metrics.get("private") == metrics.get("private_metrics")

        # Public metrics structure.
        pub = metrics["public_metrics"]
        assert "dimensions" in pub
        assert "cell_key" in pub
        assert "classification" in pub
        for dim in DIMENSION_NAMES:
            assert dim in pub["dimensions"]

        # Private metrics structure.
        priv = metrics["private_metrics"]
        assert "judge_evaluations" in priv
        assert "total_cost" in priv
        assert "aggregate" in priv

    @pytest.mark.asyncio
    async def test_correct_json_structure(self, genome_file, results_dir):
        """correct.json must have {"correct": bool}."""

        async def mock_query(model_name, msg, system_msg, output_model=None, **kw):
            if output_model is JudgeScores:
                return _FakeResult(MOCK_SCORES)
            return _FakeResult(MOCK_CLASSIFICATION, model_name="claude-haiku-4-5-20251001", cost=0.001)

        with patch("owtn.evaluation.stage_1.query_async", side_effect=mock_query):
            await evaluate(str(genome_file), str(results_dir), "configs/stage_1_default.yaml")

        correct = json.loads((results_dir / "correct.json").read_text())
        assert "correct" in correct
        assert isinstance(correct["correct"], bool)
        assert correct["correct"] is True

    @pytest.mark.asyncio
    async def test_failure_correct_json(self, tmp_path):
        """Failed evaluations must also write correct.json with correct=false."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{")
        results_dir = tmp_path / "results"

        await evaluate(str(bad_file), str(results_dir), "configs/stage_1_default.yaml")

        correct = json.loads((results_dir / "correct.json").read_text())
        assert correct["correct"] is False
