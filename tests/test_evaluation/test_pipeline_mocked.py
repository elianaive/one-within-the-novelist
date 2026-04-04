"""Mocked end-to-end tests for the evaluation pipeline.

Tests the full evaluate() flow with mocked LLM calls. Validates result
structure, output files, and ShinkaEvolve contract compatibility.
"""

import json
from unittest.mock import patch

import pytest

from owtn.evaluation.models import DIMENSION_NAMES
from owtn.evaluation.stage_1 import evaluate

from tests.conftest import MOCK_JUDGE_SCORES


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_gate1_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{")
        results_dir = tmp_path / "results"
        result = await evaluate(str(bad_file), str(results_dir), "configs/stage_1/medium.yaml")
        assert result.correct is False
        assert "Invalid JSON" in result.error
        assert (results_dir / "correct.json").exists()
        correct = json.loads((results_dir / "correct.json").read_text())
        assert correct["correct"] is False

    @pytest.mark.asyncio
    async def test_gate1_missing_premise(self, tmp_path):
        bad_file = tmp_path / "no_premise.json"
        bad_file.write_text(json.dumps({"target_effect": "Something heavy."}))
        results_dir = tmp_path / "results"
        result = await evaluate(str(bad_file), str(results_dir), "configs/stage_1/medium.yaml")
        assert result.correct is False
        assert "validation failed" in result.error.lower() or "premise" in result.error.lower()

    @pytest.mark.asyncio
    async def test_gate1_trivial_content(self, tmp_path):
        trivial = tmp_path / "trivial.json"
        trivial.write_text(json.dumps({
            "premise": "A person faces a challenge.",
            "target_effect": "Something unsettling and existential.",
        }))
        results_dir = tmp_path / "results"
        result = await evaluate(str(trivial), str(results_dir), "configs/stage_1/medium.yaml")
        assert result.correct is False
        assert "Trivial" in result.error

    @pytest.mark.asyncio
    async def test_full_pipeline(self, genome_file, results_dir, mock_query_async):
        """Full pipeline with mocked LLM calls."""
        with patch("owtn.evaluation.stage_1.query_async", side_effect=mock_query_async):
            result = await evaluate(str(genome_file), str(results_dir), "configs/stage_1/medium.yaml")

        assert result.correct is True
        assert result.combined_score > 0
        assert result.holder_score > 0

        # Public metrics.
        assert "dimensions" in result.public_metrics
        assert "cell_key" in result.public_metrics
        dims = result.public_metrics["dimensions"]
        for name in DIMENSION_NAMES:
            assert name in dims
            assert 0 <= dims[name] <= 5

        # Private metrics.
        assert "judge_evaluations" in result.private_metrics
        assert len(result.private_metrics["judge_evaluations"]) == 3
        assert result.private_metrics["total_cost"] > 0

        # Text feedback.
        assert "Strong concept" in result.text_feedback

        # Output files.
        assert (results_dir / "correct.json").exists()
        assert (results_dir / "metrics.json").exists()
        correct = json.loads((results_dir / "correct.json").read_text())
        assert correct["correct"] is True
        assert isinstance(correct["correct"], bool)
        metrics = json.loads((results_dir / "metrics.json").read_text())
        assert metrics["combined_score"] == result.combined_score

    @pytest.mark.asyncio
    async def test_classification_cell_key(self, genome_file, results_dir, mock_query_async):
        """Verify classification produces correct cell_key."""
        with patch("owtn.evaluation.stage_1.query_async", side_effect=mock_query_async):
            result = await evaluate(str(genome_file), str(results_dir), "configs/stage_1/medium.yaml")

        cell_key = result.public_metrics["cell_key"]
        assert cell_key == ["voice_constraint", "fall"]


class TestShinkaContract:
    """Verify serialized JSON matches the keys ShinkaEvolve reads.

    ShinkaEvolve reads metrics.json via:
        metrics_val.get("public_metrics", {})
        metrics_val.get("private_metrics", {})
        metrics_val.get("combined_score", 0.0)
        metrics_val.get("text_feedback", "")

    And correct.json via:
        results.get("correct", {}).get("correct", False)

    This test catches field-name drift between the two systems.
    """

    @pytest.mark.asyncio
    async def test_metrics_json_field_names(self, genome_file, results_dir, mock_query_async):
        """The keys in metrics.json must match what async_runner.py reads."""
        with patch("owtn.evaluation.stage_1.query_async", side_effect=mock_query_async):
            await evaluate(str(genome_file), str(results_dir), "configs/stage_1/medium.yaml")

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

        # MAP-Elites archive reads these from public_metrics.
        assert "map_elites_cell" in pub
        cell = pub["map_elites_cell"]
        assert "concept_type" in cell
        assert "arc_shape" in cell
        assert "holder_score" in pub
        assert isinstance(pub["holder_score"], float)

        # Private metrics structure.
        priv = metrics["private_metrics"]
        assert "judge_evaluations" in priv
        assert "total_cost" in priv
        assert "aggregate" in priv
