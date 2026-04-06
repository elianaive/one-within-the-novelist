"""Mocked end-to-end tests for the evaluation pipeline.

With pairwise selection, evaluate() only does validation (gates 1 & 2).
Pairwise comparison happens in the runner, not here. These tests verify
the validation contract and output file format.
"""

import json

import pytest

from owtn.evaluation.stage_1 import evaluate


class TestGateValidation:
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


class TestValidConceptPassesGates:
    @pytest.mark.asyncio
    async def test_valid_concept_returns_correct(self, genome_file, results_dir):
        """Valid concept passes gates, returns correct=True with placeholder score."""
        result = await evaluate(str(genome_file), str(results_dir), "configs/stage_1/medium.yaml")

        assert result.correct is True
        assert result.combined_score == 0.0  # Placeholder — pairwise sets real signal
        assert result.text_feedback == ""  # Pairwise fills this in the runner

    @pytest.mark.asyncio
    async def test_output_files_written(self, genome_file, results_dir):
        """Both correct.json and metrics.json are written."""
        await evaluate(str(genome_file), str(results_dir), "configs/stage_1/medium.yaml")

        assert (results_dir / "correct.json").exists()
        assert (results_dir / "metrics.json").exists()
        correct = json.loads((results_dir / "correct.json").read_text())
        assert correct["correct"] is True


class TestShinkaContract:
    """Verify serialized JSON matches the keys ShinkaEvolve reads.

    ShinkaEvolve reads metrics.json via:
        metrics_val.get("combined_score", 0.0)
        metrics_val.get("public_metrics", {})
        metrics_val.get("private_metrics", {})
        metrics_val.get("text_feedback", "")

    And correct.json via:
        results.get("correct", {}).get("correct", False)
    """

    @pytest.mark.asyncio
    async def test_metrics_json_field_names(self, genome_file, results_dir):
        """The keys in metrics.json must match what async_runner.py reads."""
        await evaluate(str(genome_file), str(results_dir), "configs/stage_1/medium.yaml")

        metrics = json.loads((results_dir / "metrics.json").read_text())

        # Top-level keys that async_runner.py reads.
        assert "combined_score" in metrics
        assert "public_metrics" in metrics
        assert "private_metrics" in metrics
        assert "text_feedback" in metrics
        assert "correct" in metrics

    @pytest.mark.asyncio
    async def test_correct_json_format(self, genome_file, results_dir):
        """correct.json must have {"correct": bool}."""
        await evaluate(str(genome_file), str(results_dir), "configs/stage_1/medium.yaml")

        correct = json.loads((results_dir / "correct.json").read_text())
        assert "correct" in correct
        assert isinstance(correct["correct"], bool)
