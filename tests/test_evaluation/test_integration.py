"""Live API smoke test for validation-only evaluation.

Confirms the evaluation pipeline validates a concept and returns the
correct structure. Pairwise comparison is tested separately via the runner.

Cost: ~$0.00 (no LLM calls — validation is local).
"""

import pytest

from owtn.evaluation.stage_1 import evaluate


@pytest.mark.live_api
class TestLiveEvaluation:
    @pytest.mark.asyncio
    async def test_validation_runs(self, genome_file, tmp_path):
        """A valid genome should pass validation."""
        results_dir = tmp_path / "results"
        result = await evaluate(
            str(genome_file),
            str(results_dir),
            "configs/stage_1/medium.yaml",
        )

        assert result.correct is True
        assert result.combined_score == 0.0  # Placeholder — pairwise sets real score
        assert (results_dir / "correct.json").exists()
        assert (results_dir / "metrics.json").exists()
