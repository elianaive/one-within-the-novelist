"""Live API smoke test for the evaluation pipeline.

Confirms the pipeline runs end-to-end and produces valid output structure.
Does NOT assert on score values — LLM judgment is non-deterministic.

Cost: ~$0.03 per run.
Requires API keys in .env (OPENAI_API_KEY for judges, ANTHROPIC_API_KEY for classifier).
"""

import pytest

from owtn.evaluation.models import DIMENSION_NAMES
from owtn.evaluation.stage_1 import evaluate


@pytest.mark.live_api
class TestLiveEvaluation:
    @pytest.mark.asyncio
    async def test_pipeline_runs(self, genome_file, tmp_path):
        """A valid genome should evaluate successfully with real LLM calls."""
        results_dir = tmp_path / "results"
        result = await evaluate(
            str(genome_file),
            str(results_dir),
            "configs/stage_1/medium.yaml",
        )

        assert result.correct is True
        assert result.combined_score > 0
        assert result.holder_score > 0

        dims = result.public_metrics["dimensions"]
        for name in DIMENSION_NAMES:
            assert name in dims
            assert 0 <= dims[name] <= 5

        assert len(result.public_metrics["cell_key"]) == 3
        assert (results_dir / "correct.json").exists()
        assert (results_dir / "metrics.json").exists()
