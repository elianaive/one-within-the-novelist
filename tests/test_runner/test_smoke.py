"""Smoke test for the Stage 1 concept evolution pipeline.

Tests the integration path: operator prompt → LLM → JSON genome → evaluation.
Requires API keys. Skipped automatically if keys are not set.

Cost: ~$0.05-0.10 per run.
"""

from __future__ import annotations

import json
import os

import pytest

HAS_API_KEYS = bool(os.environ.get("ANTHROPIC_API_KEY")) or bool(
    os.environ.get("OPENAI_API_KEY")
)

pytestmark = pytest.mark.skipif(
    not HAS_API_KEYS,
    reason="No API keys available for smoke test",
)


class TestEvaluationSmoke:
    """Test that a concept genome can be evaluated end-to-end."""

    @pytest.mark.asyncio
    async def test_evaluate_concept(self, tmp_path):
        from owtn.evaluation.stage_1 import evaluate

        genome = {
            "premise": "A linguist discovers that a dying language contains a grammatical tense that allows speakers to describe events that happened to people who were never born — and she realizes her own unborn daughter has a past tense history in this language.",
            "target_effect": "The reader should feel vertigo at the boundary between existence and non-existence, then a spreading ache for lives that almost happened.",
            "character_seeds": [
                {
                    "label": "Dr. Elena Vasquez",
                    "sketch": "Field linguist, 42, meticulous recorder",
                }
            ],
            "thematic_tension": "preservation vs. letting go",
            "constraints": ["No exposition about how the tense works — show it in use"],
        }

        program_path = tmp_path / "concept.json"
        program_path.write_text(json.dumps(genome))
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        result = await evaluate(
            program_path=str(program_path),
            results_dir=str(results_dir),
            config_path="configs/stage_1/medium.yaml",
        )

        assert result.correct
        assert result.combined_score > 0
        assert result.holder_score > 0
        cell = result.public_metrics.get("cell_key") or result.public_metrics.get("map_elites_cell")
        assert cell is not None, f"No cell key in public_metrics: {list(result.public_metrics.keys())}"
