"""Integration tests for the evaluation pipeline — makes real API calls.

Run with: uv run pytest tests/test_evaluation/test_integration.py -v
Requires API keys in .env (OPENAI_API_KEY for judges, ANTHROPIC_API_KEY for classifier).
Costs ~$0.03-0.05 per test (3 judge calls + 1 classifier call).
"""

import json
from pathlib import Path

import pytest

from owtn.evaluation.models import DIMENSION_NAMES
from owtn.evaluation.stage_1 import evaluate


SAMPLE_CONCEPTS = {
    "plant_anesthesia": {
        "premise": "A veterinary anesthesiologist moonlighting at a botanical research lab discovers that the same drugs she uses to put dogs under surgery render plants unconscious — and that the plants respond to dosage the way her patients do. She begins to suspect the ficus in her apartment has been aware of her this whole time.",
        "target_effect": "A slow, queasy dissolution of the boundary between sentience and mechanism — the feeling of realizing you may have been observed by something you never considered capable of observation.",
        "character_seeds": [
            {
                "label": "Dr. Lena Mäkelä",
                "sketch": "Precise, skeptical, uncomfortable with ambiguity. Has euthanized hundreds of animals and made peace with it by trusting the line between conscious and not.",
                "wound": "Her certainty about that line is the thing that lets her sleep.",
                "want": "To prove the plant responses are mechanical, not experiential.",
                "need": "To sit with not knowing.",
            }
        ],
        "thematic_tension": "the ethics of a consciousness boundary that may not exist",
        "constraints": [
            "No anthropomorphizing the plants — their responses must remain ambiguous.",
            "The story never resolves whether plants are conscious.",
        ],
        "style_hint": "Clinical precision that slowly becomes inadequate to describe what she's observing.",
    },
    "prison_chess": {
        "premise": "Two former political prisoners — one who tapped chess moves through cell walls for a decade, one who was the guard tasked with confiscating the soap chess pieces — meet at a reunion event thirty years later. The guard, now elderly and remorseful, asks to play a game.",
        "target_effect": "The vertigo of forgiveness offered to someone who hasn't earned it, and the question of whether the game itself — the thing that was forbidden — can carry the weight of what happened between them.",
        "character_seeds": [
            {
                "label": "Sipho",
                "sketch": "Former prisoner. Quiet authority. Plays chess competitively now — turned the survival skill into a public identity.",
                "want": "To be seen as more than what was done to him.",
                "need": "To stop performing recovery.",
            },
            {
                "label": "Van der Merwe",
                "sketch": "Former guard. Shrunken, apologetic, carries the confiscated soap king in his jacket pocket.",
                "want": "Absolution through the game.",
                "need": "To understand that absolution isn't his to request.",
            },
        ],
        "setting_seeds": "A community center in Cape Town. Fluorescent lights. Folding chairs. A cheap tournament chess set between them.",
        "thematic_tension": "justice vs. reconciliation — whether shared ritual can bridge what power made unbridgeable",
        "constraints": [
            "The chess game must be technically accurate.",
            "No flashbacks — the past exists only in what the characters say and don't say.",
        ],
    },
    "weak_cliche": {
        "premise": "A young woman discovers she has magical powers and must save her village from an ancient evil.",
        "target_effect": "A sense of wonder and empowerment.",
        "thematic_tension": "good vs. evil",
    },
}


CONFIG_PATH = "configs/stage_1/medium.yaml"


@pytest.fixture
def concept_dir(tmp_path):
    """Write all sample concepts to JSON files."""
    paths = {}
    for name, genome in SAMPLE_CONCEPTS.items():
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(genome))
        paths[name] = p
    return paths


class TestLiveEvaluation:
    """Full pipeline tests with real LLM calls."""

    @pytest.mark.asyncio
    async def test_strong_concept(self, concept_dir, tmp_path):
        """A strong concept (plant anesthesia) should score well."""
        results_dir = tmp_path / "results_strong"
        result = await evaluate(
            str(concept_dir["plant_anesthesia"]),
            str(results_dir),
            CONFIG_PATH,
        )

        assert result.correct is True
        assert result.combined_score > 0
        assert result.holder_score > 0

        # All dimensions should be scored.
        dims = result.public_metrics["dimensions"]
        for name in DIMENSION_NAMES:
            assert name in dims, f"Missing dimension: {name}"
            assert 0 <= dims[name] <= 5, f"{name} out of range: {dims[name]}"

        # Classification should produce a valid cell_key.
        cell_key = result.public_metrics["cell_key"]
        assert len(cell_key) == 3
        classification = result.public_metrics["classification"]
        assert classification["concept_type"] in (
            "thought_experiment", "situation_with_reveal", "voice_constraint",
            "character_collision", "atmospheric_associative", "constraint_driven",
        )

        # Judge reasoning should be present.
        assert len(result.text_feedback) > 100

        # Per-judge breakdowns in private metrics.
        judge_evals = result.private_metrics["judge_evaluations"]
        assert len(judge_evals) == 3
        for ev in judge_evals:
            assert ev["holder_score"] > 0
            assert ev["cost"] > 0

        # Output files written.
        assert (results_dir / "correct.json").exists()
        assert (results_dir / "metrics.json").exists()

        print(f"\n--- Plant Anesthesia ---")
        print(f"Combined: {result.combined_score:.2f}  Hölder: {result.holder_score:.2f}")
        for name in DIMENSION_NAMES:
            print(f"  {name}: {dims[name]:.1f}")
        print(f"Cell: {cell_key}")

    @pytest.mark.asyncio
    async def test_character_driven_concept(self, concept_dir, tmp_path):
        """A character-driven concept (prison chess) should score well."""
        results_dir = tmp_path / "results_character"
        result = await evaluate(
            str(concept_dir["prison_chess"]),
            str(results_dir),
            CONFIG_PATH,
        )

        assert result.correct is True
        assert result.combined_score > 0

        dims = result.public_metrics["dimensions"]
        for name in DIMENSION_NAMES:
            assert 0 <= dims[name] <= 5

        print(f"\n--- Prison Chess ---")
        print(f"Combined: {result.combined_score:.2f}  Hölder: {result.holder_score:.2f}")
        for name in DIMENSION_NAMES:
            print(f"  {name}: {dims[name]:.1f}")
        print(f"Cell: {result.public_metrics['cell_key']}")

    @pytest.mark.asyncio
    async def test_weak_concept_scores_lower(self, concept_dir, tmp_path):
        """A deliberately cliche concept should score noticeably lower."""
        results_dir = tmp_path / "results_weak"
        result = await evaluate(
            str(concept_dir["weak_cliche"]),
            str(results_dir),
            CONFIG_PATH,
        )

        assert result.correct is True
        dims = result.public_metrics["dimensions"]

        # Originality and anti_cliche should be low for this blatant cliche.
        assert dims["originality"] <= 3.0, f"Cliche rated too high on originality: {dims['originality']}"
        assert dims["anti_cliche"] <= 3.0, f"Cliche rated too high on anti_cliche: {dims['anti_cliche']}"

        print(f"\n--- Weak Cliche ---")
        print(f"Combined: {result.combined_score:.2f}  Hölder: {result.holder_score:.2f}")
        for name in DIMENSION_NAMES:
            print(f"  {name}: {dims[name]:.1f}")


class TestScoreDiscrimination:
    """Verify that the pipeline discriminates between strong and weak concepts."""

    @pytest.mark.asyncio
    async def test_strong_beats_weak(self, concept_dir, tmp_path):
        """Strong concepts should outscore weak ones."""
        results_strong = tmp_path / "results_strong2"
        results_weak = tmp_path / "results_weak2"

        result_strong = await evaluate(
            str(concept_dir["plant_anesthesia"]),
            str(results_strong),
            CONFIG_PATH,
        )
        result_weak = await evaluate(
            str(concept_dir["weak_cliche"]),
            str(results_weak),
            CONFIG_PATH,
        )

        assert result_strong.combined_score > result_weak.combined_score, (
            f"Strong ({result_strong.combined_score:.2f}) should beat "
            f"weak ({result_weak.combined_score:.2f})"
        )
        print(f"\nStrong: {result_strong.combined_score:.2f} vs Weak: {result_weak.combined_score:.2f}")
