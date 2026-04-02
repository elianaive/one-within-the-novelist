import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from owtn.evaluation.models import DIMENSION_NAMES, JudgeScores
from owtn.evaluation.prompts import build_judge_system, build_judge_user
from owtn.evaluation.stage_1 import _is_trivial, evaluate
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome


HILLS_GENOME = {
    "premise": "Two people at a train station discuss something they never name.",
    "target_effect": "The weight of what remains unsaid — dread, helplessness, the slow realization that silence is a form of violence.",
    "character_seeds": [
        {
            "label": "the man",
            "sketch": "Confident on the surface, steering the conversation.",
            "want": "For her to agree without him having to say what he wants.",
        },
        {
            "label": "the woman",
            "sketch": "Deflecting with imagery — the hills, the drinks.",
            "want": "To not have this conversation.",
        },
    ],
    "thematic_tension": "autonomy vs. obligation",
    "constraints": [
        "The word 'abortion' never appears.",
        "No interiority — only dialogue and physical action.",
        "Single scene, near-real-time.",
    ],
}

MINIMAL_GENOME = {
    "premise": "A lighthouse keeper discovers the light has been signaling someone.",
    "target_effect": "Creeping dread and the vertigo of complicity.",
}

MOCK_JUDGE_SCORES = JudgeScores(
    reasoning="Strong concept with clear tension and specificity.",
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

MOCK_CLASSIFICATION_JSON = json.dumps({
    "concept_type": "voice_constraint",
    "concept_type_confidence": "high",
    "arc_shape": "fall",
    "arc_shape_confidence": "medium",
    "tonal_register": "matter_of_fact",
    "tonal_register_confidence": "high",
    "thematic_domain": "interpersonal",
    "thematic_domain_confidence": "high",
})


# --- Gate 1: Validation ---


class TestGate1Validation:
    def test_valid_genome_passes(self):
        genome = ConceptGenome.model_validate(HILLS_GENOME)
        assert _is_trivial(genome) is None

    def test_minimal_genome_passes(self):
        genome = ConceptGenome.model_validate(MINIMAL_GENOME)
        assert _is_trivial(genome) is None

    def test_todo_premise(self):
        genome = ConceptGenome(
            premise="TODO write a real premise here later",
            target_effect="Something unsettling and existential.",
        )
        assert _is_trivial(genome) is not None

    def test_lorem_ipsum(self):
        genome = ConceptGenome(
            premise="Lorem ipsum dolor sit amet, this is a test premise.",
            target_effect="Something unsettling and existential.",
        )
        assert _is_trivial(genome) is not None

    def test_placeholder_premise(self):
        genome = ConceptGenome(
            premise="Insert your creative premise here please.",
            target_effect="Something unsettling and existential.",
        )
        assert _is_trivial(genome) is not None

    def test_generic_premise(self):
        genome = ConceptGenome(
            premise="A person faces a challenge.",
            target_effect="Something unsettling and existential.",
        )
        assert _is_trivial(genome) is not None

    def test_meta_commentary_premise(self):
        genome = ConceptGenome(
            premise="This is a story about loss and redemption.",
            target_effect="Something unsettling and existential.",
        )
        assert _is_trivial(genome) is not None

    def test_trivial_target_effect(self):
        genome = ConceptGenome(
            premise="A lighthouse keeper discovers the light has been signaling someone.",
            target_effect="Write a story about this premise.",
        )
        assert _is_trivial(genome) is not None


# --- Prompt Assembly ---


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
        # No unfilled template placeholders should remain.
        assert "{" not in prompt or "{{" in prompt  # allow literal braces
        assert judge.name in prompt
        assert "Grip" in prompt
        assert "use your best judgment" in prompt  # moderate harshness

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


# --- End-to-End (Mocked LLM) ---


class _FakeQueryResult:
    """Minimal mock of QueryResult."""

    def __init__(self, content, model_name="gpt-4o", cost=0.005):
        self.content = content
        self.model_name = model_name
        self.cost = cost


class TestEndToEnd:
    @pytest.fixture
    def genome_file(self, tmp_path):
        p = tmp_path / "concept.json"
        p.write_text(json.dumps(HILLS_GENOME))
        return p

    @pytest.fixture
    def config_path(self):
        return "configs/stage_1_default.yaml"

    @pytest.fixture
    def results_dir(self, tmp_path):
        return tmp_path / "results"

    @pytest.mark.asyncio
    async def test_gate1_invalid_json(self, tmp_path, config_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{")
        results_dir = tmp_path / "results"
        result = await evaluate(str(bad_file), str(results_dir), config_path)
        assert result.correct is False
        assert "Invalid JSON" in result.error
        assert (results_dir / "correct.json").exists()
        correct = json.loads((results_dir / "correct.json").read_text())
        assert correct["correct"] is False

    @pytest.mark.asyncio
    async def test_gate1_missing_premise(self, tmp_path, config_path):
        bad_file = tmp_path / "no_premise.json"
        bad_file.write_text(json.dumps({"target_effect": "Something heavy."}))
        results_dir = tmp_path / "results"
        result = await evaluate(str(bad_file), str(results_dir), config_path)
        assert result.correct is False
        assert "validation failed" in result.error.lower() or "premise" in result.error.lower()

    @pytest.mark.asyncio
    async def test_gate1_trivial_content(self, tmp_path, config_path):
        trivial = tmp_path / "trivial.json"
        trivial.write_text(json.dumps({
            "premise": "A person faces a challenge.",
            "target_effect": "Something unsettling and existential.",
        }))
        results_dir = tmp_path / "results"
        result = await evaluate(str(trivial), str(results_dir), config_path)
        assert result.correct is False
        assert "Trivial" in result.error

    @pytest.mark.asyncio
    async def test_full_pipeline(self, genome_file, results_dir, config_path):
        """Full pipeline with mocked LLM calls."""

        async def mock_query_async(model_name, msg, system_msg, output_model=None, **kwargs):
            if output_model is JudgeScores:
                return _FakeQueryResult(MOCK_JUDGE_SCORES)
            # Classification call (Anthropic, no output_model).
            return _FakeQueryResult(MOCK_CLASSIFICATION_JSON, model_name="claude-haiku-4-5-20251001", cost=0.001)

        with patch("owtn.evaluation.stage_1.query_async", side_effect=mock_query_async):
            result = await evaluate(str(genome_file), str(results_dir), config_path)

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
        metrics = json.loads((results_dir / "metrics.json").read_text())
        assert metrics["combined_score"] == result.combined_score

    @pytest.mark.asyncio
    async def test_classification_cell_key(self, genome_file, results_dir, config_path):
        """Verify classification produces correct cell_key."""

        async def mock_query_async(model_name, msg, system_msg, output_model=None, **kwargs):
            if output_model is JudgeScores:
                return _FakeQueryResult(MOCK_JUDGE_SCORES)
            return _FakeQueryResult(MOCK_CLASSIFICATION_JSON, model_name="claude-haiku-4-5-20251001", cost=0.001)

        with patch("owtn.evaluation.stage_1.query_async", side_effect=mock_query_async):
            result = await evaluate(str(genome_file), str(results_dir), config_path)

        cell_key = result.public_metrics["cell_key"]
        assert cell_key == ["voice_constraint", "fall", "heavy"]
