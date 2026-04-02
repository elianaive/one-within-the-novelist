import pytest
from pydantic import ValidationError

from owtn.evaluation.models import (
    DIMENSION_NAMES,
    EvaluationResult,
    JudgeEvaluation,
    JudgeScores,
)


VALID_SCORES = {
    "reasoning": "This concept is strong because...",
    "originality": 4.0,
    "transportation_potential": 3.5,
    "narrative_tension": 4.0,
    "thematic_resonance": 3.0,
    "scope_calibration": 4.5,
    "anti_cliche": 3.5,
    "concept_coherence": 4.0,
    "generative_fertility": 3.0,
    "over_explanation_resistance": 4.0,
}


class TestJudgeScores:
    def test_valid(self):
        scores = JudgeScores(**VALID_SCORES)
        assert scores.originality == 4.0
        assert scores.reasoning.startswith("This concept")

    def test_score_below_zero(self):
        data = {**VALID_SCORES, "originality": -0.1}
        with pytest.raises(ValidationError):
            JudgeScores(**data)

    def test_score_above_five(self):
        data = {**VALID_SCORES, "originality": 5.1}
        with pytest.raises(ValidationError):
            JudgeScores(**data)

    def test_boundary_values(self):
        data = {**VALID_SCORES, "originality": 0.0, "scope_calibration": 5.0}
        scores = JudgeScores(**data)
        assert scores.originality == 0.0
        assert scores.scope_calibration == 5.0

    def test_to_list_order(self):
        scores = JudgeScores(**VALID_SCORES)
        score_list = scores.to_list()
        assert len(score_list) == 9
        for i, name in enumerate(DIMENSION_NAMES):
            assert score_list[i] == getattr(scores, name)

    def test_to_dict_excludes_reasoning(self):
        scores = JudgeScores(**VALID_SCORES)
        d = scores.to_dict()
        assert "reasoning" not in d
        assert len(d) == 9
        assert d["originality"] == 4.0


class TestJudgeEvaluation:
    def test_construction(self):
        scores = JudgeScores(**VALID_SCORES)
        ev = JudgeEvaluation(
            judge_id="mira-okonkwo",
            scores=scores,
            holder_score=3.7,
            model_used="gpt-4o",
            cost=0.005,
        )
        assert ev.judge_id == "mira-okonkwo"
        assert ev.scores.originality == 4.0


class TestEvaluationResult:
    def test_failure_result(self):
        result = EvaluationResult(correct=False, error="Invalid JSON")
        assert result.combined_score == 0.0
        assert result.text_feedback == ""

    def test_serialization_round_trip(self):
        result = EvaluationResult(
            correct=True,
            combined_score=3.5,
            holder_score=3.4,
            public_metrics={"dimensions": {"originality": 4.0}},
            private_metrics={"total_cost": 0.01},
            text_feedback="Good concept.",
        )
        json_str = result.model_dump_json()
        restored = EvaluationResult.model_validate_json(json_str)
        assert restored.combined_score == 3.5
        assert restored.public_metrics["dimensions"]["originality"] == 4.0
