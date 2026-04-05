"""Shared test fixtures and data for the OWTN test suite."""

import json

import pytest

from owtn.evaluation.models import JudgeScores


# --- Canonical test genomes ---

HILLS_GENOME = {
    "premise": "Two people at a train station discuss something they never name.",
    "target_effect": "The weight of what remains unsaid — dread, helplessness, the slow realization that silence is a form of violence.",
    "character_seeds": [
        {
            "label": "the man",
            "sketch": "Confident on the surface, steering the conversation with practiced ease.",
            "want": "For her to agree without him having to say what he wants.",
        },
        {
            "label": "the woman",
            "sketch": "Deflecting with imagery — the hills, the drinks, the beaded curtain.",
            "want": "To not have this conversation.",
        },
    ],
    "setting_seeds": "A train station in Spain. Hot. A bar with a beaded curtain. Two lines of rails in the sun.",
    "thematic_tension": "autonomy vs. obligation",
    "constraints": [
        "The word 'abortion' never appears.",
        "No interiority — only dialogue and physical action.",
        "Single scene, near-real-time.",
    ],
    "style_hint": "Spare, concrete, almost journalistic. The horror is in the contrast between flat tone and devastating content.",
}

MINIMAL_GENOME = {
    "premise": "A lighthouse keeper discovers the light has been signaling someone.",
    "target_effect": "Creeping dread and the vertigo of complicity.",
}


# --- Mock evaluation data ---

MOCK_JUDGE_SCORES = JudgeScores(
    reasoning="Strong concept with clear tension and specificity.",
    novelty=4.0,
    grip=3.5,
    tension_architecture=4.0,
    emotional_depth=3.0,
    thematic_resonance=3.0,
    concept_coherence=4.0,
    generative_fertility=3.0,
    scope_calibration=4.5,
    indelibility=3.5,
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


class FakeQueryResult:
    """Minimal mock of QueryResult for pipeline tests."""

    def __init__(self, content, model_name="gpt-4o", cost=0.005):
        self.content = content
        self.model_name = model_name
        self.cost = cost


# --- Shared fixtures ---

@pytest.fixture
def genome_file(tmp_path):
    """Write the canonical Hills genome to a temp JSON file."""
    p = tmp_path / "concept.json"
    p.write_text(json.dumps(HILLS_GENOME))
    return p


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path / "results"


@pytest.fixture
def mock_query_async():
    """Returns a mock LLM query function for pipeline tests."""
    async def _mock(model_name, msg, system_msg, output_model=None, **kwargs):
        if output_model is JudgeScores:
            return FakeQueryResult(MOCK_JUDGE_SCORES)
        return FakeQueryResult(
            MOCK_CLASSIFICATION_JSON,
            model_name="claude-haiku-4-5-20251001",
            cost=0.001,
        )
    return _mock
