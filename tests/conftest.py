"""Shared test fixtures and data for the OWTN test suite."""

import json

import pytest

from owtn.evaluation.models import DIMENSION_NAMES
from owtn.models.stage_1.config import PairwiseAggregationConfig


# --- Canonical test genomes ---

HILLS_GENOME = {
    "premise": "Two people at a train station discuss something they never name.",
    "anchor_scene": {
        "sketch": "She asks if they'll be fine afterward and he says yes; neither looks at the other, and the reader registers that both of them know he is lying.",
        "role": "reveal",
    },
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
    "thematic_engine": "held tension — the story refuses to choose between her right to decide and his refusal to let her, and the silence between them is the place where the unchosen sits and grows.",
    "constraints": [
        "The word 'abortion' never appears.",
        "No interiority — only dialogue and physical action.",
        "Single scene, near-real-time.",
    ],
    "style_hint": "Spare, concrete, almost journalistic. The horror is in the contrast between flat tone and devastating content.",
}

MINIMAL_GENOME = {
    "premise": "A lighthouse keeper discovers the light has been signaling someone.",
    "anchor_scene": {
        "sketch": "The keeper finds his own signature pattern in the previous keeper's logbook — the light has been signaling him across years, since before he arrived.",
        "role": "reveal",
    },
    "target_effect": "Creeping dread and the vertigo of complicity.",
}


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


# --- Pairwise aggregation config fixtures ---
#
# These mirror the shape of `evaluation.pairwise` in configs/stage_1/*.yaml.
# Keep `default_pairwise_cfg` in sync with the YAMLs — if weights change there,
# update here so test scenarios stay meaningful.

@pytest.fixture
def default_pairwise_cfg():
    """Canonical weights matching configs/stage_1/*.yaml. Explicit here so
    tests don't silently drift if the YAML values change."""
    return PairwiseAggregationConfig(
        dim_weights={
            "indelibility": 2.00,
            "grip": 1.75,
            "novelty": 1.75,
            "generative_fertility": 1.25,
            "tension_architecture": 1.00,
            "emotional_depth": 1.00,
            "thematic_resonance": 1.00,
            "concept_coherence": 0.50,
            "scope_calibration": 0.50,
        },
        tiebreaker_threshold=1.0,
        tiebreaker_dims=["indelibility", "grip"],
    )


@pytest.fixture
def uniform_pairwise_cfg():
    """All weights = 1.0, threshold = 0.0 — reproduces pre-change behavior.
    Used for regression tests that assert weighted selection degenerates to
    the old integer-majority rule."""
    return PairwiseAggregationConfig(
        dim_weights={d: 1.0 for d in DIMENSION_NAMES},
        tiebreaker_threshold=0.0,
        tiebreaker_dims=[],
    )
