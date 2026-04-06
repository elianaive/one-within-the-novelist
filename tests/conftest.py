"""Shared test fixtures and data for the OWTN test suite."""

import json

import pytest


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
