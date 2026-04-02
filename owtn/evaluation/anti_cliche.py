from __future__ import annotations

from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_1.config import AntiCliqueConfig


def gate_2_anti_cliche(genome: ConceptGenome, config: AntiCliqueConfig) -> dict:
    """Stub — returns not-flagged.

    Will embed premise and compare against convergence patterns
    (data/convergence-patterns.yaml) when embedding client lands.
    """
    return {
        "flagged": False,
        "stub": True,
        "similarity": 0.0,
        "matched_pattern": None,
    }
