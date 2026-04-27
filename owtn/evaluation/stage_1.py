"""Stage 1 evaluation: validation only.

Selection is handled by pairwise comparison in the runner, not by
pointwise scoring here. This module validates the genome (gates 1 & 2)
and returns correct/incorrect. No judge panel, no scoring, no classification.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import ValidationError

from owtn.evaluation.anti_cliche import gate_2_anti_cliche
from owtn.evaluation.models import EvaluationResult
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_1.config import StageConfig

logger = logging.getLogger(__name__)

# Patterns that indicate trivial/placeholder content.
_TRIVIAL_PATTERNS = [
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\blorem ipsum\b", re.IGNORECASE),
    re.compile(r"\binsert .{0,40}here\b", re.IGNORECASE),
    re.compile(r"\bplaceholder\b", re.IGNORECASE),
    re.compile(r"^a (?:person|man|woman|character) (?:faces|deals with|encounters) a (?:challenge|problem|situation)\.?$", re.IGNORECASE),
    re.compile(r"^this is a story about\b", re.IGNORECASE),
    re.compile(r"^write a story\b", re.IGNORECASE),
    re.compile(r"^generate a\b", re.IGNORECASE),
]


def _is_trivial(genome: ConceptGenome) -> str | None:
    """Return error message if genome is trivial/placeholder, else None."""
    for pattern in _TRIVIAL_PATTERNS:
        if pattern.search(genome.premise):
            return f"Trivial premise (matched: {pattern.pattern})"
        if pattern.search(genome.target_effect):
            return f"Trivial target_effect (matched: {pattern.pattern})"
        if pattern.search(genome.anchor_scene.sketch):
            return f"Trivial anchor_scene sketch (matched: {pattern.pattern})"
    return None


def _fail(error: str, results_dir: Path) -> EvaluationResult:
    """Write failure output and return result."""
    result = EvaluationResult(correct=False, error=error)
    _write_outputs(result, results_dir)
    return result


def _write_outputs(result: EvaluationResult, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "correct.json").write_text(
        json.dumps({"correct": result.correct})
    )
    (results_dir / "metrics.json").write_text(
        result.model_dump_json(indent=2)
    )


async def evaluate(
    program_path: str,
    results_dir: str,
    config_path: str,
) -> EvaluationResult:
    """Validate a concept genome. No scoring — selection is pairwise.

    Called as a subprocess by ShinkaEvolve. Writes metrics.json and
    correct.json to results_dir. Pairwise comparison against the island
    champion happens in the runner after this returns.
    """
    results_path = Path(results_dir)
    config = StageConfig.from_yaml(config_path)

    # --- GATE 1: Validation ---
    program = Path(program_path)
    try:
        raw = json.loads(program.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return _fail(f"Invalid JSON: {e}", results_path)

    try:
        genome = ConceptGenome.model_validate(raw)
    except ValidationError as e:
        return _fail(f"Genome validation failed: {e}", results_path)

    trivial_error = _is_trivial(genome)
    if trivial_error:
        return _fail(trivial_error, results_path)

    # --- GATE 2: Anti-Cliche (stub) ---
    anti_cliche = gate_2_anti_cliche(genome, config.evaluation.anti_cliche)

    # --- Result: valid concept, no scoring ---
    result = EvaluationResult(
        correct=True,
        combined_score=0.0,  # Placeholder — pairwise sets the real signal
        public_metrics={},
        private_metrics={"anti_cliche": anti_cliche},
        text_feedback="",  # Pairwise reasoning fills this in the runner
    )

    _write_outputs(result, results_path)
    logger.debug("Concept validated: %s", program_path)
    return result
