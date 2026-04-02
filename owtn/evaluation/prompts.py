from __future__ import annotations

import json
from pathlib import Path

from owtn.evaluation.models import JudgeScores
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "stage_1"

HARSHNESS_INSTRUCTIONS = {
    "lenient": (
        "When a concept partially meets the criteria for two adjacent "
        "score levels, score to the higher level. If a rubric level "
        "lists multiple conditions (e.g., 'A and B'), meeting most of "
        "them is sufficient. Read implied potential as present — if "
        "the concept doesn't preclude a quality, treat it as available."
    ),
    "moderate": (
        "When a concept partially meets the criteria for two adjacent "
        "score levels, use your best judgment without rounding in "
        "either direction. If a rubric level lists multiple conditions, "
        "the concept should meet the majority. Only credit qualities "
        "that are clearly implied, not merely possible."
    ),
    "demanding": (
        "When a concept partially meets the criteria for two adjacent "
        "score levels, score to the lower level. If a rubric level "
        "lists multiple conditions, the concept must meet all of them. "
        "Only credit qualities that are explicitly present on the page. "
        "Potential is not evidence."
    ),
}


def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def build_judge_system(persona: JudgePersona) -> str:
    template = load_prompt("judge_system.txt")
    rubric_anchors = load_prompt("rubric_anchors.txt")

    values_str = "\n".join(f"- {v}" for v in persona.values)
    exemplars_str = "\n".join(f"- {e}" for e in persona.exemplars)
    harshness_instruction = HARSHNESS_INSTRUCTIONS[persona.harshness]

    return template.format(
        judge_name=persona.name,
        judge_identity=persona.identity,
        judge_values=values_str,
        judge_exemplars=exemplars_str,
        judge_harshness=persona.harshness,
        harshness_instruction=harshness_instruction,
        rubric_anchors=rubric_anchors,
    )


def build_judge_user(genome: ConceptGenome) -> str:
    template = load_prompt("judge_user.txt")
    return template.format(**genome.to_prompt_fields())


def build_classification_prompt(
    genome: ConceptGenome, avg_scores: dict[str, float]
) -> str:
    template = load_prompt("classification.txt")
    return template.format(
        genome_json=json.dumps(genome.model_dump(exclude_none=True), indent=2),
        scores_json=json.dumps(avg_scores, indent=2),
    )
