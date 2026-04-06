from __future__ import annotations

import json
from pathlib import Path

from owtn.evaluation.models import JudgeScores, PairwiseJudgment
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "stage_1"

HARSHNESS_INSTRUCTIONS = {
    "lenient": (
        "You are a generous reader. Most competent work lands around 3. "
        "You give 4s when something genuinely stands out. You've given "
        "a handful of 5s in your career — those were for work that "
        "changed how you thought about fiction."
    ),
    "moderate": (
        "You have high standards. Most work lands between 2 and 3. "
        "A 4 from you is rare — maybe a few times a year, for concepts "
        "that stay with you after you've closed the file. You've given "
        "a 5 once or twice in your life. A 3 is not a failure — it's "
        "the expected performance of competent work."
    ),
    "demanding": (
        "You are the toughest reader on the panel. Most work lands "
        "between 1.5 and 3. You almost never give above a 4. A 4 from "
        "you means the concept is exceptional — it genuinely surprised "
        "you, and you've read thousands. You have never given a 5. "
        "You're not sure a 5 exists."
    ),
}


def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def build_judge_system(persona: JudgePersona) -> str:
    template = load_prompt("judge_system.txt")
    rubric_anchors = load_prompt("rubric_anchors.txt")

    values_str = "\n".join(f"- {v}" for v in persona.values)
    exemplars_str = "\n".join(f"- {e}" for e in persona.exemplars)
    # harshness_instruction = HARSHNESS_INSTRUCTIONS[persona.harshness]

    return template.format(
        judge_name=persona.name,
        judge_identity=persona.identity,
        judge_values=values_str,
        judge_exemplars=exemplars_str,
        # judge_harshness=persona.harshness,
        # harshness_instruction=harshness_instruction,
        rubric_anchors=rubric_anchors,
    )


def build_judge_user(genome: ConceptGenome) -> str:
    template = load_prompt("judge_user.txt")
    return template.format(**genome.to_prompt_fields())


def build_pairwise_system(persona: JudgePersona) -> str:
    template = load_prompt("pairwise_system.txt")
    rubric_anchors = load_prompt("rubric_anchors.txt")

    values_str = "\n".join(f"- {v}" for v in persona.values)
    exemplars_str = "\n".join(f"- {e}" for e in persona.exemplars)

    return template.format(
        judge_name=persona.name,
        judge_identity=persona.identity,
        judge_values=values_str,
        judge_exemplars=exemplars_str,
        rubric_anchors=rubric_anchors,
    )


def build_pairwise_user(
    genome_a: ConceptGenome, genome_b: ConceptGenome
) -> str:
    template = load_prompt("pairwise_user.txt")
    fields_a = genome_a.to_prompt_fields()
    fields_b = genome_b.to_prompt_fields()
    return template.format(
        **{f"{k}_a": v for k, v in fields_a.items()},
        **{f"{k}_b": v for k, v in fields_b.items()},
    )


def build_classification_prompt(
    genome: ConceptGenome, avg_scores: dict[str, float]
) -> str:
    template = load_prompt("classification.txt")
    return template.format(
        genome_json=json.dumps(genome.model_dump(exclude_none=True), indent=2),
        scores_json=json.dumps(avg_scores, indent=2),
    )
