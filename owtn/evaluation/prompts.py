from __future__ import annotations

from pathlib import Path

from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "stage_1"


def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def build_pairwise_system(persona: JudgePersona) -> str:
    template = load_prompt("pairwise_system.txt")
    rubric_anchors = load_prompt("rubric_anchors.txt")
    harshness_instruction = load_prompt(f"harshness/{persona.harshness}.txt").strip()

    values_str = "\n".join(f"- {v}" for v in persona.values)
    exemplars_str = "\n".join(f"- {e}" for e in persona.exemplars)

    return template.format(
        judge_name=persona.name,
        judge_identity=persona.identity,
        judge_values=values_str,
        judge_exemplars=exemplars_str,
        judge_harshness=persona.harshness,
        harshness_instruction=harshness_instruction,
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
