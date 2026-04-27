"""Stage 2 prompt assembly.

Mirrors `owtn.prompts.stage_1.registry` in spirit (template files loaded from
disk, placeholders substituted at call time). Phase 3 shipped seed_root's
merged motif + demand extraction; Phase 4 adds the judge pairwise prompts.

Each loader is a one-liner returning the file contents; substitution happens
in the `build_*` functions. Templates live alongside this module as `.txt`
files; placeholder convention is `{UPPER_SNAKE_CASE}` for top-level
substitutions and `{lower_snake_case}` for fields fed via `.format()`.
"""

from __future__ import annotations

import json
from pathlib import Path

from owtn.evaluation.models import STAGE_2_DIMENSION_NAMES
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome


_PROMPTS_DIR = Path(__file__).resolve().parent
_RUBRIC_DIR = _PROMPTS_DIR / "rubric_anchors"

# Harshness templates: reused from Stage 1. Same vocabulary works for either
# stage's judges — the templates talk about "picking a winner per dimension"
# in dimension-set-agnostic language. If Stage-2-specific harshness text is
# ever needed, copy these into `owtn/prompts/stage_2/harshness/` and point
# `_HARSHNESS_DIR` here instead.
_HARSHNESS_DIR = _PROMPTS_DIR.parent / "stage_1" / "harshness"


def _load(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def load_base_system() -> str:
    """Stage 2 base system message — the typed-edge taxonomy and beat-sketch
    contract. Shared between operators (this phase) and judges (Phase 4)."""
    return _load("base_system.txt")


def load_seed_motif_template() -> str:
    """Raw template for the merged motif + concept_demand extraction prompt.
    Placeholders: `{CONCEPT_JSON}`, `{ANCHOR_SKETCH}`, `{ANCHOR_ROLE}`."""
    return _load("seed_motif.txt")


def build_seed_motif_prompt(concept: ConceptGenome) -> tuple[str, str]:
    """Assemble (system_msg, user_msg) for one seed_root extraction call.

    The user message is the seed_motif template with concept fields filled in.
    The system message is the Stage 2 base — establishes typed-edge context
    so the LLM produces motifs/demands consistent with the DAG schema the
    rest of Stage 2 will operate on.
    """
    system_msg = load_base_system()
    template = load_seed_motif_template()
    concept_json = json.dumps(concept.model_dump(exclude_none=True), indent=2)
    user_msg = (
        template
        .replace("{CONCEPT_JSON}", concept_json)
        .replace("{ANCHOR_SKETCH}", concept.anchor_scene.sketch)
        .replace("{ANCHOR_ROLE}", concept.anchor_scene.role)
    )
    return system_msg, user_msg


# ----- Pairwise judging (Phase 4) -----


def load_rubric_anchors() -> str:
    """Concatenate the 8 dimension rubric files in canonical order.

    Each file lives at `rubric_anchors/<dim>.txt`. Concatenation order
    matches `STAGE_2_DIMENSION_NAMES`, so the per-dim reasoning blocks the
    judge produces will appear in the same order in their structured output.
    """
    parts: list[str] = []
    for dim in STAGE_2_DIMENSION_NAMES:
        text = (_RUBRIC_DIR / f"{dim}.txt").read_text().rstrip()
        parts.append(text)
    return "\n\n".join(parts)


def build_stage2_pairwise_system(persona: JudgePersona) -> str:
    """Build the system message for one judge evaluating two structures.

    Substitutes the persona's identity, values, exemplars, lean-in signals,
    harshness instruction, and the concatenated 8-dim rubric anchors into
    `judge_system.txt`. The rendered message is stable across all comparisons
    by this judge, so callers should pass it as `system_prefix` to
    `query_async` for prompt-cache benefits.
    """
    template = (_PROMPTS_DIR / "judge_system.txt").read_text()
    rubric_anchors = load_rubric_anchors()
    harshness_text = (_HARSHNESS_DIR / f"{persona.harshness}.txt").read_text().strip()

    values_str = "\n".join(f"- {v}" for v in persona.values)
    exemplars_str = "\n".join(f"- {e}" for e in persona.exemplars)
    lean_in_str = "\n".join(f"- {s}" for s in persona.lean_in_signals)

    return template.format(
        judge_name=persona.name,
        judge_identity=persona.identity,
        judge_values=values_str,
        judge_exemplars=exemplars_str,
        judge_lean_in_signals=lean_in_str,
        judge_harshness=persona.harshness,
        harshness_instruction=harshness_text,
        rubric_anchors=rubric_anchors,
    )


_PHASE_DESCRIPTIONS = {
    "forward": (
        "FORWARD phase. The seed is the anchor (the role-bearing node) and "
        "MCTS is growing beats DOWNSTREAM of the anchor — expanding the "
        "falling-action / coda region. New beats sit temporally after the "
        "anchor."
    ),
    "backward": (
        "BACKWARD phase. The forward winner has been carried forward as "
        "the root; MCTS is now growing beats UPSTREAM of the anchor — "
        "expanding the rising-action / setup region. New beats sit "
        "temporally before the anchor."
    ),
    "refinement": (
        "REFINEMENT phase (Phase 3). Both upstream and downstream of the "
        "anchor are now established. ONLY add_edge actions are allowed — "
        "no new beats, no rewrites. Edges must SPAN the anchor: at least "
        "one endpoint upstream of the anchor, at least one downstream "
        "(or one endpoint at the anchor with the other on either side). "
        "Same-side edges should have been added during forward or backward."
    ),
}


_DIRECTION_RULES = {
    "forward": (
        "`anchor_id` (the source) must be the anchor or a descendant of "
        "the anchor. `direction` must be 'downstream'. The new beat is "
        "the target of the new edge — it sits downstream of the anchor."
    ),
    "backward": (
        "`anchor_id` (the target) must be the anchor or an ancestor of "
        "the anchor. `direction` must be 'upstream'. The new beat is the "
        "source of the new edge — it sits upstream of the anchor."
    ),
    "refinement": (
        "Not applicable in this phase — `add_beat` actions are rejected "
        "during refinement."
    ),
}


_ADD_EDGE_RULES = {
    "forward": (
        "Both endpoints must be the anchor or descendants of the anchor "
        "(no spanning edges in this phase)."
    ),
    "backward": (
        "Both endpoints must be the anchor or ancestors of the anchor "
        "(no spanning edges in this phase)."
    ),
    "refinement": (
        "Edges MUST span the anchor — at least one endpoint upstream, at "
        "least one endpoint downstream (or one endpoint at the anchor "
        "with the other on either side). Non-spanning add_edge actions "
        "are rejected."
    ),
}


def build_expansion_prompt(
    concept: ConceptGenome,
    dag_rendering: str,
    *,
    phase: str,
    permitted_edge_types: list[str],
    pacing_hint: str,
    champion_brief: str = "(brief not yet available)",
    extra_context: str = "",
    k: int = 4,
) -> tuple[str, str]:
    """Assemble (system_msg, user_msg) for one MCTS expansion call.

    `dag_rendering` is the incident-encoded outline from
    `owtn.stage_2.rendering.render(dag)`. `phase` is one of "forward",
    "backward", "refinement" — selects phase-specific instructions.
    `champion_brief` is the rendered text from
    `owtn.stage_2.champion_brief.get_or_compute_brief` — defaults to a
    cold-start placeholder so callers without brief state still produce
    valid prompts. `extra_context` is per-call DAG-state context; refinement
    uses it to pass the anchor's upstream/downstream topology so the LLM
    can target spanning edges directly. Forward/backward leave it empty.
    """
    system_msg = load_base_system()
    template = (_PROMPTS_DIR / "expansion.txt").read_text()
    concept_json = json.dumps(concept.model_dump(exclude_none=True), indent=2)
    extra_block = f"\n{extra_context}\n" if extra_context else ""
    user_msg = (
        template
        .replace("{K}", str(k))
        .replace("{PHASE_DESCRIPTION}", _PHASE_DESCRIPTIONS.get(phase, phase))
        .replace("{EXTRA_CONTEXT}", extra_block)
        .replace("{PERMITTED_EDGE_TYPES}", ", ".join(sorted(permitted_edge_types)))
        .replace("{ADD_BEAT_DIRECTION_RULE}", _DIRECTION_RULES.get(phase, ""))
        .replace("{ADD_EDGE_PHASE_RULE}", _ADD_EDGE_RULES.get(phase, ""))
        .replace("{CONCEPT_JSON}", concept_json)
        .replace("{DAG_RENDERING}", dag_rendering)
        .replace("{PACING_HINT}", pacing_hint or "(no preset hint)")
        .replace("{CHAMPION_BRIEF}", champion_brief)
    )
    return system_msg, user_msg


def build_stage2_pairwise_user(
    concept: ConceptGenome,
    structure_a_rendering: str,
    structure_b_rendering: str,
) -> str:
    """Build the user message for one pairwise comparison.

    `structure_*_rendering` are pre-rendered incident-encoded outlines from
    `owtn.stage_2.rendering.render(dag, label="A"/"B")`. We do not render
    inline here — the renderer is the single source of truth for layout and
    judges must see the same format every comparison.

    `concept` is the Stage 1 concept both structures realize. It's included
    so judges can evaluate concept-fidelity dimensions properly.
    """
    template = (_PROMPTS_DIR / "judge_user.txt").read_text()
    fields = concept.to_prompt_fields()
    return template.format(
        structure_a_rendering=structure_a_rendering,
        structure_b_rendering=structure_b_rendering,
        **fields,
    )
