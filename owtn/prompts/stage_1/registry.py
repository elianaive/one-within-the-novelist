"""Operator prompt registry for Stage 1 concept evolution.

Maps operator names to their prompt templates, patch routing, and seed types.
Consumed by ShinkaEvolve's sampler (Edit 4) to dispatch operator prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from owtn.models.stage_1.seed_bank import OPERATOR_SEED_TYPES, SeedBank

_PROMPTS_DIR = Path(__file__).resolve().parent
_OPERATORS_DIR = _PROMPTS_DIR / "operators"

from owtn.prompts import sample_tonal_steering  # noqa: E402 (after Path import)


@dataclass(frozen=True)
class OperatorDef:
    name: str
    routing: str  # "full" or "diff"
    needs_inspiration: bool  # cross-type: needs a second parent
    seed_types: list[str]
    sys_format: str  # appended to base system message
    operator_instructions: str  # fills {operator_instructions} in templates
    mutation_preamble: str = ""  # prepended to instructions in mutation mode


OPERATOR_DEFS: dict[str, dict] = {
    "collision":          {"routing": "full", "cross": True},
    "noun_list":          {"routing": "full", "cross": False},
    "thought_experiment": {"routing": "full", "cross": False},
    "compost":            {"routing": "full", "cross": True},
    "crossover":          {"routing": "full", "cross": True},
    "inversion":          {"routing": "diff", "cross": False},
    "discovery":          {"routing": "full", "cross": False},
    "compression":        {"routing": "full", "cross": False},
    "constraint_first":   {"routing": "full", "cross": False},
    "anti_premise":       {"routing": "full", "cross": False},
    "real_world_seed":    {"routing": "full", "cross": False},
}


def _load_text(path: Path) -> str:
    return path.read_text()


def _load_output_format() -> str:
    return _load_text(_PROMPTS_DIR / "output_format.txt")


def _load_iteration_template() -> str:
    return _load_text(_PROMPTS_DIR / "iteration.txt")


def _load_initial_template() -> str:
    return _load_text(_PROMPTS_DIR / "initial.txt")


def _load_base_system() -> str:
    return _load_text(_PROMPTS_DIR / "base_system.txt")


def _load_run_prompt_template() -> str:
    return _load_text(_PROMPTS_DIR / "run_prompt.txt")


def load_registry() -> dict[str, OperatorDef]:
    """Load all operator definitions with resolved prompt templates."""
    output_format = _load_output_format()
    registry = {}
    for name, defn in OPERATOR_DEFS.items():
        raw = _load_text(_OPERATORS_DIR / f"{name}.txt")
        seed_types = OPERATOR_SEED_TYPES.get(name, [])
        # The operator prompt file is the instructions portion.
        # Resolve {output_format} now; {seed_content} resolved at call time.
        instructions = raw.replace("{output_format}", output_format)
        # The sys_format is the first paragraph (identity line) of the operator.
        # ShinkaEvolve appends this to the base system message.
        first_break = instructions.find("\n\n---INSTRUCTIONS---\n\n")
        if first_break > 0:
            sys_format = instructions[:first_break]
            operator_body = instructions[first_break + len("\n\n---INSTRUCTIONS---\n\n"):]
        else:
            sys_format = ""
            operator_body = instructions

        # Split mutation preamble from operator body.
        mutation_preamble = ""
        mutation_break = operator_body.find("\n\n---MUTATION---\n\n")
        if mutation_break > 0:
            mutation_preamble = operator_body[mutation_break + len("\n\n---MUTATION---\n\n"):]
            operator_body = operator_body[:mutation_break]

        registry[name] = OperatorDef(
            name=name,
            routing=defn["routing"],
            needs_inspiration=defn["cross"],
            seed_types=seed_types,
            sys_format=sys_format,
            operator_instructions=operator_body,
            mutation_preamble=mutation_preamble,
        )
    return registry


def inject_seed(
    operator: str,
    seed_bank: SeedBank,
    exclude_ids: set[str] | None = None,
) -> str:
    """Select and format a seed for injection into an operator prompt.

    Returns formatted seed text, or empty string if no matching seed exists.
    """
    seed_types = OPERATOR_SEED_TYPES.get(operator, [])
    if not seed_types:
        return ""
    seed = seed_bank.select(seed_types, exclude_ids=exclude_ids)
    if seed is None:
        return ""
    content = seed.content if isinstance(seed.content, str) else "\n".join(seed.content)
    return f"\nUse this as your starting point:\n\n{content}"


# Dimension name → regex pattern for matching headers in judge reasoning.
# Judges format dimension names inconsistently (plain, numbered, bold, with
# hyphens or underscores), so patterns use . to match any separator.
_DIM_PATTERNS = {
    "novelty": "NOVELTY",
    "grip": "GRIP",
    "tension_architecture": "TENSION.ARCHITECTURE",
    "emotional_depth": "EMOTIONAL.DEPTH",
    "thematic_resonance": "THEMATIC.RESONANCE",
    "concept_coherence": "CONCEPT.COHERENCE",
    "generative_fertility": "GENERATIVE.FERTILITY",
    "scope_calibration": "SCOPE.CALIBRATION",
    "indelibility": "INDELIBILITY",
}

# Pre-compiled regex matching any dimension header.
_ANY_DIM_RE = re.compile(
    r'(?:^|\n)\s*(?:\*{0,2}\d*[.)]*\s*\*{0,2}\s*)?(?:'
    + "|".join(_DIM_PATTERNS.values())
    + r")",
    re.IGNORECASE,
)


def _extract_dimension_section(judge_text: str, dim_name: str) -> str | None:
    """Extract a single dimension's feedback from one judge's reasoning."""
    pattern = _DIM_PATTERNS.get(dim_name)
    if not pattern:
        return None
    header_re = re.compile(
        rf"(?:^|\n)\s*(?:\*{{0,2}}\d*[.)]*\s*\*{{0,2}}\s*)?{pattern}",
        re.IGNORECASE,
    )
    match = header_re.search(judge_text)
    if not match:
        return None
    # Find the next dimension header after this one.
    next_match = _ANY_DIM_RE.search(judge_text, match.end() + 1)
    end = next_match.start() if next_match else len(judge_text)
    return judge_text[match.start():end].strip()


_PARENT_BRIEF_CACHE_RENDER_KEY = "parent_brief_rendered"


def build_mutation_feedback(
    text_feedback: str | None,
    public_metrics: dict,
    private_metrics: dict | None = None,
) -> str:
    """Build the feedback section injected into the mutation prompt.

    Preferred path (Phase-3 feedback pipeline): if the parent's
    `private_metrics` contains a pre-rendered parent brief (set by the
    runner's async precompute step via `feedback.get_or_compute_brief`),
    return it verbatim. This is the curated, accumulated-critique summary.

    Fallback (legacy): for seeds, for programs without accumulated critiques,
    or if the precompute step was skipped, return a minimal pairwise-result
    snippet. No regex-based dimension extraction — that path produced
    corrupt, champion-praising feedback (see
    `lab/issues/2026-04-18-lazy-feedback-summarizer.md`).
    """
    if private_metrics:
        rendered = private_metrics.get(_PARENT_BRIEF_CACHE_RENDER_KEY)
        if rendered:
            return rendered

    if not text_feedback or not text_feedback.strip():
        return ""
    return text_feedback[:500]


def build_operator_prompt(
    operator: str,
    *,
    registry: dict[str, OperatorDef] | None = None,
    parent_genome: str = "",
    metrics: str = "",
    feedback: str = "",
    episodic_context: str = "",
    seed_bank: SeedBank | None = None,
    exclude_seed_ids: set[str] | None = None,
    prompt: str = "",
    tonal_steering: str = "",
    is_initial: bool = False,
) -> tuple[str, str]:
    """Build complete system and user messages for an operator.

    Returns (system_msg, user_msg) ready for ShinkaEvolve.
    """
    if registry is None:
        registry = load_registry()

    op = registry[operator]
    seed_text = ""
    if seed_bank is not None:
        seed_text = inject_seed(operator, seed_bank, exclude_ids=exclude_seed_ids)

    # System message order (see docs/prompting-guide.md):
    #   1. Operator persona — sets distributional neighborhood
    #   2. Tonal atmosphere — random affective register / literary mode
    #   3. Run-prompt block — user's directional pressure (omitted if empty)
    #   4. Base task description — structural contract last
    base_system = _load_base_system()
    tonal_atmosphere = f"\n\n{tonal_steering}" if tonal_steering else ""
    run_prompt_block = (
        _load_run_prompt_template().replace("{prompt}", prompt.strip())
        if prompt and prompt.strip()
        else ""
    )

    parts = []
    if op.sys_format:
        parts.append(op.sys_format)
    if tonal_atmosphere.strip():
        parts.append(tonal_atmosphere.strip())
    if run_prompt_block:
        parts.append(run_prompt_block.strip())
    parts.append(base_system)
    system_msg = "\n\n".join(parts)

    # Resolve {seed_content} in operator instructions.
    # {tonal_steering} placeholder kept for backward compat but now empty —
    # tonal steering lives in the system message as atmospheric context.
    instructions = (
        op.operator_instructions
        .replace("{seed_content}", seed_text)
        .replace("{tonal_steering}", "")
    )

    # In mutation mode, prepend the mutation preamble to anchor to the parent.
    if not is_initial and op.mutation_preamble:
        instructions = op.mutation_preamble + "\n\n" + instructions

    # User message: initial or iteration template.
    if is_initial:
        template = _load_initial_template()
        user_msg = template.replace("{operator_instructions}", instructions)
    else:
        template = _load_iteration_template()
        text_feedback_section = f"\n\n# Judge Feedback\n\n{feedback}" if feedback else ""
        episodic_section = f"\n\n# Context from Prior Runs\n\n{episodic_context}" if episodic_context else ""
        user_msg = (
            template
            .replace("{code_content}", parent_genome)
            .replace("{performance_metrics}", metrics)
            .replace("{text_feedback_section}", text_feedback_section)
            .replace("{episodic_context}", episodic_section)
            .replace("{operator_instructions}", instructions)
        )

    return system_msg, user_msg
