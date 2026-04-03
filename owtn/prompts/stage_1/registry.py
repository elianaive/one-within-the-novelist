"""Operator prompt registry for Stage 1 concept evolution.

Maps operator names to their prompt templates, patch routing, and seed types.
Consumed by ShinkaEvolve's sampler (Edit 4) to dispatch operator prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from owtn.models.stage_1.seed_bank import OPERATOR_SEED_TYPES, SeedBank

_PROMPTS_DIR = Path(__file__).resolve().parent
_OPERATORS_DIR = _PROMPTS_DIR / "operators"

# Tonal targets for diversity steering. Each entry is a short creative direction
# that pushes the LLM toward a different emotional register. Assigned per-concept
# to prevent the population from collapsing into a single aesthetic.
TONAL_TARGETS = [
    "Write in the register of DREAD — something closing in, inevitable, quiet horror. The reader should want to stop reading and not be able to.",
    "Write in the register of COMEDY — genuine humor, wit, absurdist logic, or the kind of joke that makes you see something true. Solemnity is not depth.",
    "Write in the register of TENDERNESS — unguarded, specific, earned warmth. Not sentimentality. The kind of gentleness that only works because it's precise.",
    "Write in the register of FURY — real anger, not melodrama. A situation so unjust the reader's jaw tightens. The story should vibrate with it.",
    "Write in the register of WONDER — genuine astonishment, the feeling of encountering something that shouldn't exist but does. Curiosity as an engine.",
    "Write in the register of ABSURDITY — dream logic, Kafka's bureaucratic nightmare, Barthelme's debris. The premise should be impossible and emotionally true.",
    "Write in the register of MELANCHOLY — not sadness, but the particular ache of time passing. Nostalgia with teeth. Things that were and won't be again.",
    "Write in the register of GIDDINESS — the rush of transgression, getting away with it, the thrill of a plan working or spectacularly failing. Energy, not stillness.",
    "Write in the register of FLATNESS — affectless, procedural, Carver-esque. The emotional weight is in what's not said. Restraint as power.",
    "Write in the register of REVULSION — not gore, but moral or existential disgust. Something the reader can't unsee. Discomfort that produces insight.",
]


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
    for seed_type in seed_types:
        seed = seed_bank.select(seed_type, exclude_ids=exclude_ids)
        if seed is not None:
            content = seed.content if isinstance(seed.content, str) else "\n".join(seed.content)
            return f"\nUse this as your starting point:\n\n{content}"
    return ""


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
    steering: str = "",
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

    # System message: base + operator identity
    base_system = _load_base_system()
    steering_section = f"\n\nCreative direction for this run: {steering}" if steering else ""
    system_msg = base_system.replace("{steering_section}", steering_section)
    if op.sys_format:
        system_msg += "\n\n" + op.sys_format

    # Resolve {seed_content} and {tonal_steering} in operator instructions.
    tonal_section = f"\n3. Tonal target: {tonal_steering}" if tonal_steering else ""
    instructions = (
        op.operator_instructions
        .replace("{seed_content}", seed_text)
        .replace("{tonal_steering}", tonal_section)
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
