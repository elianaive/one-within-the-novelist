"""Prompt templates for the OWTN pipeline.

General prompts (base_system, judge, classification) are shared across stages.
Stage-specific operator prompts live in subdirectories (stage_1/, etc.).

All prompts are plain .txt files with {placeholder} syntax for str.format().

Operator prompt files contain two sections separated by '---INSTRUCTIONS---':
- Everything BEFORE the separator is the SYS_FORMAT (appended to the system message)
- Everything AFTER is the INSTRUCTIONS (injected into the iteration template's
  {operator_instructions} slot)

The iteration template (stage_1/iteration.txt) provides evolutionary context:
{code_content}, {performance_metrics}, {text_feedback_section} from the parent.
For initial generation (no parent), use stage_1/initial.txt instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

PROMPTS_DIR = Path(__file__).parent
OPERATOR_SEPARATOR = "---INSTRUCTIONS---"

# Loaded lazily on first access.
_affective_registers: list[dict] | None = None
_literary_modes: list[dict] | None = None


def _load_yaml_list(filename: str) -> list[dict]:
    path = PROMPTS_DIR / filename
    with open(path) as f:
        return yaml.safe_load(f)


def _ensure_loaded() -> tuple[list[dict], list[dict]]:
    global _affective_registers, _literary_modes
    if _affective_registers is None:
        _affective_registers = _load_yaml_list("affective_registers.yaml")
    if _literary_modes is None:
        _literary_modes = _load_yaml_list("literary_modes.yaml")
    return _affective_registers, _literary_modes


def _lookup(entries: list[dict], name: str) -> dict | None:
    for e in entries:
        if e["name"] == name:
            return e
    return None


def _build_combined(reg: dict, mode: dict) -> str:
    return f"{reg['text'].strip()}\n\n{mode['text'].strip()}"


def sample_tonal_steering(
    *,
    parent_register: str | None = None,
    parent_mode: str | None = None,
    parent2_register: str | None = None,
    parent2_mode: str | None = None,
    is_crossover: bool = False,
    inherit_rate: float = 0.5,
    crossover_new_rate: float = 0.33,
    rng: np.random.Generator | None = None,
) -> tuple[str, str, str]:
    """Sample one affective register and one literary mode.

    Inheritance logic (register and mode roll independently):
    - Genesis (no parent): always fresh random.
    - Mutation (one parent): inherit each dimension with probability
      ``inherit_rate``, else fresh random.
    - Crossover (two parents): for each dimension, pick parent A with
      probability (1 - crossover_new_rate) / 2, parent B with same,
      or fresh random with probability crossover_new_rate.

    Returns (combined_prompt_text, register_name, mode_name).
    """
    registers, modes = _ensure_loaded()
    if rng is None:
        rng = np.random.default_rng()

    def _random_register() -> dict:
        return registers[rng.integers(len(registers))]

    def _random_mode() -> dict:
        return modes[rng.integers(len(modes))]

    def _pick_dimension(
        entries: list[dict],
        random_fn,
        p1_name: str | None,
        p2_name: str | None,
    ) -> dict:
        if is_crossover and p1_name and p2_name:
            # Crossover: parent A, parent B, or new
            parent_share = (1.0 - crossover_new_rate) / 2.0
            roll = rng.random()
            if roll < parent_share:
                found = _lookup(entries, p1_name)
                if found:
                    return found
            elif roll < parent_share * 2:
                found = _lookup(entries, p2_name)
                if found:
                    return found
            return random_fn()
        elif p1_name:
            # Mutation: inherit or re-roll
            if rng.random() < inherit_rate:
                found = _lookup(entries, p1_name)
                if found:
                    return found
            return random_fn()
        else:
            # Genesis: fresh
            return random_fn()

    reg = _pick_dimension(registers, _random_register, parent_register, parent2_register)
    mode = _pick_dimension(modes, _random_mode, parent_mode, parent2_mode)

    return _build_combined(reg, mode), reg["name"], mode["name"]


def load(name: str) -> str:
    """Load a prompt template by name. Accepts 'base_system' or 'stage_1/collision'."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text()


def load_operator(name: str) -> Tuple[str, str]:
    """Load an operator prompt, split into (sys_format, instructions).

    Operator files contain both sections separated by '---INSTRUCTIONS---'.
    The sys_format is appended to the base system message.
    The instructions are injected into the iteration template.
    """
    raw = load(name)
    if OPERATOR_SEPARATOR in raw:
        sys_format, instructions = raw.split(OPERATOR_SEPARATOR, 1)
        return sys_format.strip(), instructions.strip()
    return raw.strip(), ""
