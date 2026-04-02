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

from pathlib import Path
from typing import Tuple

PROMPTS_DIR = Path(__file__).parent
OPERATOR_SEPARATOR = "---INSTRUCTIONS---"


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
