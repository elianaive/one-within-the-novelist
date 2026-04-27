from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class JudgePersona(BaseModel):
    id: str
    name: str
    identity: str
    values: list[str]
    exemplars: list[str]
    lean_in_signals: list[str]
    harshness: Literal[
        "advancing",
        "standard",
        "demanding",
        "failing_unless_exceptional",
    ]
    priority: Literal["primary", "secondary", "contrarian"]
    model: list[str]
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    # Reasoning effort for reasoning-capable models. "disabled" suppresses
    # reasoning-mode on models that allow it; models with requires_reasoning=True
    # in pricing.csv will coerce "disabled" → "low". Default disabled because
    # reasoning mode can let the model pre-commit to gestalt verdicts before
    # filling the structural sub-criteria template, degrading rubric-alignment.
    reasoning_effort: Literal["disabled", "low", "medium", "high"] = "disabled"


def load_panel(judges_dir: str, panel_ids: list[str]) -> list[JudgePersona]:
    """Load judge personas from YAML files."""
    dir_path = Path(judges_dir)
    judges = []
    for judge_id in panel_ids:
        path = dir_path / f"{judge_id}.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        judges.append(JudgePersona.model_validate(data))
    return judges
