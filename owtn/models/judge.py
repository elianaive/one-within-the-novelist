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
    harshness: Literal["lenient", "moderate", "demanding"]
    priority: Literal["primary", "secondary", "contrarian"]
    model: list[str]


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
