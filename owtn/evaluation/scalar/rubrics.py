"""YAML loader for rubrics.

Rubrics live in `configs/scalar/rubric_*.yaml`. Each file is a sequence of
dim definitions plus optional scale parameters. This module reads + validates
into the typed `Rubric`/`Dim` dataclasses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from owtn.evaluation.scalar.types import Dim, Rubric

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "scalar"


def load_rubric(rubric_name: str, *, config_dir: Path | None = None) -> Rubric:
    """Load a rubric by short name (`concept`, `dag`, ...) from its YAML.

    Looks for `<config_dir>/rubric_<name>.yaml`. Default config_dir is
    `configs/scalar/` at the repo root.
    """
    config_dir = config_dir or _DEFAULT_CONFIG_DIR
    path = config_dir / f"rubric_{rubric_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"rubric YAML not found: {path}")

    raw = yaml.safe_load(path.read_text())
    return _parse_rubric(raw, source=path)


def _parse_rubric(raw: dict[str, Any], *, source: Path) -> Rubric:
    if not isinstance(raw, dict) or "dims" not in raw:
        raise ValueError(f"{source}: rubric YAML must be a mapping with a `dims` key")

    dims = tuple(_parse_dim(d, source=source) for d in raw["dims"])
    if not dims:
        raise ValueError(f"{source}: rubric must have at least one dim")

    kwargs: dict[str, Any] = {"dims": dims}
    for key in ("scale_min", "scale_max", "scale_anchors"):
        if key in raw:
            kwargs[key] = raw[key]
    return Rubric(**kwargs)


def _parse_dim(d: dict[str, Any], *, source: Path) -> Dim:
    if not isinstance(d, dict) or "name" not in d or "description" not in d:
        raise ValueError(f"{source}: each dim needs `name` and `description`")
    return Dim(
        name=d["name"],
        description=d["description"],
        polarity=d.get("polarity", "positive"),
        weight=float(d.get("weight", 1.0)),
    )
