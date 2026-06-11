"""Critic-persona loader.

Reads YAMLs from `configs/stage_4/critics/`, validates each against
`CriticPersona`, and returns a sorted-by-id pool. Mirrors the shape of
`owtn.stage_3.personas.load_persona_pool` — file naming, `_template`
skip, malformed-YAML logging — to keep the cross-stage pattern
recognizable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from owtn.models.stage_4 import CriticPersona


logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CRITICS_DIR = REPO_ROOT / "configs" / "stage_4" / "critics"


def load_critic_pool(
    critics_dir: Path | str | None = None,
) -> list[CriticPersona]:
    """Load every critic YAML in the directory, sorted by id.

    Files starting with `_` (templates, comment files) are skipped.
    Malformed YAMLs are logged and dropped without killing the pool.
    Returns an empty list when the dir is missing — caller decides policy.
    """
    critics_dir = Path(critics_dir) if critics_dir else DEFAULT_CRITICS_DIR
    if not critics_dir.is_dir():
        logger.warning("critic pool dir does not exist: %s", critics_dir)
        return []

    critics: list[CriticPersona] = []
    for path in sorted(critics_dir.glob("*.yaml")):
        if path.stem.startswith("_"):
            continue
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            logger.warning("critic YAML unreadable: %s (%s)", path, e)
            continue
        if not isinstance(raw, dict) or "id" not in raw:
            continue
        try:
            critics.append(CriticPersona.model_validate(raw))
        except Exception as e:
            logger.warning("critic %s failed validation: %s", path.name, e)
            continue

    critics.sort(key=lambda c: c.id)
    if not critics:
        logger.warning("critic pool is empty after load: %s", critics_dir)
    return critics
