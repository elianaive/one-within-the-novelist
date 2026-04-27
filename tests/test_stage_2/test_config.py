"""Stage 2 config schema test. Skipped when the YAML doesn't exist yet.

`configs/stage_2/light.yaml` ships in Phase 9 (runner). Phase 1 just defines
the schema and skips the test until the file exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path("/home/user/one-within-the-novelist")
LIGHT_YAML = REPO_ROOT / "configs" / "stage_2" / "light.yaml"


@pytest.mark.skipif(
    not LIGHT_YAML.exists(),
    reason=f"Stage 2 config YAML not yet shipped (Phase 9 work). Expected: {LIGHT_YAML}",
)
def test_light_config_validates() -> None:
    from owtn.models.stage_2.config import Stage2Config

    cfg = Stage2Config.from_yaml(LIGHT_YAML)
    assert cfg.iterations_per_phase >= 1
    assert len(cfg.dimensions) == 8
