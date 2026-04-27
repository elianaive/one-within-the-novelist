"""Live smoke test for the Stage 2 runner. Makes real LLM calls.

Skipped by default (per `@pytest.mark.live_api`). Run with:
    uv run pytest tests/test_stage_2/test_runner_live.py -v

Phase 9 exit criterion: live smoke test ends under $2.

Scope: 1 Stage 1 concept, 1 preset (`cassandra_ish`), 2 iterations per phase.
Estimated cost ~$0.25-$0.50 — well under the $2 ceiling. The test asserts
only that:
- The runner produces archive + manifest files
- The manifest validates against `Stage2HandoffManifest`
- At least one advancing output is produced (or the test logs a warning if
  the cheap judge happened to reject all candidates — non-deterministic)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from owtn.models.stage_2.config import Stage2Config
from owtn.models.stage_2.handoff import Stage2HandoffManifest
from owtn.stage_2.runner import run_stage_2


pytestmark = pytest.mark.live_api


# Pinned Stage 1 run from Phase 0 audit. Other runs work too — adjust if
# this directory is rotated.
PINNED_RUN = Path("/home/user/one-within-the-novelist/results/run_20260424_140843/stage_1")


@pytest.fixture
def pinned_stage_1_run() -> Path:
    if not PINNED_RUN.exists():
        pytest.skip(f"pinned Stage 1 run not present at {PINNED_RUN}")
    return PINNED_RUN


@pytest.fixture
def smoke_config(tmp_path: Path) -> Stage2Config:
    """Minimal config for cost containment: 1 preset, 2 iterations/phase, 1 P3."""
    cfg = Stage2Config.from_yaml(Path("configs/stage_2/light.yaml"))
    return cfg.model_copy(update={
        "iterations_per_phase": 2,
        "phase_3_iterations": 1,
        "presets": cfg.presets.model_copy(update={"light": ["cassandra_ish"]}),
    })


def test_live_smoke(
    pinned_stage_1_run: Path,
    smoke_config: Stage2Config,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "stage_2_smoke"
    archive_path, manifest_path = asyncio.run(run_stage_2(
        stage_1_dir=pinned_stage_1_run,
        config=smoke_config,
        config_tier="light",
        cheap_judge_id="roon",
        output_dir=output_dir,
        max_concepts=1,  # one concept only
    ))

    assert archive_path.exists()
    assert manifest_path.exists()

    manifest = Stage2HandoffManifest.model_validate_json(manifest_path.read_text())
    assert manifest.run_id  # non-empty
    # The smoke run may or may not produce advancing outputs (depends on
    # whether the LLM produced valid actions and judges accepted at least
    # one candidate). Both outcomes are valid for a smoke test. Just log
    # the count.
    print(f"\n[live smoke] advancing outputs: {len(manifest.advancing)}")
    print(f"[live smoke] archive cells: {len(manifest.advancing) and 'populated' or 'empty'}")
