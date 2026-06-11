"""Regression: scheduler.run must forward island_idx and parent_id.

`_evaluate_with_pairwise` strict-asserts island_idx for the per-island champion
lock (per lab/issues/2026-04-19-parallel-eval-champion-race.md). The async path
(`submit_async`) always passed island_idx; the sync `scheduler.run` path didn't.
Gen-0 island seeding in `owtn/stage_1/runner.py` calls scheduler.run, so it
silently failed for islands 1..N-1 with `correct=False`.

These tests pin that scheduler.run forwards both kwargs through to the
configured eval_function.
"""
from __future__ import annotations

import json
from pathlib import Path

from shinka.launch.scheduler import LocalJobConfig, JobScheduler


def _make_scheduler(captured: dict, results_dir: Path) -> JobScheduler:
    """Scheduler with an inline eval_function that records its kwargs and
    writes a results.json so load_results returns a dict."""
    def fake_eval(*, program_path, results_dir, parent_id=None, island_idx=None, **extra):
        captured["program_path"] = program_path
        captured["results_dir"] = results_dir
        captured["parent_id"] = parent_id
        captured["island_idx"] = island_idx
        captured["extra"] = extra
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        (Path(results_dir) / "metrics.json").write_text(json.dumps({"combined_score": 0.5}))
        (Path(results_dir) / "correct.json").write_text(json.dumps({"correct": True}))

    cfg = LocalJobConfig(eval_function=fake_eval)
    return JobScheduler(job_type="local", config=cfg, verbose=False)


def test_run_forwards_island_idx_and_parent_id(tmp_path: Path):
    """Both kwargs reach the eval function unchanged."""
    captured: dict = {}
    scheduler = _make_scheduler(captured, tmp_path)

    program_path = str(tmp_path / "main.json")
    Path(program_path).write_text("{}")
    results_dir = str(tmp_path / "results")

    results, rtime = scheduler.run(
        program_path, results_dir, parent_id="parent_pid", island_idx=3,
    )

    assert captured["island_idx"] == 3
    assert captured["parent_id"] == "parent_pid"
    assert captured["program_path"] == program_path
    assert captured["results_dir"] == results_dir
    assert results["correct"]["correct"] is True


def test_run_defaults_to_none_when_kwargs_omitted(tmp_path: Path):
    """Subprocess-based evals that don't need island_idx still work."""
    captured: dict = {}
    scheduler = _make_scheduler(captured, tmp_path)

    program_path = str(tmp_path / "main.json")
    Path(program_path).write_text("{}")
    results_dir = str(tmp_path / "results")

    scheduler.run(program_path, results_dir)

    assert captured["island_idx"] is None
    assert captured["parent_id"] is None
