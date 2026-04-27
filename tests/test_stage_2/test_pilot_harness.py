"""Pilot harness orchestration tests. All LLM seams mocked.

These cover the *plumbing* of the pilot — does the 3-arm structure run
end-to-end, do head-to-head matchups produce the expected dict shape,
does `metrics.json` validate against the documented schema. The actual
go/no-go decision happens AFTER the pilot runs against real LLMs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from owtn.evaluation.models import STAGE_2_DIMENSION_NAMES, PairwiseResult
from owtn.models.stage_2.config import Stage2Config
from owtn.models.stage_2.dag import DAG
from owtn.models.stage_2.handoff import Stage1Winner

from lab.scripts.stage_2_pilot import _aggregate, run_pilot
from tests.conftest import HILLS_GENOME


# ----- Fixtures -----


@pytest.fixture
def fast_config() -> Stage2Config:
    """Real config trimmed to minimum cost so the integration test runs fast.
    Phase 9's test_runner.py uses the same approach."""
    cfg = Stage2Config.from_yaml(Path("configs/stage_2/light.yaml"))
    return cfg.model_copy(update={
        "iterations_per_phase": 1,
        "phase_3_iterations": 1,
    })


def _seed_dag(concept_id: str = "test_pid", preset: str = "p") -> DAG:
    from owtn.models.stage_1.concept_genome import ConceptGenome
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    return DAG(
        concept_id=concept_id, preset=preset,
        motif_threads=["m1", "m2"], concept_demands=[],
        nodes=[{
            "id": "anchor",
            "sketch": concept.anchor_scene.sketch,
            "role": [concept.anchor_scene.role],
            "motifs": [],
        }],
        edges=[], character_arcs=[], story_constraints=[],
        target_node_count=5,
    )


def _make_stage_1_winner(program_id: str = "test_pid") -> Stage1Winner:
    from owtn.models.stage_1.concept_genome import ConceptGenome
    return Stage1Winner(
        program_id=program_id,
        genome=ConceptGenome.model_validate(HILLS_GENOME),
        combined_score=0.85,
        affective_register="JOY",
        literary_mode="REALIST",
        patch_type="collision",
        source_run="run_test",
    )


def _canned_pairwise_result(winner_letter: str = "a") -> PairwiseResult:
    return PairwiseResult(
        winner=winner_letter,
        dimension_wins={d: winner_letter for d in STAGE_2_DIMENSION_NAMES},
        a_wins=8 if winner_letter == "a" else 0,
        b_wins=8 if winner_letter == "b" else 0,
        ties=0,
        a_weighted=8.0 if winner_letter == "a" else 0.0,
        b_weighted=8.0 if winner_letter == "b" else 0.0,
        tie_weighted=0.0,
        judgments=[],
    )


# ----- Aggregation -----


class TestAggregation:
    def test_aggregates_arm1_wins(self) -> None:
        per_concept = [
            {
                "head_to_head": [
                    {"label": "arm1_vs_arm2", "winner": "arm1"},
                    {"label": "arm1_vs_arm3", "winner": "arm1"},
                    {"label": "arm2_vs_arm3", "winner": "arm3"},
                ],
            },
            {
                "head_to_head": [
                    {"label": "arm1_vs_arm2", "winner": "arm1"},
                    {"label": "arm1_vs_arm3", "winner": "arm3"},
                    {"label": "arm2_vs_arm3", "winner": "arm2"},
                ],
            },
        ]
        agg = _aggregate(per_concept)
        assert agg["n_concepts"] == 2
        assert agg["arm1_vs_arm2"]["arm1"] == 2
        assert agg["arm1_vs_arm2"]["arm2"] == 0
        assert agg["arm1_vs_arm3"]["arm1"] == 1
        assert agg["arm1_vs_arm3"]["arm3"] == 1
        assert agg["arm2_vs_arm3"]["arm2"] == 1
        assert agg["arm2_vs_arm3"]["arm3"] == 1

    def test_skipped_matches_counted(self) -> None:
        per_concept = [{
            "head_to_head": [
                {"label": "arm1_vs_arm2", "winner": "arm1"},
                {"label": "arm1_vs_arm3", "winner": None,
                 "skipped_reason": "missing arm: arm1=present, arm3=failed"},
                {"label": "arm2_vs_arm3", "winner": None,
                 "skipped_reason": "missing arm: arm2=present, arm3=failed"},
            ],
        }]
        agg = _aggregate(per_concept)
        assert agg["arm1_vs_arm3"]["skipped"] == 1
        assert agg["arm2_vs_arm3"]["skipped"] == 1


# ----- End-to-end pilot run with mocks -----


def _write_fake_stage_1_directory(stage_1_dir: Path, n_concepts: int = 2) -> None:
    """Build a minimal Stage 1 results directory with N champions."""
    champions_dir = stage_1_dir / "champions"
    champions_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_concepts):
        champion = {
            "id": f"pilot_concept_{i}",
            "code": json.dumps(HILLS_GENOME),
            "metadata": {
                "patch_type": "collision",
                "affective_register": "JOY",
                "literary_mode": "REALIST",
            },
            "combined_score": 0.85,
        }
        (champions_dir / f"island_{i}.json").write_text(json.dumps(champion))


class TestPilotEndToEnd:
    def test_pilot_runs_and_writes_metrics(
        self, fast_config: Stage2Config, tmp_path: Path,
    ) -> None:
        stage_1_dir = tmp_path / "run_pilot" / "stage_1"
        _write_fake_stage_1_directory(stage_1_dir, n_concepts=2)
        output_dir = tmp_path / "pilot_output"

        # Mock all LLM-touching seams.
        async def fake_run_concept(*, winner, **kwargs):
            from owtn.models.stage_2.handoff import Stage2Output
            return [Stage2Output(
                concept_id=winner.program_id, preset="cassandra_ish",
                tournament_rank=1, qd_cell=(2, 1),
                genome=_seed_dag(winner.program_id, "cassandra_ish"),
                stage_1_forwarded=winner, mcts_reward=0.7,
            )]

        async def fake_single_shot(concept, *, concept_id, **kwargs):
            return _seed_dag(concept_id, "single_shot")

        async def fake_iterated(concept, *, concept_id, **kwargs):
            return _seed_dag(concept_id, "iterated")

        async def fake_compare(inputs, *, panel, dim_weights=None):
            return _canned_pairwise_result("a")

        # Use a fake panel — load_panel mock returns it without real YAML files.
        from owtn.models.judge import JudgePersona
        fake_panel = [JudgePersona(
            id="testjudge", name="Test", identity="t", values=["v"],
            exemplars=["e"], lean_in_signals=["s"], harshness="standard",
            priority="primary", model=["claude-sonnet-4-6"], temperature=0.0,
        )]

        with (
            patch("lab.scripts.stage_2_pilot.run_concept", side_effect=fake_run_concept),
            patch("lab.scripts.stage_2_pilot.generate_single_shot_dag", side_effect=fake_single_shot),
            patch("lab.scripts.stage_2_pilot.run_iterated_baseline", side_effect=fake_iterated),
            patch("lab.scripts.stage_2_pilot.compare_stage2", side_effect=fake_compare),
            patch("lab.scripts.stage_2_pilot.load_panel", return_value=fake_panel),
        ):
            metrics_path = asyncio.run(run_pilot(
                stage_1_dir=stage_1_dir,
                config=fast_config,
                config_tier="light",
                cheap_judge_id="testjudge",
                judges_dir="dummy",
                output_dir=output_dir,
                max_concepts=2,
            ))

        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert metrics["config_tier"] == "light"
        assert len(metrics["per_concept"]) == 2

        # Each per-concept result has the 3 arms + 3 head-to-head matchups.
        for r in metrics["per_concept"]:
            assert r["arm1"] is not None
            assert r["arm2"] is not None
            assert r["arm3"] is not None
            assert len(r["head_to_head"]) == 3

        # Aggregate reflects winners.
        agg = metrics["aggregate"]
        assert agg["n_concepts"] == 2
        # All matches resolved to "a" (challenger wins). The challenger
        # in each matchup is the lower-named arm (arm1 in arm1_vs_arm2,
        # arm1 in arm1_vs_arm3, arm2 in arm2_vs_arm3).
        assert agg["arm1_vs_arm2"]["arm1"] == 2
        assert agg["arm1_vs_arm3"]["arm1"] == 2
        assert agg["arm2_vs_arm3"]["arm2"] == 2

    def test_pilot_handles_arm_failure_gracefully(
        self, fast_config: Stage2Config, tmp_path: Path,
    ) -> None:
        """If one arm fails on a concept, the matchups involving that arm
        are skipped (winner=None, skipped_reason set). Other arms still run."""
        stage_1_dir = tmp_path / "run_pilot" / "stage_1"
        _write_fake_stage_1_directory(stage_1_dir, n_concepts=1)
        output_dir = tmp_path / "pilot_output"

        async def fake_run_concept(*, winner, **kwargs):
            return []  # arm 1 failed (no advancing outputs)

        async def fake_single_shot(concept, *, concept_id, **kwargs):
            return _seed_dag(concept_id, "single_shot")

        async def fake_iterated(concept, *, concept_id, **kwargs):
            return _seed_dag(concept_id, "iterated")

        async def fake_compare(inputs, *, panel, dim_weights=None):
            return _canned_pairwise_result("a")

        from owtn.models.judge import JudgePersona
        fake_panel = [JudgePersona(
            id="testjudge", name="Test", identity="t", values=["v"],
            exemplars=["e"], lean_in_signals=["s"], harshness="standard",
            priority="primary", model=["claude-sonnet-4-6"], temperature=0.0,
        )]

        with (
            patch("lab.scripts.stage_2_pilot.run_concept", side_effect=fake_run_concept),
            patch("lab.scripts.stage_2_pilot.generate_single_shot_dag", side_effect=fake_single_shot),
            patch("lab.scripts.stage_2_pilot.run_iterated_baseline", side_effect=fake_iterated),
            patch("lab.scripts.stage_2_pilot.compare_stage2", side_effect=fake_compare),
            patch("lab.scripts.stage_2_pilot.load_panel", return_value=fake_panel),
        ):
            metrics_path = asyncio.run(run_pilot(
                stage_1_dir=stage_1_dir,
                config=fast_config,
                config_tier="light",
                cheap_judge_id="testjudge",
                judges_dir="dummy",
                output_dir=output_dir,
                max_concepts=1,
            ))

        metrics = json.loads(metrics_path.read_text())
        per_concept = metrics["per_concept"][0]
        assert per_concept["arm1"] is None
        assert per_concept["arm2"] is not None
        assert per_concept["arm3"] is not None

        # Matchups involving arm1 are skipped; arm2_vs_arm3 still runs.
        h2h_by_label = {h["label"]: h for h in per_concept["head_to_head"]}
        assert h2h_by_label["arm1_vs_arm2"]["winner"] is None
        assert h2h_by_label["arm1_vs_arm3"]["winner"] is None
        assert h2h_by_label["arm2_vs_arm3"]["winner"] in ("arm2", "arm3")
