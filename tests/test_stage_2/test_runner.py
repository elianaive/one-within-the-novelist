"""End-to-end runner integration tests. All LLM calls mocked.

Phase 9 exit criterion covered:
- Fully mocked integration test runs the complete Stage 2 stack (seed →
  forward → backward → Phase 3 → tournament → archive → handoff) without
  touching the network.

Mocking strategy: patch the four LLM-touching seams — `seed_root` (in
`owtn.stage_2.orchestration`), `propose_actions_via_llm` (in
`owtn.stage_2.operators`), `evaluate_rollout` (in
`owtn.stage_2.tree_runtime`), and `compare_stage2` (in
`owtn.stage_2.tournament`). Each returns a canned response sized for the
test (one action per expansion, canned cheap-judge outcomes). Asserts
focus on STRUCTURE: the run produces manifest + archive files, with the
expected schemas, without crashing.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from owtn.evaluation.models import (
    STAGE_2_DIMENSION_NAMES,
    PairwiseResult,
    Stage2PairwiseJudgment,
)
from owtn.evaluation.stage_2 import CheapJudgeOutcome, RolloutEvaluation
from owtn.models.stage_2.actions import AddBeatAction
from owtn.models.stage_2.config import Stage2Config
from owtn.models.stage_2.dag import DAG, Node
from owtn.models.stage_2.handoff import Stage1Winner, Stage2HandoffManifest
from owtn.stage_2.orchestration import run_concept
from owtn.stage_2.archive import Stage2Archive
from owtn.stage_2.runner import run_stage_2
from tests.conftest import HILLS_GENOME


# ----- Mock helpers -----


def _seed_dag_for_concept(program_id: str = "test_pid") -> DAG:
    """Canned 1-node seed DAG mimicking what `seed_root` returns."""
    from owtn.models.stage_1.concept_genome import ConceptGenome
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    return DAG(
        concept_id=program_id,
        preset="cassandra_ish",
        motif_threads=["the hills", "the operation never named"],
        concept_demands=[],
        nodes=[Node(
            id="anchor",
            sketch=concept.anchor_scene.sketch,
            role=[concept.anchor_scene.role],
            motifs=[],
        )],
        edges=[],
        character_arcs=[],
        story_constraints=[],
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
        tournament_rank=1,
    )


def _add_beat_action(node_id: str, anchor_id: str = "anchor") -> AddBeatAction:
    return AddBeatAction(
        anchor_id=anchor_id,
        direction="downstream",
        new_node_id=node_id,
        sketch=f"A canned beat sketch for {node_id} with enough words to satisfy validation.",
        edge_type="causal",
        edge_payload={
            "realizes": "this is a substantive realizes payload field with at least four tokens",
        },
    )


def _all_a_judgment() -> Stage2PairwiseJudgment:
    return Stage2PairwiseJudgment(
        reasoning="canned reasoning for the test",
        edge_logic="a_clear",
        motivational_coherence="a_clear",
        tension_information_arch="a_clear",
        post_dictability="a_clear",
        arc_integrity_ending="a_clear",
        structural_coherence="a_clear",
        beat_quality="a_clear",
        concept_fidelity_thematic="a_clear",
    )


# ----- Fixtures -----


@pytest.fixture
def light_config(tmp_path: Path) -> Stage2Config:
    """Real config loaded from configs/stage_2/light.yaml, then trimmed
    to a single preset and 2 iterations per phase to keep the integration
    test fast."""
    cfg = Stage2Config.from_yaml(Path("configs/stage_2/light.yaml"))
    return cfg.model_copy(update={
        "iterations_per_phase": 2,
        "phase_3_iterations": 1,
    })


@pytest.fixture
def fake_panel():
    """Single-judge panel — minimum viable to exercise tournament + commitment paths."""
    from owtn.models.judge import JudgePersona
    return [JudgePersona(
        id="testjudge",
        name="Test Judge",
        identity="A test judge",
        values=["v"],
        exemplars=["e"],
        lean_in_signals=["s"],
        harshness="standard",
        priority="primary",
        model=["claude-sonnet-4-6"],
        temperature=0.0,
    )]


# ----- run_concept end-to-end -----


class TestRunConceptIntegration:
    """Phase 9 exit: complete stack runs end-to-end with mocked LLM."""

    def test_run_concept_produces_advancing_outputs(
        self, light_config: Stage2Config, fake_panel: list,
    ) -> None:
        winner = _make_stage_1_winner()
        archive = Stage2Archive()

        seed = _seed_dag_for_concept(winner.program_id)

        async def fake_seed_root(concept, *, concept_id, preset, target_node_count, **kwargs):
            return seed.model_copy(update={"preset": preset, "concept_id": concept_id})

        # Each expansion call returns a single causal beat downstream of
        # whatever the latest beat is, naming the new beat by call count.
        call_count = [0]

        async def fake_propose(*args, **kwargs):
            call_count[0] += 1
            return [_add_beat_action(f"beat_{call_count[0]}")]

        # Cheap judge always returns "challenger wins" → backprop = 0.7,
        # promotes challenger.
        async def fake_evaluate_rollout(inputs, **kwargs):
            return RolloutEvaluation(
                backprop_reward=0.7,
                cheap_outcome=CheapJudgeOutcome(
                    challenger_wins=True,
                    reward=0.7,
                    resolved_votes={d: "a_clear" for d in STAGE_2_DIMENSION_NAMES},
                    cost=0.001,
                ),
                verified_partial=inputs.challenger,
                full_panel_outcome=None,  # skip full-panel verification in this test
                promoted=True,
                cheap_full_agreement=None,
                total_cost=0.001,
                notes=["cheap_says_win:no_full_panel_configured:promoted"],
            )

        # Tournament uses compare_stage2 directly. Return a canned PairwiseResult
        # with all judges agreeing on "a wins".
        async def fake_compare_stage2(inputs, *, panel, dim_weights=None):
            return PairwiseResult(
                winner="a",
                dimension_wins={d: "a" for d in STAGE_2_DIMENSION_NAMES},
                a_wins=8, b_wins=0, ties=0,
                a_weighted=8.0, b_weighted=0.0, tie_weighted=0.0,
                judgments=[{
                    "judge_id": p.id,
                    "forward_votes": {d: "a_clear" for d in STAGE_2_DIMENSION_NAMES},
                    "reverse_votes": {d: "a_clear" for d in STAGE_2_DIMENSION_NAMES},
                    "resolved": {d: "a_clear" for d in STAGE_2_DIMENSION_NAMES},
                    "forward_reasoning": "test",
                    "reverse_reasoning": "test",
                    "cost": 0.001,
                } for p in panel],
            )

        with (
            patch("owtn.stage_2.orchestration.seed_root", side_effect=fake_seed_root),
            patch("owtn.stage_2.operators.propose_actions_via_llm", side_effect=fake_propose),
            patch("owtn.stage_2.tree_runtime.evaluate_rollout", side_effect=fake_evaluate_rollout),
            patch("owtn.stage_2.tournament.compare_stage2", side_effect=fake_compare_stage2),
        ):
            outputs = asyncio.run(run_concept(
                winner=winner,
                config=light_config,
                config_tier="light",
                cheap_judge=fake_panel[0],
                full_panel=fake_panel,
                classifier_model="dummy",
                archive=archive,
            ))

        # Light tier ships 2 presets; top_k=1 with near_tie_promoted=True.
        # In this test all matches resolve unanimously to A on the LAST entry
        # (since `winner=='a'` means challenger wins; in tournament dispatch
        # entry_a is the lower-indexed preset). With both entries tied at 1
        # win each? Actually compare_stage2 always returns "a" → first iter
        # cassandra (vs randy) = a wins → cassandra wins. Then in handoff:
        # cassandra = rank 1.
        assert len(outputs) >= 1
        assert outputs[0].concept_id == winner.program_id
        # Archive received both presets (advancing + non-advancing).
        assert len(archive.cells) >= 1


# ----- Full run_stage_2 end-to-end -----


def _write_fake_stage_1_directory(stage_1_dir: Path) -> None:
    """Build a minimal Stage 1 results directory: one champion + an empty
    tournament.json (sibling). `Stage1Winner.from_champion_file` reads both."""
    champions_dir = stage_1_dir / "champions"
    champions_dir.mkdir(parents=True, exist_ok=True)
    champion = {
        "id": "test_pid_e2e",
        "code": json.dumps(HILLS_GENOME),
        "metadata": {
            "patch_type": "collision",
            "affective_register": "JOY",
            "literary_mode": "REALIST",
        },
        "combined_score": 0.85,
    }
    (champions_dir / "island_0.json").write_text(json.dumps(champion))
    # tournament.json is optional — Stage1Winner.from_champion_file handles its absence.


class TestRunStage2Integration:
    """Top-level `run_stage_2` smoke test: reads from disk, runs concept(s),
    writes archive + manifest. All LLM calls mocked."""

    def test_full_run_writes_archive_and_manifest(
        self, light_config: Stage2Config, fake_panel: list, tmp_path: Path,
    ) -> None:
        stage_1_dir = tmp_path / "run_test" / "stage_1"
        _write_fake_stage_1_directory(stage_1_dir)
        output_dir = tmp_path / "run_test" / "stage_2"

        seed = _seed_dag_for_concept("test_pid_e2e")

        async def fake_seed_root(concept, *, concept_id, preset, target_node_count, **kwargs):
            return seed.model_copy(update={"preset": preset, "concept_id": concept_id})

        async def fake_propose(*args, **kwargs):
            return [_add_beat_action(f"beat_{id(args)}")]

        async def fake_evaluate_rollout(inputs, **kwargs):
            return RolloutEvaluation(
                backprop_reward=0.7,
                cheap_outcome=CheapJudgeOutcome(
                    challenger_wins=True, reward=0.7,
                    resolved_votes={d: "a_clear" for d in STAGE_2_DIMENSION_NAMES},
                    cost=0.001,
                ),
                verified_partial=inputs.challenger,
                full_panel_outcome=None,
                promoted=True,
                cheap_full_agreement=None,
                total_cost=0.001,
                notes=[],
            )

        async def fake_compare_stage2(inputs, *, panel, dim_weights=None):
            return PairwiseResult(
                winner="a",
                dimension_wins={d: "a" for d in STAGE_2_DIMENSION_NAMES},
                a_wins=8, b_wins=0, ties=0,
                a_weighted=8.0, b_weighted=0.0, tie_weighted=0.0,
                judgments=[{
                    "judge_id": p.id,
                    "forward_votes": {d: "a_clear" for d in STAGE_2_DIMENSION_NAMES},
                    "reverse_votes": {d: "a_clear" for d in STAGE_2_DIMENSION_NAMES},
                    "resolved": {d: "a_clear" for d in STAGE_2_DIMENSION_NAMES},
                    "forward_reasoning": "test",
                    "reverse_reasoning": "test",
                    "cost": 0.001,
                } for p in panel],
            )

        # Patch `load_panel` so we don't need real judge YAMLs in the test env.
        def fake_load_panel(judges_dir, panel_ids):
            return list(fake_panel)

        with (
            patch("owtn.stage_2.runner.load_panel", side_effect=fake_load_panel),
            patch("owtn.stage_2.orchestration.seed_root", side_effect=fake_seed_root),
            patch("owtn.stage_2.operators.propose_actions_via_llm", side_effect=fake_propose),
            patch("owtn.stage_2.tree_runtime.evaluate_rollout", side_effect=fake_evaluate_rollout),
            patch("owtn.stage_2.tournament.compare_stage2", side_effect=fake_compare_stage2),
        ):
            archive_path, manifest_path = asyncio.run(run_stage_2(
                stage_1_dir=stage_1_dir,
                config=light_config,
                config_tier="light",
                cheap_judge_id=fake_panel[0].id,
                output_dir=output_dir,
            ))

        assert archive_path.exists()
        assert manifest_path.exists()

        # Manifest validates against schema.
        loaded = Stage2HandoffManifest.model_validate_json(manifest_path.read_text())
        assert loaded.run_id  # non-empty
        # At least one advancing output (light tier top_k=1 with near-tie expansion possible).
        assert len(loaded.advancing) >= 1

        # Archive validates as JSON with the expected top-level shape.
        archive_data = json.loads(archive_path.read_text())
        assert archive_data["grid_size"] == [5, 5]
        assert "axes" in archive_data
        assert "cells" in archive_data
