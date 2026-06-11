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
from owtn.stage_2.runner import load_stage_1_winners, run_stage_2
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
        # `captured_pacing_hints` records the hint each call received so the
        # assertions below can verify run_bidirectional → expand_factory →
        # propose_actions_via_llm threads the preset's expansion_hint
        # through (regression: earlier code defaulted them all to "").
        call_count = [0]
        captured_pacing_hints: list[str] = []

        async def fake_propose(*args, **kwargs):
            call_count[0] += 1
            captured_pacing_hints.append(kwargs.get("pacing_hint", ""))
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
        # Pacing hints reached the expansion call. At least one forward or
        # backward expansion fired with a non-empty hint matching one of
        # the two light-tier presets' written `expansion_hint`s.
        from owtn.models.stage_2.pacing import get_preset
        expected = {
            get_preset("cassandra_ish").expansion_hint,
            get_preset("randy_ish").expansion_hint,
        }
        forward_backward_hints = [h for h in captured_pacing_hints if h]
        assert forward_backward_hints, (
            "no expansion call received a non-empty pacing hint; "
            "tree_runtime → run_bidirectional plumbing regressed"
        )
        assert set(forward_backward_hints).issubset(expected), (
            "captured pacing hints don't match the light-tier preset hints"
        )


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


# ----- Handoff ordering -----


def _write_island_champion(champions_dir: Path, island_idx: int, pid: str) -> None:
    champions_dir.mkdir(parents=True, exist_ok=True)
    (champions_dir / f"island_{island_idx}.json").write_text(json.dumps({
        "id": pid,
        "code": json.dumps(HILLS_GENOME),
        "metadata": {
            "patch_type": "collision",
            "affective_register": "JOY",
            "literary_mode": "REALIST",
        },
        "combined_score": 0.85,
    }))


class TestLoadStage1WinnersOrdering:
    """Stage 1→2 handoff orders by Swiss tournament rank (best first), so
    `--max-concepts N` slices the top N. Champions without a rank (no
    tournament happened) sort last."""

    def test_loads_in_tournament_rank_order_not_island_order(self, tmp_path: Path):
        stage_1_dir = tmp_path / "stage_1"
        champions = stage_1_dir / "champions"
        # Island indices [0,1,2,3] but tournament ranks intentionally shuffled:
        # island_0 → rank 3, island_1 → rank 1 (winner), island_2 → rank 4, island_3 → rank 2.
        _write_island_champion(champions, 0, "pid_island0")
        _write_island_champion(champions, 1, "pid_island1")
        _write_island_champion(champions, 2, "pid_island2")
        _write_island_champion(champions, 3, "pid_island3")
        (stage_1_dir / "tournament.json").write_text(json.dumps([
            {"rank": 1, "program_id": "pid_island1", "wins": 3, "losses": 0, "buchholz": 6, "matches": []},
            {"rank": 2, "program_id": "pid_island3", "wins": 2, "losses": 1, "buchholz": 5, "matches": []},
            {"rank": 3, "program_id": "pid_island0", "wins": 1, "losses": 2, "buchholz": 4, "matches": []},
            {"rank": 4, "program_id": "pid_island2", "wins": 0, "losses": 3, "buchholz": 3, "matches": []},
        ]))

        winners = load_stage_1_winners(stage_1_dir)

        assert [w.program_id for w in winners] == [
            "pid_island1", "pid_island3", "pid_island0", "pid_island2",
        ]
        assert [w.tournament_rank for w in winners] == [1, 2, 3, 4]

    def test_falls_back_to_island_order_when_tournament_absent(self, tmp_path: Path):
        """No tournament.json (e.g. fewer than 2 island survivors) → ranks
        are all None and stable filename order (island index) is preserved."""
        stage_1_dir = tmp_path / "stage_1"
        champions = stage_1_dir / "champions"
        _write_island_champion(champions, 0, "pid_island0")
        _write_island_champion(champions, 1, "pid_island1")

        winners = load_stage_1_winners(stage_1_dir)

        assert [w.program_id for w in winners] == ["pid_island0", "pid_island1"]
        assert [w.tournament_rank for w in winners] == [None, None]

    def test_unranked_champions_sort_last(self, tmp_path: Path):
        """Pathological: tournament.json exists but is missing one of the
        island champions. The unranked one goes last; ranked ones come first
        in rank order."""
        stage_1_dir = tmp_path / "stage_1"
        champions = stage_1_dir / "champions"
        _write_island_champion(champions, 0, "pid_island0")  # no rank
        _write_island_champion(champions, 1, "pid_island1")  # rank 2
        _write_island_champion(champions, 2, "pid_island2")  # rank 1
        (stage_1_dir / "tournament.json").write_text(json.dumps([
            {"rank": 1, "program_id": "pid_island2", "wins": 2, "losses": 0, "buchholz": 4, "matches": []},
            {"rank": 2, "program_id": "pid_island1", "wins": 1, "losses": 1, "buchholz": 3, "matches": []},
        ]))

        winners = load_stage_1_winners(stage_1_dir)

        assert [w.program_id for w in winners] == ["pid_island2", "pid_island1", "pid_island0"]
        assert [w.tournament_rank for w in winners] == [1, 2, None]


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
