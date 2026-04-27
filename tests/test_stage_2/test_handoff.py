"""Handoff tests: Stage1Winner loading + Stage 2 → Stage 3 manifest construction.

Two surfaces:
- `Stage1Winner.from_champion_file`: loads Stage 1 champions for Stage 2's
  consumption. Integration-tested against a real Stage 1 run (skips if the
  pinned run directory isn't present locally).
- `build_handoff_for_concept` + `write_manifest`: construct the manifest
  Stage 3 reads. Phase 8 exit criterion — `Stage2HandoffManifest` validates
  against the dataclass and round-trips through JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from pathlib import Path

from owtn.models.stage_2.dag import DAG
from owtn.models.stage_2.handoff import (
    Stage1Winner,
    Stage2HandoffManifest,
    Stage2Output,
)
from owtn.stage_2.archive import Stage2Archive
from owtn.stage_2.handoff import (
    build_handoff_for_concept,
    write_manifest,
)
from owtn.stage_2.tournament import TournamentEntry


# Pinned Stage 1 run directory used during Phase 0 handoff audit.
PINNED_RUN = Path("/home/user/one-within-the-novelist/results/run_20260424_140843/stage_1")


@pytest.fixture
def champion_path() -> Path:
    path = PINNED_RUN / "champions" / "island_0.json"
    if not path.exists():
        pytest.skip(f"pinned Stage 1 champion not present at {path}")
    return path


def test_loads_champion_with_genome(champion_path: Path) -> None:
    winner = Stage1Winner.from_champion_file(champion_path)
    assert winner.program_id, "program_id must be present"
    assert winner.combined_score > 0
    # Genome must parse cleanly through ConceptGenome (Stage 1 model).
    assert len(winner.genome.premise) > 20
    assert winner.genome.anchor_scene.role in {"climax", "reveal", "pivot"}


def test_metadata_passthrough(champion_path: Path) -> None:
    winner = Stage1Winner.from_champion_file(champion_path)
    # The pinned run was generated with patch_type="compression"; we don't
    # assert the specific value (drift over time) but the field must exist.
    assert winner.patch_type is not None
    assert winner.affective_register is not None
    assert winner.literary_mode is not None


def test_tournament_enrichment_when_sibling_present(champion_path: Path) -> None:
    winner = Stage1Winner.from_champion_file(champion_path)
    # Sibling tournament.json exists in this run, so rank should be set.
    assert winner.tournament_rank is not None
    assert winner.tournament_dimension_wins is not None
    # First match should have an opponent and dimension_wins dict.
    first_match = winner.tournament_dimension_wins[0]
    assert "opponent" in first_match
    assert "dimension_wins" in first_match


# ----- Manifest construction (Phase 8) -----


def _make_stage_1_winner(program_id: str = "test_pid") -> Stage1Winner:
    """Build a Stage1Winner directly (not via from_champion_file) so manifest
    tests don't depend on a real run directory."""
    from owtn.models.stage_1.concept_genome import ConceptGenome
    from tests.conftest import HILLS_GENOME

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


def _make_entry(preset: str, dag: DAG, *, mcts_reward: float = 0.5,
                wins: int = 0, dim_wins_total: int = 0) -> TournamentEntry:
    return TournamentEntry(
        preset=preset, dag=dag, mcts_reward=mcts_reward,
        wins=wins, dim_wins_total=dim_wins_total,
    )


class TestPerConceptHandoff:
    """Phase 8 exit: handoff manifest validates against Stage2Output schema."""

    def test_top_k_one_advances_only_first(
        self,
        canonical_lottery: DAG,
        canonical_hemingway: DAG,
    ) -> None:
        archive = Stage2Archive()
        ranked = [
            _make_entry("cassandra_ish", canonical_lottery, wins=1, dim_wins_total=8),
            _make_entry("phoebe_ish", canonical_hemingway, wins=0, dim_wins_total=0),
        ]
        outputs, near_tie = build_handoff_for_concept(
            concept_id="c_test",
            stage_1_forwarded=_make_stage_1_winner(),
            ranked_entries=ranked,
            archive=archive,
            top_k=1,
            near_tie_promoted=False,
        )
        assert len(outputs) == 1
        assert outputs[0].preset == "cassandra_ish"
        assert outputs[0].tournament_rank == 1
        assert near_tie is False

    def test_top_k_all_advances_everyone(
        self,
        canonical_lottery: DAG,
        canonical_hemingway: DAG,
        canonical_chiang: DAG,
        canonical_oconnor: DAG,
    ) -> None:
        archive = Stage2Archive()
        ranked = [
            _make_entry("cassandra_ish", canonical_lottery),
            _make_entry("phoebe_ish", canonical_hemingway),
            _make_entry("randy_ish", canonical_chiang),
            _make_entry("winston_ish", canonical_oconnor),
        ]
        outputs, _ = build_handoff_for_concept(
            concept_id="c_test",
            stage_1_forwarded=_make_stage_1_winner(),
            ranked_entries=ranked,
            archive=archive,
            top_k=None,
        )
        assert len(outputs) == 4

    def test_near_tie_promotion(
        self,
        canonical_lottery: DAG,
        canonical_hemingway: DAG,
        canonical_chiang: DAG,
    ) -> None:
        """At the K=1 boundary, if rank-1 and rank-2 differ by ≤1 dim wins
        across all matches, both advance."""
        archive = Stage2Archive()
        ranked = [
            _make_entry("cassandra_ish", canonical_lottery, wins=1, dim_wins_total=8),
            _make_entry("phoebe_ish", canonical_hemingway, wins=1, dim_wins_total=7),  # gap=1
            _make_entry("randy_ish", canonical_chiang, wins=0, dim_wins_total=0),
        ]
        outputs, near_tie = build_handoff_for_concept(
            concept_id="c_test",
            stage_1_forwarded=_make_stage_1_winner(),
            ranked_entries=ranked,
            archive=archive,
            top_k=1,
            near_tie_promoted=True,
            near_tie_gap=1,
        )
        assert len(outputs) == 2
        assert near_tie is True

    def test_archive_receives_all_entries_advancing_or_not(
        self,
        canonical_lottery: DAG,
        canonical_hemingway: DAG,
    ) -> None:
        """Per qd-archive.md §Population Rules: non-advancing DAGs are
        archived too. The archive should see ALL entries, not just advancing."""
        archive = Stage2Archive()
        ranked = [
            _make_entry("cassandra_ish", canonical_lottery),
            _make_entry("phoebe_ish", canonical_hemingway),
        ]
        build_handoff_for_concept(
            concept_id="c_test",
            stage_1_forwarded=_make_stage_1_winner(),
            ranked_entries=ranked,
            archive=archive,
            top_k=1,
        )
        # Both DAGs landed in the archive (different cells: lottery=(2,1), hemingway=(3,1)).
        assert (2, 1) in archive.cells
        assert (3, 1) in archive.cells

    def test_outputs_carry_qd_cell_and_metadata(
        self, canonical_lottery: DAG,
    ) -> None:
        archive = Stage2Archive()
        ranked = [_make_entry("cassandra_ish", canonical_lottery, mcts_reward=0.72)]
        outputs, _ = build_handoff_for_concept(
            concept_id="c_test",
            stage_1_forwarded=_make_stage_1_winner(),
            ranked_entries=ranked,
            archive=archive,
            top_k=None,
        )
        out = outputs[0]
        assert out.qd_cell == (2, 1)  # Lottery's cell
        assert out.mcts_reward == 0.72
        assert out.stage_1_forwarded.program_id == "test_pid"


class TestManifestSerialization:
    def test_manifest_round_trips_through_json(
        self, canonical_lottery: DAG, tmp_path: Path,
    ) -> None:
        archive = Stage2Archive()
        ranked = [_make_entry("cassandra_ish", canonical_lottery, mcts_reward=0.7)]
        outputs, _ = build_handoff_for_concept(
            concept_id="c_test",
            stage_1_forwarded=_make_stage_1_winner(),
            ranked_entries=ranked,
            archive=archive,
            top_k=None,
        )
        out_path = write_manifest(
            run_id="run_test",
            advancing=outputs,
            run_dir=tmp_path,
        )
        assert out_path == tmp_path / "handoff_manifest.json"
        assert out_path.exists()

        # Round-trip through Pydantic.
        loaded = Stage2HandoffManifest.model_validate_json(out_path.read_text())
        assert loaded.run_id == "run_test"
        assert len(loaded.advancing) == 1
        assert loaded.advancing[0].concept_id == "c_test"
        assert loaded.advancing[0].genome.concept_id == canonical_lottery.concept_id

    def test_empty_advancing_list_still_writes_manifest(
        self, tmp_path: Path,
    ) -> None:
        out_path = write_manifest(
            run_id="run_empty",
            advancing=[],
            run_dir=tmp_path,
        )
        loaded = Stage2HandoffManifest.model_validate_json(out_path.read_text())
        assert loaded.advancing == []
