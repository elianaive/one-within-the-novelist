"""QD archive tests. Pure-functional, no LLM.

Phase 8 exit criterion covered:
- Archive flushes to `qd_archive.json` with the schema in
  `qd-archive.md` §Archive Visualization (`TestSerialization`).

Plus per-component tests: cell assignment correctness on canonicals, bin
boundary handling, multi-entry cells, JSON round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from owtn.models.stage_2.dag import DAG
from owtn.stage_2.archive import (
    DEFAULT_DENSITY_CUTS,
    DEFAULT_DISCLOSURE_CUTS,
    DENSITY_LABELS,
    DISCLOSURE_LABELS,
    Stage2Archive,
    compute_cell,
    disclosure_ratio,
    structural_density,
)


# ----- Cell assignment -----


class TestCellAssignment:
    """Cell coords match what the canonical scratch doc declares."""

    def test_lottery_lands_in_balanced_simple(self, canonical_lottery: DAG) -> None:
        # 6 edges, 5 nodes. disclosure 2/6 = 0.333 → bin 2 (Balanced).
        # density 6/5 = 1.2 → bin 1 (Simple — lower bound is inclusive).
        assert compute_cell(canonical_lottery) == (2, 1)

    def test_hemingway_lands_in_heavy_simple(self, canonical_hemingway: DAG) -> None:
        # 5 edges, 4 nodes. disclosure 2/5 = 0.40 → bin 3 (Heavy, [0.40, 0.55)).
        # density 5/4 = 1.25 → bin 1 (Simple).
        assert compute_cell(canonical_hemingway) == (3, 1)

    def test_oconnor_lands_in_light_skeletal(self, canonical_oconnor: DAG) -> None:
        # 12 edges, 11 nodes. disclosure 2/12 = 0.167 → bin 1 (Light).
        # density 12/11 ≈ 1.09 → bin 0 (Skeletal).
        assert compute_cell(canonical_oconnor) == (1, 0)

    def test_disclosure_ratio_calculation(
        self, canonical_lottery: DAG, canonical_hemingway: DAG,
    ) -> None:
        assert disclosure_ratio(canonical_lottery) == pytest.approx(2 / 6)
        assert disclosure_ratio(canonical_hemingway) == pytest.approx(2 / 5)

    def test_structural_density_calculation(self, canonical_chiang: DAG) -> None:
        # 8 edges, 9 nodes after the rev 8 fix → density 8/9 ≈ 0.89.
        # Canonical's actual edge count after rev-8 fix is 9 (not 8).
        # density = edges / nodes
        assert structural_density(canonical_chiang) == pytest.approx(
            len(canonical_chiang.edges) / len(canonical_chiang.nodes)
        )


class TestBinBoundaries:
    """The bin function uses `<` so values exactly at a cut land in the
    higher bin. Verify on the documented bin specs."""

    def test_value_at_lower_bound_of_bin_lands_in_that_bin(self) -> None:
        # disclosure_ratio == 0.10 should land in bin 1 (Light), not bin 0.
        # Implementation: cuts are (0.10, 0.25, ...). _bin_index returns the
        # first i where value < cuts[i]. value=0.10 is not < 0.10, so i=1.
        from owtn.stage_2.archive import _bin_index
        assert _bin_index(0.10, DEFAULT_DISCLOSURE_CUTS) == 1
        assert _bin_index(0.25, DEFAULT_DISCLOSURE_CUTS) == 2
        assert _bin_index(0.55, DEFAULT_DISCLOSURE_CUTS) == 4

    def test_value_below_first_cut_lands_in_bin_zero(self) -> None:
        from owtn.stage_2.archive import _bin_index
        assert _bin_index(0.05, DEFAULT_DISCLOSURE_CUTS) == 0
        assert _bin_index(0.0, DEFAULT_DISCLOSURE_CUTS) == 0

    def test_value_above_top_cut_lands_in_top_bin(self) -> None:
        from owtn.stage_2.archive import _bin_index
        assert _bin_index(0.99, DEFAULT_DISCLOSURE_CUTS) == 4
        assert _bin_index(10.0, DEFAULT_DENSITY_CUTS) == 4


# ----- Archive add/flush -----


class TestArchiveAdd:
    def test_add_records_entry_in_correct_cell(self, canonical_lottery: DAG) -> None:
        archive = Stage2Archive()
        cell = archive.add(
            canonical_lottery,
            concept_id="c_test", preset="cassandra_ish", tournament_rank=1,
        )
        assert cell == (2, 1)
        assert len(archive.cells[(2, 1)]) == 1

    def test_multi_entry_cell_aggregates(
        self, canonical_lottery: DAG, canonical_hemingway: DAG,
    ) -> None:
        """Two DAGs with different cells should occupy different entries.
        Two DAGs with the same cell stack in one cell."""
        archive = Stage2Archive()
        archive.add(
            canonical_lottery,
            concept_id="c_a", preset="cassandra_ish", tournament_rank=1,
        )
        archive.add(
            canonical_hemingway,
            concept_id="c_b", preset="phoebe_ish", tournament_rank=1,
        )
        assert (2, 1) in archive.cells
        assert (3, 1) in archive.cells
        assert len(archive.cells[(2, 1)]) == 1
        assert len(archive.cells[(3, 1)]) == 1

    def test_same_cell_two_entries_stacked(self, canonical_oconnor: DAG) -> None:
        archive = Stage2Archive()
        archive.add(
            canonical_oconnor,
            concept_id="c_a", preset="cassandra_ish", tournament_rank=1,
        )
        archive.add(
            canonical_oconnor,
            concept_id="c_b", preset="phoebe_ish", tournament_rank=2,
        )
        # Same DAG → same cell. Both entries kept (write-only, no
        # competitive insertion in v1).
        assert len(archive.cells[compute_cell(canonical_oconnor)]) == 2


# ----- Serialization -----


class TestSerialization:
    """Phase 8 exit: archive flushes to JSON with the design-doc schema."""

    def test_to_dict_has_required_top_level_keys(
        self, canonical_lottery: DAG,
    ) -> None:
        archive = Stage2Archive()
        archive.add(
            canonical_lottery,
            concept_id="c_test", preset="cassandra_ish", tournament_rank=1,
        )
        d = archive.to_dict(run_id="run_test")
        assert d["run_id"] == "run_test"
        assert d["grid_size"] == [5, 5]
        assert "axes" in d
        assert "cells" in d

    def test_axis_metadata_includes_bins_and_labels(self) -> None:
        archive = Stage2Archive()
        d = archive.to_dict(run_id="run_test")
        disc = d["axes"]["disclosure_ratio"]
        dens = d["axes"]["structural_density"]
        assert len(disc["bins"]) == 5
        assert len(dens["bins"]) == 5
        assert disc["labels"] == list(DISCLOSURE_LABELS)
        assert dens["labels"] == list(DENSITY_LABELS)

    def test_cell_entry_carries_genome_and_metadata(
        self, canonical_lottery: DAG,
    ) -> None:
        archive = Stage2Archive()
        archive.add(
            canonical_lottery,
            concept_id="c_test", preset="cassandra_ish",
            tournament_rank=2, mcts_reward=0.83,
        )
        d = archive.to_dict(run_id="run_test")
        cell = d["cells"][0]
        entry = cell["entries"][0]
        assert entry["concept_id"] == "c_test"
        assert entry["preset"] == "cassandra_ish"
        assert entry["tournament_rank"] == 2
        assert entry["mcts_reward"] == 0.83
        # genome is the full DAG dump — Stage 3 needs to reconstruct from it.
        assert entry["genome"]["concept_id"] == canonical_lottery.concept_id
        assert len(entry["genome"]["nodes"]) == len(canonical_lottery.nodes)

    def test_flush_writes_qd_archive_json(
        self, canonical_lottery: DAG, tmp_path: Path,
    ) -> None:
        archive = Stage2Archive()
        archive.add(
            canonical_lottery,
            concept_id="c_test", preset="cassandra_ish", tournament_rank=1,
        )
        out_path = archive.flush(tmp_path, run_id="run_test")
        assert out_path == tmp_path / "qd_archive.json"
        assert out_path.exists()
        # Round-trip through JSON parser.
        loaded = json.loads(out_path.read_text())
        assert loaded["run_id"] == "run_test"
        assert len(loaded["cells"]) == 1
