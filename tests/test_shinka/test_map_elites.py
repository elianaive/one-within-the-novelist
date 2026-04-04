"""Tests for MAP-Elites archive logic."""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from shinka.database.dbase import ProgramDatabase, DatabaseConfig


def _make_program(id, cell, holder_score, correct=True):
    """Create a mock Program with MAP-Elites cell data."""
    prog = MagicMock()
    prog.id = id
    prog.correct = correct
    prog.combined_score = holder_score
    prog.public_metrics = {
        "holder_score": holder_score,
        "map_elites_cell": {
            "concept_type": cell[0],
            "arc_shape": cell[1],
        },
    }
    return prog


def _make_program_no_cell(id, holder_score):
    prog = MagicMock()
    prog.id = id
    prog.correct = True
    prog.combined_score = holder_score
    prog.public_metrics = {"holder_score": holder_score}
    return prog


@pytest.fixture
def map_elites_db(tmp_path):
    """Set up a ProgramDatabase with MAP-Elites strategy."""
    config = DatabaseConfig(
        db_path=str(tmp_path / "test.db"),
        archive_size=100,
        archive_selection_strategy="map_elites",
        num_islands=1,
    )
    db = ProgramDatabase(config=config)
    # Disable FK constraints so we can insert into archive/cells without programs rows
    db.cursor.execute("PRAGMA foreign_keys = OFF")
    return db


class TestGetCellKey:
    def test_valid_cell(self):
        prog = _make_program("p1", ("thought_experiment", "rise"), 3.0)
        key = ProgramDatabase._get_cell_key(prog)
        assert key == ("thought_experiment", "rise")

    def test_missing_dimension_returns_none(self):
        prog = MagicMock()
        prog.public_metrics = {
            "map_elites_cell": {"concept_type": "thought_experiment"},
        }
        assert ProgramDatabase._get_cell_key(prog) is None

    def test_no_cell_data_returns_none(self):
        prog = MagicMock()
        prog.public_metrics = {"holder_score": 3.0}
        assert ProgramDatabase._get_cell_key(prog) is None

    def test_no_public_metrics_returns_none(self):
        prog = MagicMock()
        prog.public_metrics = None
        assert ProgramDatabase._get_cell_key(prog) is None


class TestMapElitesArchive:
    def test_insert_into_empty_cell(self, map_elites_db):
        prog = _make_program("p1", ("thought_experiment", "rise"), 3.0)
        # Mock _get_program_by_id since we don't have a full programs table
        map_elites_db._get_program_by_id = MagicMock(return_value=None)

        map_elites_db._update_archive_map_elites(prog)

        cell_key_str = json.dumps(("thought_experiment", "rise"))
        map_elites_db.cursor.execute(
            "SELECT program_id FROM map_elites_cells WHERE cell_key = ?",
            (cell_key_str,),
        )
        row = map_elites_db.cursor.fetchone()
        assert row is not None
        assert row[0] == "p1"

    def test_better_program_replaces_occupant(self, map_elites_db):
        cell = ("voice_constraint", "fall")
        occupant = _make_program("p1", cell, 2.0)
        challenger = _make_program("p2", cell, 4.0)

        # Insert occupant first
        map_elites_db._update_archive_map_elites(occupant)

        # Mock _get_program_by_id to return the occupant
        map_elites_db._get_program_by_id = MagicMock(return_value=occupant)

        map_elites_db._update_archive_map_elites(challenger)

        cell_key_str = json.dumps(cell)
        map_elites_db.cursor.execute(
            "SELECT program_id FROM map_elites_cells WHERE cell_key = ?",
            (cell_key_str,),
        )
        row = map_elites_db.cursor.fetchone()
        assert row[0] == "p2"

    def test_worse_program_does_not_replace(self, map_elites_db):
        cell = ("situation_with_reveal", "rise_fall")
        occupant = _make_program("p1", cell, 4.0)
        challenger = _make_program("p2", cell, 2.0)

        map_elites_db._update_archive_map_elites(occupant)
        map_elites_db._get_program_by_id = MagicMock(return_value=occupant)

        map_elites_db._update_archive_map_elites(challenger)

        cell_key_str = json.dumps(cell)
        map_elites_db.cursor.execute(
            "SELECT program_id FROM map_elites_cells WHERE cell_key = ?",
            (cell_key_str,),
        )
        row = map_elites_db.cursor.fetchone()
        assert row[0] == "p1"

    def test_different_cells_coexist(self, map_elites_db):
        p1 = _make_program("p1", ("thought_experiment", "rise"), 3.0)
        p2 = _make_program("p2", ("collision", "fall"), 2.5)

        map_elites_db._update_archive_map_elites(p1)
        map_elites_db._update_archive_map_elites(p2)

        map_elites_db.cursor.execute("SELECT COUNT(*) FROM map_elites_cells")
        assert map_elites_db.cursor.fetchone()[0] == 2

    def test_no_cell_skipped(self, map_elites_db):
        prog = _make_program_no_cell("p1", 3.0)
        map_elites_db._update_archive_map_elites(prog)

        map_elites_db.cursor.execute("SELECT COUNT(*) FROM map_elites_cells")
        assert map_elites_db.cursor.fetchone()[0] == 0

    def test_incorrect_program_not_archived(self, map_elites_db):
        prog = _make_program("p1", ("thought_experiment", "rise"), 3.0)
        prog.correct = False

        map_elites_db._update_archive(prog)

        map_elites_db.cursor.execute("SELECT COUNT(*) FROM map_elites_cells")
        assert map_elites_db.cursor.fetchone()[0] == 0


class TestUpdateArchiveStrategyDispatch:
    def test_map_elites_strategy_dispatches(self, map_elites_db):
        """_update_archive routes to _update_archive_map_elites for map_elites strategy."""
        prog = _make_program("p1", ("thought_experiment", "rise"), 3.0)
        map_elites_db._update_archive(prog)

        cell_key_str = json.dumps(("thought_experiment", "rise"))
        map_elites_db.cursor.execute(
            "SELECT program_id FROM map_elites_cells WHERE cell_key = ?",
            (cell_key_str,),
        )
        assert map_elites_db.cursor.fetchone() is not None

    def test_map_elites_bypasses_archive_size(self, map_elites_db):
        """MAP-Elites doesn't check archive_size — inserts regardless."""
        map_elites_db.config.archive_size = 0  # would block fitness strategy

        prog = _make_program("p1", ("thought_experiment", "rise"), 3.0)
        map_elites_db._update_archive(prog)

        cell_key_str = json.dumps(("thought_experiment", "rise"))
        map_elites_db.cursor.execute(
            "SELECT program_id FROM map_elites_cells WHERE cell_key = ?",
            (cell_key_str,),
        )
        assert map_elites_db.cursor.fetchone() is not None
