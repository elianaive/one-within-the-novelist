"""Stage 4 test fixtures.

Reuses the Stage 2 canonical DAGs as scaffolding inputs — Stage 4's
manuscript scaffolder reads any valid DAG, and the four canonicals cover
the structural shapes Stage 4 needs to handle (multi-disclosure,
constraint-density, motivates-locality, single-anchor).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from owtn.models.stage_2.dag import DAG


CANONICAL_FIXTURES_DIR = Path(__file__).parent.parent / "test_stage_2" / "fixtures"


def _load(name: str) -> DAG:
    return DAG.model_validate_json((CANONICAL_FIXTURES_DIR / name).read_text())


@pytest.fixture(scope="session")
def canonical_lottery() -> DAG:
    return _load("canonical_lottery.json")


@pytest.fixture(scope="session")
def canonical_hemingway() -> DAG:
    return _load("canonical_hemingway.json")
