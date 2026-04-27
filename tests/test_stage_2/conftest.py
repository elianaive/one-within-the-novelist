"""Stage 2 test fixtures.

Diverges from Stage 1's `tests/conftest.py` pattern (HILLS_GENOME et al. as
hardcoded Python dicts) by loading canonicals from external JSON files. The
rev-7 canonicals are too large to encode as inline dicts (~100+ lines each
with deeply nested motif mentions and edge payloads), and the same JSON files
double as inputs to the standalone CLI (`python -m owtn.models.stage_2.dag`).

Fixtures load each canonical once via `DAG.model_validate_json` so a syntax
error in any JSON fixture fails fast at collection time, not deep inside a
test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from owtn.models.stage_2.dag import DAG


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> DAG:
    return DAG.model_validate_json((FIXTURES_DIR / name).read_text())


@pytest.fixture(scope="session")
def canonical_lottery() -> DAG:
    return _load("canonical_lottery.json")


@pytest.fixture(scope="session")
def canonical_hemingway() -> DAG:
    return _load("canonical_hemingway.json")


@pytest.fixture(scope="session")
def canonical_chiang() -> DAG:
    return _load("canonical_chiang.json")


@pytest.fixture(scope="session")
def canonical_oconnor() -> DAG:
    return _load("canonical_oconnor.json")


@pytest.fixture(scope="session")
def canonical_path_lottery() -> Path:
    """Path to the Lottery canonical JSON, for tests that need a file path
    (e.g. exercising the standalone CLI)."""
    return FIXTURES_DIR / "canonical_lottery.json"


@pytest.fixture(scope="session", params=[
    "canonical_lottery.json",
    "canonical_hemingway.json",
    "canonical_chiang.json",
    "canonical_oconnor.json",
])
def any_canonical(request) -> DAG:
    """Parametrized over all 4 canonicals, for tests that should hold for each."""
    return _load(request.param)
