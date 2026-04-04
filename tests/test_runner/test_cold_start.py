"""Tests for ConceptEvolutionRunner cold-start allocation."""

from collections import Counter

import numpy as np
import pytest

from owtn.runner import (
    COLD_START_OPERATORS,
    _CS_NAMES,
    _CS_PROBS,
    _build_shinka_configs,
)
from owtn.models.stage_1.config import StageConfig


def test_cold_start_probs_sum_to_one():
    assert np.isclose(_CS_PROBS.sum(), 1.0, atol=1e-9)


def test_cold_start_excludes_cross_types():
    """compost and crossover require existing population — excluded at cold start."""
    assert "compost" not in COLD_START_OPERATORS
    assert "crossover" not in COLD_START_OPERATORS


def test_cold_start_distribution():
    """Sampling should roughly match expected distribution over many draws."""
    np.random.seed(42)
    draws = [np.random.choice(_CS_NAMES, p=_CS_PROBS) for _ in range(10000)]
    counts = Counter(draws)

    # collision and thought_experiment should be most frequent (~20% each)
    assert counts["collision"] > 1500
    assert counts["thought_experiment"] > 1500
    # compression and real_world_seed should be least frequent (~5% each)
    assert counts["compression"] < 1000
    assert counts["real_world_seed"] < 1000


def test_build_shinka_configs():
    cfg = StageConfig.from_yaml("configs/stage_1/medium.yaml")
    evo, db, job = _build_shinka_configs(cfg, "configs/stage_1/medium.yaml")

    assert evo.language == "json"
    assert len(evo.patch_types) > 0
    assert db.num_islands > 0
    assert job.eval_program_path.endswith("__main__.py")
