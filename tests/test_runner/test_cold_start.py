"""Tests for genesis-operator filtering and runner config translation."""

from collections import Counter

import numpy as np
import pytest

from owtn.models.stage_1.config import StageConfig
from owtn.prompts.stage_1.registry import (
    OPERATOR_DEFS,
    filter_genesis_eligible,
    is_genesis_eligible,
)
from owtn.stage_1.runner import _build_shinka_configs


NON_GENESIS = {"inversion", "compost", "crossover"}


def test_non_genesis_operators_flagged():
    """Inversion, compost, crossover cannot run without a parent — marked genesis=False."""
    for name in NON_GENESIS:
        assert not is_genesis_eligible(name), f"{name} should not be genesis-eligible"


def test_all_other_operators_are_genesis_eligible():
    for name, defn in OPERATOR_DEFS.items():
        if name in NON_GENESIS:
            continue
        assert defn["genesis"], f"{name} should be genesis-eligible"


def test_filter_removes_non_genesis():
    types = list(OPERATOR_DEFS.keys())
    probs = [1.0 / len(types)] * len(types)
    filtered_types, filtered_probs = filter_genesis_eligible(types, probs)
    for bad in NON_GENESIS:
        assert bad not in filtered_types
    assert np.isclose(sum(filtered_probs), 1.0, atol=1e-9)


def test_filter_renormalizes_weights():
    """Dropping operators should proportionally redistribute their weight."""
    types = ["collision", "inversion", "noun_list"]
    probs = [0.5, 0.3, 0.2]
    filtered_types, filtered_probs = filter_genesis_eligible(types, probs)
    assert filtered_types == ["collision", "noun_list"]
    # 0.5 and 0.2 renormalized: 5/7, 2/7
    assert np.isclose(filtered_probs[0], 5 / 7)
    assert np.isclose(filtered_probs[1], 2 / 7)


def test_filter_preserves_input_when_nothing_eligible():
    """Degenerate case: if caller passes only non-genesis types, return them
    unchanged rather than crashing on empty probs."""
    types = ["inversion", "compost"]
    probs = [0.5, 0.5]
    filtered_types, filtered_probs = filter_genesis_eligible(types, probs)
    assert filtered_types == types
    assert filtered_probs == probs


def test_genesis_sampling_never_produces_non_genesis():
    """Drawn many times from a full-operator pool, genesis filter must exclude
    inversion/compost/crossover."""
    np.random.seed(42)
    types = list(OPERATOR_DEFS.keys())
    probs = [1.0 / len(types)] * len(types)
    filtered_types, filtered_probs = filter_genesis_eligible(types, probs)
    draws = [np.random.choice(filtered_types, p=filtered_probs) for _ in range(10000)]
    counts = Counter(draws)
    for bad in NON_GENESIS:
        assert counts[bad] == 0


def test_build_shinka_configs():
    cfg = StageConfig.from_yaml("configs/stage_1/medium.yaml")
    evo, db, job = _build_shinka_configs(cfg, "configs/stage_1/medium.yaml")

    assert evo.language == "json"
    assert len(evo.patch_types) > 0
    assert db.num_islands > 0
    assert job.extra_cmd_args.get("config_path") is not None
