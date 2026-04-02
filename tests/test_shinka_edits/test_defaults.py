"""Tests for Edit 3: Concept operators in defaults.py."""

import numpy as np

from shinka.defaults import default_patch_types, default_patch_type_probs


def test_eleven_operators():
    types = default_patch_types()
    assert len(types) == 11


def test_probs_sum_to_one():
    probs = default_patch_type_probs()
    assert np.isclose(sum(probs), 1.0, atol=1e-6)


def test_probs_match_types():
    assert len(default_patch_types()) == len(default_patch_type_probs())


def test_expected_operators_present():
    types = set(default_patch_types())
    expected = {
        "collision", "noun_list", "thought_experiment", "compost",
        "crossover", "inversion", "discovery", "compression",
        "constraint_first", "anti_premise", "real_world_seed",
    }
    assert types == expected
