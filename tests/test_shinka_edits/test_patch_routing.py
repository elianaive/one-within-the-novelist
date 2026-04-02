"""Tests for Edit 5: Patch routing in async_apply.py."""

import pytest

from shinka.edit.async_apply import _FULL_PATCH_TYPES, _DIFF_PATCH_TYPES
from owtn.prompts.stage_1.registry import OPERATOR_DEFS


FULL_OPERATORS = [
    "collision", "noun_list", "thought_experiment", "compost",
    "crossover", "discovery", "compression",
    "constraint_first", "anti_premise", "real_world_seed",
]

DIFF_OPERATORS = ["inversion"]


@pytest.mark.parametrize("op", FULL_OPERATORS)
def test_full_operators_route_correctly(op):
    assert op in _FULL_PATCH_TYPES


@pytest.mark.parametrize("op", DIFF_OPERATORS)
def test_diff_operators_route_correctly(op):
    assert op in _DIFF_PATCH_TYPES


def test_legacy_types_preserved():
    assert "full" in _FULL_PATCH_TYPES
    assert "cross" in _FULL_PATCH_TYPES
    assert "diff" in _DIFF_PATCH_TYPES


def test_no_overlap():
    assert _FULL_PATCH_TYPES.isdisjoint(_DIFF_PATCH_TYPES)


def test_all_operators_routed():
    all_routed = _FULL_PATCH_TYPES | _DIFF_PATCH_TYPES
    for name in OPERATOR_DEFS:
        assert name in all_routed, f"Operator {name} not routed"
