"""Tests for Edit 4: Sampler dispatch in sampler.py."""

from unittest.mock import MagicMock

import pytest

from shinka.core.sampler import PromptSampler
from shinka.database import Program
from owtn.prompts.stage_1.registry import OPERATOR_DEFS


_next_id = 0

def _make_parent():
    """Create a mock parent Program."""
    global _next_id
    _next_id += 1
    parent = MagicMock(spec=Program)
    parent.id = f"mock-{_next_id}"
    parent.code = '{"premise": "What if...", "target_effect": "wonder"}'
    parent.combined_score = 2.5
    parent.public_metrics = {"holder_score": 2.5}
    parent.text_feedback = "Good originality, weak coherence."
    return parent


@pytest.mark.parametrize("operator", list(OPERATOR_DEFS.keys()))
def test_operator_dispatch_returns_triple(operator):
    """Every concept operator produces a (sys_msg, user_msg, patch_type) triple."""
    sampler = PromptSampler(
        language="json",
        patch_types=[operator],
        patch_type_probs=[1.0],
        use_text_feedback=True,
    )

    parent = _make_parent()
    # Provide inspirations so cross-type operators work
    inspirations = [_make_parent()] * 2

    sys_msg, user_msg, returned_type = sampler.sample(
        parent, inspirations, inspirations,
    )

    assert returned_type == operator
    assert isinstance(sys_msg, str)
    assert isinstance(user_msg, str)
    assert len(sys_msg) > 50
    assert len(user_msg) > 50


def test_cross_operators_filtered_when_no_inspirations():
    """Cross-type operators excluded when no inspirations available."""
    cross_ops = [name for name, d in OPERATOR_DEFS.items() if d["cross"]]
    non_cross = [name for name, d in OPERATOR_DEFS.items() if not d["cross"]]

    sampler = PromptSampler(
        language="json",
        patch_types=cross_ops + non_cross[:1],
        patch_type_probs=[1.0 / (len(cross_ops) + 1)] * (len(cross_ops) + 1),
    )

    parent = _make_parent()

    # With empty inspirations, should never return a cross-type operator
    for _ in range(20):
        _, _, patch_type = sampler.sample(parent, [], [])
        assert patch_type not in cross_ops


def test_legacy_diff_still_works():
    """Legacy 'diff' patch type still dispatches correctly."""
    sampler = PromptSampler(
        language="python",
        patch_types=["diff"],
        patch_type_probs=[1.0],
    )
    parent = _make_parent()
    parent.code = "def foo(): pass"

    sys_msg, user_msg, patch_type = sampler.sample(parent, [], [])
    assert patch_type == "diff"
    assert "SEARCH/REPLACE" in sys_msg or "diff" in sys_msg.lower()


def test_legacy_full_still_works():
    """Legacy 'full' patch type still dispatches correctly."""
    sampler = PromptSampler(
        language="python",
        patch_types=["full"],
        patch_type_probs=[1.0],
    )
    parent = _make_parent()
    parent.code = "def foo(): pass"

    sys_msg, user_msg, patch_type = sampler.sample(parent, [], [])
    assert patch_type == "full"


def test_operator_prompt_contains_parent_genome():
    """Operator user message includes the parent genome."""
    sampler = PromptSampler(
        language="json",
        patch_types=["noun_list"],
        patch_type_probs=[1.0],
    )
    parent = _make_parent()

    _, user_msg, _ = sampler.sample(parent, [], [])
    assert "What if..." in user_msg


def test_seed_bank_threaded_to_operator():
    """Seed bank injected into operator prompt when available."""
    mock_bank = MagicMock()
    mock_seed = MagicMock()
    mock_seed.content = "A physicist discovers her equations predict her own death."
    mock_bank.select.return_value = mock_seed

    sampler = PromptSampler(
        language="json",
        patch_types=["thought_experiment"],
        patch_type_probs=[1.0],
        seed_bank=mock_bank,
    )
    parent = _make_parent()

    _, user_msg, _ = sampler.sample(parent, [], [])
    assert "physicist" in user_msg
