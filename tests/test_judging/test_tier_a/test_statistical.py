"""Tests for owtn.judging.tier_a.statistical."""

import pytest

from owtn.judging.tier_a.preprocessing import get_nlp
from owtn.judging.tier_a.statistical import (
    mattr,
    score_mattr,
    score_participial_openings,
)


@pytest.fixture(scope="module")
def nlp():
    return get_nlp()


class TestMATTR:
    def test_repetitive_text_high_score(self, nlp):
        text = " ".join(["the cat sat on the mat"] * 100)
        doc = nlp(text)
        assert score_mattr(doc) > 0.5

    def test_mattr_function_short_text(self):
        tokens = ["the", "cat", "sat", "on", "a", "mat"]
        result = mattr(tokens, window_size=500)
        assert result == len(set(tokens)) / len(tokens)

    def test_mattr_empty(self):
        assert mattr([]) == 0.0


class TestParticipialOpenings:
    def test_many_ing_starts(self, nlp):
        text = (
            "Walking down the street, he noticed the sign. "
            "Turning the corner, she saw the house. "
            "Reaching for the handle, he paused. "
            "Glancing over her shoulder, she quickened her pace. "
            "Breathing heavily, he climbed the stairs. "
            "Opening the door, she stepped inside. "
            "Sitting at the table, he waited. "
            "Standing by the window, she watched. "
            "Running through the rain, he arrived. "
            "Looking up, she smiled."
        )
        doc = nlp(text)
        assert score_participial_openings(doc) > 0.5

    def test_normal_openings(self, nlp):
        text = (
            "He walked down the street. "
            "The sign was old and faded. "
            "She turned the corner. "
            "A house stood at the end of the lane. "
            "He reached for the handle."
        )
        doc = nlp(text)
        assert score_participial_openings(doc) < 0.3
