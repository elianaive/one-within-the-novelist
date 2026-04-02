"""Tests for owtn.judging.tier_a.anti_patterns."""

import pytest

from owtn.judging.tier_a.anti_patterns import (
    score_balanced_antithesis,
    score_cold_start,
    score_negative_assertion,
    score_section_break_overuse,
    score_simile_crutch,
    score_triadic_listing,
)
from owtn.judging.tier_a.preprocessing import get_nlp


@pytest.fixture(scope="module")
def nlp():
    return get_nlp()


class TestColdStart:
    def test_it_was_pattern(self):
        text = "It was the kind of town that time forgot. The streets were empty."
        assert score_cold_start(text) > 0

    def test_clean_opening(self):
        text = "Mara found the letter behind the radiator."
        assert score_cold_start(text) == 0.0


class TestTriadicListing:
    def test_triads_flagged(self, nlp):
        text = (
            "He was tall, dark, and handsome. "
            "The room was cold, damp, and dark. "
            "She felt tired, angry, and lost. "
            "The food was stale, bland, and cold. "
        ) * 3
        doc = nlp(text)
        assert score_triadic_listing(doc, len(text.split())) > 0


class TestNegativeAssertion:
    def test_did_not_repetition(self, nlp):
        text = (
            "He did not look back. He did not think about the room. "
            "She did not answer. She didn't move. He didn't breathe."
        ) * 3
        doc = nlp(text)
        assert score_negative_assertion(doc, len(text.split())) > 0


class TestSimileCrutch:
    def test_the_way_overuse(self, nlp):
        text = (
            "The way she held the cup. The way he turned away. "
            "The way light fell across the table. The way doors close."
        ) * 3
        doc = nlp(text)
        assert score_simile_crutch(doc, len(text.split())) > 0


class TestSectionBreakOveruse:
    def test_excessive_breaks(self):
        text = "Some text.\n\n---\n\nMore text.\n\n---\n\nEven more.\n\n---\n\nAnd more."
        assert score_section_break_overuse(text, len(text.split())) > 0


class TestBalancedAntithesis:
    def test_multiple_antithesis_in_dialogue(self):
        dialogues = [
            '"Not fear, but anticipation"',
            '"Not anger, but sadness"',
            '"Not hate, but indifference"',
        ]
        assert score_balanced_antithesis(dialogues) > 0

    def test_no_antithesis(self):
        dialogues = ['"Hello"', '"Goodbye"']
        assert score_balanced_antithesis(dialogues) == 0.0
