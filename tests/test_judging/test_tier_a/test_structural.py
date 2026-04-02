"""Tests for owtn.judging.tier_a.structural."""

import pytest

from owtn.judging.tier_a.preprocessing import get_nlp
from owtn.judging.tier_a.structural import (
    score_burstiness,
    score_em_dash,
    score_three_short_declaratives,
    score_transition_chains,
)


@pytest.fixture(scope="module")
def nlp():
    return get_nlp()


class TestBurstiness:
    def test_uniform_sentences_high_score(self, nlp):
        text = ". ".join(["The dog walked down the road slowly"] * 20) + "."
        doc = nlp(text)
        assert score_burstiness(doc) > 0.5

    def test_varied_sentences_low_score(self, nlp):
        text = (
            "Stop. "
            "The old man crossed the street slowly, his cane tapping against "
            "the cracked concrete with each careful step. "
            "Rain. "
            "She had not spoken to her mother in fourteen years, and the silence "
            "between them had calcified into something neither could name. "
            "He ran. "
            "Why? "
            "Because the alternative was to stand in that kitchen and listen to "
            "the clock on the wall mark the seconds until everything fell apart."
        )
        doc = nlp(text)
        assert score_burstiness(doc) < 0.5

    def test_too_few_sentences(self, nlp):
        doc = nlp("One. Two. Three.")
        assert score_burstiness(doc) == 0.0


class TestTransitionChains:
    def test_no_transitions(self):
        paras = ["The cat sat.", "A dog barked.", "Birds flew."]
        assert score_transition_chains(paras) == 0.0

    def test_transition_chain_flagged(self):
        paras = [
            "However, the rain continued.",
            "Furthermore, the wind picked up.",
            "Additionally, the temperature dropped.",
            "Moreover, the roads were flooding.",
        ]
        assert score_transition_chains(paras) > 0.5


class TestThreeShortDeclaratives:
    def test_short_burst_flagged(self, nlp):
        text = "He stood. He turned. He left. He ran. He stopped."
        doc = nlp(text)
        assert score_three_short_declaratives(doc) > 0

    def test_normal_prose(self, nlp):
        text = (
            "The morning was cold and grey, with frost on the windows. "
            "She opened the door carefully, checking the street both ways. "
            "Nothing moved except the old flag on the post office."
        )
        doc = nlp(text)
        assert score_three_short_declaratives(doc) == 0.0


class TestEmDash:
    def test_no_dashes(self):
        assert score_em_dash("Normal text here.", 100) == 0.0

    def test_excessive_dashes(self):
        text = "word \u2014 " * 20
        assert score_em_dash(text, 50) > 0.5
