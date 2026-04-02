"""Tests for owtn.judging.tier_a.construction."""

import pytest

from owtn.judging.tier_a.construction import score_construction_patterns
from owtn.judging.tier_a.preprocessing import get_nlp


@pytest.fixture(scope="module")
def nlp():
    return get_nlp()


class TestConstructionPatterns:
    def test_not_x_but_y(self, nlp):
        text = (
            "It was not just a house, but a monument to ambition. "
            "She was not merely tired, but fundamentally exhausted."
        ) * 5
        doc = nlp(text)
        score = score_construction_patterns(text, doc, len(text.split()))
        assert score > 0

    def test_clean_prose(self, nlp):
        text = "The road turned and dust rose. He kept walking."
        doc = nlp(text)
        score = score_construction_patterns(text, doc, len(text.split()))
        assert score < 0.3

    def test_sensation_through(self, nlp):
        text = (
            "His words sent a chill through her spine. "
            "The touch sent electricity coursing through his veins. "
            "A jolt of fear shooting through her body."
        ) * 5
        doc = nlp(text)
        score = score_construction_patterns(text, doc, len(text.split()))
        assert score > 0

    def test_voice_quality(self, nlp):
        text = (
            "Her voice was a low rumble. His voice dropped to a husky whisper. "
            "Her voice was trembling and soft."
        ) * 5
        doc = nlp(text)
        score = score_construction_patterns(text, doc, len(text.split()))
        assert score > 0

    def test_autonomic_emotion(self, nlp):
        """Sage v1.6.1: no pulse/breath/heartbeat as emotion indicators."""
        text = (
            "Her pulse quickened. His breath hitched. "
            "Her heart raced as she watched him leave. "
            "His heart pounded against his ribs."
        ) * 5
        doc = nlp(text)
        score = score_construction_patterns(text, doc, len(text.split()))
        assert score > 0

    def test_electric_sensation(self, nlp):
        """Sage v1.6.1: no jolt/shock/electricity + body parts."""
        text = (
            "A jolt of electricity ran through her. "
            "A bolt of awareness hit him. "
            "A spark of energy coursed through the room."
        ) * 5
        doc = nlp(text)
        score = score_construction_patterns(text, doc, len(text.split()))
        assert score > 0

    def test_dead_metaphors(self, nlp):
        """Sage v1.6.1: no dead metaphors or stock similes."""
        text = (
            "Her sparkling eyes met his across the room. "
            "He had a razor-sharp wit. "
            "Her laugh was like wind chimes. "
            "The silence was thick enough to cut with a knife."
        ) * 5
        doc = nlp(text)
        score = score_construction_patterns(text, doc, len(text.split()))
        assert score > 0
