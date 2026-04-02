"""Tests for owtn.judging.tier_a.vocabulary."""

import pytest

from owtn.judging.tier_a.preprocessing import get_nlp
from owtn.judging.tier_a.vocabulary import score_banned_vocabulary


@pytest.fixture(scope="module")
def nlp():
    return get_nlp()


class TestBannedVocabulary:
    def test_clean_text_low_score(self, nlp):
        text = "The cat sat on the mat. It was a quiet afternoon."
        doc = nlp(text)
        score, flagged = score_banned_vocabulary(doc, len(text.split()))
        assert score < 0.1
        assert len(flagged) == 0

    def test_tier1_words_flagged(self, nlp):
        text = ("She delved into the tapestry of the realm, "
                "embarking on a multifaceted endeavor to encompass the paradigm.")
        doc = nlp(text)
        score, flagged = score_banned_vocabulary(doc, len(text.split()))
        assert score > 0
        assert any(item["tier"] == 1 for item in flagged)

    def test_tier3_phrase_flagged(self, nlp):
        text = "It's worth noting that this speaks volumes about the situation."
        doc = nlp(text)
        score, flagged = score_banned_vocabulary(doc, len(text.split()))
        assert any(item["tier"] == 3 for item in flagged)

    def test_fiction_tourism_flagged(self, nlp):
        text = "The breathtaking vista was truly majestic and fascinating."
        doc = nlp(text)
        score, flagged = score_banned_vocabulary(doc, len(text.split()))
        assert any(item["tier"] == 1 for item in flagged)

    def test_weasel_phrases_flagged(self, nlp):
        text = "Experts agree that studies show the answer is clear."
        doc = nlp(text)
        score, flagged = score_banned_vocabulary(doc, len(text.split()))
        assert any(item["tier"] == 3 for item in flagged)

    def test_expanded_tier3_phrases(self, nlp):
        text = "Furthermore, it is important to note that additionally, in conclusion, we should note this."
        doc = nlp(text)
        score, flagged = score_banned_vocabulary(doc, len(text.split()))
        assert len([f for f in flagged if f["tier"] == 3]) >= 2

    def test_reinhart_tier1_words(self, nlp):
        """Reinhart et al. 50x+ words should be Tier 1."""
        text = "The fleeting solace amidst the palpable unease was unspoken."
        doc = nlp(text)
        score, flagged = score_banned_vocabulary(doc, len(text.split()))
        tier1_lemmas = {f["lemma"] for f in flagged if f["tier"] == 1}
        assert "fleeting" in tier1_lemmas or "solace" in tier1_lemmas or "palpable" in tier1_lemmas

    def test_fiction_ai_tells(self, nlp):
        """Nous FICTION_AI_TELLS regex patterns should be detected."""
        text = (
            "She couldn't help but feel a sense of dread. "
            "A wave of relief washed over him. "
            "She let out a breath she didn't know she was holding. "
            "Something dark stirred within."
        )
        doc = nlp(text)
        score, flagged = score_banned_vocabulary(doc, len(text.split()))
        fiction_tells = [f for f in flagged if f.get("tier") == "fiction_tell"]
        assert len(fiction_tells) > 0

    def test_telling_adverbs(self, nlp):
        """Emotion adverbs that signal telling-not-showing."""
        text = (
            "She angrily slammed the door. He nervously paced the room. "
            "She sadly looked away. He desperately reached for the phone. "
            "She furiously typed the message."
        ) * 3
        doc = nlp(text)
        score, _flagged = score_banned_vocabulary(doc, len(text.split()))
        # Dense telling adverbs should push the score up
        assert score > 0
