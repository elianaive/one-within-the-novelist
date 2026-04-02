"""Tests for owtn.judging.tier_a.ngrams."""

from owtn.judging.tier_a.ngrams import score_slop_ngrams


class TestSlopNgrams:
    def test_clean_text(self):
        text = "The morning was cold and gray. She drank her coffee in silence."
        assert score_slop_ngrams(text, len(text.split())) == 0.0

    def test_slop_detected(self):
        text = (
            "She took deep breath as her heart pounding chest. "
            "The dimly lit room cast long shadows. "
            "A growing sense unease filled the air thick tension. "
            "Her breath caught throat as the blood ran cold. "
            "The silence hung heavy and the door creaked open. "
            "She couldn't shake feeling that the truth finally dawned."
        ) * 10
        word_count = len(text.split())
        assert score_slop_ngrams(text, word_count) > 0

    def test_business_slop_in_fiction(self):
        text = (
            "She thought about how data driven decision making plays crucial role "
            "in long term success. The multi faceted approach was a testament "
            "enduring power of the tapestry woven threads."
        ) * 5
        word_count = len(text.split())
        assert score_slop_ngrams(text, word_count) > 0
