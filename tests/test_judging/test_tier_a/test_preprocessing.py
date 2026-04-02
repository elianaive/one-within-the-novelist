"""Tests for owtn.judging.tier_a.preprocessing."""

from owtn.judging.tier_a.preprocessing import extract_dialogue, get_paragraphs


class TestExtractDialogue:
    def test_double_quotes(self):
        text = 'He said "hello there" and she replied "goodbye now".'
        dialogues, narrative = extract_dialogue(text)
        assert len(dialogues) == 2
        assert '"hello there"' in dialogues[0]
        assert "hello" not in narrative

    def test_empty_text(self):
        dialogues, narrative = extract_dialogue("")
        assert dialogues == []


class TestGetParagraphs:
    def test_double_newline_split(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird."
        paras = get_paragraphs(text)
        assert len(paras) == 3

    def test_empty(self):
        assert get_paragraphs("") == []

    def test_single_paragraph(self):
        assert len(get_paragraphs("Just one paragraph.")) == 1
