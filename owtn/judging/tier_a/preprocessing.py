"""Text preprocessing and spaCy setup for Tier A filters.

Single spaCy pass shared across all filters. Dialogue extraction and
paragraph segmentation utilities.
"""

import re

import spacy

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


_DIALOGUE_PATTERN = re.compile(
    r'\u201c[^\u201d]*\u201d|"[^"]*"|\'[^\']*\'|\u2014[^\u2014\n]*(?:\u2014|\n)'
)


def extract_dialogue(text: str) -> tuple[list[str], str]:
    """Extract dialogue segments; return (dialogue_list, narrative_only)."""
    dialogues = _DIALOGUE_PATTERN.findall(text)
    narrative = _DIALOGUE_PATTERN.sub(' ', text)
    return dialogues, narrative


def get_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double newlines."""
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
