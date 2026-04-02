"""Filter 3: Structural Patterns.

Paragraph and section-level tells. AI prose has suspiciously uniform
sentence lengths, paragraph lengths, and mechanical rhythm.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBJECT_PRONOUNS = {"he", "she", "it", "they", "i", "we"}

TRANSITION_WORDS = {
    "however", "furthermore", "additionally", "moreover", "consequently",
    "nevertheless", "therefore", "meanwhile", "similarly", "conversely",
    "nonetheless", "subsequently", "accordingly",
}


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------

def score_burstiness(doc) -> float:
    """Low sentence-length variation = AI-like. Returns 0-1.

    Human fiction: CV 0.5-0.8+. AI: 0.2-0.4.
    """
    sentence_lengths = [len(sent) for sent in doc.sents]
    if len(sentence_lengths) < 5:
        return 0.0

    mean_len = np.mean(sentence_lengths)
    if mean_len == 0:
        return 1.0

    cv = float(np.std(sentence_lengths) / mean_len)
    if cv >= 0.6:
        return 0.0
    elif cv <= 0.2:
        return 1.0
    return 1.0 - (cv - 0.2) / 0.4


def score_paragraph_uniformity(paragraphs: list[str]) -> float:
    """Low paragraph-length variation = AI-like. Returns 0-1."""
    if len(paragraphs) < 3:
        return 0.0

    lengths = [len(p.split()) for p in paragraphs]
    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 1.0

    cv = float(np.std(lengths) / mean_len)
    if cv >= 0.5:
        return 0.0
    elif cv <= 0.2:
        return 1.0
    return 1.0 - (cv - 0.2) / 0.3


def score_consecutive_pronouns(doc) -> float:
    """Flag 3+ consecutive sentences starting with the same pronoun."""
    max_consecutive = 0
    current_pronoun = None
    current_count = 0

    for sent in doc.sents:
        first_token = next((t for t in sent if not t.is_space), None)
        if first_token and first_token.text.lower() in SUBJECT_PRONOUNS:
            if first_token.text.lower() == current_pronoun:
                current_count += 1
                max_consecutive = max(max_consecutive, current_count)
            else:
                current_pronoun = first_token.text.lower()
                current_count = 1
        else:
            current_pronoun = None
            current_count = 0

    if max_consecutive >= 5:
        return 1.0
    elif max_consecutive >= 3:
        return 0.6
    return 0.0


def score_em_dash(text: str, word_count: int) -> float:
    """Em-dash overuse. >2 per 250 words = flag."""
    em_dashes = text.count('\u2014') + text.count('--')
    pages = max(word_count / 250, 1)
    per_page = em_dashes / pages

    if per_page <= 2:
        return 0.0
    elif per_page >= 5:
        return 1.0
    return (per_page - 2) / 3


def score_transition_chains(paragraphs: list[str]) -> float:
    """Flag 3+ consecutive paragraphs opened with transition words."""
    max_consecutive = 0
    current = 0

    for para in paragraphs:
        first_word = para.split()[0].lower().rstrip(",") if para.split() else ""
        if first_word in TRANSITION_WORDS:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 0

    if max_consecutive >= 4:
        return 1.0
    elif max_consecutive >= 3:
        return 0.6
    return 0.0


def score_three_short_declaratives(doc) -> float:
    """Flag 3+ consecutive sentences under 8 words.

    Signals the model has "lost the thread" — Sage GLM-4.6.
    """
    max_consecutive = 0
    current = 0

    for sent in doc.sents:
        wc = len([t for t in sent if not t.is_punct and not t.is_space])
        if wc < 8:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 0

    if max_consecutive >= 5:
        return 1.0
    elif max_consecutive >= 3:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def score_structural_patterns(doc, text: str, word_count: int, paragraphs: list[str]) -> float:
    """Composite structural patterns score."""
    scores = [
        score_paragraph_uniformity(paragraphs),
        score_consecutive_pronouns(doc),
        score_em_dash(text, word_count),
        score_transition_chains(paragraphs),
        score_three_short_declaratives(doc),
    ]
    return float(np.mean(scores))
