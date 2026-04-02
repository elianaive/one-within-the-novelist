"""Filter 4: Statistical Signals.

Corpus-level metrics that distinguish human from AI prose. Sources:
Reinhart et al. (PNAS 2025), Voss et al. (Nature 2025), GPTZero methodology.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERSONAL_PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
}

NOMINALIZATION_SUFFIXES = ("tion", "ment", "ness", "ity", "ence", "ance")


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def mattr(tokens: list[str], window_size: int = 500) -> float:
    """Moving Average Type-Token Ratio. Literary fiction: >0.80. AI: 0.55-0.75."""
    if not tokens:
        return 0.0
    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens)

    ttrs = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        ttrs.append(len(set(window)) / window_size)
    return float(np.mean(ttrs))


def score_mattr(doc) -> float:
    """Score 0-1 where higher = less lexically diverse (AI-like)."""
    tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]
    m = mattr(tokens)

    if m >= 0.80:
        return 0.0
    elif m <= 0.60:
        return 1.0
    return 1.0 - (m - 0.60) / 0.20


def score_pronoun_density(doc, word_count: int) -> float:
    """Fewer personal pronouns = AI-like. Returns 0-1.

    Human fiction baseline: ~60-80 per 1,000 words.
    AI fiction: ~40-55 per 1,000 words.
    Voss et al. (Nature 2025): mediates reduced reader transportation.
    """
    pronoun_count = sum(1 for t in doc if t.text.lower() in PERSONAL_PRONOUNS)
    density = pronoun_count / max(word_count / 1000, 1)

    if density >= 65:
        return 0.0
    elif density <= 40:
        return 1.0
    return 1.0 - (density - 40) / 25


def score_nominalization(doc, word_count: int) -> float:
    """AI prose over-nominalizes (1.5-2x human rate, d=1.23). Returns 0-1."""
    nom_count = 0
    for token in doc:
        if token.pos_ == "NOUN" and len(token.text) > 5:
            if token.text.lower().endswith(NOMINALIZATION_SUFFIXES):
                nom_count += 1

    density = nom_count / max(word_count / 1000, 1)

    if density <= 25:
        return 0.0
    elif density >= 45:
        return 1.0
    return (density - 25) / 20


def score_participial_openings(doc) -> float:
    """Flag sentences opening with present participial (-ing) clauses.

    GPT-4o at 5.3x human rate (d=1.38, Reinhart et al. PNAS 2025).
    """
    sentences = list(doc.sents)
    if not sentences:
        return 0.0

    vbg_starts = 0
    for sent in sentences:
        first_content = next((t for t in sent if not t.is_space and not t.is_punct), None)
        if first_content and first_content.tag_ == "VBG":
            vbg_starts += 1

    ratio = vbg_starts / len(sentences)
    # Human baseline: ~3-5% of sentences. AI: 15-25%.
    if ratio <= 0.05:
        return 0.0
    elif ratio >= 0.20:
        return 1.0
    return (ratio - 0.05) / 0.15
