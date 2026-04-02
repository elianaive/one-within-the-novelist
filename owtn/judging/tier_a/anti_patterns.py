"""Filter 5: Fiction Anti-Patterns (Nous Research).

Structural patterns that "survive prompt engineering and surface-level slop
detection." Sources: Nous Research ANTI-PATTERNS.md (12 patterns),
Sage GLM-4.6 v1.6.1.
"""

import re

import numpy as np

# ---------------------------------------------------------------------------
# Shared constants (also used by construction.py)
# ---------------------------------------------------------------------------

EMOTION_ADJECTIVES = {
    "angry", "sad", "happy", "afraid", "scared", "nervous",
    "frustrated", "relieved", "anxious", "terrified", "furious",
    "devastated", "elated", "confused", "desperate", "hopeless",
    "excited", "ashamed", "guilty", "jealous", "lonely",
}

_TELL_VERBS = {"felt", "was", "seemed", "appeared", "looked", "realized"}

_SCENE_ACTION_PATTERN = re.compile(
    r'\b(?:said|asked|walked|ran|opened|closed|picked|grabbed|turned|looked)\b',
    re.IGNORECASE,
)

_COLD_START_PATTERNS = [
    re.compile(r'^It was(?:n\'t)?\s+', re.IGNORECASE),
    re.compile(r'^The kind of\s+\w+\s+that', re.IGNORECASE),
    re.compile(r'^It (?:is|was) a truth', re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Individual anti-pattern scorers
# ---------------------------------------------------------------------------

def detect_over_explanation(doc, word_count: int) -> float:
    """Nous #1 — show-then-tell: action followed by explicit emotion tell."""
    sentences = list(doc.sents)
    over_explain_count = 0

    for i in range(len(sentences) - 1):
        sent = sentences[i]
        next_sent = sentences[i + 1]

        has_action = any(t.pos_ == "VERB" and t.dep_ == "ROOT" for t in sent)
        next_tokens = {t.text.lower() for t in next_sent}
        has_tell_verb = bool(next_tokens & _TELL_VERBS)
        has_emotion = bool(next_tokens & EMOTION_ADJECTIVES)

        if has_action and has_tell_verb and has_emotion:
            over_explain_count += 1

    density = over_explain_count / max(word_count / 1000, 1)
    return min(density / 2.0, 1.0)


def score_triadic_listing(doc, word_count: int) -> float:
    """Nous #2 — groups of three: "X, Y, and Z" enumerations.

    Threshold: >2 per 1,000 words = flag.
    """
    triadic = re.findall(r'\b\w+,\s+\w+,\s+and\s+\w+\b', doc.text)
    density = len(triadic) / max(word_count / 1000, 1)
    if density <= 2:
        return 0.0
    elif density >= 5:
        return 1.0
    return (density - 2) / 3


def score_negative_assertion(doc, word_count: int) -> float:
    """Nous #3 — "did not X" repetition. Threshold: >1 per 1,000 words."""
    pattern = re.compile(r'\b(?:did not|didn\'t)\s+\w+', re.IGNORECASE)
    matches = pattern.findall(doc.text)
    density = len(matches) / max(word_count / 1000, 1)
    if density <= 1:
        return 0.0
    elif density >= 4:
        return 1.0
    return (density - 1) / 3


def score_simile_crutch(doc, word_count: int) -> float:
    """Nous #5 — "the way X did Y" overuse. Threshold: >2 per 1,000 words."""
    pattern = re.compile(r'\bthe way\s+\w+\s+\w+', re.IGNORECASE)
    matches = pattern.findall(doc.text)
    density = len(matches) / max(word_count / 1000, 1)
    if density <= 2:
        return 0.0
    elif density >= 5:
        return 1.0
    return (density - 2) / 3


def score_section_break_overuse(text: str, word_count: int) -> float:
    """Nous #6 — excessive section breaks. Threshold: >2 per 2,000 words."""
    breaks = len(re.findall(r'\n\s*[-*_]{3,}\s*\n', text))
    density = breaks / max(word_count / 2000, 1)
    if density <= 2:
        return 0.0
    elif density >= 5:
        return 1.0
    return (density - 2) / 3


def score_balanced_antithesis(dialogues: list[str]) -> float:
    """Nous #9 — multiple characters using "not X, but Y" in dialogue."""
    if len(dialogues) < 2:
        return 0.0
    antithesis_count = sum(
        1 for d in dialogues
        if re.search(r'\bnot\s+\w+[\w\s,]*,?\s*but\s+', d, re.IGNORECASE)
    )
    if antithesis_count >= 3:
        return 1.0
    elif antithesis_count >= 2:
        return 0.5
    return 0.0


def score_dialogue_naturalness(dialogues: list[str]) -> float:
    """Nous #10 — AI dialogue: too grammatical, too long, no interruptions."""
    if not dialogues:
        return 0.0

    lengths = [len(d.split()) for d in dialogues]
    mean_length = np.mean(lengths)

    fragments = sum(
        1 for d in dialogues
        if not d.rstrip('."\'!?\u201d').endswith(('.', '!', '?'))
    )
    fragment_ratio = fragments / max(len(dialogues), 1)

    score = 0.0
    if mean_length > 20:
        score += 0.5
    elif mean_length > 15:
        score += 0.25

    if fragment_ratio < 0.1:
        score += 0.3
    if fragment_ratio < 0.05:
        score += 0.2

    return min(score, 1.0)


def score_scene_summary_ratio(paragraphs: list[str]) -> float:
    """Nous #11 — flag if <70% of prose is in-scene."""
    if not paragraphs:
        return 0.0

    scene_paras = 0
    for para in paragraphs:
        has_dialogue = '"' in para or '\u201c' in para or "'" in para
        has_action = bool(_SCENE_ACTION_PATTERN.search(para))
        if has_dialogue or has_action:
            scene_paras += 1

    ratio = scene_paras / len(paragraphs)

    if ratio >= 0.7:
        return 0.0
    elif ratio <= 0.4:
        return 1.0
    return 1.0 - (ratio - 0.4) / 0.3


def score_cold_start(text: str) -> float:
    """Nous #12 — cold start failure patterns in opening."""
    opening = text[:1200]
    score = 0.0
    for pattern in _COLD_START_PATTERNS:
        if pattern.search(opening):
            score += 0.3
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def score_fiction_anti_patterns(
    doc, text: str, word_count: int, dialogues: list[str], paragraphs: list[str],
) -> float:
    """Composite of Nous Research fiction anti-patterns (9 of 12)."""
    scores = [
        detect_over_explanation(doc, word_count),
        score_triadic_listing(doc, word_count),
        score_negative_assertion(doc, word_count),
        score_simile_crutch(doc, word_count),
        score_section_break_overuse(text, word_count),
        score_balanced_antithesis(dialogues),
        score_dialogue_naturalness(dialogues),
        score_scene_summary_ratio(paragraphs),
        score_cold_start(text),
    ]
    return float(np.mean(scores))
