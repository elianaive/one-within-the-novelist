"""Filter 2: Construction Patterns.

Syntactic patterns at the sentence level. Harder to game than vocabulary
because they target structure, not words. Sources: Sage GLM-4.6 v1.6.1
(scenario.json system prompt — ~200 hours of systematic testing),
Nous ANTI-SLOP.md, EQ-Bench (25% of composite slop score from Not-X-But-Y).
"""

import re

import numpy as np

from .anti_patterns import EMOTION_ADJECTIVES

# ---------------------------------------------------------------------------
# Pattern constants
# ---------------------------------------------------------------------------

# "The single most overused LLM rhetorical pattern" — Nous Research.
NOT_X_BUT_Y = re.compile(
    r'\b(?:not\s+(?:just\s+|merely\s+|only\s+|simply\s+)?[\w\s,]+,?\s*but\s+'
    r"|it'?s\s+not\s+[\w\s,]+\s*[—\-]+\s*it'?s\s+"
    r"|more\s+than\s+just\s+)",
    re.IGNORECASE,
)

# "Ban the pattern, not the phrase" — Sage GLM-4.6 v1.6.1
BODY_PARTS = {
    "heart", "chest", "stomach", "gut", "spine", "jaw", "fist",
    "hand", "hands", "throat", "pulse", "blood", "knuckles",
    "eyes", "fists",
}

EMOTION_VERBS = {
    "hammer", "pound", "race", "clench", "tighten", "flutter",
    "sink", "drop", "lurch", "twist", "knot", "squeeze",
    "ache", "burn", "freeze", "tremble", "shiver", "sparkle",
    "darken", "widen", "narrow",
}

# Eyes as active subject — eyes don't do things, people do.
EYES_AS_SUBJECT = re.compile(
    r'\b(?:his|her|their|the)\s+eyes?\s+(?:searched|scanned|traced|found|'
    r'locked|darted|swept|roamed|traveled|roved|bore into|bored into|'
    r'followed|drifted|flicked|narrowed on|settled on|met\s+)',
    re.IGNORECASE,
)

# "His words sent a chill through her spine."
SENSATION_THROUGH = re.compile(
    r'\b(?:sent|sending|shoot|shooting|cours(?:ed|ing)|ran|running|jolt(?:ed|ing))'
    r'\s+[\w\s]*?\b(?:through|down)\s+(?:his|her|their|the)\s+'
    r'(?:spine|body|veins|bones|core|back|chest|frame)',
    re.IGNORECASE,
)

# Temperature as emotion proxy.
TEMPERATURE_WORDS = {
    "warm", "warmth", "cold", "coldness", "ice", "icy", "fire", "fiery",
    "heat", "chill", "chilly", "freeze", "freezing", "burn", "burning",
    "hot", "cool", "frost", "frosty",
}

# Voice quality as emotion indicator.
VOICE_QUALITY = re.compile(
    r'\bvoice\s+(?:was\s+)?(?:a\s+)?(?:low|husky|trembling|soft|firm|steady|'
    r'sharp|throaty|breathier|huskier|rough|gravelly|silky|velvety|'
    r'barely\s+(?:a\s+)?whisper|a\s+low\s+rumble)',
    re.IGNORECASE,
)

# Hedging qualifiers — AI hedges where humans state.
HEDGING = re.compile(
    r'\b(?:seemed to|appeared to|as if|as though|kind of|sort of|almost\s+'
    r'(?:as if|like)|a hint of|a flicker of|a shadow of|a touch of|'
    r'a note of|somehow|something like)\b',
    re.IGNORECASE,
)

# Smile/voice as primary action.
SMILE_ACTION = re.compile(
    r'\b(?:a\s+)?(?:smile|grin|smirk|frown)\s+(?:spread|crept|played|tugged|'
    r'curved|crossed|touched|lit|formed|flickered|appeared)',
    re.IGNORECASE,
)

# Stacked adjectives after commas.
STACKED_ADJ = re.compile(r',\s+\w+\s+and\s+\w+(?:\s*[.!?]|\s*,)')

# The "drawing gaze" family.
DRAWING_GAZE = re.compile(
    r'\b(?:draw(?:ing|n)?|pull(?:ing|ed)?)\s+(?:his|her|their|the)\s+'
    r'(?:gaze|eyes?|attention)',
    re.IGNORECASE,
)

MAGNETIC_WORDS = re.compile(
    r'\b(?:intoxicating|magnetic|irresistible|mesmerizing|hypnotic)\b',
    re.IGNORECASE,
)

# "Voiced a [adj] [noun]" construction. Sage v1.6.1 explicitly bans this.
# "She voiced a sultry purr." "He voiced a low growl."
VOICED_CONSTRUCTION = re.compile(
    r'\bvoiced\s+a\s+\w+\s+\w+',
    re.IGNORECASE,
)

# Pulse/breath/heartbeat as emotion/arousal indicators. Sage v1.6.1:
# "No pulse/breath/heartbeat as arousal indicators (pulse quickening,
# breath hitching, heart racing)."
AUTONOMIC_EMOTION = re.compile(
    r'\b(?:pulse\s+(?:quicken|race|pound|throb|hammer)|'
    r'breath\s+(?:hitch|catch|quicken|come\s+in\s+(?:short|ragged))|'
    r'heart\s+(?:race|pound|hammer|thunder|stutter|slam)|'
    r'heartbeat\s+(?:quicken|race|pound|stutter))',
    re.IGNORECASE,
)

# "jolt/shock/electricity" as sensation. Sage v1.6.1:
# "No jolt/shock/electricity + body parts."
ELECTRIC_SENSATION = re.compile(
    r'\b(?:(?:a\s+)?(?:jolt|shock|bolt|spark|current)\s+of\s+(?:electricity|energy|'
    r'awareness|recognition|pain|pleasure|heat|desire)|'
    r'electric(?:ity|al)?\s+(?:shock|current|charge|jolt))',
    re.IGNORECASE,
)

# Dead metaphors / stock similes. Sage v1.6.1:
# "Avoid dead metaphors (sparkling eyes, razor-sharp wit).
# No stock similes (laugh like wind chimes)."
DEAD_METAPHORS = re.compile(
    r'\b(?:sparkling\s+eyes|razor[- ]sharp\s+wit|'
    r'laugh(?:ed|ing|s)?\s+like\s+(?:wind\s+chimes|bells|music)|'
    r'eyes\s+like\s+(?:pools|oceans|stars|diamonds)|'
    r'silence\s+(?:thick|heavy)\s+enough\s+to\s+cut|'
    r'cut\s+the\s+(?:tension|silence)\s+with\s+a\s+knife)',
    re.IGNORECASE,
)

# Physical feature catalog words.
PHYSICAL_WORDS = {
    "hair", "eyes", "lips", "face", "skin", "shoulders", "body",
    "figure", "frame", "cheeks", "chin", "nose", "brow", "forehead",
    "jaw", "neck", "arms", "legs", "hands", "fingers",
}


# ---------------------------------------------------------------------------
# Sub-scorers
# ---------------------------------------------------------------------------

def _score_body_emotion(doc, word_count: int) -> float:
    """Detect BODY_PART + EMOTION_VERB in the same sentence."""
    count = 0
    for sent in doc.sents:
        tokens_lower = {t.lemma_.lower() for t in sent}
        if tokens_lower & BODY_PARTS and tokens_lower & EMOTION_VERBS:
            count += 1
    density = count / max(word_count / 1000, 1)
    if density <= 2:
        return 0.0
    elif density >= 5:
        return 1.0
    return (density - 2) / 3


def _score_temperature_emotion(doc, word_count: int) -> float:
    """Detect temperature words used as emotion proxies."""
    count = 0
    tokens = list(doc)
    for i, token in enumerate(tokens):
        if token.lemma_.lower() not in TEMPERATURE_WORDS:
            continue
        window_start = max(0, i - 3)
        window_end = min(len(tokens), i + 4)
        window_lemmas = {tokens[j].lemma_.lower() for j in range(window_start, window_end) if j != i}
        has_character = any(
            tokens[j].ent_type_ == "PERSON" or tokens[j].dep_ in ("nsubj", "nsubjpass")
            for j in range(window_start, window_end) if j != i
        )
        has_emotion = bool(window_lemmas & EMOTION_ADJECTIVES)
        if has_character or has_emotion:
            count += 1

    density = count / max(word_count / 1000, 1)
    if density <= 1:
        return 0.0
    elif density >= 4:
        return 1.0
    return (density - 1) / 3


def _score_catalog_descriptions(doc) -> float:
    """Detect 3+ consecutive physical detail sentences about the same person."""
    sentences = list(doc.sents)
    max_consecutive = 0
    current = 0

    for sent in sentences:
        lemmas = {t.lemma_.lower() for t in sent}
        if lemmas & PHYSICAL_WORDS:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 0

    if max_consecutive >= 4:
        return 1.0
    elif max_consecutive >= 3:
        return 0.6
    return 0.0


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_construction_patterns(text: str, doc, word_count: int) -> float:
    """Score 0-1 where higher = more AI construction patterns detected.

    Covers: Not-X-But-Y, body+emotion, eyes-as-subject, sensation-through,
    temperature-as-emotion, voice quality, hedging, smile-action, stacked
    adjectives, drawing gaze, catalog descriptions, voiced-construction,
    autonomic-emotion, electric-sensation, dead metaphors.
    """
    scores = []

    # Not-X-But-Y — any occurrence in fiction is suspicious
    not_x_count = len(NOT_X_BUT_Y.findall(text))
    not_x_density = not_x_count / max(word_count / 1000, 1)
    scores.append(min(not_x_density / 1.0, 1.0))

    scores.append(_score_body_emotion(doc, word_count))

    # Eyes as active subject
    eyes_count = len(EYES_AS_SUBJECT.findall(text))
    eyes_density = eyes_count / max(word_count / 1000, 1)
    scores.append(min(eyes_density / 1.0, 1.0))

    # Sensation-through constructions
    sensation_count = len(SENSATION_THROUGH.findall(text))
    sensation_density = sensation_count / max(word_count / 1000, 1)
    scores.append(min(sensation_density / 1.0, 1.0))

    scores.append(_score_temperature_emotion(doc, word_count))

    # Voice quality as emotion indicator (>1 per 1,000 words = flag)
    voice_count = len(VOICE_QUALITY.findall(text))
    voice_density = voice_count / max(word_count / 1000, 1)
    scores.append(min(voice_density / 1.0, 1.0))

    # Hedging qualifiers (human baseline ~1-2 per 1k; >3 = flag)
    hedge_count = len(HEDGING.findall(text))
    hedge_density = hedge_count / max(word_count / 1000, 1)
    if hedge_density <= 3:
        scores.append(0.0)
    elif hedge_density >= 6:
        scores.append(1.0)
    else:
        scores.append((hedge_density - 3) / 3)

    # Smile/voice as primary action
    smile_count = len(SMILE_ACTION.findall(text))
    smile_density = smile_count / max(word_count / 1000, 1)
    scores.append(min(smile_density / 1.0, 1.0))

    # Stacked adjectives after commas
    stacked_count = len(STACKED_ADJ.findall(text))
    stacked_density = stacked_count / max(word_count / 1000, 1)
    if stacked_density <= 1:
        scores.append(0.0)
    elif stacked_density >= 4:
        scores.append(1.0)
    else:
        scores.append((stacked_density - 1) / 3)

    # Drawing gaze family + magnetic/intoxicating words
    gaze_count = len(DRAWING_GAZE.findall(text)) + len(MAGNETIC_WORDS.findall(text))
    gaze_density = gaze_count / max(word_count / 1000, 1)
    scores.append(min(gaze_density / 1.0, 1.0))

    scores.append(_score_catalog_descriptions(doc))

    # "Voiced a [adj] [noun]" — Sage v1.6.1 explicitly bans
    voiced_count = len(VOICED_CONSTRUCTION.findall(text))
    voiced_density = voiced_count / max(word_count / 1000, 1)
    scores.append(min(voiced_density / 1.0, 1.0))

    # Pulse/breath/heartbeat as emotion indicators
    autonomic_count = len(AUTONOMIC_EMOTION.findall(text))
    autonomic_density = autonomic_count / max(word_count / 1000, 1)
    scores.append(min(autonomic_density / 1.0, 1.0))

    # Jolt/shock/electricity as sensation
    electric_count = len(ELECTRIC_SENSATION.findall(text))
    electric_density = electric_count / max(word_count / 1000, 1)
    scores.append(min(electric_density / 1.0, 1.0))

    # Dead metaphors / stock similes
    dead_count = len(DEAD_METAPHORS.findall(text))
    dead_density = dead_count / max(word_count / 1000, 1)
    scores.append(min(dead_density / 1.0, 1.0))

    return float(np.mean(scores))
