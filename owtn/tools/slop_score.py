"""Slop score — port of EQ-Bench's slop-score for agent-facing use.

Four-component composite detecting AI-generated prose:
- **slop words** — per-1k-word rate against a 1648-word list of LLM-overrepresented
  vocabulary (50% weight)
- **contrast patterns** — per-1k-character rate of "not X, but Y" rhetorical
  structures via 10 surface regexes + 35 POS-tagged regexes (20% weight)
- **slop trigrams** — per-1k-word rate against a 430-trigram list of overrepresented
  3-word phrases (15% weight)
- **slop bigrams** — per-1k-word rate against a 200-bigram list of overrepresented
  2-word phrases (15% weight)

Source: https://github.com/sam-paech/slop-score (MIT). Lists derived from a
benchmark across 67 LLMs vs human writing baselines. The upstream tool's
composite is 60/25/15 over words/contrast/trigrams and never reads the
bigram list; we fold bigrams in here because they catch high-signal phrases
("deep breath", "heart pounding") that the trigram list misses.

Composite uses empirical normalization ranges drawn from the public
leaderboard's observed model spread, so scores fall on a 0-100 scale where
~20-30 is current-frontier-LLM territory and 50+ is conspicuous AI register.

Note: the slop-word list contains many invented fantasy names ("aelara",
"aethelred", etc.) — fantasy prose using such names will score as slop.
Treat the score as a register signal, not an authority.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from owtn.judging.tier_a.preprocessing import get_nlp


REPO_ROOT = Path(__file__).resolve().parents[2]
SLOP_WORDS_PATH = REPO_ROOT / "data" / "slop-score" / "slop_list.json"
SLOP_BIGRAMS_PATH = REPO_ROOT / "data" / "slop-score" / "slop_list_bigrams.json"
SLOP_TRIGRAMS_PATH = REPO_ROOT / "data" / "slop-score" / "slop_list_trigrams.json"


# ─── Slop list loading ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_slop_words() -> frozenset[str]:
    """1648 single words overrepresented in LLM output."""
    with SLOP_WORDS_PATH.open() as f:
        raw = json.load(f)
    return frozenset(item[0].lower() for item in raw if item)


@lru_cache(maxsize=1)
def _load_slop_bigrams() -> frozenset[str]:
    """200 two-word phrases overrepresented in LLM output."""
    with SLOP_BIGRAMS_PATH.open() as f:
        raw = json.load(f)
    return frozenset(item[0].lower() for item in raw if item)


@lru_cache(maxsize=1)
def _load_slop_trigrams() -> frozenset[str]:
    """430 three-word phrases overrepresented in LLM output."""
    with SLOP_TRIGRAMS_PATH.open() as f:
        raw = json.load(f)
    return frozenset(item[0].lower() for item in raw if item)


# ─── Tokenization (matches the JS port) ────────────────────────────────────

_QUOTE_NORMALIZE = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "′": "'", "ʼ": "'", "＇": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "″": '"', "«": '"', "»": '"', "＂": '"',
})

_DASH_NORMALIZE = str.maketrans({"—": "-", "–": "-"})

_WORD_TOKEN_RE = re.compile(r"[a-z']+")


def _normalize_text(text: str) -> str:
    """Normalize curly quotes and en/em dashes — matches JS normalizeText
    used for Stage 1 regexes."""
    return text.translate(_QUOTE_NORMALIZE).translate(_DASH_NORMALIZE)


def _tokenize_words(text: str) -> list[str]:
    """Lowercase word tokens, stripping leading/trailing apostrophes.
    Matches the JS `wordsOnlyLower` used for slop-word matching."""
    lowered = text.lower().translate(_QUOTE_NORMALIZE)
    out = []
    for tok in _WORD_TOKEN_RE.findall(lowered):
        stripped = tok.strip("'")
        if stripped:
            out.append(stripped)
    return out


# ─── Stage 1 contrast regexes (10 surface patterns) ────────────────────────
# Ported from lab/references/slop-score/slop-score/js/regexes-stage1.js.
# These run on raw normalized text — no POS tagging.

_MAXG = 160
_PRON = r"(?:it|they|this|that)"
_BE = r"(?:is|are|was|were)"
_BE_NEG = r"(?:is\s+not|are\s+not|was\s+not|were\s+not|isn't|aren't|wasn't|weren't|ain't)"

_STAGE1_RAW: dict[str, str] = {
    "RE_NOT_BUT": (
        rf"\b(?:(?:{_BE_NEG})|not(?!\s+(?:that|only)\b))\s+"
        rf"(?:(?!\bbut\b|[.?!]).){{1,100}}?"
        rf"[,;:]\s*but\s+"
        r"(?!when\b|while\b|which\b|who\b|whom\b|whose\b|where\b|if\b|that\b"
        r"|as\b|because\b|although\b|though\b|till\b|until\b|unless\b"
        r"|here\b|there\b|then\b|my\b|we\b|I\b|you\b"
        r"|it\s+seems\b|it\s+appears\b|it\s+felt\b|it\s+looks?\b|anything\b)"
    ),
    "RE_NOT_DASH": (
        r"\b(?:\w+n't|not)\s+(?:just|only|merely)?\s+"
        rf"(?:(?![.?!]).){{1,{_MAXG}}}?"
        r"(?:-|\s-\s|[—–])\s*"
        rf"{_PRON}\s+(?:(?:'re|are|'s|is|were|was)\b"
        r"|(?!'re|are|'s|is|were|was)[*_~]*[a-z]\w*)"
    ),
    "RE_PRON_BE_NOT_SEP_BE": (
        r"(?:(?<=^)|(?<=[.?!]\s))\s*[\"']?"
        rf"(?:(?:{_PRON}\s+{_BE}\s+not)|(?:{_PRON}\s+{_BE}n't)"
        r"|(?:it's|they're|that's)\s+not)\b"
        rf"[^.?!]{{0,{_MAXG}}}[.;:?!]\s*[\"']?"
        rf"{_PRON}\s+(?:{_BE}|(?:'s|'re))\b(?!\s+not\b)"
    ),
    "RE_NP_BE_NOT_SEP_THEY_BE": (
        r"(?:(?<=^)|(?<=[.?!]\s))\s*"
        r"(?![^.?!]{0,80}\b(?:knew|know|thought|think|said|says|told|heard|learned)\b"
        r"[^.?!]{0,40}?\bthat\b)"
        r"(?!\s*not\s+without\b)"
        r"(?![^.?!]{0,50}\bnot\s+put\b)"
        rf"[^.?!]{{0,{_MAXG}}}?\b(?:{_BE_NEG})\b[^.?!]{{0,{_MAXG}}}[.;:?!]\s*"
        rf"[\"']?{_PRON}\b(?:'re|\s+(?:are|were|is|was))\b(?!\s+not\b)"
    ),
    "RE_NO_LONGER": (
        rf"(?:(?<=^)|(?<=[.?!]\s))\s*[^.?!]{{0,{_MAXG}}}\bno\s+longer\b"
        rf"[^.;:?!]{{0,{_MAXG}}}"
        r"[.;:?!]\s*(?:it|they|this|that)\s+(?:is|are|was|were)\b(?!\s+not\b)"
    ),
    "RE_NOT_JUST_SEP": (
        r"(?:(?<=^)|(?<=[.?!]\s))\s*[\"']?"
        rf"{_PRON}\b(?:'s|'re|\s+(?:is|are|was|were))?\s+not\s+just\b"
        rf"[^.?!]{{0,{_MAXG}}}[.?!]\s*[\"']?"
        rf"{_PRON}\b(?:'s|'re|\s+(?:is|are|was|were))\b(?!\s+not\b)"
    ),
    "RE_NOT_PERIOD_SAMEVERB": (
        r"(?:(?<=^)|(?<=[.?!]\s))[^.?!]*?\b(?:do|does|did)n't\b\s+"
        r"(?:(?:\w+\s+){0,2})([a-z]{3,})\b[^.?!]*[.?!]\s*"
        rf"{_PRON}\s+\1(?:ed|es|s|ing)?\b"
    ),
    "RE_SIMPLE_BE_NOT_IT_BE": (
        r"(?:(?<=^)|(?<=[.?!]\s))\s*[\"']?"
        r"(?!he\b|she\b|i\b|you\b|we\b)"
        r"(?![^.?!]{0,80}\b(?:knew|know|thought|think|said|says|told|heard|learned)\b"
        r"[^.?!]{0,40}?\bthat\b)"
        rf"[^.?!]{{0,{_MAXG}}}?\b{_BE_NEG}\b[^.?!]{{0,{_MAXG}}}[.;:?!]\s*"
        r"[\"']?it(?:'s|\s+(?:is|are|was|were))\b"
    ),
    "RE_EMBEDDED_NOT_JUST_SEP": (
        r"(?:(?<=^)|(?<=[.?!]\s))"
        r"[^.?!]{0,80}?\b(?:(?:it|they)\s+(?:is|are)|(?:it's|they're))\s+not\s+just\b"
        rf"[^.?!]{{0,{_MAXG}}}[.?!]\s*"
        r"(?:(?:it|they)\s+(?:is|are)|(?:it's|they're))\b"
    ),
    "RE_DIALOGUE_NOT_JUST": (
        rf"[\"']?{_PRON}(?:'re|'s|\s+(?:are|is|was|were))\s+not\s+just\b"
        rf"[^\"']{{0,{_MAXG}}}[\"']?\s*"
        r"(?:[^.?!]{0,80}\b(?:said|asked|whispered|muttered|replied|added|shouted|cried)\b"
        r"[^.?!]{0,80}[.?!]\s*)?"
        rf"[\"']?{_PRON}(?:'re|'s|\s+(?:are|is|was|were))\s+[*_~]?[a-z]\w*"
    ),
}

STAGE1_REGEXES: dict[str, re.Pattern] = {
    name: re.compile(pat, re.IGNORECASE) for name, pat in _STAGE1_RAW.items()
}


# ─── Stage 2 POS-tagged regexes (35 patterns) ──────────────────────────────
# Ported from lab/references/slop-score/slop-score/js/regexes-stage2.js.
# These run against a stream where verb tokens have been replaced by the
# literal "VERB". We use spaCy's Penn-Treebank `tag_` to identify verbs,
# matching the wink-pos-tagger tag set the JS port targets.

_VERB_PT_TAGS = frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"})

_STAGE2_RAW: dict[str, str] = {
    "POS_DOESNT_VERB":
        r"[\"']\s*(?:[Tt]he\s+\w+|[Ii]t|[Tt]hey|[Yy]ou)\s+doesn't\s+VERB[^.!?]*?[.!?]"
        r"\s*(?:it|they|you|that)\s+[*_~]?(?:VERB|whispers?|reminds?|signals?|tests?|speaks?)",
    "POS_DONT_JUST_VERB":
        r"[\"']\s*(?:[Tt]hey|[Yy]ou|[Ii]t)\s+don't\s+just\s+VERB[^.!?]*?[—-]\s*they\s+[*_~]?VERB",
    "POS_GERUND_FRAGMENT":
        r"[\"']\s*Not\s+just\s+VERB[.!?]\s+[*_~]?VERB[.!?]",
    "POS_NOT_ADJ":
        r"\bnot\s+(random|passive|simple|normal)[.!?;]\s+(?:[Tt]hey|[Ii]t)\s+"
        r"(?:were|was|are|is|'re|'s)\s+[*_~]?(intentional|active|complex|different|\w{8,})",
    "POS_DASH_VERB":
        r"\b(?:wasn't|weren't|isn't|aren't)\s+just\s+(?:VERB|a\s+\w+)[^-]{0,30}?-\s*"
        r"(?:it|they)\s+(?:was|were|is|are|'s|'re)\s+[*_~]?(?:VERB|a\s+[*_~]?\w+)",
    "POS_NOT_JUST_VERB_PAST":
        r"\b(?:was|were)\s+not\s+just\s+(?:VERB|a\s+\w+)[.!?]\s+(?:[Ii]t|[Tt]hey)\s+"
        r"(?:was|were)\s+[*_~]?(?:VERB|a\s+[*_~]?\w+)",
    "POS_COLON_VERB":
        r":\s+(?:the\s+\w+|it|they)\s+(?:was|were)\s+not\s+just\s+VERB[.!?]\s+"
        r"(?:[Ii]t|[Tt]hey)\s+(?:was|were)\s+[*_~]?VERB",
    "POS_ISNT_JUST_VERB":
        r"[\"'](?:[^\"']{0,100}?\b)?(?:The\s+\w+|It|They|You)\s+"
        r"(?:isn't|aren't|wasn't|weren't)\s+just\s+VERB[^\"'.!?]{0,40}?[—-]\s*"
        r"(?:it's|they're)\s+[*_~]?VERB",
    "POS_QUOTE_MULTI_VERB":
        r"[\"']\s*[^\"']{0,150}?\b(?:not\s+just|isn't|aren't)\s+(?:VERB|a\s+\w+)"
        r"[^\"'.!?]{0,60}?[.!?]\s+(?:[^\"']{0,40}?\b)?"
        r"(?:It's|They're|You're|That's)\s+[*_~]?(?:VERB|a\s+[*_~]?\w+)",
    "POS_ELLIPSIS_VERB":
        r"[\"']\s*[^\"']{0,100}?\b(?:not\s+just|isn't)\s+VERB[^\"']{0,30}?[.…]\s*"
        r"[.…]\s*(?:they're|it's|you're)\s+[*_~]?VERB",
    "POS_NOT_NOUN":
        r"[\"']\s*(?:That's|It's)\s+not\s+(?:a\s+)?"
        r"(sign|message|warning|pattern|test|phenomenon|one\s+\w+)[.!?]\s+"
        r"(?:That's|It's)\s+(?:a\s+|\*?all\s+)?[*_~]?"
        r"(warning|question|language|symbol|test|presence|story|challenge|\w+)",
    "POS_DOESNT_VERB_EMPHASIS":
        r"[\"']\s*(?:The\s+\w+|It|They)\s+doesn't\s+(?:VERB|react|warn|speak)[.!?]\s+"
        r"It\s+\*(?:VERB|whispers?|reminds?|signals?)",
    "POS_DASH_VERB_BROAD":
        r"\b(?:wasn't|weren't|isn't|aren't|don't|doesn't)\s+just\s+"
        r"(?:VERB|(?:the|a)\s+\w+)[^-]{0,40}?-\s*(?:it|they)\s+"
        r"(?:was|were|is|are|'s|'re)?\s*[*_~]?(?:VERB|(?:the|a)\s+[*_~]?\w+)",
    "POS_ELLIPSIS_BROAD":
        r"[\"']\s*(?:[^\"']{0,100}?\b)?(?:They're|You're|This)\s+"
        r"(?:not\s+just|isn't)\s+(?:VERB|a\s+\w+)[^\"']{0,40}?[.…]\s*"
        r"[.…]\s*(?:they're|it's|you're|this)\s+(?:VERB|a\s+\w+)",
    "POS_NOT_BECAUSE":
        r"\bit's\s+not\s+because\s+[^.!?]{5,60}?[.!?]\s+(?:It's|That's)\s+because\s+[^.!?]{5,60}",
    "POS_GERUND_BROAD":
        r"[\"']\s*Not\s+just\s+VERB[.!?]\s+\*VERB[.!?]?",
    "POS_QUOTE_VERBING":
        r"[\"']\s*(?:You're|They're|It's)\s+not\s+(?:just\s+)?VERB[^\"'.!?]{0,30}?[.,]\s+"
        r"[^\"']{0,50}?(?:You're|They're|It's)\s+(?:VERB|waiting)",
    "POS_DOESNT_LITERAL":
        r"[\"']\s*(?:The\s+\w+|It|They)\s+doesn't\s+(?:VERB|react|warn|speak|listen)\s*[.!?]\s+"
        r"It\s+\*\w+\*",
    "POS_DASH_NOUN_SWAP":
        r"\b(?:was|were|is|are)\s+not\s+just\s+a\s+\w+[^-]{0,10}?-\s*"
        r"(?:it|they)\s+(?:was|were|is|are)\s+(?:a\s+)?\*\w+\*",
    "POS_ISNT_DASH_EMPHASIS":
        r"[\"']\s*(?:The\s+\w+|It|They)\s+(?:isn't|aren't|wasn't|weren't)\s+just\s+"
        r"(?:VERB|a\s+\w+)[^-]{0,40}?-\s*(?:it's|they're)\s+\*\w+\*",
    "POS_THATS_NOT_NOUN":
        r"[\"']\s*That's\s+not\s+(?:a\s+)?"
        r"(?:sign|message|pattern|phenomenon|test|one\s+\w+|\w+)[.!?]\s+"
        r"(?:That's|It's)\s+(?:a\s+)?\*\w+\*",
    "POS_GERUND_EMPHASIS":
        r"[\"']\s*Not\s+just\s+(?:VERB|reacting|dying|\w+ing)[.!?]\s+\*[A-Z]\w+\*",
    "POS_QUOTE_ATTRIBUTION_VERB":
        r"[\"']\s*(?:The\s+\w+|They)\s+(?:are|were|'re)\s+not\s+just\s+VERB,\""
        r"\s+[^\"']{0,30}?\.\s+\"They're\s+\*?VERB",
    "POS_ISNT_NOUN":
        r"[\"']\s*(?:This|That|It)\s+isn't\s+just\s+a\s+\w+[.!?]\s+It's\s+(?:a\s+)?\*\w+\*",
    "POS_ITS_NOT_JUST":
        r"[\"']\s*It's\s+not\s+just\s+(?:one\s+)?(\w+)[.!?]\s+It's\s+\*(?:all|every|each|\w+)\*",
    "POS_DASH_GERUND_OBJ":
        r"[\"']\s*(?:They're|You're|It's)\s+not\s+just\s+"
        r"(?:VERB|emitting|dying|\w+ing)\s+(?:a|an|the)\s+\w+[^-]{0,10}?-\s*"
        r"(?:they're|you're|it's)\s+\*\w+\*",
    "POS_ELLIPSIS_DIALOGUE":
        r"[\"']\s*(?:They're|You're|It's)\s+not\s+just\s+VERB,\""
        r"\s+[^\"']{5,40}?\.\s+\"(?:They're|You're|It's)[…\s]+(?:VERB|\w+ing)",
    "POS_SEMI_NOUN":
        r"\b(?:were|was|are|is)\s+not\s+just\s+(?:folklore|\w+);\s+"
        r"(?:they|it)\s+(?:were|was|are|is)\s+a\s+\w+",
    "POS_ISNT_ADJ_NOUN":
        r"[\"']\s*(?:[^\"']{0,30}?\b)?(?:this|that|it)\s+isn't\s+just\s+a\s+"
        r"(?:natural\s+)?\w+[.!?]\s+It's\s+(?:a\s+)?\*\w+\*",
    "POS_DIALOGUE_ATTR":
        r"[\"']\s*(?:You're|They're|It's|The\s+\w+)\s+"
        r"(?:(?:are|is|'re|'s)\s+)?not\s+just\s+(?:VERB(?:\s+\w+)?|a\s+\w+),\""
        r"\s+[^\"']{3,50}?\.\s+\"(?:You're|They're|It's)\s+(?:a\s+)?\*\w+\*",
    "POS_TO_VERB_ISNT":
        r"[\"']\s*To\s+VERB\s+(?:that\s+)?[^\"']{5,50}?isn't\s+just\s+a\s+\w+[.!?]\s+"
        r"It's\s+(?:a\s+)?\*\w+\*",
    "POS_I_AM_NOT_SEMI":
        r"\bI\s+am\s+not\s+VERB[^;]{5,80}?;\s*it\s+is\b",
    "POS_NOT_ANYMORE_ITS":
        r"\bIt's\s+not\s+[A-Z]\w+\s+anymore[.!?]\s+It's\s+[A-Z]\w+",
    "POS_AINT_SIMPLE":
        r"\b(?:That|This)\s+ain't\s+[^.!?]{3,40}?[.!?]\s+(?:They|It)\s+\w+",
    "LEMMA_SAME_VERB":
        r"\b(REACT|SPEAK|LISTEN|LEARN|SIGNAL|WARN|DIE|LIVE|TEST|TEACH|"
        r"AMPLIFY|INTERPRET|TRANSLATE|DECODE|EMIT)\b[^.!?]{5,80}?[.!?;—-]\s*"
        r"[^.!?]{0,40}?\b\1\b",
}

# JS source uses /g (case-sensitive) on these four; everything else is /gi.
_STAGE2_CASE_SENSITIVE = frozenset({
    "POS_GERUND_FRAGMENT",
    "POS_GERUND_BROAD",
    "POS_GERUND_EMPHASIS",
    "POS_NOT_ANYMORE_ITS",
})

STAGE2_REGEXES: dict[str, re.Pattern] = {
    name: re.compile(
        pat,
        0 if name in _STAGE2_CASE_SENSITIVE else re.IGNORECASE,
    )
    for name, pat in _STAGE2_RAW.items()
}


def _build_verb_stream(doc) -> str:
    """Produce a tagged stream where verb tokens are replaced with the
    literal 'VERB' and all other text (including whitespace) is preserved.

    spaCy's `token.tag_` returns Penn Treebank tags, matching the wink-pos-
    tagger tag set the original JS port targets.
    """
    parts: list[str] = []
    for tok in doc:
        if tok.tag_ in _VERB_PT_TAGS:
            parts.append("VERB")
        else:
            parts.append(tok.text)
        parts.append(tok.whitespace_)
    return "".join(parts)


# ─── Sentence span + interval merge (matches JS contrast-detector.js) ──────

_SENT_SPLIT = re.compile(r"[^.!?]*[.!?]", re.DOTALL)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    last_end = 0
    for m in _SENT_SPLIT.finditer(text):
        spans.append((m.start(), m.end()))
        last_end = m.end()
    if last_end < len(text):
        spans.append((last_end, len(text)))
    return spans


def _covered_sentences(spans: list[tuple[int, int]], start: int, end: int):
    if not spans or start >= end:
        return None
    starts = [s[0] for s in spans]
    ends = [s[1] for s in spans]
    # bisect_right(ends, start)
    lo, hi = 0, len(ends)
    while lo < hi:
        mid = (lo + hi) // 2
        if ends[mid] <= start:
            lo = mid + 1
        else:
            hi = mid
    sl = lo
    # bisect_left(starts, end) - 1
    lo, hi = 0, len(starts)
    while lo < hi:
        mid = (lo + hi) // 2
        if starts[mid] < end:
            lo = mid + 1
        else:
            hi = mid
    sh = lo - 1
    if sl >= len(spans) or sh < 0 or sl > sh:
        return None
    return sl, sh


def _merge_intervals(items: list[dict]) -> list[dict]:
    if not items:
        return []
    items_sorted = sorted(items, key=lambda d: (d["lo"], d["hi"], d["raw_start"]))
    merged = [dict(items_sorted[0])]
    for it in items_sorted[1:]:
        cur = merged[-1]
        if it["lo"] <= cur["hi"]:
            cur["hi"] = max(cur["hi"], it["hi"])
            cur["raw_end"] = max(cur["raw_end"], it["raw_end"])
        else:
            merged.append(dict(it))
    return merged


# ─── Component scorers ─────────────────────────────────────────────────────

def _score_slop_words(tokens: list[str]) -> tuple[int, dict[str, int]]:
    slop = _load_slop_words()
    hits: dict[str, int] = {}
    for tok in tokens:
        if tok in slop:
            hits[tok] = hits.get(tok, 0) + 1
    return sum(hits.values()), hits


def _score_slop_bigrams(tokens: list[str]) -> tuple[int, dict[str, int]]:
    slop = _load_slop_bigrams()
    if len(tokens) < 2:
        return 0, {}
    hits: dict[str, int] = {}
    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]} {tokens[i+1]}"
        if bg in slop:
            hits[bg] = hits.get(bg, 0) + 1
    return sum(hits.values()), hits


def _score_slop_trigrams(tokens: list[str]) -> tuple[int, dict[str, int]]:
    slop = _load_slop_trigrams()
    if len(tokens) < 3:
        return 0, {}
    hits: dict[str, int] = {}
    for i in range(len(tokens) - 2):
        tg = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
        if tg in slop:
            hits[tg] = hits.get(tg, 0) + 1
    return sum(hits.values()), hits


def _score_contrast_patterns(norm_text: str, doc) -> tuple[int, list[dict]]:
    """Run Stage 1 + Stage 2 contrast regexes; return (count, examples).

    Mirrors the JS extractContrastMatches: collect all candidate matches,
    map to sentence spans, merge overlapping intervals, and report the
    full sentence span for each merged hit. Count = post-merge length.

    Expects `norm_text` to already be normalized (curly quotes/dashes
    converted to ASCII) and `doc` to be the spaCy parse of that same
    normalized text.
    """
    spans = _sentence_spans(norm_text)
    candidates: list[dict] = []

    for name, regex in STAGE1_REGEXES.items():
        for m in regex.finditer(norm_text):
            rng = _covered_sentences(spans, m.start(), m.end())
            if rng is None:
                continue
            lo, hi = rng
            candidates.append({
                "lo": lo, "hi": hi,
                "raw_start": m.start(), "raw_end": m.end(),
                "pattern_name": f"S1_{name}",
                "match_text": m.group(0).strip(),
            })

    stream = _build_verb_stream(doc)
    # Stage 2 matches against the verb-tagged stream; we report the matched
    # *stream* substring as the example since back-mapping to raw character
    # offsets is brittle when spaCy and the original tokenizer differ. The
    # match text still reads clearly with `VERB` placeholders for agents.
    for name, regex in STAGE2_REGEXES.items():
        for m in regex.finditer(stream):
            # No reliable offset mapping into raw text; treat the entire
            # passage as the covered range so merge doesn't drop these,
            # and synthesize a sentence-level reference using the surrounding
            # context if the stream offset happens to fall inside a span.
            rng = _covered_sentences(spans, min(m.start(), len(norm_text) - 1),
                                            min(m.end(), len(norm_text)))
            if rng is None:
                # Stream-only match — record without sentence-mapping
                candidates.append({
                    "lo": -1, "hi": -1,
                    "raw_start": -1, "raw_end": -1,
                    "pattern_name": f"S2_{name}",
                    "match_text": m.group(0).strip(),
                })
            else:
                lo, hi = rng
                candidates.append({
                    "lo": lo, "hi": hi,
                    "raw_start": m.start(), "raw_end": m.end(),
                    "pattern_name": f"S2_{name}",
                    "match_text": m.group(0).strip(),
                })

    # Merge intervals for Stage-1-style mapped candidates only; stream-only
    # candidates (lo=-1) bypass merging since they have no mapped sentence.
    mapped = [c for c in candidates if c["lo"] >= 0]
    unmapped = [c for c in candidates if c["lo"] < 0]
    merged = _merge_intervals(mapped)
    final = merged + unmapped

    examples = [
        {
            "pattern": it["pattern_name"],
            "match": it["match_text"][:200],
        }
        for it in final[:8]
    ]
    return len(final), examples


# ─── Composite + report ────────────────────────────────────────────────────

# Empirical normalization ranges. Words/trigrams/contrast are anchored to
# the public slop-score leaderboard's observed model spread; bigrams are
# anchored to a calibration sweep over the project's voice-references corpus
# (DeepSeek-v4-pro, Sonnet-4-6, GPT-4o defaults vs. literary baselines).
# Reference distribution — composite-score quantiles across the project's
# voice-references corpus. Hard-coded from a one-shot calibration sweep
# (lab/scripts/slop_score_calibration_sweep.py); rerun and update if the
# corpus, lists, or weights change. Surfaced in every report so an agent
# can interpret its own score against concrete anchors instead of guessing
# what the 0-100 scale means.
REFERENCE_DISTRIBUTION: dict[str, dict] = {
    "human_literary":         {"n": 203, "median": 5.3,  "p90": 15.1},
    "human_amateur":          {"n":  13, "median": 8.3,  "p90": 25.1},
    "frontier_llm_default":   {"n":  34, "median": 17.3, "p90": 27.4},
    "older_llm_default":      {"n": 339, "median": 25.6, "p90": 56.2},
}


_NORM_RANGE_WORDS = (0.0, 50.0)        # per 1000 words
_NORM_RANGE_TRIGRAMS = (0.0, 1.5)      # per 1000 words
_NORM_RANGE_BIGRAMS = (0.0, 15.0)      # per 1000 words
_NORM_RANGE_CONTRAST = (0.0, 1.5)      # per 1000 chars

_WEIGHT_WORDS = 0.50
_WEIGHT_CONTRAST = 0.20
_WEIGHT_TRIGRAMS = 0.15
_WEIGHT_BIGRAMS = 0.15

# Threshold calibrated against the project's voice-references corpus:
#   DeepSeek-v4-pro defaults: median 17, p90 27
#   Sonnet (leaderboard-derived):  ~16
#   Older LLMs (Gemma, Llama-3, etc.): median 25, p90 55
#   Human literary exemplars:  median 5, p90 15
#   Human amateur (litbench/wp): median 7, p90 24
# 20 sits above frontier-default median, below the older-LLM median, and
# above the human-writing p90 — flags conspicuous slop without penalising
# decent frontier register.
SLOP_THRESHOLD = 20.0


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


@dataclass
class SlopScoreReport:
    """Compact tool response for an agent. Stays well under 4 kB serialized.

    `composite` is the 0-100 weighted score (higher = more slop). `components`
    has the raw per-1k rates so the agent can see which axis is the problem.
    `top_hits` provides up to ~10 concrete examples per axis for revision.
    `reference_distribution` gives the median + p90 of the composite across
    the project's calibration corpus so an agent can place its score against
    concrete anchors. `placement` is the closest matching bucket.
    `comparison` is None unless `compare_to` was passed; when populated, it
    holds the averaged reference score plus per-axis deltas (candidate minus
    reference) so an agent can see whether a revision moved the score.
    """
    composite: float
    is_slop: bool
    placement: str
    components: dict
    top_hits: dict
    reference_distribution: dict
    interpretation_notes: str
    comparison: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _placement(score: float) -> str:
    """Closest matching bucket from REFERENCE_DISTRIBUTION."""
    rd = REFERENCE_DISTRIBUTION
    if score <= rd["human_literary"]["median"]:
        return "below human-literary median"
    if score <= rd["human_literary"]["p90"]:
        return "within human-literary range"
    if score <= rd["human_amateur"]["p90"]:
        return "within human-amateur range"
    if score <= rd["frontier_llm_default"]["p90"]:
        return "within frontier-LLM-default range"
    if score <= rd["older_llm_default"]["p90"]:
        return "within older-LLM-default range"
    return "above older-LLM p90 — conspicuous slop"


def _interpretation(
    composite: float,
    sw_per_1k: float,
    sb_per_1k: float,
    st_per_1k: float,
    cx_per_1k_chars: float,
    placement: str,
    word_count: int,
    word_hits: dict[str, int],
    bi_hits: dict[str, int],
    tri_hits: dict[str, int],
    contrast_examples: list[dict],
    comparison: dict | None = None,
) -> str:
    """Per-axis dynamic commentary, keyed to calibration-corpus thresholds.

    Each axis gets its own sentence that translates the numeric value into
    a register characterisation, with concrete revision targets (top hits)
    where applicable. Silent on axes that fired no hits — no point telling
    the agent what *isn't* present. Closes with a meta-target when the
    composite exceeds the threshold.
    """
    def _top_n(hits: dict[str, int], n: int) -> str:
        items = sorted(hits.items(), key=lambda kv: -kv[1])[:n]
        return ", ".join(f'"{k}"' for k, _ in items)

    parts: list[str] = []

    # Headline: composite + threshold + corpus placement
    threshold_clause = "exceeds" if composite > SLOP_THRESHOLD else "is below"
    parts.append(
        f"Composite {composite:.1f}/100 {threshold_clause} the "
        f"{SLOP_THRESHOLD:.0f}-point threshold; sits {placement}."
    )

    # Slop words — 50% weight, the highest-leverage axis
    if sw_per_1k < 5:
        parts.append(
            f"Slop word rate {sw_per_1k:.1f}/1k is at human-literary level "
            f"— vocabulary is clean."
        )
    elif sw_per_1k < 14:
        parts.append(
            f"Slop word rate {sw_per_1k:.1f}/1k is in human-amateur / "
            f"frontier-LLM territory (frontier median 14, p90 23). Top "
            f"hits: {_top_n(word_hits, 3)} — common AI tells."
        )
    elif sw_per_1k < 23:
        parts.append(
            f"Slop word rate {sw_per_1k:.1f}/1k is at frontier-LLM level "
            f"(median 14, p90 23). Top hits: {_top_n(word_hits, 3)}."
        )
    else:
        parts.append(
            f"Slop word rate {sw_per_1k:.1f}/1k is above frontier-LLM p90 "
            f"(23) — vocabulary is conspicuously AI-default. Top hits: "
            f"{_top_n(word_hits, 4)}. Highest-leverage revision axis."
        )

    # Contrast patterns — 20% weight; Nous Research calls "not X but Y"
    # the single most overused LLM rhetorical structure.
    if cx_per_1k_chars > 0:
        ex = contrast_examples[0]["match"][:60] if contrast_examples else ""
        if cx_per_1k_chars < 0.3:
            parts.append(
                f"Contrast patterns at {cx_per_1k_chars:.2f}/1k chars "
                f"— within human range (literary p90 ≈ 0.3)."
            )
        else:
            parts.append(
                f"Contrast patterns at {cx_per_1k_chars:.2f}/1k chars "
                f"— above human p90 (0.3); e.g. \"{ex}\"."
            )

    # Trigrams — 15% weight, very diagnostic when present (LLM median ≈0)
    if st_per_1k > 0:
        parts.append(
            f"Slop trigram rate {st_per_1k:.2f}/1k (LLM median ≈ 0): "
            f"{_top_n(tri_hits, 2)} — fiction-cliché matches; high-priority "
            f"to revise."
        )

    # Bigrams — 15% weight, only call out when above human p90 (≈3.8)
    if sb_per_1k >= 4:
        parts.append(
            f"Slop bigram rate {sb_per_1k:.1f}/1k is above human p90 (≈3.8): "
            f"{_top_n(bi_hits, 3)}."
        )

    # Closing meta-statement, only when over threshold
    if composite > SLOP_THRESHOLD:
        # Identify the dominant weighted contributor for the agent
        contributions = [
            ("slop words", _normalize(sw_per_1k, *_NORM_RANGE_WORDS) * _WEIGHT_WORDS),
            ("contrast patterns", _normalize(cx_per_1k_chars, *_NORM_RANGE_CONTRAST) * _WEIGHT_CONTRAST),
            ("slop trigrams", _normalize(st_per_1k, *_NORM_RANGE_TRIGRAMS) * _WEIGHT_TRIGRAMS),
            ("slop bigrams", _normalize(sb_per_1k, *_NORM_RANGE_BIGRAMS) * _WEIGHT_BIGRAMS),
        ]
        top_axis = max(contributions, key=lambda x: x[1])[0]
        parts.append(
            f"Reducing {top_axis} will lower the composite fastest; that "
            f"axis is the dominant weighted contributor."
        )

    # Reliability caveat for short passages — slop rates are noisy at <200w
    if word_count < 200:
        parts.append(
            f"(Word count {word_count} is below 200 — per-1k rates are "
            f"noisy at this scale; treat axis values as advisory.)"
        )

    # Comparison delta — name the dominant axis change vs the reference
    if comparison is not None:
        delta_c = comparison["delta_composite"]
        direction = "above" if delta_c > 0 else "below"
        # Pick the axis with the largest weighted absolute delta
        axis_weights = {
            "slop_words_per_1k_words": (_NORM_RANGE_WORDS, _WEIGHT_WORDS, "slop words"),
            "contrast_patterns_per_1k_chars": (_NORM_RANGE_CONTRAST, _WEIGHT_CONTRAST, "contrast patterns"),
            "slop_trigrams_per_1k_words": (_NORM_RANGE_TRIGRAMS, _WEIGHT_TRIGRAMS, "slop trigrams"),
            "slop_bigrams_per_1k_words": (_NORM_RANGE_BIGRAMS, _WEIGHT_BIGRAMS, "slop bigrams"),
        }
        weighted_deltas = []
        for k, delta in comparison["delta_rates"].items():
            (lo, hi), w, name = axis_weights[k]
            denom = max(hi - lo, 1e-9)
            weighted_deltas.append((name, delta, abs(delta / denom * w)))
        top_name, top_delta, _ = max(weighted_deltas, key=lambda x: x[2])
        parts.append(
            f"Compared to {comparison['n_passages']}-passage reference "
            f"(composite {comparison['reference_composite']:.1f}): "
            f"{abs(delta_c):.1f} points {direction}; dominant change is "
            f"{top_name} ({top_delta:+.2f}/1k)."
        )

    return " ".join(parts)


def slop_score(
    passage: str,
    compare_to: str | list[str] | None = None,
) -> SlopScoreReport:
    """Compute the EQ-Bench slop score on `passage`.

    Returns a `SlopScoreReport` with the 0-100 composite, per-axis raw
    rates, top-N concrete hits per axis, and a short interpretation note.
    Designed as a stand-alone agent tool: deterministic, no API calls,
    sub-4kB JSON output.

    Args:
        passage: Prose to score. Empty strings return a zero-filled report.
        compare_to: Optional reference passage (or list of passages) to
            compare against. When provided, the report's `comparison` field
            holds the averaged reference score and per-axis deltas
            (candidate minus reference). Useful for revision tracking
            ("did v2 reduce slop vs v1?") or exemplar-targeting ("how much
            sloppier am I than this Hemingway passage?"). The agent can
            use `lookup_exemplar()` to retrieve corpus texts to pass here.
    """
    norm = _normalize_text(passage)
    nlp = get_nlp()
    doc = nlp(norm)

    tokens = _tokenize_words(norm)
    word_count = len(tokens)
    char_count = len(norm)

    if word_count == 0 or char_count == 0:
        return SlopScoreReport(
            composite=0.0, is_slop=False,
            placement="empty passage",
            components={
                "slop_words_per_1k_words": 0.0,
                "slop_bigrams_per_1k_words": 0.0,
                "slop_trigrams_per_1k_words": 0.0,
                "contrast_patterns_per_1k_chars": 0.0,
                "word_count": word_count, "char_count": char_count,
                "match_counts": {"slop_words": 0, "slop_bigrams": 0,
                                 "slop_trigrams": 0, "contrast_patterns": 0},
            },
            top_hits={"slop_words": [], "slop_bigrams": [],
                      "slop_trigrams": [], "contrast_patterns": []},
            reference_distribution=REFERENCE_DISTRIBUTION,
            interpretation_notes="Empty passage — no slop signal computed.",
        )

    word_hits_n, word_hits = _score_slop_words(tokens)
    bi_hits_n, bi_hits = _score_slop_bigrams(tokens)
    tri_hits_n, tri_hits = _score_slop_trigrams(tokens)
    contrast_n, contrast_examples = _score_contrast_patterns(norm, doc)

    sw_per_1k = (word_hits_n / word_count) * 1000.0
    sb_per_1k = (bi_hits_n / word_count) * 1000.0
    st_per_1k = (tri_hits_n / word_count) * 1000.0
    cx_per_1k_chars = (contrast_n / char_count) * 1000.0

    composite = (
        _normalize(sw_per_1k, *_NORM_RANGE_WORDS) * _WEIGHT_WORDS
        + _normalize(cx_per_1k_chars, *_NORM_RANGE_CONTRAST) * _WEIGHT_CONTRAST
        + _normalize(st_per_1k, *_NORM_RANGE_TRIGRAMS) * _WEIGHT_TRIGRAMS
        + _normalize(sb_per_1k, *_NORM_RANGE_BIGRAMS) * _WEIGHT_BIGRAMS
    ) * 100.0

    top_word_hits = sorted(word_hits.items(), key=lambda kv: -kv[1])[:10]
    top_bi_hits = sorted(bi_hits.items(), key=lambda kv: -kv[1])[:10]
    top_tri_hits = sorted(tri_hits.items(), key=lambda kv: -kv[1])[:10]

    components = {
        "slop_words_per_1k_words": round(sw_per_1k, 2),
        "slop_bigrams_per_1k_words": round(sb_per_1k, 2),
        "slop_trigrams_per_1k_words": round(st_per_1k, 2),
        "contrast_patterns_per_1k_chars": round(cx_per_1k_chars, 3),
        "word_count": word_count,
        "char_count": char_count,
        "match_counts": {
            "slop_words": word_hits_n,
            "slop_bigrams": bi_hits_n,
            "slop_trigrams": tri_hits_n,
            "contrast_patterns": contrast_n,
        },
    }

    top_hits = {
        "slop_words": [[w, c] for w, c in top_word_hits],
        "slop_bigrams": [[b, c] for b, c in top_bi_hits],
        "slop_trigrams": [[t, c] for t, c in top_tri_hits],
        "contrast_patterns": contrast_examples,
    }

    placement = _placement(composite)

    # Comparison against one or more reference passages (revision tracking
    # or exemplar-targeting). Recursive call passes compare_to=None so we
    # don't loop. Multiple references are averaged into a single anchor.
    comparison_data = None
    if compare_to is not None:
        refs = [compare_to] if isinstance(compare_to, str) else list(compare_to)
        ref_reports = [slop_score(t) for t in refs]
        rate_keys = [
            "slop_words_per_1k_words",
            "slop_bigrams_per_1k_words",
            "slop_trigrams_per_1k_words",
            "contrast_patterns_per_1k_chars",
        ]
        avg_composite = sum(r.composite for r in ref_reports) / len(ref_reports)
        avg_rates = {
            k: sum(r.components[k] for r in ref_reports) / len(ref_reports)
            for k in rate_keys
        }
        comparison_data = {
            "n_passages": len(ref_reports),
            "reference_composite": round(avg_composite, 2),
            "reference_rates": {k: round(v, 3) for k, v in avg_rates.items()},
            "delta_composite": round(composite - avg_composite, 2),
            "delta_rates": {
                k: round(components[k] - avg_rates[k], 3) for k in rate_keys
            },
        }

    notes = _interpretation(
        composite, sw_per_1k, sb_per_1k, st_per_1k, cx_per_1k_chars,
        placement, word_count, word_hits, bi_hits, tri_hits, contrast_examples,
        comparison_data,
    )

    return SlopScoreReport(
        composite=round(composite, 2),
        is_slop=composite > SLOP_THRESHOLD,
        placement=placement,
        components=components,
        top_hits=top_hits,
        reference_distribution=REFERENCE_DISTRIBUTION,
        interpretation_notes=notes,
        comparison=comparison_data,
    )
