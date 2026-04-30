"""Thesaurus tool — Datamuse API wrapper for word-level diction work.

Lexical lookup (synonyms, phonetic neighbors, related words, adjective/noun
pairings, antonyms) backing diction revision in voice sessions. Free, no
API key. Queries are deterministic per `(word, mode, max_results)`, so we
cache responses in-process and elide repeat round-trips during a session.

Modes — each maps to a single Datamuse query parameter:

| Mode             | Datamuse | Use                                    |
|------------------|----------|----------------------------------------|
| `means_like`     | `ml`     | synonyms / semantic neighbours         |
| `sounds_like`    | `sl`     | phonetic similarity (sound work)       |
| `related_to`     | `rel_trg`| words that often co-occur              |
| `adjective_for`  | `rel_jjb`| adjectives commonly modifying a noun   |
| `noun_for`       | `rel_jja`| nouns commonly modified by an adjective|
| `antonyms`       | `rel_ant`| opposites                              |

The doc-table `topics` mode is omitted: in Datamuse semantics `topics` is a
filter that piggybacks on another query (e.g. `?ml=word&topics=ocean`),
not a standalone lookup. Add it back when an agent needs paired filtering.

Reference: https://www.datamuse.com/api/
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import httpx


DATAMUSE_URL = "https://api.datamuse.com/words"

MODE_TO_PARAM: dict[str, str] = {
    "means_like": "ml",
    "sounds_like": "sl",
    "related_to": "rel_trg",
    "adjective_for": "rel_jjb",
    "noun_for": "rel_jja",
    "antonyms": "rel_ant",
}

DEFAULT_MAX_RESULTS = 20

# (word_lower, mode, max_results) → trimmed result list. Datamuse responses
# are deterministic; a process-scoped cache eliminates repeat traffic when
# an agent revises diction across multiple turns.
_CACHE: dict[tuple[str, str, int], list[dict]] = {}


@dataclass
class ThesaurusReport:
    """Compact tool response. `results` holds the Datamuse rows trimmed to
    `word`, `score`, and any POS / synonym `tags` Datamuse attaches —
    enough for an agent to pick a candidate, drop the noisy internal fields.
    """
    word: str
    mode: str
    n_results: int
    results: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def thesaurus(
    word: str,
    mode: str = "means_like",
    max_results: int = DEFAULT_MAX_RESULTS,
    *,
    client: httpx.Client | None = None,
) -> ThesaurusReport:
    """Query Datamuse for words related to `word` under the given `mode`.

    Args:
        word: Query word. Whitespace-stripped and lowercased before lookup.
        mode: One of the keys in `MODE_TO_PARAM`. See module docstring.
        max_results: Cap on returned entries (Datamuse `max` parameter).
        client: Optional `httpx.Client` for connection reuse or tests.
            When omitted, a one-shot client is created and closed.

    Returns:
        ThesaurusReport with `results = [{word, score, tags?}, ...]`
        ordered by Datamuse's relevance score (highest first).
    """
    if mode not in MODE_TO_PARAM:
        raise ValueError(
            f"unknown thesaurus mode {mode!r}; "
            f"valid modes: {sorted(MODE_TO_PARAM)}"
        )

    word_norm = word.strip().lower()
    if not word_norm:
        return ThesaurusReport(word=word, mode=mode, n_results=0, results=[])

    cache_key = (word_norm, mode, max_results)
    if cache_key in _CACHE:
        cached = _CACHE[cache_key]
        return ThesaurusReport(
            word=word_norm, mode=mode, n_results=len(cached), results=cached,
        )

    params = {MODE_TO_PARAM[mode]: word_norm, "max": str(max_results)}
    owns_client = client is None
    client = client or httpx.Client(timeout=10.0)
    try:
        resp = client.get(DATAMUSE_URL, params=params)
        resp.raise_for_status()
        raw = resp.json()
    finally:
        if owns_client:
            client.close()

    results = [
        {k: row[k] for k in ("word", "score", "tags") if k in row}
        for row in raw
    ]
    _CACHE[cache_key] = results
    return ThesaurusReport(
        word=word_norm, mode=mode, n_results=len(results), results=results,
    )


def clear_cache() -> None:
    """Drop the in-process Datamuse response cache. For tests and CLI."""
    _CACHE.clear()
