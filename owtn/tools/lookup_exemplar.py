"""Lookup tool — retrieve specific reference passages from the corpus.

Two surfaces:
  - `lookup_exemplar(query, ...)`         sync, deterministic (id/author/tag).
  - `lookup_exemplar_async(query, ...)`   async, NL via the haiku resolver.

The async path is what voice agents call — it accepts natural-language
queries ("Morrison's incantatory mode") and dispatches to a fast resolver
that translates intent into existing corpus keys. The sync path is for
tests and the CLI; it does plain key-based resolution without an LLM call.

Both paths return the same dict shape and both restrict to `exemplars` +
`baselines` categories. LLM defaults are blocked — reading them would just
teach the agent to imitate the register it's trying to escape.
"""

from __future__ import annotations

from typing import Optional

from ._corpus import ReferenceCorpus, ReferenceEntry, load_corpus
from ._lookup_resolver import LookupResolution, resolve_query_async


# ─── Result shape ────────────────────────────────────────────────────────


def _format_passage(entry: ReferenceEntry, max_words: int) -> dict:
    words = entry.text.split()
    truncated = len(words) > max_words
    text = " ".join(words[:max_words]) + (" […]" if truncated else "")
    return {
        "id": entry.id,
        "category": entry.resolved_category,
        "source": entry.source,
        "tags": entry.tags,
        "text": text,
        "truncated": truncated,
        "word_count": len(words),
    }


def _empty_result(query: str, match: str, note: str, **extra) -> dict:
    out = {
        "query": query,
        "match": match,
        "interpretation": "",
        "authors": [],
        "tags": [],
        "n_returned": 0,
        "n_available": 0,
        "passages": [],
        "note": note,
    }
    out.update(extra)
    return out


def _format_result(
    *,
    query: str,
    match: str,
    interpretation: str,
    authors: list[str],
    tags: list[str],
    candidates: list[ReferenceEntry],
    n: int,
    max_words: int,
    note: str,
) -> dict:
    """Sort candidates by id, take n, format passages, return the standard shape."""
    candidates = sorted(candidates, key=lambda e: e.id)
    picks = candidates[:n]
    return {
        "query": query,
        "match": match,
        "interpretation": interpretation,
        "authors": authors,
        "tags": tags,
        "n_returned": len(picks),
        "n_available": len(candidates),
        "passages": [_format_passage(e, max_words) for e in picks],
        "note": note,
    }


# ─── Entry-id fast path ──────────────────────────────────────────────────


def _entry_by_id(query: str, corpus: ReferenceCorpus) -> Optional[ReferenceEntry]:
    for e in corpus.entries:
        if e.id == query:
            return e
    return None


def _id_lookup_result(
    entry: ReferenceEntry, query: str, max_words: int,
) -> dict:
    """Result for an exact-id match. Blocks defaults explicitly so an agent
    can't reach an LLM-default sample by guessing its id."""
    cat = entry.resolved_category
    if cat not in ("exemplars", "baselines"):
        return _empty_result(
            query, match="blocked",
            note=(
                f"entry {query!r} is in `{cat}` (LLM-default register). Not "
                "lookup-accessible — reading it would teach the agent to imitate "
                "the register it's trying to escape."
            ),
        )
    return _format_result(
        query=query,
        match="id",
        interpretation=f"Direct entry-id lookup for {query!r}.",
        authors=[],
        tags=[],
        candidates=[entry],
        n=1,
        max_words=max_words,
        note="Exact entry-id match.",
    )


# ─── Author/tag fetch ────────────────────────────────────────────────────


def _entries_for(
    *,
    corpus: ReferenceCorpus,
    authors: list[str],
    tags: list[str],
    mode: str,
) -> list[ReferenceEntry]:
    """Apply the resolver's match mode against the corpus.

    `intersect` falls through to author-only when the strict intersection
    is empty — better to return *something* by the named author than nothing,
    and the resolver's note already explains the substitution.
    """
    pool = corpus.reachable_entries()
    if mode == "intersect":
        strict = [
            e for e in pool
            if e.author_slug in authors and any(t in e.tags for t in tags)
        ]
        if strict:
            return strict
        # Fall through: empty intersection, return author-only.
        return [e for e in pool if e.author_slug in authors]
    if mode == "authors_only":
        return [e for e in pool if e.author_slug in authors]
    if mode == "tags_only":
        return [e for e in pool if any(t in e.tags for t in tags)]
    return []  # `none`


# ─── Sync deterministic path (tests + CLI) ───────────────────────────────


def lookup_exemplar(
    query: str,
    n: int = 2,
    max_words: int = 400,
    corpus: ReferenceCorpus | None = None,
) -> dict:
    """Deterministic lookup — exact id, then author slug, then tag.

    No LLM round-trip. Used by tests and the CLI. Voice agents go through
    `lookup_exemplar_async` instead, which adds the NL resolver in front.
    """
    corpus = corpus or load_corpus()

    # 1. Exact id
    entry = _entry_by_id(query, corpus)
    if entry is not None:
        return _id_lookup_result(entry, query, max_words)

    # 2. Author slug
    pool = corpus.reachable_entries()
    pool_ids = {e.id for e in pool}
    author_hits = [e for e in corpus.by_author(query) if e.id in pool_ids]
    if author_hits:
        return _format_result(
            query=query,
            match="authors_only",
            interpretation=f"Author-slug lookup for {query!r}.",
            authors=[query],
            tags=[],
            candidates=author_hits,
            n=n,
            max_words=max_words,
            note=f"{len(author_hits)} entries by author slug {query!r}.",
        )

    # 3. Tag
    tag_hits = [e for e in pool if query in e.tags]
    if tag_hits:
        return _format_result(
            query=query,
            match="tags_only",
            interpretation=f"Tag lookup for {query!r}.",
            authors=[],
            tags=[query],
            candidates=tag_hits,
            n=n,
            max_words=max_words,
            note=f"{len(tag_hits)} entries tagged {query!r}.",
        )

    return _empty_result(
        query, match="none",
        note=f"no entry id, author slug, or tag matches {query!r} in this corpus.",
    )


# ─── Async NL path (voice agents) ────────────────────────────────────────


async def lookup_exemplar_async(
    query: str,
    n: int = 2,
    max_words: int = 400,
    corpus: ReferenceCorpus | None = None,
    resolution: LookupResolution | None = None,
) -> dict:
    """Natural-language lookup. Pipeline:

      1. If `query` exactly matches an entry id, fast-path to that entry
         (no LLM call). Agents see entry ids in prior results, so re-fetching
         a known passage shouldn't pay the resolver tax.
      2. Otherwise, call the haiku resolver to translate intent into
         author slugs + tags.
      3. Apply the resolver's `match` mode against the corpus and return.

    `resolution` parameter is for tests — pass a pre-built LookupResolution
    to skip the LLM call. Production callers leave it None.
    """
    corpus = corpus or load_corpus()

    entry = _entry_by_id(query, corpus)
    if entry is not None:
        return _id_lookup_result(entry, query, max_words)

    if resolution is None:
        resolution = await resolve_query_async(query, corpus=corpus)

    if resolution.match == "none":
        return _empty_result(
            query, match="none",
            interpretation=resolution.interpretation,
            note=resolution.note,
        )

    candidates = _entries_for(
        corpus=corpus,
        authors=resolution.authors,
        tags=resolution.tags,
        mode=resolution.match,
    )
    if not candidates:
        return _empty_result(
            query, match="none",
            interpretation=resolution.interpretation,
            note=(resolution.note + " (no entries matched the resolved keys).").strip(),
        )

    return _format_result(
        query=query,
        match=resolution.match,
        interpretation=resolution.interpretation,
        authors=list(resolution.authors),
        tags=list(resolution.tags),
        candidates=candidates,
        n=n,
        max_words=max_words,
        note=resolution.note,
    )
