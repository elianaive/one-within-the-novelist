"""Tests for owtn.tools.lookup_exemplar — sync deterministic + async NL paths.

The async path is what voice agents actually call. We mock the resolver
subagent for unit tests (no live LLM dependency) and verify each match
mode (intersect / authors_only / tags_only / none) flows through the
deterministic fetch correctly. A `live_api`-marked smoke test exercises
the real resolver against the gap log from `run_20260429_131414`.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from owtn.tools import _corpus as corpus_module
from owtn.tools._corpus import load_corpus
from owtn.tools._lookup_resolver import LookupResolution, reset_catalog_cache
from owtn.tools.lookup_exemplar import lookup_exemplar, lookup_exemplar_async


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def lookup_corpus(tmp_path, monkeypatch):
    """Tiny synthetic corpus shaped like the real one — a couple of authors
    with multiple works and overlapping tags so we can test intersection."""
    passages_dir = tmp_path / "passages"
    passages_dir.mkdir()

    # (id, tags, category-or-None, text)
    refs = [
        ("morrison-beloved-s0",
         ["literary", "lyric_african_american_modernism", "third_person", "polyphonic"],
         None,
         "She was crawling already when I got there. One mile of dust road between her and me."),
        ("morrison-song-of-solomon-s0",
         ["literary", "lyric_african_american_modernism", "third_person", "incantatory"],
         None,
         "The North Carolina Mutual Life Insurance agent promised to fly from Mercy to the other side of Lake Superior at three o'clock."),
        ("saunders-pastoralia-s0",
         ["literary", "register_collision", "third_person", "satirical"],
         None,
         "Janet's looking better. She's stopped the meds and is back to being herself, mostly. The shift bell rings."),
        ("austen-emma-s0",
         ["literary", "comic_realism", "free_indirect_discourse", "third_person"],
         None,
         "Emma Woodhouse, handsome, clever, and rich, with a comfortable home and happy disposition."),
        ("scp-173",
         ["procedural_bureaucratic", "institutional_voice", "containment_doc"],
         "exemplars",
         "Item #: SCP-173. Object Class: Euclid. Special Containment Procedures: Item SCP-173 is to be kept."),
        ("llm-sample-1",
         ["llm_default", "model:test"],
         None,
         "She walked through the morning light, feeling the weight of everything she had been carrying."),
        ("expo-1",
         ["expository"],
         None,
         "Photosynthesis converts light energy into chemical energy via the Calvin cycle."),
    ]
    for eid, _, _, text in refs:
        (passages_dir / f"{eid}.txt").write_text(text)

    yaml_data = {
        "references": [
            {
                "id": eid, "tags": tags, "source": f"test source for {eid}",
                "license": "test", "text_file": f"passages/{eid}.txt",
                **({"category": cat} if cat else {}),
            }
            for eid, tags, cat, _ in refs
        ]
    }
    yaml_path = tmp_path / "voice-references-stylometric.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_data))

    monkeypatch.setattr(corpus_module, "CORPUS_YAML", yaml_path)
    monkeypatch.setattr(corpus_module, "PASSAGES_DIR", tmp_path)
    monkeypatch.setattr(corpus_module, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(corpus_module, "_CORPUS_CACHE", None)
    reset_catalog_cache()

    return load_corpus(force_reload=True)


def _resolution(**overrides) -> LookupResolution:
    base = {
        "interpretation": "test",
        "authors": [],
        "tags": [],
        "match": "none",
        "note": "test note",
    }
    base.update(overrides)
    return LookupResolution(**base)


# ─── Sync deterministic path ─────────────────────────────────────────────


class TestSyncLookup:
    def test_exact_id_returns_entry(self, lookup_corpus):
        r = lookup_exemplar("morrison-beloved-s0", corpus=lookup_corpus)
        assert r["match"] == "id"
        assert r["passages"][0]["id"] == "morrison-beloved-s0"

    def test_default_id_blocked(self, lookup_corpus):
        r = lookup_exemplar("llm-sample-1", corpus=lookup_corpus)
        assert r["match"] == "blocked"
        assert r["n_returned"] == 0

    def test_author_slug(self, lookup_corpus):
        r = lookup_exemplar("morrison", n=5, corpus=lookup_corpus)
        assert r["match"] == "authors_only"
        assert r["n_returned"] == 2
        assert {p["id"] for p in r["passages"]} == {
            "morrison-beloved-s0", "morrison-song-of-solomon-s0",
        }

    def test_tag(self, lookup_corpus):
        r = lookup_exemplar("institutional_voice", corpus=lookup_corpus)
        assert r["match"] == "tags_only"
        assert r["passages"][0]["id"] == "scp-173"

    def test_unknown_query(self, lookup_corpus):
        r = lookup_exemplar("does-not-exist", corpus=lookup_corpus)
        assert r["match"] == "none"
        assert r["n_returned"] == 0


# ─── Async NL path with mocked resolver ──────────────────────────────────


@pytest.mark.asyncio
class TestAsyncLookup:
    """Each test mocks `resolve_query_async` to return a specific
    LookupResolution, then checks the deterministic fetch picks the right
    entries. The fast-path tests use a query that's an exact id and never
    invoke the resolver mock."""

    async def test_entry_id_fast_path_skips_resolver(self, lookup_corpus):
        called = False

        async def boom(*a, **kw):
            nonlocal called
            called = True
            return _resolution()

        with patch("owtn.tools.lookup_exemplar.resolve_query_async", side_effect=boom):
            r = await lookup_exemplar_async(
                "morrison-beloved-s0", corpus=lookup_corpus,
            )
        assert r["match"] == "id"
        assert not called, "fast-path must not call the resolver"

    async def test_intersect_returns_strict_subset(self, lookup_corpus):
        # Morrison + incantatory → only morrison-song-of-solomon-s0 has both.
        r = await lookup_exemplar_async(
            "Morrison's incantatory mode",
            corpus=lookup_corpus,
            resolution=_resolution(
                authors=["morrison"], tags=["incantatory"], match="intersect",
                interpretation="Morrison + incantatory tag",
            ),
        )
        assert r["match"] == "intersect"
        assert r["n_returned"] == 1
        assert r["passages"][0]["id"] == "morrison-song-of-solomon-s0"

    async def test_intersect_falls_through_when_empty(self, lookup_corpus):
        # Morrison + free_indirect_discourse → no Morrison entry has FID.
        # Should fall through to authors-only (both Morrison entries).
        r = await lookup_exemplar_async(
            "Morrison's free indirect discourse",
            corpus=lookup_corpus,
            resolution=_resolution(
                authors=["morrison"], tags=["free_indirect_discourse"], match="intersect",
            ),
        )
        assert r["match"] == "intersect"
        assert r["n_returned"] == 2

    async def test_authors_only(self, lookup_corpus):
        r = await lookup_exemplar_async(
            "Saunders",
            corpus=lookup_corpus,
            resolution=_resolution(
                authors=["saunders"], match="authors_only",
            ),
        )
        assert r["match"] == "authors_only"
        assert r["passages"][0]["id"] == "saunders-pastoralia-s0"

    async def test_tags_only(self, lookup_corpus):
        r = await lookup_exemplar_async(
            "second-person address",
            corpus=lookup_corpus,
            resolution=_resolution(
                tags=["institutional_voice"], match="tags_only",
            ),
        )
        assert r["match"] == "tags_only"
        assert r["passages"][0]["id"] == "scp-173"

    async def test_none_passes_resolver_note(self, lookup_corpus):
        r = await lookup_exemplar_async(
            "Beckett's reductive monologue",
            corpus=lookup_corpus,
            resolution=_resolution(
                match="none",
                interpretation="agent reaching for an absent author",
                note="no Beckett entries; closest: morrison (incantatory).",
            ),
        )
        assert r["match"] == "none"
        assert "Beckett" in r["note"]
        assert r["n_returned"] == 0

    async def test_resolver_filters_out_of_catalog_keys(self, lookup_corpus):
        """If the resolver returns slugs/tags absent from the catalog, the
        validator drops them so the deterministic fetcher never sees
        hallucinated keys."""
        from owtn.llm.result import QueryResult
        from owtn.tools._lookup_resolver import resolve_query_async

        async def fake(**kwargs):
            return QueryResult(
                content=LookupResolution(
                    interpretation="x",
                    authors=["morrison", "made_up_author"],
                    tags=["incantatory", "made_up_tag"],
                    match="intersect", note="x",
                ),
                msg="", system_msg="", new_msg_history=[],
                model_name="test", kwargs={},
                input_tokens=0, output_tokens=0,
            )

        with patch("owtn.llm.api.query_async", side_effect=fake):
            res = await resolve_query_async("Morrison's incantatory mode", corpus=lookup_corpus)

        assert res.authors == ["morrison"]
        assert res.tags == ["incantatory"]


# ─── Catalog construction ────────────────────────────────────────────────


class TestCatalog:
    def test_excludes_defaults(self, lookup_corpus):
        from owtn.tools._lookup_resolver import get_catalog
        cat = get_catalog(lookup_corpus)
        assert "llm-sample-1" not in cat.system_prefix
        # Reachable entries should be present.
        assert "morrison-beloved-s0" in cat.system_prefix

    def test_authors_header_lists_literary_slugs(self, lookup_corpus):
        from owtn.tools._lookup_resolver import get_catalog
        cat = get_catalog(lookup_corpus)
        assert "morrison" in cat.valid_authors
        assert "saunders" in cat.valid_authors
        assert "austen" in cat.valid_authors
        # SCP isn't an author concept — no slug for non-literary entries.
        assert "scp" not in cat.valid_authors



# ─── Live-API smoke test (the gap-log pilot) ─────────────────────────────


# 17 queries from results/run_20260429_131414/.../corpus_gaps.jsonl.
# Each query and what we expect — false gaps should now resolve, real
# gaps should still come back as `match: none` with sensible notes.
LIVE_PILOT_CASES = [
    # False gaps in the original run — content exists, slug interface couldn't reach it.
    ("morrison_toni",                     "should_match",  ["morrison"]),
    ("toni_morrison",                     "should_match",  ["morrison"]),
    ("toni_morrison_incantation",         "should_match",  ["morrison"]),
    ("saunders_george",                   "should_match",  ["saunders"]),
    ("sebald_wg",                         "should_match",  ["sebald"]),
    ("incantation",                       "should_match",  None),  # tag near-miss → incantatory
    ("third_person_close",                "should_match",  None),  # tag near-miss → third_person
    # Real gaps — should report `match: none` cleanly.
    ("beckett",                           "should_be_none", None),
    ("beckett_company",                   "should_be_none", None),
    ("legal_prose",                       "should_be_none", None),
    ("present_tense",                     "should_be_none", None),
    # Agent confusions (not corpus gaps).
    ("deodand",                           "should_be_none", None),
    ("saunders_timing",                   "should_match",   ["saunders"]),
]


@pytest.mark.live_api
@pytest.mark.asyncio
@pytest.mark.parametrize("query, expectation, expected_authors", LIVE_PILOT_CASES)
async def test_live_pilot_resolution(query, expectation, expected_authors):
    """End-to-end against the real haiku resolver. Verifies the gap log
    from `run_20260429_131414` now resolves correctly. Costs <$0.01 total."""
    r = await lookup_exemplar_async(query)
    if expectation == "should_match":
        assert r["match"] != "none", f"{query!r}: expected match, got note: {r.get('note')!r}"
        assert r["n_returned"] > 0
        if expected_authors:
            for a in expected_authors:
                assert a in r["authors"], f"{query!r}: expected author {a} in {r['authors']}"
    else:
        assert r["match"] == "none", f"{query!r}: expected none, got match={r['match']}"
        assert r["note"], "must surface a useful note"
