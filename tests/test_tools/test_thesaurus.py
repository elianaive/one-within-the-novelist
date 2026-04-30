"""Tests for owtn.tools.thesaurus — Datamuse wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from owtn.tools import thesaurus
from owtn.tools.thesaurus import (
    DATAMUSE_URL,
    MODE_TO_PARAM,
    clear_cache,
)


# ─── Fake httpx client ────────────────────────────────────────────────────

@dataclass
class FakeResponse:
    payload: list[dict]

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


class FakeClient:
    """Stand-in for httpx.Client that records requests and returns a scripted
    payload. Tests inject the client directly to avoid network access."""

    def __init__(self, payload: list[dict]):
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict) -> FakeResponse:
        self.calls.append((url, dict(params)))
        return FakeResponse(self.payload)

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


# ─── Mode handling ────────────────────────────────────────────────────────

class TestModeMapping:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="unknown thesaurus mode"):
            thesaurus("happy", mode="not_a_mode")

    @pytest.mark.parametrize("mode,param", MODE_TO_PARAM.items())
    def test_mode_maps_to_datamuse_param(self, mode, param):
        client = FakeClient(payload=[])
        thesaurus("happy", mode=mode, client=client)
        assert client.calls[0][0] == DATAMUSE_URL
        assert param in client.calls[0][1]
        assert client.calls[0][1][param] == "happy"


# ─── Empty / whitespace input ─────────────────────────────────────────────

class TestEmptyInput:
    def test_empty_word_returns_empty_report_without_request(self):
        client = FakeClient(payload=[{"word": "x", "score": 1}])
        report = thesaurus("", client=client)
        assert report.n_results == 0
        assert report.results == []
        assert client.calls == []

    def test_whitespace_only_returns_empty(self):
        client = FakeClient(payload=[])
        report = thesaurus("   ", client=client)
        assert report.n_results == 0
        assert client.calls == []


# ─── Result trimming ──────────────────────────────────────────────────────

class TestResultShape:
    def test_trims_to_word_score_tags(self):
        # Datamuse can return defHeadword, defs, numSyllables, etc. — drop them.
        client = FakeClient(payload=[
            {"word": "pleased", "score": 40004395, "tags": ["syn", "adj"],
             "defHeadword": "x", "numSyllables": 2},
            {"word": "glad", "score": 40003904, "tags": ["syn"]},
        ])
        report = thesaurus("happy", client=client)
        assert report.n_results == 2
        for row in report.results:
            assert set(row.keys()) <= {"word", "score", "tags"}
        assert report.results[0]["word"] == "pleased"

    def test_missing_tags_field_is_dropped(self):
        client = FakeClient(payload=[{"word": "glad", "score": 100}])
        report = thesaurus("happy", client=client)
        assert "tags" not in report.results[0]

    def test_to_dict_round_trips_through_json(self):
        client = FakeClient(payload=[{"word": "glad", "score": 100, "tags": ["syn"]}])
        report = thesaurus("happy", client=client)
        d = report.to_dict()
        assert d["mode"] == "means_like"
        assert d["n_results"] == 1
        # JSON-serializable
        json.dumps(d)


# ─── Caching ──────────────────────────────────────────────────────────────

class TestCache:
    def test_repeat_call_skips_network(self):
        client = FakeClient(payload=[{"word": "glad", "score": 100}])
        thesaurus("happy", client=client)
        thesaurus("happy", client=client)
        assert len(client.calls) == 1

    def test_case_and_whitespace_collapse_to_same_key(self):
        client = FakeClient(payload=[{"word": "glad", "score": 100}])
        thesaurus("happy", client=client)
        thesaurus("  HAPPY  ", client=client)
        assert len(client.calls) == 1

    def test_distinct_modes_are_separate_cache_entries(self):
        client = FakeClient(payload=[{"word": "x", "score": 1}])
        thesaurus("happy", mode="means_like", client=client)
        thesaurus("happy", mode="antonyms", client=client)
        assert len(client.calls) == 2

    def test_distinct_max_results_are_separate_cache_entries(self):
        client = FakeClient(payload=[{"word": "x", "score": 1}])
        thesaurus("happy", max_results=10, client=client)
        thesaurus("happy", max_results=20, client=client)
        assert len(client.calls) == 2

    def test_clear_cache_forces_refetch(self):
        client = FakeClient(payload=[{"word": "x", "score": 1}])
        thesaurus("happy", client=client)
        clear_cache()
        thesaurus("happy", client=client)
        assert len(client.calls) == 2


# ─── max_results plumbing ─────────────────────────────────────────────────

class TestMaxResultsParam:
    def test_max_param_passed_to_datamuse(self):
        client = FakeClient(payload=[])
        thesaurus("happy", max_results=5, client=client)
        assert client.calls[0][1]["max"] == "5"


# ─── Live API smoke ───────────────────────────────────────────────────────

@pytest.mark.live_api
def test_live_datamuse_synonyms():
    """Real Datamuse round-trip — confirms wire format hasn't drifted."""
    report = thesaurus("happy", mode="means_like", max_results=5)
    assert report.n_results > 0
    assert all("word" in r for r in report.results)
