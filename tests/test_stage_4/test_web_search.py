"""Tests for owtn.tools.web_search — Exa wrapper with mocked HTTP."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from owtn.tools.web_search import SearchHit, SearchResult, web_search


# ─── Helpers ─────────────────────────────────────────────────────────────


class _MockResponse:
    def __init__(self, status_code: int, json_body: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json = json_body or {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if 400 <= self.status_code:
            raise httpx.HTTPStatusError(
                f"Server returned {self.status_code}",
                request=httpx.Request("POST", "https://api.exa.ai/search"),
                response=httpx.Response(self.status_code, text=self.text),
            )


class _MockClient:
    """Minimal AsyncClient stand-in. Returns the fixture response on
    `.post(...)`."""
    def __init__(self, response: _MockResponse):
        self._response = response
        self.last_request: dict | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def post(self, url, *, json=None, headers=None):
        self.last_request = {"url": url, "json": json, "headers": headers}
        return self._response


# ─── No API key — graceful degradation ──────────────────────────────────


@pytest.mark.asyncio
async def test_returns_empty_with_note_when_no_api_key(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    result = await web_search("quantum optics measurement")
    assert isinstance(result, SearchResult)
    assert result.hits == []
    assert "EXA_API_KEY" in result.note


@pytest.mark.asyncio
async def test_returns_empty_with_note_on_empty_query():
    result = await web_search("   ")
    assert result.hits == []
    assert "empty query" in result.note


# ─── Successful Exa response ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parses_exa_results_into_search_hits():
    body = {
        "results": [
            {
                "title": "Measurement back-action in optical interferometers",
                "url": "https://arxiv.org/abs/2101.01234",
                "text": "Measurement back-action arises when the act of observation perturbs the system being observed.",
                "publishedDate": "2021-01-15",
            },
            {
                "title": "Quantum optics primer",
                "url": "https://example.org/primer",
                "text": "An introduction to quantum optics for physicists.",
            },
        ]
    }
    mock_client = _MockClient(_MockResponse(200, json_body=body))
    with patch("owtn.tools.web_search.httpx.AsyncClient", return_value=mock_client):
        result = await web_search("quantum optics", n=2, api_key="fake-key-for-testing")

    assert result.note == ""
    assert len(result.hits) == 2
    assert result.hits[0].title.startswith("Measurement back-action")
    assert result.hits[0].url == "https://arxiv.org/abs/2101.01234"
    assert "back-action" in result.hits[0].snippet
    assert result.hits[0].published == "2021-01-15"
    # Request payload was constructed correctly
    assert mock_client.last_request["headers"]["x-api-key"] == "fake-key-for-testing"
    assert mock_client.last_request["json"]["query"] == "quantum optics"
    assert mock_client.last_request["json"]["num_results"] == 2


@pytest.mark.asyncio
async def test_truncates_to_n_results():
    body = {
        "results": [
            {"title": f"Result {i}", "url": f"https://example.com/{i}"}
            for i in range(20)
        ]
    }
    mock_client = _MockClient(_MockResponse(200, json_body=body))
    with patch("owtn.tools.web_search.httpx.AsyncClient", return_value=mock_client):
        result = await web_search("anything", n=3, api_key="fake")
    assert len(result.hits) == 3


@pytest.mark.asyncio
async def test_clamps_n_to_max_20():
    """The wrapper's `num_results` is clamped to [1, 20] before sending."""
    mock_client = _MockClient(_MockResponse(200, json_body={"results": []}))
    with patch("owtn.tools.web_search.httpx.AsyncClient", return_value=mock_client):
        await web_search("x", n=999, api_key="fake")
    assert mock_client.last_request["json"]["num_results"] == 20


# ─── HTTP errors ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_http_error_returns_note_with_status():
    mock_client = _MockClient(_MockResponse(401, text="invalid api key"))
    with patch("owtn.tools.web_search.httpx.AsyncClient", return_value=mock_client):
        result = await web_search("anything", api_key="bad-key")
    assert result.hits == []
    assert "401" in result.note


@pytest.mark.asyncio
async def test_request_error_returns_note():
    """Network/connect errors land in the note, not as exceptions."""
    class _RaisingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("connection refused")

    with patch("owtn.tools.web_search.httpx.AsyncClient", return_value=_RaisingClient()):
        result = await web_search("anything", api_key="fake")
    assert result.hits == []
    assert "ConnectError" in result.note


# ─── Snippet truncation ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_long_snippet_capped_at_600_chars():
    body = {
        "results": [{
            "title": "Long article",
            "url": "https://example.com",
            "text": "x" * 5000,
        }],
    }
    mock_client = _MockClient(_MockResponse(200, json_body=body))
    with patch("owtn.tools.web_search.httpx.AsyncClient", return_value=mock_client):
        result = await web_search("x", api_key="fake")
    assert len(result.hits[0].snippet) == 600


@pytest.mark.asyncio
async def test_handles_results_without_text_field():
    """Exa sometimes returns results without `text` (no contents requested)."""
    body = {
        "results": [{
            "title": "No-text result",
            "url": "https://example.com",
        }],
    }
    mock_client = _MockClient(_MockResponse(200, json_body=body))
    with patch("owtn.tools.web_search.httpx.AsyncClient", return_value=mock_client):
        result = await web_search("x", api_key="fake")
    assert result.hits[0].snippet == ""
    assert result.hits[0].title == "No-text result"
