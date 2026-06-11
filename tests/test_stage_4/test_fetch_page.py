"""Tests for owtn.tools.fetch_page — httpx + trafilatura wrapper."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from owtn.tools.fetch_page import (
    DEFAULT_CHAR_CAP,
    FETCH_PAGE,
    FetchedPage,
    fetch_page,
    fetch_page_handler,
)


# ─── Helpers ─────────────────────────────────────────────────────────────


class _MockResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"Server returned {self.status_code}",
                request=httpx.Request("GET", "https://example.com"),
                response=httpx.Response(self.status_code, text=self.text),
            )


class _MockClient:
    def __init__(self, response: _MockResponse):
        self._response = response
        self.last_url: str | None = None
        self.last_headers: dict | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def get(self, url):
        self.last_url = url
        return self._response


# Minimal HTML that trafilatura can extract content from.
_GOOD_HTML = """<!DOCTYPE html>
<html>
  <head><title>The Doctrine of Deodand</title></head>
  <body>
    <nav>navigation links here</nav>
    <main>
      <article>
        <h1>The Doctrine of Deodand</h1>
        <p>Where any movable thing whatsoever, animate or inanimate, occasions
        the death of a reasonable creature, the thing is forfeited to the Crown,
        and is to be applied to pious uses by the king's almoner.</p>
        <p>This was the doctrine of deodand, set forth by Blackstone in his
        Commentaries on the Laws of England, Book I, Chapter 8.</p>
        <p>The moving cart, the running horse, the falling tree — each was
        treated as the agent of the harm, and each was forfeit.</p>
      </article>
    </main>
    <footer>copyright notice</footer>
  </body>
</html>
"""


# ─── Validation paths ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_note_when_url_empty():
    result = await fetch_page("   ")
    assert isinstance(result, FetchedPage)
    assert result.text == ""
    assert "empty url" in result.note


@pytest.mark.asyncio
async def test_rejects_non_http_url():
    result = await fetch_page("ftp://example.com/file")
    assert result.text == ""
    assert "http://" in result.note


# ─── Happy path ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_extracts_main_content_as_markdown():
    mock_client = _MockClient(_MockResponse(200, text=_GOOD_HTML))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client):
        result = await fetch_page("https://example.com/deodand")

    assert result.url == "https://example.com/deodand"
    assert result.title == "The Doctrine of Deodand"
    assert "Blackstone" in result.text
    assert "Commentaries" in result.text
    # Boilerplate should be stripped
    assert "navigation links" not in result.text
    assert "copyright notice" not in result.text
    assert result.note == ""


@pytest.mark.asyncio
async def test_extracts_title_when_present():
    html = "<html><head><title>Hello World</title></head><body><p>some content here that trafilatura will keep.</p><p>another paragraph for body content.</p></body></html>"
    mock_client = _MockClient(_MockResponse(200, text=html))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client):
        result = await fetch_page("https://example.com")
    assert result.title == "Hello World"


# ─── Char cap ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_char_cap_truncates_long_pages():
    big_text = "<p>" + ("an idea that recurs at length. " * 1000) + "</p>"
    html = f"<html><head><title>Big</title></head><body><article>{big_text}</article></body></html>"
    mock_client = _MockClient(_MockResponse(200, text=html))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client):
        result = await fetch_page("https://example.com/big", char_cap=500)
    assert len(result.text) <= 500
    assert "truncated" in result.note
    # The new note format names total length and next-offset value
    assert "remaining" in result.note
    assert "offset=" in result.note


@pytest.mark.asyncio
async def test_default_char_cap_applies_when_unspecified():
    big_text = "<p>" + ("a " * 50000) + "</p>"
    html = f"<html><head><title>Big</title></head><body><article>{big_text}</article></body></html>"
    mock_client = _MockClient(_MockResponse(200, text=html))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client):
        result = await fetch_page("https://example.com/big")
    assert len(result.text) <= DEFAULT_CHAR_CAP


# ─── Offset (page through long content) ─────────────────────────────────


@pytest.mark.asyncio
async def test_offset_skips_prefix():
    """offset > 0 lets the caller skip past front matter / boilerplate."""
    big_text = "<p>" + ("frontmatter. " * 200) + "</p>" + "<p>" + ("real-content. " * 200) + "</p>"
    html = f"<html><head><title>Doc</title></head><body><article>{big_text}</article></body></html>"
    mock_client = _MockClient(_MockResponse(200, text=html))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client):
        # First read: front matter
        first = await fetch_page("https://example.com/doc", char_cap=500, offset=0)
    # Note carries next-offset hint
    assert "offset=" in first.note

    # Second read: skip past first window
    mock_client2 = _MockClient(_MockResponse(200, text=html))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client2):
        second = await fetch_page("https://example.com/doc", char_cap=500, offset=500)
    assert len(second.text) <= 500
    assert "skipped first 500 chars" in second.note
    # Different content from the first window (since we offset past it)
    assert second.text != first.text


@pytest.mark.asyncio
async def test_offset_past_end_returns_empty_with_note():
    """offset >= total length returns no text + a note explaining the gap."""
    html = "<html><head><title>Short</title></head><body><article><p>brief content here that trafilatura extracts fine.</p></article></body></html>"
    mock_client = _MockClient(_MockResponse(200, text=html))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client):
        result = await fetch_page("https://example.com/short", offset=10000)
    assert result.text == ""
    assert "beyond the extracted length" in result.note


@pytest.mark.asyncio
async def test_offset_zero_does_not_emit_skip_note():
    """offset=0 (the default) shouldn't add a skipped-first-N note."""
    html = "<html><head><title>Tiny</title></head><body><article><p>this fits in one window comfortably.</p></article></body></html>"
    mock_client = _MockClient(_MockResponse(200, text=html))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client):
        result = await fetch_page("https://example.com/tiny")
    assert "skipped first" not in result.note
    assert result.note == ""


# ─── Error paths ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_http_status_error_returns_note():
    mock_client = _MockClient(_MockResponse(404, text="not found body"))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client):
        result = await fetch_page("https://example.com/missing")
    assert result.text == ""
    assert "HTTP 404" in result.note


@pytest.mark.asyncio
async def test_request_error_returns_note():
    class _RaisingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def get(self, url):
            raise httpx.ConnectError("connection refused")

    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=_RaisingClient()):
        result = await fetch_page("https://example.com")
    assert result.text == ""
    assert "ConnectError" in result.note


@pytest.mark.asyncio
async def test_extraction_failure_returns_note():
    """trafilatura returning None (e.g. JS-only shell) → empty text + note."""
    js_only = "<html><head><title>SPA</title></head><body><div id='root'></div></body></html>"
    mock_client = _MockClient(_MockResponse(200, text=js_only))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client):
        result = await fetch_page("https://spa.example.com")
    assert result.text == ""
    assert "trafilatura" in result.note or "no main content" in result.note
    # Title still recovered from <title> tag
    assert result.title == "SPA"


# ─── Tool handler ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handler_returns_json_payload():
    mock_client = _MockClient(_MockResponse(200, text=_GOOD_HTML))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client):
        out = await fetch_page_handler({"url": "https://example.com/deodand"}, ctx=None)  # type: ignore[arg-type]
    payload = json.loads(out)
    assert payload["url"] == "https://example.com/deodand"
    assert payload["title"] == "The Doctrine of Deodand"
    assert "Blackstone" in payload["text"]


@pytest.mark.asyncio
async def test_handler_rejects_empty_url():
    out = await fetch_page_handler({"url": "  "}, ctx=None)  # type: ignore[arg-type]
    assert out.startswith("ERROR")


@pytest.mark.asyncio
async def test_handler_rejects_invalid_char_cap():
    out = await fetch_page_handler(
        {"url": "https://example.com", "char_cap": "not-a-number"}, ctx=None,  # type: ignore[arg-type]
    )
    assert out.startswith("ERROR")


@pytest.mark.asyncio
async def test_handler_passes_char_cap_through():
    big_text = "<p>" + ("x " * 10000) + "</p>"
    html = f"<html><head><title>Big</title></head><body><article>{big_text}</article></body></html>"
    mock_client = _MockClient(_MockResponse(200, text=html))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client):
        out = await fetch_page_handler(
            {"url": "https://example.com", "char_cap": 1000}, ctx=None,  # type: ignore[arg-type]
        )
    payload = json.loads(out)
    assert len(payload["text"]) <= 1000
    assert "truncated" in payload.get("note", "")


@pytest.mark.asyncio
async def test_handler_passes_offset_through():
    big_text = "<p>" + ("frontmatter. " * 200) + "</p>" + "<p>" + ("body. " * 500) + "</p>"
    html = f"<html><head><title>Doc</title></head><body><article>{big_text}</article></body></html>"
    mock_client = _MockClient(_MockResponse(200, text=html))
    with patch("owtn.tools.fetch_page.httpx.AsyncClient", return_value=mock_client):
        out = await fetch_page_handler(
            {"url": "https://example.com", "char_cap": 500, "offset": 800},
            ctx=None,  # type: ignore[arg-type]
        )
    payload = json.loads(out)
    assert "skipped first 800 chars" in payload["note"]


@pytest.mark.asyncio
async def test_handler_rejects_invalid_offset():
    out = await fetch_page_handler(
        {"url": "https://example.com", "offset": "not-a-number"}, ctx=None,  # type: ignore[arg-type]
    )
    assert out.startswith("ERROR")


# ─── Tool spec ───────────────────────────────────────────────────────────


def test_tool_spec_shape():
    assert FETCH_PAGE.name == "fetch_page"
    schema = FETCH_PAGE.parameters
    assert schema["type"] == "object"
    assert "url" in schema["properties"]
    assert "char_cap" in schema["properties"]
    assert "offset" in schema["properties"]
    assert schema["required"] == ["url"]
