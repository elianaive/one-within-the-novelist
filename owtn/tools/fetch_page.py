"""Fetch a single URL and return its main content as clean Markdown.

Cross-stage; lives alongside `web_search` so domain-expert critics that
need to verify a quote, a doctrinal passage, or a technical claim can
follow a search hit to the actual page text. Stage 4's domain_expert
critics are the current consumer.

The web is messy: nav bars, ads, footers, comments, JS-rendered shells.
`trafilatura` handles the extraction — the same library Common Crawl
and many archival pipelines use for "give me the article from this URL."
We ask for Markdown output with links and images stripped, then cap the
result so a single fetch can't blow the tool-result token budget.

Graceful when fetching fails (DNS, 4xx/5xx, timeout) or when extraction
returns empty: the FetchedPage carries an empty `text` and a `note`
describing what happened. Critics read the note and proceed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping

import httpx
import trafilatura

from owtn.orchestration import ToolContext, ToolSpec


logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_S = 20.0
DEFAULT_CHAR_CAP = 12000
"""~3K tokens, comparable to a long search-result-page block. Critics that
need more can pass a larger cap explicitly via the `char_cap` param."""


_DEFAULT_USER_AGENT = (
    "OneWithinTheNovelist/0.1.0 "
    "(research; +https://github.com/elianaive/one-within-the-novelist) "
    "httpx"
)
"""Wikimedia-policy-compliant UA: tool name, version, contact endpoint
(repo URL), and library tag. Wikimedia and many other large sites refuse
generic / unidentified UAs (Wikipedia returns 403 with a pointer to
their robot policy). Override via the `OWTN_USER_AGENT` env var if you
deploy this in a different context."""


def _user_agent() -> str:
    return os.environ.get("OWTN_USER_AGENT") or _DEFAULT_USER_AGENT


@dataclass(frozen=True, slots=True)
class FetchedPage:
    """One fetched-and-cleaned page. `text` is empty when extraction failed
    or the URL did not return HTML; `note` is non-empty whenever the call
    surfaced something the caller should know."""
    url: str
    title: str = ""
    text: str = ""
    note: str = ""


async def fetch_page(
    url: str,
    *,
    char_cap: int = DEFAULT_CHAR_CAP,
    offset: int = 0,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> FetchedPage:
    """Fetch `url`, extract main content, return as Markdown.

    Pipeline: httpx GET → trafilatura.extract(output_format='markdown')
    → slice `[offset : offset + char_cap]`. Returns a `FetchedPage` whose
    `text` is empty on any failure path; the `note` field describes the
    failure.

    `offset` lets a critic skip past a known prefix (license boilerplate,
    table of contents, irrelevant chapters) and read deeper into a long
    page. Combine with a previous fetch's `note` (which carries the total
    extracted length) to page through a document.
    """
    if not url.strip():
        return FetchedPage(url=url, note="empty url — nothing to fetch")
    if not url.lower().startswith(("http://", "https://")):
        return FetchedPage(
            url=url,
            note="url must start with http:// or https://",
        )

    headers = {"User-Agent": _user_agent(), "Accept": "text/html,application/xhtml+xml"}
    try:
        async with httpx.AsyncClient(
            timeout=timeout_s, follow_redirects=True, headers=headers,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text
    except httpx.HTTPStatusError as e:
        body = e.response.text[:200]
        logger.warning("fetch_page %s returned %s: %s", url, e.response.status_code, body)
        return FetchedPage(
            url=url,
            note=f"HTTP {e.response.status_code}: {body}",
        )
    except httpx.RequestError as e:
        logger.warning("fetch_page request failed (%s: %s)", type(e).__name__, e)
        return FetchedPage(url=url, note=f"request failed ({type(e).__name__}: {e})")

    extracted = trafilatura.extract(
        html,
        output_format="markdown",
        include_links=False,
        include_images=False,
        include_tables=True,
        include_comments=False,
        favor_precision=True,
    )

    title = _extract_title(html)

    if not extracted or not extracted.strip():
        return FetchedPage(
            url=url,
            title=title,
            note=(
                "trafilatura returned no main content — page may be "
                "JS-rendered, paywalled, or otherwise unreachable to the "
                "extractor. Try a different source."
            ),
        )

    text = extracted.strip()
    total_len = len(text)

    # Apply offset, then cap. The note describes both ends so the caller
    # can compose follow-up reads (next offset = current offset + len(text))
    # without having to recompute.
    if offset < 0:
        offset = 0
    sliced = text[offset:offset + char_cap]
    sliced = sliced.rstrip()

    note_parts: list[str] = []
    if offset >= total_len and total_len > 0:
        # Caller asked for content past the end. Return nothing + a note
        # naming the actual length so the caller can reduce offset.
        note_parts.append(
            f"offset {offset} is beyond the extracted length {total_len}; "
            f"reduce offset to read the page."
        )
    else:
        if offset > 0:
            note_parts.append(
                f"skipped first {offset} chars via offset (extracted length "
                f"{total_len})."
            )
        end_pos = offset + len(sliced)
        if end_pos < total_len:
            remaining = total_len - end_pos
            note_parts.append(
                f"page content truncated at {end_pos} of {total_len} chars "
                f"({remaining} chars remaining). Pass offset={end_pos} to "
                f"read further, or a larger char_cap, or fetch a more "
                f"specific URL."
            )

    return FetchedPage(
        url=url,
        title=title,
        text=sliced,
        note=" ".join(note_parts),
    )


def _extract_title(html: str) -> str:
    """Pull the document title from the HTML's <title> tag. Trafilatura's
    metadata extraction is heavier than we need here; a simple substring
    scan works for ~all cases and stays robust when the metadata path
    fails."""
    lower = html.lower()
    start = lower.find("<title")
    if start == -1:
        return ""
    open_end = lower.find(">", start)
    if open_end == -1:
        return ""
    close = lower.find("</title>", open_end)
    if close == -1:
        return ""
    title = html[open_end + 1:close].strip()
    return title[:200]


# ─── Tool handler + spec ─────────────────────────────────────────────────


async def fetch_page_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    """Tool handler — wraps `fetch_page()` for the LLM tool-use channel.
    Returns a JSON-shaped string with `url`, `title`, `text`, and an
    optional `note` (set on truncation, fetch failure, or extraction
    failure)."""
    url = params.get("url", "")
    if not isinstance(url, str) or not url.strip():
        return "ERROR: fetch_page requires non-empty `url`"

    char_cap = params.get("char_cap")
    char_cap_int: int = DEFAULT_CHAR_CAP
    if char_cap is not None:
        try:
            char_cap_int = max(1000, int(char_cap))
        except (TypeError, ValueError):
            return "ERROR: fetch_page `char_cap` must be an integer"

    offset = params.get("offset", 0)
    offset_int: int = 0
    if offset is not None:
        try:
            offset_int = max(0, int(offset))
        except (TypeError, ValueError):
            return "ERROR: fetch_page `offset` must be an integer"

    page = await fetch_page(url, char_cap=char_cap_int, offset=offset_int)
    payload: dict[str, Any] = {
        "url": page.url,
        "title": page.title,
        "text": page.text,
    }
    if page.note:
        payload["note"] = page.note
    return json.dumps(payload, ensure_ascii=False, indent=2)


FETCH_PAGE = ToolSpec(
    name="fetch_page",
    description=(
        "Fetch a URL and return its main content as clean Markdown. "
        "Use this to follow up on a `web_search` hit when the snippet is "
        "not enough — e.g., to read the actual passage of a doctrinal "
        "source, verify a quoted claim, or check a technical definition "
        "in context. Returns title, text (a window of length up to "
        "char_cap starting at offset), and a `note` field that is "
        "non-empty when the window is truncated at either end, when "
        "fetching failed, or when no main content could be extracted. "
        "Page through long pages by combining `offset` and `char_cap`: "
        "the note from the first call carries the total extracted length "
        "and the next-offset value to continue reading."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL to fetch (must start with http:// or https://).",
            },
            "char_cap": {
                "type": "integer",
                "description": (
                    "Max characters of main content to return in this window. "
                    "Default 12000 (~3K tokens). Min 1000."
                ),
                "default": DEFAULT_CHAR_CAP,
                "minimum": 1000,
            },
            "offset": {
                "type": "integer",
                "description": (
                    "Starting character index inside the extracted Markdown. "
                    "Default 0 (read from the top). Use this to skip past "
                    "front matter, license boilerplate, or already-read "
                    "sections; the previous call's note tells you the next "
                    "offset to use."
                ),
                "default": 0,
                "minimum": 0,
            },
        },
        "required": ["url"],
    },
    handler=fetch_page_handler,
)
