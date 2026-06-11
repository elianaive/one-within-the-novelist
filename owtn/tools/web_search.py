"""Web search tool — thin Python wrapper around Exa's HTTP API.

Cross-stage; lives alongside `stylometry`/`slop_score`/`writing_style`
so any stage that needs factual lookup gets the same surface. Stage 4's
domain-expert critics are the current consumer.

Provider-agnostic by design: no native Anthropic web_search call (the
dev default DeepSeek has no equivalent), so we hit Exa's HTTP API
directly and the wrapper looks the same regardless of which model
family the calling critic runs on.

Graceful when `EXA_API_KEY` is missing or the provider returns an
error: the call returns an empty list with a `note` field describing
what happened. Critics that depend on web_search read the note and
proceed accordingly; the session doesn't crash on a configuration gap.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx

from owtn.orchestration import ToolContext, ToolSpec


logger = logging.getLogger(__name__)


EXA_SEARCH_URL = "https://api.exa.ai/search"
DEFAULT_TIMEOUT_S = 15.0


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One search result. `snippet` may be empty when the provider
    returns a result without text content."""
    title: str
    url: str
    snippet: str = ""
    published: str | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The full result of a `web_search` call. `note` is non-empty when
    the call surfaced something the caller should know — missing API
    key, provider error, zero results."""
    query: str
    hits: list[SearchHit] = field(default_factory=list)
    note: str = ""


async def web_search(
    query: str,
    *,
    n: int = 5,
    api_key: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> SearchResult:
    """Run one Exa web search. Returns a `SearchResult` with up to `n`
    `SearchHit`s. The `api_key` argument overrides `EXA_API_KEY`; pass
    explicitly in tests, leave None for production."""
    if not query.strip():
        return SearchResult(query=query, note="empty query — nothing to search")
    key = api_key or os.environ.get("EXA_API_KEY")
    if not key:
        return SearchResult(
            query=query,
            note=(
                "EXA_API_KEY not set; web_search returned no results. "
                "Set EXA_API_KEY in .env or call with api_key=."
            ),
        )

    payload = {"query": query, "num_results": max(1, min(n, 20)), "type": "auto", "contents": {"text": True}}
    headers = {"x-api-key": key, "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(EXA_SEARCH_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        body = e.response.text[:200]
        logger.warning("Exa returned %s: %s", e.response.status_code, body)
        return SearchResult(query=query, note=f"Exa error {e.response.status_code}: {body}")
    except httpx.RequestError as e:
        logger.warning("Exa request failed (%s: %s)", type(e).__name__, e)
        return SearchResult(query=query, note=f"Exa request failed ({type(e).__name__}: {e})")

    return SearchResult(query=query, hits=_parse_hits(data, n))


def _parse_hits(data: Any, n: int) -> list[SearchHit]:
    """Parse Exa's response into SearchHits. Defensive — Exa's response
    shape is mostly stable but field names occasionally shift; we read
    the common fields and tolerate unknowns."""
    raw_hits = data.get("results") if isinstance(data, dict) else None
    if not isinstance(raw_hits, list):
        return []
    hits: list[SearchHit] = []
    for entry in raw_hits[:n]:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or entry.get("url") or "(untitled)")
        url = str(entry.get("url") or "")
        snippet = str(entry.get("text") or entry.get("snippet") or "").strip()
        published = entry.get("publishedDate") or entry.get("published_date")
        hits.append(SearchHit(
            title=title.strip(),
            url=url.strip(),
            snippet=snippet[:600],
            published=str(published) if published else None,
        ))
    return hits


# ─── Tool handler + spec ─────────────────────────────────────────────────


async def web_search_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    """Tool handler — wraps `web_search()` for the LLM tool-use channel.
    Returns a JSON-shaped string with `query`, `hits`, and an optional
    `note` (set when the call surfaced something worth knowing —
    missing API key, provider error, zero results)."""
    query = params.get("query", "")
    n = int(params.get("n", 5))
    if not isinstance(query, str) or not query.strip():
        return "ERROR: web_search requires non-empty `query`"

    result = await web_search(query, n=n)
    payload: dict[str, Any] = {
        "query": result.query,
        "hits": [
            {"title": h.title, "url": h.url, "snippet": h.snippet, "published": h.published}
            for h in result.hits
        ],
    }
    if result.note:
        payload["note"] = result.note
    return json.dumps(payload, ensure_ascii=False, indent=2)


WEB_SEARCH = ToolSpec(
    name="web_search",
    description=(
        "Search the web for factual verification. Provider-agnostic "
        "Python wrapper around Exa. Returns top results with titles, "
        "URLs, snippets, and (when available) published dates. When "
        "EXA_API_KEY is missing, returns an empty hit list with a "
        "`note` field describing the configuration gap; callers that "
        "depend on web verification read the note and proceed accordingly."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "n": {"type": "integer", "description": "Max results (default 5).", "default": 5, "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
    },
    handler=web_search_handler,
)
