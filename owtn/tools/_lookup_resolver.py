"""Natural-language resolver for `lookup_reference`.

Translates an agent's NL query (e.g. "Morrison's incantatory mode") into
existing corpus keys (author slugs + tag tokens) via a fast LLM
(`claude-haiku-4-5-20251001`). The deterministic loader in `_corpus.py`
still does passage IO; this module's only job is intent → keys.

Catalog + instructions are sent as a single `cache_control: ephemeral`
system prefix so per-call cost amortizes across the 4-agent fan-out in
a Stage 3 phase. See `lab/issues/2026-04-29-natural-language-lookup-reference.md`
for the design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field

from owtn.tools._corpus import ReferenceCorpus, load_corpus

logger = logging.getLogger(__name__)


RESOLVER_MODEL = "claude-haiku-4-5-20251001"


# ─── Pydantic output schema ──────────────────────────────────────────────


class LookupResolution(BaseModel):
    """Translation of a voice-agent's NL lookup_reference query into existing corpus keys.

    Returns the authors and tags that match the agent's intent. The outer
    fetcher will validate them against the catalog and apply intersection /
    fallback per `match`. Don't invent slugs or tags — only return tokens
    that appear verbatim in the catalog provided.
    """

    interpretation: str = Field(
        ...,
        description="One-sentence paraphrase of what the agent is asking for.",
    )
    authors: list[str] = Field(
        default_factory=list,
        description="Author slugs from the catalog (e.g. 'morrison', 'bronte-c'). Empty if the query is style-only or unmatched.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tag tokens from the catalog (e.g. 'incantatory'). Empty if the query is author-only or unmatched.",
    )
    match: Literal["intersect", "authors_only", "tags_only", "none"] = Field(
        ...,
        description=(
            "intersect: both authors+tags filled, return entries by author AND with tag. "
            "authors_only: return all entries by the named author(s). "
            "tags_only: return entries with the named tag(s). "
            "none: query doesn't match any author or style in this corpus."
        ),
    )
    note: str = Field(
        ...,
        description=(
            "Short message surfaced back to the calling agent. Explain interpretation, any "
            "substitutions, and (when match is `none`) closest available alternatives."
        ),
    )


# ─── Catalog construction ────────────────────────────────────────────────


@dataclass(frozen=True)
class CatalogPayload:
    """Built once per process. `system_prefix` is the bytewise-identical
    text sent on every resolver call so the prompt cache hits."""
    system_prefix: str
    valid_authors: frozenset[str]
    valid_tags: frozenset[str]


RESOLVER_INSTRUCTIONS = """\
You translate creative-writing voice-agents' natural-language queries into existing corpus keys.

A voice agent calls `lookup_reference(query: str)` to find prose passages they want to study. Their query is natural language describing a voice, style, tradition, or author — for example:

- "Toni Morrison's incantatory mode"
- "second-person direct address"
- "Saunders' tonal-disjunction maximalism"
- "minimalist Carver-Hempel register"
- "Kafka allegorical institutional voice"

The corpus catalog below lists the authors and tags that exist. Your job is to return a `LookupResolution` whose `authors` and `tags` are tokens drawn FROM THE CATALOG (do not invent slugs or tags).

# Picking `match`

- `intersect` — query names an author AND a style they exhibit. Both `authors` and `tags` filled. The outer fetcher returns entries by that author AND with at least one of those tags. If you suspect the intersection will be empty (the author exists but doesn't carry that tag in this corpus), prefer `authors_only` and explain in `note`.
- `authors_only` — query names an author without a specific style, OR the named style isn't a tag axis in this corpus (e.g. "Saunders' timing" — return `authors=[saunders]`, `tags=[]`, explain that timing isn't a corpus tag and the agent can read Saunders directly).
- `tags_only` — query names a style/register without a specific author.
- `none` — query doesn't match any author or style in this corpus. Use for genuine corpus gaps (an absent author) and for queries that aren't stylistic asks at all (vocabulary lookups, topic research, the agent's own concept title). Explain the closest available alternatives in `note`.

# Style guidance

- Prefer fewer, more accurate keys over many speculative ones. Author-only: 1 author, 0 tags. Style-only: 1-3 tags. Compound: 1 author + 1-3 tags.
- Tag form must match the catalog exactly. If the query says "incantation" and the catalog has `incantatory`, return `incantatory`.
- The catalog's `Authors` header lists every legal author slug. Don't return slugs that aren't in that list. If you think you remember an author who isn't in the catalog, that author is genuinely absent — return `match: none` and name them in `note` so the corpus curator sees the gap.
- Author slugs are usually last-name only (`morrison`, `saunders`). Special cases: Brontë sisters use `bronte-c` (Charlotte) and `bronte-e` (Emily); PG19 entries use last-name only (e.g. `conrad`, not `pg19-conrad`).

# When the query isn't a stylistic ask

Some queries reach for vocabulary meaning, topic information, or the agent's own concept title (e.g. an agent working on a concept titled "deodand" might query `deodand` looking for a definition). This corpus indexes voice/style for prose study, not topics or definitions. Return `match: none` with a `note` explaining the tool's purpose.

# `note` field

Short — 1-3 sentences. Always state your interpretation. Mention any substitutions ("incantation isn't a tag; used `incantatory` and `polyphonic`"). When `match` is `none`, name the closest available authors or tags so the agent can pivot.
"""


def _build_catalog(corpus: ReferenceCorpus) -> CatalogPayload:
    reachable = corpus.reachable_entries()
    authors: set[str] = set()
    all_tags: set[str] = set()
    entry_lines: list[str] = []
    for e in sorted(reachable, key=lambda x: x.id):
        if e.author_slug:
            authors.add(e.author_slug)
        all_tags.update(e.tags)
        src = e.source if len(e.source) <= 120 else e.source[:117] + "..."
        entry_lines.append(f"- {e.id} | {e.author_slug or ''} | {src} | {', '.join(e.tags)}")

    catalog = "\n\n".join([
        "# Authors (literary-tagged entries; pass these tokens in `authors`)\n"
        + ", ".join(sorted(authors)),
        "# Tags (exact tokens you can return in `tags`)\n"
        + ", ".join(sorted(all_tags)),
        "# Catalog entries\n# format: id | author-slug-or-blank | source | tags\n"
        + "\n".join(entry_lines),
    ])
    return CatalogPayload(
        system_prefix=RESOLVER_INSTRUCTIONS + "\n\n" + catalog,
        valid_authors=frozenset(authors),
        valid_tags=frozenset(all_tags),
    )


# Module-level cache: one CatalogPayload per process. The system_prefix is
# bytewise-identical across calls so the prompt cache hits — that's the
# whole point of this design.
_CATALOG: Optional[CatalogPayload] = None


def get_catalog(corpus: Optional[ReferenceCorpus] = None) -> CatalogPayload:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _build_catalog(corpus or load_corpus())
    return _CATALOG


def reset_catalog_cache() -> None:
    """Force a rebuild on next call. Used by tests and after corpus reloads."""
    global _CATALOG
    _CATALOG = None


# ─── Resolver call ───────────────────────────────────────────────────────


async def resolve_query_async(
    query: str,
    corpus: Optional[ReferenceCorpus] = None,
    model: str = RESOLVER_MODEL,
) -> LookupResolution:
    """Call the resolver subagent and return the validated keys.

    Tokens not in the catalog are filtered out — the resolver is supposed
    to draw from the catalog, but if it slips, the deterministic fetcher
    must not see hallucinated keys.
    """
    from owtn.llm.api import query_async
    from owtn.llm.call_logger import llm_context

    catalog = get_catalog(corpus)
    base_ctx = dict(llm_context.get({}))
    role_token = llm_context.set({**base_ctx, "role": "stage_3_lookup_resolver"})
    try:
        result = await query_async(
            model_name=model,
            msg=query,
            system_msg="",
            system_prefix=catalog.system_prefix,
            output_model=LookupResolution,
        )
    finally:
        llm_context.reset(role_token)
    resolution: LookupResolution = result.content  # type: ignore[assignment]

    filtered_authors = [a for a in resolution.authors if a in catalog.valid_authors]
    filtered_tags = [t for t in resolution.tags if t in catalog.valid_tags]
    if filtered_authors != resolution.authors or filtered_tags != resolution.tags:
        dropped = {
            "authors": sorted(set(resolution.authors) - set(filtered_authors)),
            "tags": sorted(set(resolution.tags) - set(filtered_tags)),
        }
        logger.info("resolver returned out-of-catalog keys (dropped): %s", dropped)
        resolution = resolution.model_copy(update={
            "authors": filtered_authors,
            "tags": filtered_tags,
        })
    return resolution
