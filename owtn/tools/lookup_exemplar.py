"""Lookup tool — retrieve specific reference passages from the corpus.

Restricted to `exemplars` and `baselines` categories only. Defaults
(LLM-generated samples) are blocked — reading them would just teach the
agent to imitate the register it's trying to escape.

Resolution order for `target`:
  1. Exact entry id match (e.g. "austen-emma-s0")
  2. Author slug (e.g. "austen") → returns up to N passages by that author
  3. Style/tradition tag (e.g. "free_indirect_discourse", "gothic")

Cross-stage: any agent that wants to see specific reference passages can
call this. Currently used by Stage 3 voice agents during revision phases.
"""

from __future__ import annotations

from ._corpus import ReferenceCorpus, load_corpus


def lookup_exemplar(
    target: str,
    n: int = 2,
    max_words: int = 400,
    corpus: ReferenceCorpus | None = None,
) -> dict:
    """Return up to N reference passages matching `target` for the agent
    to read as study material.

    **Restricted to `exemplars` and `baselines` categories only.** Defaults
    are blocked — the agent should not read LLM-default samples since
    that would just teach it to imitate the register it's trying to escape.

    Resolution order for `target`:
      1. Exact entry id match (e.g. "austen-emma-s0")
      2. Author slug (e.g. "austen") → returns up to N passages by that author
      3. Style/tradition tag (e.g. "free_indirect_discourse", "gothic")

    Returns:
      {
        "target": "<input>",
        "kind": "id" | "author" | "tag" | "not_found",
        "n_returned": int,
        "n_available": int,
        "passages": [
          {
            "id": "...",
            "category": "exemplars" | "baselines",
            "source": "...",
            "tags": [...],
            "text": "...",  (truncated to max_words)
            "truncated": bool,
            "word_count": int,
          },
          ...
        ],
      }

    On a defaults-only target (e.g. caller passes "model:sonnet-4-6"),
    returns a `kind: "blocked"` response with no passages — the rule is
    that LLM-default samples are not study material.
    """
    if corpus is None:
        corpus = load_corpus()

    def _allowed(entry) -> bool:
        cat = next((t for t in ["exemplars", "baselines"] if t in entry.tags), None)
        if cat:
            return True
        # Tag membership check is the source of truth for the category. If
        # neither exemplars nor baselines tag is present, the entry's
        # text_file path tells us which folder it lives in.
        return "/exemplars/" in entry.text_sha256 or False  # placeholder

    # We can't read tags for category since the corpus YAML uses a
    # `category` field, not a tag. Use the text_file path or stored
    # category instead.
    def _category(entry) -> str:
        # corpus loader doesn't expose `category` directly, so infer from
        # the resolved text_file path stored at load time. ReferenceEntry
        # doesn't carry it explicitly — we infer from id/tags.
        if "llm_default" in entry.tags or "llm_simple" in entry.tags:
            return "defaults"
        if "literary" in entry.tags or "antislop_chosen" in entry.tags or "pg_excerpt" in entry.tags:
            return "exemplars"
        if "human_amateur" in entry.tags or "expository" in entry.tags:
            return "baselines"
        return "unknown"

    # Try exact id match
    matches: list = []
    kind = "not_found"
    for e in corpus.entries:
        if e.id == target:
            matches = [e]
            kind = "id"
            break

    if not matches:
        # Try author slug
        ents = corpus.by_author(target)
        if ents:
            matches = ents
            kind = "author"

    if not matches:
        # Try literary tag
        ents = [e for e in corpus.by_tag(target) if "literary" in e.tags]
        if ents:
            matches = ents
            kind = "tag"

    if not matches:
        return {
            "target": target, "kind": "not_found", "n_returned": 0,
            "n_available": 0, "passages": [],
            "note": f"no entry id, author slug, or literary tag matches {target!r}",
        }

    # Filter to allowed categories (exemplars + baselines only)
    allowed = [(e, _category(e)) for e in matches]
    allowed = [(e, c) for e, c in allowed if c in ("exemplars", "baselines")]
    if not allowed:
        return {
            "target": target, "kind": "blocked", "n_returned": 0,
            "n_available": len(matches), "passages": [],
            "note": "matches found but all are in `defaults/` (LLM-default register). "
                    "Not lookup-accessible — reading these would teach the agent to "
                    "imitate the register it's trying to escape.",
        }

    # Stable, deterministic order
    allowed.sort(key=lambda ec: ec[0].id)
    picks = allowed[:n]

    out_passages = []
    for e, cat in picks:
        words_list = e.text.split()
        truncated = len(words_list) > max_words
        text = " ".join(words_list[:max_words]) + (" […]" if truncated else "")
        out_passages.append({
            "id": e.id,
            "category": cat,
            "source": e.source,
            "tags": e.tags,
            "text": text,
            "truncated": truncated,
            "word_count": len(words_list),
        })

    return {
        "target": target,
        "kind": kind,
        "n_returned": len(out_passages),
        "n_available": len(allowed),
        "passages": out_passages,
    }

