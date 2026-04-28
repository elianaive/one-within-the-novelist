"""Seed embeddings for farthest-first seed sampling.

Computes a per-seed dense vector via Qwen3-Embedding-0.6B with an instruction
phrased to elicit *conceptual* (not lexical) similarity. Persisted as JSONL so
the file is grep-able, line-diffable, and tolerant of partial writes (the
header line carries cache metadata; each subsequent line is one seed).

Background: see `lab/issues/2026-04-28-seed-embedding-diversity.md`. The
pre-flight on Phase 4 found Qwen3-Embedding's seed-distance correlates with
concept-distance (r=0.46, p=0.027) where text-embedding-3-small showed no
relationship — OpenAI's retrieval-tuned embeddings clustered concepts on
topic/keywords, not on underlying creative idea. Qwen3 is instruction-tunable
and captures the conceptual similarity we actually want.

Cache file format (`data/seed-bank-embeddings.jsonl`):

    {"_meta": {"model": "...", "instruction": "..."}}
    {"seed_id": "...", "hash": "...", "vector": [...]}
    {"seed_id": "...", "hash": "...", "vector": [...]}
    ...

The first line is metadata; if its `model` or `instruction` doesn't match the
constants below, the entire file is treated as stale and rewritten. Otherwise
each subsequent line is one seed's embedding; per-seed `hash` allows
fine-grained invalidation when one seed's content is edited.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from owtn.models.stage_1.seed_bank import Seed

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

# Instruction is part of the cache key — changing it invalidates all cached
# embeddings. The phrasing was validated empirically in the pre-flight; do not
# rephrase casually.
INSTRUCTION = (
    "Given a creative-writing prompt seed, embed it so that two seeds are "
    "close iff a literary-fiction author would likely produce conceptually "
    "similar story concepts when given them."
)


def _seed_text(seed: Seed) -> str:
    """The text we feed to the embedding model. Type tag in the prefix
    contributes context (anti_target seeds produce different outputs than
    real_world seeds even with similar surface vocabulary)."""
    content = seed.content if isinstance(seed.content, str) else "\n".join(seed.content)
    return f"[{seed.type}] {content}"


def _content_hash(seed: Seed) -> str:
    return hashlib.sha256(_seed_text(seed).encode("utf-8")).hexdigest()[:16]


def _format_for_qwen(text: str) -> str:
    return f"Instruct: {INSTRUCTION}\nQuery: {text}"


def compute_embeddings(seeds: list[Seed]) -> dict[str, list[float]]:
    """Embed all given seeds. Loads the model once; processes in a single
    batch on the available device (CUDA if present, else CPU)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    if not seeds:
        return {}

    logger.info("Loading embedding model %s ...", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    texts = [_format_for_qwen(_seed_text(s)) for s in seeds]
    encoded = tokenizer(
        texts, padding=True, truncation=True,
        return_tensors="pt", max_length=512,
    ).to(device)
    with torch.no_grad():
        out = model(**encoded)
    # Last-token pooling (Qwen3-Embedding standard) + L2 normalization.
    seq_lens = encoded.attention_mask.sum(dim=1) - 1
    pooled = out.last_hidden_state[torch.arange(out.last_hidden_state.size(0)), seq_lens]
    normed = F.normalize(pooled, p=2, dim=1)
    vectors = normed.cpu().tolist()
    return {s.id: v for s, v in zip(seeds, vectors)}


def _read_cache(path: Path) -> tuple[dict, dict[str, dict]]:
    """Read a JSONL cache file. Returns (meta, entries_by_seed_id).

    Returns ({}, {}) if the file is missing, empty, or unreadable. Returns
    ({}, {}) (signalling stale) if the meta line doesn't match expected
    model/instruction — caller will rewrite from scratch.
    """
    if not path.exists():
        return {}, {}
    try:
        lines = path.read_text().splitlines()
    except Exception as e:
        logger.warning("Embedding cache unreadable, recomputing: %s", e)
        return {}, {}
    if not lines:
        return {}, {}

    try:
        first = json.loads(lines[0])
    except json.JSONDecodeError:
        logger.warning("Embedding cache header malformed, recomputing.")
        return {}, {}
    meta = first.get("_meta") or {}
    if meta.get("model") != MODEL_ID or meta.get("instruction") != INSTRUCTION:
        logger.info("Embedding cache invalidated (model/instruction changed).")
        return {}, {}

    entries: dict[str, dict] = {}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        try:
            entry = json.loads(ln)
        except json.JSONDecodeError:
            continue  # tolerate corrupt rows; the seed gets recomputed
        sid = entry.get("seed_id")
        if sid:
            entries[sid] = entry
    return meta, entries


def _write_cache(path: Path, entries: dict[str, dict]) -> None:
    """Atomically rewrite the cache file with the given entries.

    Writes to a sibling tempfile then renames so a crash mid-write doesn't
    corrupt the existing cache. JSONL: meta header followed by one line per
    seed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".tmp-", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"_meta": {
                "model": MODEL_ID, "instruction": INSTRUCTION,
            }}) + "\n")
            for sid in sorted(entries):
                fh.write(json.dumps({
                    "seed_id": sid,
                    "hash": entries[sid]["hash"],
                    "vector": entries[sid]["vector"],
                }) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the tempfile if we crashed before rename.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_or_compute(seeds: list[Seed], cache_path: Path) -> dict[str, list[float]]:
    """Return embeddings for `seeds`, hitting the JSONL disk cache where
    possible.

    Embeddings are recomputed only when:
        - The cache is missing, empty, or its meta doesn't match.
        - A seed's `content` hash differs from its cached entry.
        - A seed has no cached entry.

    The cache file is rewritten when any seed is recomputed; rewrites are
    atomic (tempfile + rename). The hot path (cache fully fresh) does no
    write.
    """
    _meta, entries = _read_cache(cache_path)

    stale: list[Seed] = []
    fresh: dict[str, list[float]] = {}
    for seed in seeds:
        entry = entries.get(seed.id)
        if entry and entry.get("hash") == _content_hash(seed):
            fresh[seed.id] = entry["vector"]
        else:
            stale.append(seed)

    if not stale:
        return fresh

    logger.info(
        "Computing embeddings for %d/%d seeds (cache hit on %d).",
        len(stale), len(seeds), len(fresh),
    )
    new_vectors = compute_embeddings(stale)
    for seed in stale:
        vec = new_vectors[seed.id]
        entries[seed.id] = {"hash": _content_hash(seed), "vector": vec}
        fresh[seed.id] = vec

    _write_cache(cache_path, entries)
    logger.info("Embedding cache written -> %s", cache_path)
    return fresh
