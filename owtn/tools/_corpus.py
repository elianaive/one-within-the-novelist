"""Reference corpus loader and signal cache for the voice stylometry tool.

The corpus is metadata in `data/voice-references-stylometric.yaml` plus
per-passage text files in `data/voice-references/passages/`. This module:

- Loads the YAML and resolves each entry's text from disk
- Computes stylometric signals for each passage (lazily, with caching)
- Persists the cache to `data/voice-references-stylometric.cache.json`
- Invalidates per-passage by SHA-256 of the text file

Loaded once per process (module-level cache); `load_corpus(force_reload=True)`
forces a fresh load if you've edited the corpus mid-session.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from owtn.tools import stylometry as _signal_module
from owtn.tools.stylometry import (
    StylometricSignals,
    aggregate_function_word_distribution,
    build_mfw_stats,
    compute_signals,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_YAML = REPO_ROOT / "data" / "voice-references-stylometric.yaml"
PASSAGES_DIR = REPO_ROOT / "data" / "voice-references"
CACHE_PATH = REPO_ROOT / "data" / "voice-references-stylometric.cache.json"


def _compute_signal_version() -> str:
    """Hash of the source code that produces cached signals.

    Auto-invalidates the cache when signal logic changes — no manual
    version bumps. Includes:
      - FUNCTION_POS constant (defines what counts as a function word)
      - Source of `function_word_distribution`, `burstiness`, `mattr`,
        `compute_signals` (the functions whose output is cached)

    Whitespace-only changes will invalidate the cache, but that's
    acceptable — full rebuild is ~30s for 549 entries with no API calls.
    """
    parts = [
        repr(sorted(_signal_module.FUNCTION_POS)),
        inspect.getsource(_signal_module.function_word_distribution),
        inspect.getsource(_signal_module.burstiness),
        inspect.getsource(_signal_module.mattr),
        inspect.getsource(_signal_module.compute_signals),
    ]
    # Also include the spaCy version + model name — POS tags can shift between
    # spaCy releases or model upgrades, which would silently change signals
    # without triggering a code-hash change.
    try:
        import spacy
        from owtn.judging.tier_a.preprocessing import get_nlp
        nlp = get_nlp()
        parts.append(f"spacy:{spacy.__version__}")
        parts.append(f"model:{nlp.meta.get('name', '?')}-{nlp.meta.get('version', '?')}")
    except Exception:
        pass  # If spaCy isn't loadable, fall back to source-only hashing
    blob = "\n---\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


CACHE_VERSION = _compute_signal_version()


@dataclass
class ReferenceEntry:
    id: str
    tags: list[str]
    source: str
    license: str
    url: str | None
    text: str
    text_sha256: str
    signals: StylometricSignals = field(repr=False)


@dataclass
class ReferenceCorpus:
    entries: list[ReferenceEntry]
    # Most-frequent-word statistics for Burrows' Delta — top-N MFW lemmas
    # across the corpus with per-word mean+stdev. Computed once at load
    # time, used by `function_word_burrows_delta` for Z-score normalization.
    mfw_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def by_tag(self, tag: str) -> list[ReferenceEntry]:
        """Entries containing the given tag."""
        return [e for e in self.entries if tag in e.tags]

    def by_model_tag(self, model_tag: str) -> list[ReferenceEntry]:
        """LLM-default entries for a given model (matches `model:<tag>`)."""
        marker = f"model:{model_tag}"
        return [e for e in self.entries if marker in e.tags]

    def by_author(self, author: str) -> list[ReferenceEntry]:
        """Literary entries whose id begins with the author slug (e.g.
        'austen', 'joyce', 'kafka'). Special-cased for the few authors
        whose ids use a different prefix:
          - PG19 entries → id is 'pg19-<author>' (conrad, hardy, twain, etc.)
          - Brontë sisters → 'bronte-c-jane-eyre' / 'bronte-e-wuthering-heights'
        """
        author = author.lower().strip()
        out = []
        for e in self.entries:
            if "literary" not in e.tags:
                continue
            eid = e.id.lower()
            # Direct author-slug prefix
            if eid.startswith(f"{author}-") or eid == author:
                out.append(e); continue
            # PG19 case
            if eid.startswith(f"pg19-{author}"):
                out.append(e); continue
            # Brontë special-case
            if author == "bronte-c" and "bronte-c-" in eid:
                out.append(e)
            elif author == "bronte-e" and "bronte-e-" in eid:
                out.append(e)
        return out

    def aggregate_fw_distribution(self, entries: list[ReferenceEntry]) -> dict[str, float]:
        return aggregate_function_word_distribution(
            [e.signals.fw_distribution for e in entries]
        )

    def aggregate_signals(self, entries: list[ReferenceEntry]) -> dict[str, float]:
        """Mean of scalar signals across entries."""
        if not entries:
            return {"n_samples": 0}
        n = len(entries)
        return {
            "n_samples": n,
            "burstiness": sum(e.signals.burstiness for e in entries) / n,
            "mattr": sum(e.signals.mattr for e in entries) / n,
            "word_count_mean": sum(e.signals.word_count for e in entries) / n,
        }

    def intra_cluster_spread(self, entries: list[ReferenceEntry]) -> dict[str, float]:
        """Distribution of sample-to-centroid function-word distances within
        a cluster. Returned stats:
          - intra_dist_mean: average sample distance from the cluster centroid
          - intra_dist_p50: median (typical sample distance)
          - intra_dist_p95: 95th percentile — natural-variance threshold; a
            candidate with distance > p95 is genuinely outside the cluster
          - intra_dist_max: worst-case sample distance

        Used by the stylometry tool to calibrate "meaningful departure"
        framing — e.g., a candidate at dist=0.17 from its own model centroid
        is only "departed" if 0.17 > p95 of the model's intra-cluster spread.
        Centroids built from very few samples (n<5) yield unreliable stats.
        """
        if len(entries) < 2:
            return {
                "n_samples": len(entries),
                "intra_dist_mean": 0.0,
                "intra_dist_p50": 0.0,
                "intra_dist_p95": 0.0,
                "intra_dist_max": 0.0,
                "calibration_reliable": False,
            }
        centroid = self.aggregate_fw_distribution(entries)
        from owtn.tools.stylometry import function_word_cosine_distance
        dists = sorted(
            function_word_cosine_distance(e.signals.fw_distribution, centroid)
            for e in entries
        )
        n = len(dists)
        # Percentile via linear interpolation (no numpy dependency)
        def percentile(p: float) -> float:
            if n == 1:
                return dists[0]
            idx = (p / 100) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return dists[lo] * (1 - frac) + dists[hi] * frac
        return {
            "n_samples": n,
            "intra_dist_mean": sum(dists) / n,
            "intra_dist_p50": percentile(50),
            "intra_dist_p95": percentile(95),
            "intra_dist_max": dists[-1],
            "calibration_reliable": n >= 5,
        }


_CORPUS_CACHE: ReferenceCorpus | None = None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_disk_cache() -> dict:
    if not CACHE_PATH.exists():
        return {"version": CACHE_VERSION, "passages": {}}
    try:
        cache = json.loads(CACHE_PATH.read_text())
        if cache.get("version") != CACHE_VERSION:
            logger.info("Stylometric cache version mismatch; rebuilding")
            return {"version": CACHE_VERSION, "passages": {}}
        return cache
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Stylometric cache unreadable ({e}); rebuilding")
        return {"version": CACHE_VERSION, "passages": {}}


def _write_disk_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _signals_to_cache_entry(sha: str, sig: StylometricSignals) -> dict:
    return {
        "sha256": sha,
        "burstiness": sig.burstiness,
        "mattr": sig.mattr,
        "fw_distribution": sig.fw_distribution,
        "word_count": sig.word_count,
        "sentence_count": sig.sentence_count,
    }


def _cache_entry_to_signals(entry: dict) -> StylometricSignals:
    return StylometricSignals(
        burstiness=entry["burstiness"],
        mattr=entry["mattr"],
        fw_distribution=entry["fw_distribution"],
        word_count=entry["word_count"],
        sentence_count=entry["sentence_count"],
    )


def load_corpus(force_reload: bool = False) -> ReferenceCorpus:
    """Load the reference corpus, resolving texts and signals (with caching).

    Missing passage files are logged and skipped. The corpus is usable with
    whatever subset of entries is populated.
    """
    global _CORPUS_CACHE
    if _CORPUS_CACHE is not None and not force_reload:
        return _CORPUS_CACHE

    if not CORPUS_YAML.exists():
        logger.warning(f"No stylometric corpus YAML at {CORPUS_YAML}; returning empty corpus")
        _CORPUS_CACHE = ReferenceCorpus(entries=[])
        return _CORPUS_CACHE

    raw = yaml.safe_load(CORPUS_YAML.read_text())
    refs = raw.get("references", [])
    disk_cache = _read_disk_cache()
    cache_passages = disk_cache.setdefault("passages", {})
    cache_dirty = False
    entries: list[ReferenceEntry] = []

    for ref in refs:
        entry_id = ref["id"]
        text_file = PASSAGES_DIR / ref["text_file"]
        if not text_file.exists():
            logger.warning(f"Missing passage file for {entry_id}: {text_file}; skipping")
            continue
        text = text_file.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning(f"Empty passage file for {entry_id}; skipping")
            continue
        sha = _sha256_text(text)
        cached = cache_passages.get(entry_id)
        if cached and cached.get("sha256") == sha:
            signals = _cache_entry_to_signals(cached)
        else:
            signals = compute_signals(text)
            cache_passages[entry_id] = _signals_to_cache_entry(sha, signals)
            cache_dirty = True

        entries.append(ReferenceEntry(
            id=entry_id,
            tags=list(ref.get("tags", [])),
            source=ref.get("source", ""),
            license=ref.get("license", ""),
            url=ref.get("url"),
            text=text,
            text_sha256=sha,
            signals=signals,
        ))

    # Drop cache entries whose corpus YAML id no longer exists
    yaml_ids = {ref["id"] for ref in refs}
    stale = [k for k in cache_passages if k not in yaml_ids]
    for k in stale:
        del cache_passages[k]
        cache_dirty = True

    if cache_dirty:
        _write_disk_cache(disk_cache)

    # Build the most-frequent-word stats table for Burrows' Delta.
    # Top 150 MFW across the corpus + per-word mean/stdev. Computed once
    # per corpus load; reused for every Delta distance calculation.
    mfw_stats = build_mfw_stats(
        [e.signals.fw_distribution for e in entries],
        top_n=150,
    )

    _CORPUS_CACHE = ReferenceCorpus(entries=entries, mfw_stats=mfw_stats)
    return _CORPUS_CACHE


def rebuild_cache() -> ReferenceCorpus:
    """Force a full cache rebuild and reload."""
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
    return load_corpus(force_reload=True)
