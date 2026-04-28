"""Stylometry tool — signal primitives plus an agent-facing entry point.

Primitives at the top (burstiness, MATTR, function-word distributions,
cosine and Burrows' Delta distances) are pure computations on a spaCy
doc. The `stylometry()` entry point at the bottom builds an agent-facing
report by computing those signals on a candidate passage and positioning
it against centroids from the reference corpus.

The two metrics for function-word distance:
- **cosine** — vector-angle distance, sensitive to absolute frequency.
- **Burrows' Delta** — Z-score-normalized, less content-contaminated
  (Burrows 2002, classic authorship-attribution measure).
Both are returned for every centroid; agents can weight as appropriate.

Calibration: per-model "departure" thresholds use the model's empirical
intra-cluster cosine p95. A candidate at distance > p95 is outside the
model's natural variance.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import asdict, dataclass, field

from owtn.judging.tier_a.preprocessing import get_nlp


# Function words = closed-class grammatical tokens, identified by spaCy's
# Universal POS tagger. DET (the/a), ADP (of/in/on), PRON (he/she/it),
# AUX (is/was/will), CCONJ (and/but), SCONJ (if/because), PART (to/not).
FUNCTION_POS: frozenset[str] = frozenset({
    "DET", "ADP", "PRON", "AUX", "CCONJ", "SCONJ", "PART",
})


def burstiness(doc) -> float:
    """Coefficient of variation (stdev / mean) of sentence length in words.

    Higher = more variable rhythm. <0.4 is LLM-flat; >0.4 is human-like;
    >0.6 is rich variation.
    """
    lengths = [
        sum(1 for t in sent if not t.is_punct and not t.is_space)
        for sent in doc.sents
    ]
    lengths = [n for n in lengths if n > 0]
    if len(lengths) < 2:
        return 0.0
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    variance = sum((n - mean) ** 2 for n in lengths) / len(lengths)
    return math.sqrt(variance) / mean


def mattr(doc, window: int = 500) -> float:
    """Moving Average Type-Token Ratio over a sliding window. In [0, 1].

    Mikros 2025 finds MATTR unreliable for voice distinctiveness in
    isolation — GPT-4o-Hemingway scores *higher* MATTR than actual
    Hemingway by imitating the cliché of "spare prose" with excess
    vocabulary variety. Treat as one signal among many.
    """
    tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]
    if not tokens:
        return 0.0
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens)
    return sum(
        len(set(tokens[i:i + window])) / window
        for i in range(len(tokens) - window + 1)
    ) / (len(tokens) - window + 1)


def function_word_distribution(doc) -> dict[str, float]:
    """Function-word frequency keyed by lowercased token text.

    Function-word identity comes from `token.pos_ in FUNCTION_POS`.
    Frequencies are normalized by total non-punct/non-space token count,
    so "the" appearing 5x in a 100-token doc → 0.05.

    Returns a sparse dict; vocabulary varies per document. Cosine
    distance and aggregation take the union of keys across distributions.
    """
    total = sum(1 for t in doc if not t.is_punct and not t.is_space)
    if total == 0:
        return {}
    fw = [
        t.text.lower() for t in doc
        if t.pos_ in FUNCTION_POS and not t.is_punct and not t.is_space
    ]
    if not fw:
        return {}
    return {w: c / total for w, c in Counter(fw).items()}


def function_word_cosine_distance(
    candidate_dist: dict[str, float],
    reference_dist: dict[str, float],
) -> float:
    """Cosine distance ∈ [0, 1] between two sparse function-word
    distributions over the union of keys."""
    keys = set(candidate_dist) | set(reference_dist)
    if not keys:
        return 1.0
    cand_vec = [candidate_dist.get(k, 0.0) for k in keys]
    ref_vec = [reference_dist.get(k, 0.0) for k in keys]
    dot = sum(c * r for c, r in zip(cand_vec, ref_vec))
    cand_norm = math.sqrt(sum(c * c for c in cand_vec))
    ref_norm = math.sqrt(sum(r * r for r in ref_vec))
    if cand_norm == 0 or ref_norm == 0:
        return 1.0
    return 1.0 - (dot / (cand_norm * ref_norm))


def aggregate_function_word_distribution(
    distributions: list[dict[str, float]],
) -> dict[str, float]:
    """Mean function-word frequencies across N distributions, over the
    union of keys. A word missing from a distribution counts as 0 in
    its mean — a word appearing in only some documents has a smaller
    mean, matching the fixed-vocab semantics."""
    if not distributions:
        return {}
    n = len(distributions)
    keys = {k for d in distributions for k in d}
    return {k: sum(d.get(k, 0.0) for d in distributions) / n for k in keys}


def build_mfw_stats(
    distributions: list[dict[str, float]],
    top_n: int = 150,
) -> dict[str, dict[str, float]]:
    """Most-frequent-word statistics for Burrows' Delta. Returns
    {word: {mean, stdev}} for the top_n most frequent function-word
    lemmas across the input distributions.

    Stdev is floored at 1e-9 to avoid division by zero for words that
    happen to be perfectly constant across the corpus.
    """
    if not distributions:
        return {}
    total = Counter()
    for d in distributions:
        for w, freq in d.items():
            total[w] += freq
    mfw = [w for w, _ in total.most_common(top_n)]
    n = len(distributions)
    stats = {}
    for w in mfw:
        vals = [d.get(w, 0.0) for d in distributions]
        mean = sum(vals) / n
        if n > 1:
            stdev = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
        else:
            stdev = 0.0
        stats[w] = {"mean": mean, "stdev": max(stdev, 1e-9)}
    return stats


def function_word_burrows_delta(
    candidate_dist: dict[str, float],
    reference_dist: dict[str, float],
    mfw_stats: dict[str, dict[str, float]],
) -> float:
    """Burrows' Delta — mean absolute Z-score difference over the MFW
    vocabulary. Returns 0 when no MFW stats are available.

    Z-scoring (value − μ) / σ per word normalizes for per-word variance,
    so each MFW contributes proportionally to its stylistic
    informativeness rather than to its raw frequency. Less content-
    contaminated than cosine at our sample sizes.

    Reference: Burrows (2002), Literary and Linguistic Computing.
    """
    if not mfw_stats:
        return 0.0
    z_diffs = []
    for w, stats in mfw_stats.items():
        mu, sigma = stats["mean"], stats["stdev"]
        z_cand = (candidate_dist.get(w, 0.0) - mu) / sigma
        z_ref = (reference_dist.get(w, 0.0) - mu) / sigma
        z_diffs.append(abs(z_cand - z_ref))
    return sum(z_diffs) / len(z_diffs)


@dataclass
class StylometricSignals:
    """Raw stylometric signals for a single passage."""
    burstiness: float
    mattr: float
    fw_distribution: dict[str, float] = field(repr=False)
    word_count: int
    sentence_count: int


def compute_signals(text: str, nlp=None) -> StylometricSignals:
    """Compute the cached stylometric signals on a passage. `nlp` defaults
    to the shared spaCy pipeline from Tier-A preprocessing."""
    if nlp is None:
        nlp = get_nlp()
    doc = nlp(text)
    return StylometricSignals(
        burstiness=burstiness(doc),
        mattr=mattr(doc),
        fw_distribution=function_word_distribution(doc),
        word_count=sum(1 for t in doc if not t.is_punct and not t.is_space),
        sentence_count=sum(1 for _ in doc.sents),
    )


def near_default(
    signals: StylometricSignals,
    fw_distance_from_baseline: float,
    burstiness_floor: float = 0.4,
    fw_cosine_floor: float = 0.05,
) -> bool:
    """A candidate is "near default" if its burstiness is below the LLM-flat
    floor AND its function-word distance from the supplied baseline is
    below the meaningful-divergence threshold. Thresholds are heuristic
    starting points; calibrate from pilot data."""
    return (
        signals.burstiness < burstiness_floor
        and fw_distance_from_baseline < fw_cosine_floor
    )


# ═══ TOOL ENTRY POINT ════════════════════════════════════════════════════

# Deferred import: `_corpus` imports primitives from this module at its own
# module-load time, so importing _corpus at module-load here would form a
# circular dependency. Type annotations use TYPE_CHECKING; the actual
# `load_corpus()` call is deferred inside the `stylometry()` function body
# that needs it. `rebuild_cache` is re-exported via a thin wrapper at the
# bottom of this file for the CLI.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._corpus import ReferenceCorpus


@dataclass
class StylometricToolReport:
    """Compact tool response for a voice agent. Stays well under the 4kB
    target when serialized to JSON.

    `candidate` is raw signals on the candidate prose. `references` is the
    aggregate context: the calling model's own default centroid, the cross-
    model LLM centroid, and human-anchored centroids. `interpretation_notes`
    is dynamic guidance derived from the candidate's actual values.
    """
    candidate: dict
    references: dict
    interpretation_notes: str

    def to_dict(self) -> dict:
        return asdict(self)


def _signals_to_compact(sig: StylometricSignals) -> dict:
    return {
        "burstiness": round(sig.burstiness, 4),
        "mattr": round(sig.mattr, 4),
        "word_count": sig.word_count,
        "sentence_count": sig.sentence_count,
    }


def _aggregate_to_compact(agg: dict) -> dict:
    return {
        k: (round(v, 4) if isinstance(v, float) else v)
        for k, v in agg.items()
    }


def _build_caller_model_reference(
    corpus: ReferenceCorpus,
    caller_model: str | None,
) -> tuple[dict | None, dict[str, float] | None, dict | None]:
    """Return (compact_aggregate, fw_distribution, intra_spread_stats) for
    the caller model's default samples. (None, None, None) if absent.

    The intra_spread_stats are used by interpretation_notes to calibrate
    "departure" framing: a candidate's distance is only meaningful if it
    exceeds the model's intra-cluster p95.
    """
    if caller_model is None:
        return None, None, None
    entries = corpus.by_model_tag(caller_model)
    if not entries:
        return None, None, None
    agg = corpus.aggregate_signals(entries)
    agg["model"] = caller_model
    spread = corpus.intra_cluster_spread(entries)
    # Merge the spread fields into the aggregate (small overhead, agent-readable)
    agg["intra_dist_mean"] = round(spread["intra_dist_mean"], 4)
    agg["intra_dist_p50"] = round(spread["intra_dist_p50"], 4)
    agg["intra_dist_p95"] = round(spread["intra_dist_p95"], 4)
    agg["intra_dist_max"] = round(spread["intra_dist_max"], 4)
    agg["calibration_reliable"] = spread["calibration_reliable"]
    return _aggregate_to_compact(agg), corpus.aggregate_fw_distribution(entries), spread


def _build_centroid(
    corpus: ReferenceCorpus,
    tag: str,
) -> tuple[dict, dict[str, float]]:
    entries = corpus.by_tag(tag)
    return (
        _aggregate_to_compact(corpus.aggregate_signals(entries)),
        corpus.aggregate_fw_distribution(entries),
    )


def _interpretation(
    candidate: StylometricSignals,
    own_model_distance: float | None,
    own_model_spread: dict | None,
    llm_centroid_distance: float,
    human_literary_distance: float,
    neutral_baseline_distance: float | None,
    own_model_delta: float | None = None,
    llm_centroid_delta: float | None = None,
    human_literary_delta: float | None = None,
) -> str:
    """Short dynamic guidance based on the candidate's actual values.

    Burstiness has a robust threshold (>0.4 = human-like) backed by ~500
    samples in the corpus. Function-word distance "departure" is calibrated
    against the caller model's empirical intra-cluster cosine spread.
    Burrows' Delta is reported as a complementary signal — Z-score-
    normalized, less content-contaminated than cosine. MATTR is
    downweighted — known to be unreliable at our sample sizes.
    """
    parts: list[str] = []

    # Burstiness — strongest signal, hard threshold
    if candidate.burstiness < 0.4:
        parts.append(
            f"Burstiness {candidate.burstiness:.2f} is below the 0.4 floor — "
            f"sentence rhythm reads as LLM-flat. Vary sentence length."
        )
    else:
        parts.append(f"Burstiness {candidate.burstiness:.2f} is in human-like territory.")

    # Neutral baseline — primary signal when provided
    if neutral_baseline_distance is not None:
        if neutral_baseline_distance < 0.05:
            parts.append(
                f"Function-word cosine distance from this session's neutral "
                f"baseline ({neutral_baseline_distance:.3f}) is near zero — "
                f"your candidate is structurally indistinguishable from the "
                f"unprompted draft."
            )
        else:
            parts.append(
                f"Function-word cosine distance from neutral baseline = "
                f"{neutral_baseline_distance:.3f} (meaningful at >0.05)."
            )

    # Own-model departure — calibrated against intra-cluster spread
    if own_model_distance is not None and own_model_spread is not None:
        p95 = own_model_spread["intra_dist_p95"]
        mean = own_model_spread["intra_dist_mean"]
        reliable = own_model_spread.get("calibration_reliable", False)
        delta_str = (f" Burrows Δ = {own_model_delta:.2f}."
                     if own_model_delta is not None else "")
        if not reliable:
            parts.append(
                f"Cosine distance from your model's default = "
                f"{own_model_distance:.3f}.{delta_str} The calibration "
                f"baseline has only {own_model_spread['n_samples']} samples "
                f"(need ≥5 for reliable thresholds). Treat as advisory only."
            )
        elif own_model_distance > p95:
            ratio = own_model_distance / p95 if p95 > 0 else float("inf")
            parts.append(
                f"Cosine distance from your model's default = "
                f"{own_model_distance:.3f}.{delta_str} Your model's natural "
                f"intra-cluster spread is mean={mean:.3f}, p95={p95:.3f}. "
                f"You are {ratio:.1f}× the p95 — a genuinely meaningful "
                f"departure from your default register."
            )
        else:
            parts.append(
                f"Cosine distance from your model's default = "
                f"{own_model_distance:.3f}.{delta_str} Your model's natural "
                f"intra-cluster spread is mean={mean:.3f}, p95={p95:.3f}. "
                f"You are within natural variance — not yet a meaningful "
                f"departure."
            )

    delta_centroid_str = (
        f" (Burrows Δ: LLM={llm_centroid_delta:.2f}, "
        f"literary={human_literary_delta:.2f})"
        if llm_centroid_delta is not None and human_literary_delta is not None else ""
    )
    parts.append(
        f"Cosine distance from LLM-default centroid = {llm_centroid_distance:.3f}; "
        f"from human-literary centroid = {human_literary_distance:.3f}.{delta_centroid_str} "
        f"Selection target is departure from defaults, not proximity to any "
        f"named voice."
    )

    # MATTR last + de-emphasized — noisy at our sample sizes
    if candidate.mattr < 0.7:
        parts.append(
            f"(MATTR {candidate.mattr:.2f}; signal is unreliable at this passage "
            f"length per Mikros 2025 — interpret cautiously.)"
        )

    return " ".join(parts)


def stylometry(
    passage: str,
    caller_model: str | None = None,
    neutral_baseline: str | None = None,
    target_styles: list[str] | None = None,
    corpus: ReferenceCorpus | None = None,
) -> StylometricToolReport:
    """Compute stylometric signals on `passage` and return them alongside
    aggregate references for positioning.

    Args:
        passage: The candidate prose to analyze.
        caller_model: Short model tag (e.g. "sonnet-4-6", "deepseek-v3-2").
            If present in the corpus, the report includes that model's
            default centroid and the cosine distance from it. Falls back to
            the cross-model LLM centroid when absent or unknown.
        neutral_baseline: Optional session-level neutral-voice draft of
            the same scene. When provided, function-word distance from it
            is computed as the primary "departure from default" signal.
            Stage 3 orchestrator passes this when sessions exist; standalone
            tool calls omit it.
        target_styles: Optional list of named author or style tags. For each,
            the report adds the candidate's function-word distance from that
            author/style centroid. Resolution order per element:
              1. Author slug match (e.g. "austen" → all austen-* entries).
                 Use this for per-author distance.
              2. Tag match (e.g. "free_indirect_discourse", "gothic",
                 "russian", "minimalist") — uses any literary entry carrying
                 that tag.
            Unknown tokens are reported as `"not_found"`.
            Useful for "how close am I to Austen / FID-rich / gothic?"
            framing in voice-agent revision loops.
        corpus: Optional pre-loaded corpus (for testing). Defaults to the
            module-level singleton from `corpus.load_corpus()`.

    Returns:
        StylometricToolReport with candidate signals, references, and
        dynamic interpretation notes.
    """
    if corpus is None:
        from ._corpus import load_corpus  # deferred — see top-of-file note
        corpus = load_corpus()

    candidate_sig = compute_signals(passage)
    cand_fw = candidate_sig.fw_distribution
    mfw = corpus.mfw_stats

    def _both_distances(ref_fw: dict[str, float]) -> tuple[float, float]:
        cos = function_word_cosine_distance(cand_fw, ref_fw)
        delta = function_word_burrows_delta(cand_fw, ref_fw, mfw) if mfw else 0.0
        return cos, delta

    llm_compact, llm_fw = _build_centroid(corpus, "llm_default")
    literary_compact, literary_fw = _build_centroid(corpus, "literary")
    amateur_compact, _ = _build_centroid(corpus, "human_amateur")
    expository_compact, _ = _build_centroid(corpus, "expository")
    own_compact, own_fw, own_spread = _build_caller_model_reference(corpus, caller_model)

    llm_cos, llm_delta = _both_distances(llm_fw)
    literary_cos, literary_delta = _both_distances(literary_fw)
    own_cos, own_delta = _both_distances(own_fw) if own_fw is not None else (None, None)
    neutral_cos = neutral_delta = None
    if neutral_baseline:
        neutral_cos, neutral_delta = _both_distances(
            compute_signals(neutral_baseline).fw_distribution
        )

    candidate_compact = _signals_to_compact(candidate_sig)
    candidate_compact["fw_cosine_from_llm_centroid"] = round(llm_cos, 4)
    candidate_compact["fw_cosine_from_human_literary"] = round(literary_cos, 4)
    candidate_compact["fw_delta_from_llm_centroid"] = round(llm_delta, 4)
    candidate_compact["fw_delta_from_human_literary"] = round(literary_delta, 4)
    if own_cos is not None:
        candidate_compact["fw_cosine_from_own_model_default"] = round(own_cos, 4)
        candidate_compact["fw_delta_from_own_model_default"] = round(own_delta, 4)
    if neutral_cos is not None:
        candidate_compact["fw_cosine_from_neutral_baseline"] = round(neutral_cos, 4)
        candidate_compact["fw_delta_from_neutral_baseline"] = round(neutral_delta, 4)

    references: dict = {
        "llm_default_centroid": llm_compact,
        "human_literary_centroid": literary_compact,
        "human_amateur_centroid": amateur_compact,
        "expository_centroid": expository_compact,
    }
    if own_compact is not None:
        references["caller_model_default"] = own_compact
    elif caller_model is not None:
        references["caller_model_default"] = {
            "model": caller_model,
            "n_samples": 0,
            "note": "model not in reference corpus; falling back to LLM centroid",
        }

    # ─── Author / style distance (opt-in via target_styles) ──────────
    style_distances: dict | None = None
    if target_styles:
        style_distances = {}
        for token in target_styles:
            # Try author lookup first
            ents = corpus.by_author(token)
            kind = "author"
            if not ents:
                # Fall back to literary-tag lookup
                ents = [e for e in corpus.by_tag(token) if "literary" in e.tags]
                kind = "tag"
            if not ents:
                style_distances[token] = {"status": "not_found", "note":
                    f"no author or literary-tag entries match {token!r}"}
                continue
            sc = corpus.aggregate_fw_distribution(ents)
            spread = corpus.intra_cluster_spread(ents)
            cos = function_word_cosine_distance(cand_fw, sc)
            delta = function_word_burrows_delta(cand_fw, sc, mfw) if mfw else 0.0
            entry = {
                "kind": kind,
                "n_samples": len(ents),
                "fw_cosine": round(cos, 4),
                "fw_delta": round(delta, 4),
                "intra_dist_p95": round(spread["intra_dist_p95"], 4),
                "calibration_reliable": spread["calibration_reliable"],
                "burstiness": round(sum(e.signals.burstiness for e in ents) / len(ents), 4),
            }
            # Calibrated departure ratio: cos within style's intra-cluster p95
            # the style's own intra-cluster spread? Closer than p95 = the
            # candidate is plausibly inside that style's natural variance.
            p95 = spread["intra_dist_p95"]
            if p95 > 0:
                entry["distance_ratio_vs_p95"] = round(cos / p95, 2)
            style_distances[token] = entry
        references["style_distances"] = style_distances

    notes = _interpretation(
        candidate_sig,
        own_model_distance=own_cos,
        own_model_spread=own_spread,
        llm_centroid_distance=llm_cos,
        human_literary_distance=literary_cos,
        neutral_baseline_distance=neutral_cos,
        own_model_delta=own_delta,
        llm_centroid_delta=llm_delta,
        human_literary_delta=literary_delta,
    )

    return StylometricToolReport(
        candidate=candidate_compact,
        references=references,
        interpretation_notes=notes,
    )


# ─── Re-export of corpus rebuild for the CLI ─────────────────────────────
# Same deferred-import pattern: don't reach into _corpus until the call
# happens, to avoid the circular dependency at module load.
def rebuild_cache():
    """Force a full cache rebuild and reload — re-export of
    `_corpus.rebuild_cache` for the unified CLI in `owtn/tools/__main__.py`."""
    from ._corpus import rebuild_cache as _rebuild
    return _rebuild()
