"""Writing-style tool — surface-level register signals (vocab grade, sentence
length, paragraph length, dialogue frequency) positioned against the project
calibration corpus.

Complementary to:
- `slop_score` — catches AI-default register via slop-list matching and
  contrast patterns; says "this reads as AI."
- `stylometry` — positions voice in a function-word vector space; says
  "your voice is X distance from this author / your model's default."

This tool answers a different question: *what does the surface shape of
the prose look like?* — useful for revision signals like "paragraphs too
short, sentences too uniform, no dialogue, vocabulary too elementary."

All four metrics are scalars with no preferred direction — "high" or "low"
is good or bad depending on the target register. The tool reports raw
values plus the per-bucket median + p90 from the calibration corpus so
the agent can interpret each scalar against concrete anchors.

Metric definitions (matching slop-score upstream where applicable):
- `vocab_level`: mean syllables per word (heuristic syllable count). The
  vocabulary term of the Flesch-Kincaid formula, isolated from the
  sentence-length term — so this metric reads pure vocab sophistication
  even when the prose's sentence segmentation is far from the corpus
  median (e.g. one-sentence incantatory paragraphs).
- `avg_sentence_length`: words per sentence (spaCy sentence segmentation).
- `avg_paragraph_length`: words per paragraph (split on \\n\\s*\\n).
- `dialogue_frequency`: dialogue spans per 1000 characters of text.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from owtn.judging.tier_a.preprocessing import extract_dialogue, get_nlp, get_paragraphs


# ─── Reference distribution (calibrated on data/voice-references) ─────────
# Per-metric, per-bucket median + p90. Hard-coded from the one-shot sweep
# at lab/scripts/writing_style_calibration_sweep.py. Rerun and update if
# the corpus or metric definitions change.
REFERENCE_DISTRIBUTION: dict[str, dict[str, dict[str, float]]] = {
    # vocab_level = mean syllables per word. Bucket medians cluster tightly
    # in [1.31, 1.43] — vocabulary sophistication, measured purely, is a
    # weak register discriminator. Most of the register signal that the
    # earlier FK-grade vocab_level appeared to carry was actually the
    # sentence-length term; that signal has moved to avg_sentence_length
    # where it belongs.
    "vocab_level": {
        "human_literary":       {"median": 1.36, "p90": 1.50},
        "human_amateur":        {"median": 1.36, "p90": 1.52},
        "frontier_llm_default": {"median": 1.31, "p90": 1.42},
        "older_llm_default":    {"median": 1.43, "p90": 1.59},
    },
    "avg_sentence_length": {
        "human_literary":       {"median": 20.52, "p90": 37.89},
        "human_amateur":        {"median": 12.84, "p90": 24.00},
        "frontier_llm_default": {"median": 11.15, "p90": 15.95},
        "older_llm_default":    {"median": 16.40, "p90": 23.83},
    },
    "avg_paragraph_length": {
        # Note: p90s are noisy here — some literary excerpts are single
        # long paragraphs, inflating the upper tail. Median is the reliable
        # anchor for this metric.
        "human_literary":       {"median": 102.45, "p90": 614.0},
        "human_amateur":        {"median":  48.20, "p90": 368.0},
        "frontier_llm_default": {"median":  49.88, "p90":  75.20},
        "older_llm_default":    {"median":  55.71, "p90": 752.0},
    },
    "dialogue_frequency": {
        "human_literary":       {"median": 1.41, "p90": 5.26},
        "human_amateur":        {"median": 1.80, "p90": 4.44},
        "frontier_llm_default": {"median": 2.73, "p90": 5.71},
        "older_llm_default":    {"median": 1.60, "p90": 5.04},
    },
}

CORPUS_SIZES = {
    "human_literary": 572,
    "human_amateur": 13,
    "frontier_llm_default": 34,
    "older_llm_default": 339,
}


# ─── Metric primitives ────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\b\w+\b")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")
_TRAILING_E_RE = re.compile(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$")


def _count_syllables(word: str) -> int:
    """Heuristic syllable count — port of slop-score's JS implementation.

    Strips trailing silent-e patterns and counts remaining vowel groups.
    Sufficient for population-level FK estimates; not a phonetic dictionary.
    """
    w = word.lower()
    if len(w) <= 3:
        return 1
    w = _TRAILING_E_RE.sub("", w)
    if w.startswith("y"):
        w = w[1:]
    groups = _VOWEL_GROUP_RE.findall(w)
    return len(groups) if groups else 1


def _mean_syllables_per_word(words: list[str]) -> float:
    """Mean syllables per word — the vocabulary term of FK, isolated.

    Sentence-length-independent on purpose: this prevents passages whose
    sentence segmentation departs from the calibration corpus (e.g.
    one-sentence incantatory paragraphs) from registering false vocab
    sophistication via the FK formula's W/S term.
    """
    if not words:
        return 0.0
    return sum(_count_syllables(w) for w in words) / len(words)


# ─── Placement helpers ────────────────────────────────────────────────────

def _placement(value: float, metric: str) -> str:
    """Locate `value` against the per-bucket medians for `metric`.

    Buckets are sorted by median ascending; the placement string names where
    the value falls in that ordered ladder.
    """
    buckets = REFERENCE_DISTRIBUTION[metric]
    ladder = sorted(buckets.items(), key=lambda kv: kv[1]["median"])
    medians = [(name, stats["median"]) for name, stats in ladder]

    if value <= medians[0][1]:
        return f"at or below {medians[0][0]} median ({medians[0][1]:.1f})"
    for i in range(len(medians) - 1):
        lo_name, lo_v = medians[i]
        hi_name, hi_v = medians[i + 1]
        if lo_v < value <= hi_v:
            return (f"between {lo_name} and {hi_name} medians "
                    f"({lo_v:.1f}-{hi_v:.1f})")
    top_name, top_v = medians[-1]
    return f"above {top_name} median ({top_v:.1f})"


# ─── Report ───────────────────────────────────────────────────────────────

@dataclass
class WritingStyleReport:
    """Compact tool response for an agent. JSON serialises to ~1.5-2 kB.

    `metrics` are the raw scalar values. `placements` gives a one-line
    position string per metric so the agent can scan without parsing the
    full reference table. `reference_distribution` is the per-metric
    per-bucket median + p90 from the calibration corpus. `comparison` is
    None unless `compare_to` was passed; when populated, it holds the
    averaged reference metrics plus per-metric deltas (candidate minus
    reference) so an agent can see how a draft has shifted toward or
    away from a target passage.
    """
    metrics: dict
    placements: dict
    reference_distribution: dict
    interpretation_notes: str
    comparison: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _closest_bucket(value: float, metric: str) -> str:
    """Bucket whose median is closest to `value` for `metric`."""
    buckets = REFERENCE_DISTRIBUTION[metric]
    return min(buckets.items(), key=lambda kv: abs(kv[1]["median"] - value))[0]


def _interpretation(
    metrics: dict,
    placements: dict,
    comparison: dict | None = None,
) -> str:
    """Per-metric register translation plus a cross-metric synthesis line.

    Each metric gets a sentence keyed to its calibration thresholds, naming
    the implied register (elementary / LLM-default / mid / literary). The
    closing line detects when ≥3 of the 4 metrics align with the same
    bucket — that's the "your prose has a shape" signal.
    """
    parts: list[str] = []

    vl = metrics["vocab_level"]
    sl = metrics["avg_sentence_length"]
    pl = metrics["avg_paragraph_length"]
    df = metrics["dialogue_frequency"]

    # Vocab level — mean syllables/word. Bucket medians cluster tightly
    # (frontier 1.31, amateur 1.36, literary 1.36, older 1.43); the metric
    # is a weak register discriminator on its own. Thresholds are keyed
    # to where the candidate sits relative to the cluster.
    if vl < 1.25:
        parts.append(
            f"Vocab {vl:.2f} syllables/word is below all bucket medians "
            f"— diction is monosyllabic / clipped."
        )
    elif vl < 1.40:
        parts.append(
            f"Vocab {vl:.2f} syllables/word is at human-literary / amateur "
            f"/ frontier-LLM medians — modal range."
        )
    elif vl < 1.55:
        parts.append(
            f"Vocab {vl:.2f} syllables/word is above human medians, near "
            f"older-LLM default — Latinate / specialist diction."
        )
    else:
        parts.append(
            f"Vocab {vl:.2f} syllables/word is above all bucket p90s "
            f"except older-LLM — heavily polysyllabic / formal register."
        )

    # Sentence length — bucket medians 11.2/12.8/16.4/21.9.
    if sl < 11.0:
        parts.append(
            f"Avg sentence length {sl:.1f} wps is below all reference "
            f"medians — sentences are unusually short or fragmented."
        )
    elif sl < 14.0:
        parts.append(
            f"Avg sentence length {sl:.1f} wps is at LLM-default level — "
            f"sentences are uniform-short."
        )
    elif sl < 19.0:
        parts.append(
            f"Avg sentence length {sl:.1f} wps is in human-amateur / "
            f"older-LLM range."
        )
    else:
        parts.append(
            f"Avg sentence length {sl:.1f} wps is at human-literary level "
            f"or above — long sentences."
        )

    # Paragraph length — bucket medians 48/50/56/95 (clustered low except
    # human_literary; p90s noisy due to single-paragraph excerpts).
    if pl < 30:
        parts.append(
            f"Avg paragraph length {pl:.0f} wpp is well below all reference "
            f"medians (lowest is 48) — paragraphs are unusually short."
        )
    elif pl < 70:
        parts.append(
            f"Avg paragraph length {pl:.0f} wpp is at human-amateur / "
            f"LLM-default level."
        )
    elif pl < 130:
        parts.append(
            f"Avg paragraph length {pl:.0f} wpp is at human-literary "
            f"median (95)."
        )
    else:
        parts.append(
            f"Avg paragraph length {pl:.0f} wpp — long literary paragraphs."
        )

    # Dialogue frequency — narrow band (literary 1.5, amateur 1.8,
    # older 1.6, frontier 2.7); flag the absent/sparse/heavy extremes.
    if df < 0.1:
        parts.append("No dialogue detected — purely narrative passage.")
    elif df < 1.0:
        parts.append(
            f"Dialogue frequency {df:.1f}/1k chars is sparse — narrative-"
            f"dominant passage."
        )
    elif df < 3.5:
        parts.append(
            f"Dialogue frequency {df:.1f}/1k chars is at human-writing level."
        )
    else:
        parts.append(
            f"Dialogue frequency {df:.1f}/1k chars is dialogue-heavy "
            f"(frontier-LLM median 2.7, p90 5.7)."
        )

    # Cross-metric synthesis: if ≥3 metrics cluster around one bucket,
    # name the register. Dialogue is excluded since its bucket spread is
    # too narrow to be a register signal.
    bucket_votes = [
        _closest_bucket(vl, "vocab_level"),
        _closest_bucket(sl, "avg_sentence_length"),
        _closest_bucket(pl, "avg_paragraph_length"),
    ]
    counts: dict[str, int] = {}
    for b in bucket_votes:
        counts[b] = counts.get(b, 0) + 1
    top_bucket, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count == 3:
        parts.append(
            f"All three of vocab/sentence/paragraph cluster around "
            f"{top_bucket} medians — surface form reads as that register."
        )

    # Comparison delta line — concise per-metric diff vs reference.
    if comparison is not None:
        d = comparison["delta_metrics"]
        parts.append(
            f"Compared to {comparison['n_passages']}-passage reference: "
            f"vocab {d['vocab_level']:+.1f}, "
            f"sentences {d['avg_sentence_length']:+.1f} wps, "
            f"paragraphs {d['avg_paragraph_length']:+.0f} wpp, "
            f"dialogue {d['dialogue_frequency']:+.1f}/1k chars."
        )

    return " ".join(parts)


# ─── Tool entry point ─────────────────────────────────────────────────────

def writing_style(
    passage: str,
    compare_to: str | list[str] | None = None,
) -> WritingStyleReport:
    """Compute writing-style metrics on `passage` and position them against
    the project calibration corpus.

    Args:
        passage: Prose to analyse. Empty strings return zero-filled metrics.
        compare_to: Optional reference passage (or list of passages) to
            compare against. When provided, the report's `comparison` field
            holds the averaged reference metrics and per-metric deltas
            (candidate minus reference). Useful for revision tracking
            ("did v2 lengthen sentences vs v1?") or exemplar-targeting
            ("how far is my draft from this Hemingway in shape?"). The
            agent can use `lookup_exemplar()` to retrieve corpus texts to
            pass here.

    Returns:
        WritingStyleReport with raw metrics, placement strings, the
        reference distribution, and per-metric interpretation notes.
    """
    if not passage.strip():
        zero = {
            "vocab_level": 0.0,
            "avg_sentence_length": 0.0,
            "avg_paragraph_length": 0.0,
            "dialogue_frequency": 0.0,
        }
        return WritingStyleReport(
            metrics=zero,
            placements={k: "empty passage" for k in zero},
            reference_distribution=REFERENCE_DISTRIBUTION,
            interpretation_notes="Empty passage — no style signal computed.",
        )

    nlp = get_nlp()
    doc = nlp(passage)

    sentences = [s for s in doc.sents if s.text.strip()]
    words = _WORD_RE.findall(passage)
    paragraphs = get_paragraphs(passage)
    dialogues, _ = extract_dialogue(passage)

    n_sentences = len(sentences)
    n_words = len(words)
    n_paragraphs = max(len(paragraphs), 1)
    n_chars = len(passage)

    metrics = {
        "vocab_level": round(_mean_syllables_per_word(words), 3),
        "avg_sentence_length": round(n_words / n_sentences, 2) if n_sentences else 0.0,
        "avg_paragraph_length": round(n_words / n_paragraphs, 2),
        "dialogue_frequency": round((len(dialogues) / n_chars) * 1000, 2) if n_chars else 0.0,
    }

    placements = {m: _placement(v, m) for m, v in metrics.items()}

    # Comparison against one or more reference passages. Recursive call
    # passes compare_to=None so we don't loop. Multiple references are
    # averaged into a single anchor.
    comparison_data = None
    if compare_to is not None:
        refs = [compare_to] if isinstance(compare_to, str) else list(compare_to)
        ref_reports = [writing_style(t) for t in refs]
        avg_metrics = {
            k: round(sum(r.metrics[k] for r in ref_reports) / len(ref_reports), 2)
            for k in metrics
        }
        delta_metrics = {
            k: round(metrics[k] - avg_metrics[k], 2) for k in metrics
        }
        comparison_data = {
            "n_passages": len(ref_reports),
            "reference_metrics": avg_metrics,
            "delta_metrics": delta_metrics,
        }

    notes = _interpretation(metrics, placements, comparison_data)

    return WritingStyleReport(
        metrics=metrics,
        placements=placements,
        reference_distribution=REFERENCE_DISTRIBUTION,
        interpretation_notes=notes,
        comparison=comparison_data,
    )
