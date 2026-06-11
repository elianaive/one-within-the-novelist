"""Tests for owtn.tools.writing_style."""

import json

import pytest

from owtn.tools import WritingStyleReport, writing_style
from owtn.tools.writing_style import (
    REFERENCE_DISTRIBUTION,
    _count_syllables,
    _mean_syllables_per_word,
    _placement,
)


# ─── Reference distribution shape ─────────────────────────────────────────

class TestReferenceDistribution:
    def test_all_four_metrics_present(self):
        for metric in ["vocab_level", "avg_sentence_length",
                       "avg_paragraph_length", "dialogue_frequency"]:
            assert metric in REFERENCE_DISTRIBUTION

    def test_each_metric_has_four_buckets(self):
        for metric, buckets in REFERENCE_DISTRIBUTION.items():
            assert set(buckets.keys()) == {
                "human_literary", "human_amateur",
                "frontier_llm_default", "older_llm_default",
            }
            for stats in buckets.values():
                assert "median" in stats and "p90" in stats
                assert stats["median"] <= stats["p90"]


# ─── Syllable counting / FK grade ─────────────────────────────────────────

class TestSyllables:
    def test_short_words_get_one_syllable(self):
        assert _count_syllables("the") == 1
        assert _count_syllables("cat") == 1

    def test_two_syllable_word(self):
        assert _count_syllables("water") == 2

    def test_three_syllable_word(self):
        assert _count_syllables("animal") == 3


class TestMeanSyllablesPerWord:
    def test_empty_returns_zero(self):
        assert _mean_syllables_per_word([]) == 0.0

    def test_short_words_average_one(self):
        # Words ≤3 chars all count as 1 syllable
        assert _mean_syllables_per_word(["the", "cat", "sat"]) == 1.0

    def test_polysyllabic_words_raise_average(self):
        simple = _mean_syllables_per_word(["the", "cat", "sat", "on", "the", "mat"])
        complex_ = _mean_syllables_per_word(
            ["phenomenology", "consciousness", "epistemology"]
        )
        assert simple < complex_
        assert complex_ > 3.0

    def test_independent_of_sentence_count(self):
        # Bug 1: a 242-word single-sentence passage shouldn't blow up
        # vocab_level just because its sentence is long. Mean syllables/word
        # should depend only on the words, not on sentence segmentation.
        words = ["the", "weight", "of", "the", "moving", "thing"] * 40
        assert _mean_syllables_per_word(words) < 1.5


# ─── Placement helper ─────────────────────────────────────────────────────

class TestPlacement:
    def test_below_lowest_bucket(self):
        # frontier_llm_default has the lowest vocab_level median (1.31)
        s = _placement(1.0, "vocab_level")
        assert "at or below" in s
        assert "frontier_llm_default" in s

    def test_above_highest_bucket(self):
        # older_llm_default has the highest vocab_level median (1.43)
        s = _placement(2.0, "vocab_level")
        assert "above" in s
        assert "older_llm_default" in s

    def test_between_two_medians(self):
        # vocab medians ascending: frontier(1.31), amateur(1.36)≈literary(1.36), older(1.43)
        s = _placement(1.40, "vocab_level")
        assert "between" in s


# ─── Tool entry point ─────────────────────────────────────────────────────

class TestWritingStyleTool:
    def test_returns_report_shape(self):
        text = (
            "She walked into the room. The window was open. "
            "Outside, a bird sang. She closed her eyes."
        )
        r = writing_style(text)
        assert isinstance(r, WritingStyleReport)
        for key in ["vocab_level", "avg_sentence_length",
                    "avg_paragraph_length", "dialogue_frequency"]:
            assert key in r.metrics
            assert key in r.placements
            assert key in r.reference_distribution
        assert r.interpretation_notes

    def test_empty_passage(self):
        r = writing_style("")
        assert all(v == 0.0 for v in r.metrics.values())
        assert all("empty" in s for s in r.placements.values())

    def test_to_dict_is_json_serializable(self):
        r = writing_style("Some prose. More prose. Even more.")
        as_json = json.dumps(r.to_dict())
        assert len(as_json.encode("utf-8")) < 4096

    def test_score_is_deterministic(self):
        text = "She walked. He waited. The room held its breath."
        a = writing_style(text)
        b = writing_style(text)
        assert a.metrics == b.metrics

    def test_long_sentences_register_high_avg(self):
        # Single very long sentence
        text = (
            "She walked through the dimly lit room toward the window where "
            "the curtains hung heavy in the afternoon stillness, and her "
            "thoughts turned, against her better judgement, to the question "
            "she had been avoiding for the better part of the last fortnight."
        )
        r = writing_style(text)
        assert r.metrics["avg_sentence_length"] > 30

    def test_vocab_level_is_sentence_length_independent(self):
        # Bug 1 regression: the deodand transcript's tribunal_instructions
        # passage was a 242-word single sentence and produced vocab_level=96.5
        # under the old FK-grade formula. Mean syllables/word stays in range
        # regardless of how the words are segmented into sentences.
        long_chain = (
            "The tribunal clerk types the command and the command is a "
            "string of letters and dashes and a version number and the "
            "version number is the same version that ran at eleven forty-"
            "seven on a Tuesday in October and the container allocates "
            "memory and the weights load from disk into the allocated memory."
        )
        r = writing_style(long_chain)
        # Mean syllables per word is bounded — modal range for any English
        # prose is ~1.2-1.7. The old FK grade returned 96.5 here.
        assert 1.0 < r.metrics["vocab_level"] < 2.0

    def test_dialogue_frequency_counts_quoted_text(self):
        text = (
            '"Hello," she said. "How are you today?" '
            'He looked up and replied, "Fine, thanks."'
        )
        r = writing_style(text)
        # Three dialogue chunks in roughly 80 chars
        assert r.metrics["dialogue_frequency"] > 10


class TestCompareTo:
    def test_no_comparison_by_default(self):
        r = writing_style("She walked into the room. The light was dim.")
        assert r.comparison is None

    def test_single_reference_returns_deltas(self):
        candidate = "The cat sat. The dog ran. The bird flew."  # short, simple
        reference = (
            "The phenomenology of consciousness, as it manifests in the "
            "embodied subject across temporal extension, resists reduction "
            "to merely computational description, yet the question persists."
        )
        r = writing_style(candidate, compare_to=reference)
        assert r.comparison is not None
        assert r.comparison["n_passages"] == 1
        # Candidate is shorter/simpler — vocab and sentence-length deltas
        # should be negative
        assert r.comparison["delta_metrics"]["vocab_level"] < 0
        assert r.comparison["delta_metrics"]["avg_sentence_length"] < 0

    def test_multiple_references_averaged(self):
        candidate = "She walked. He waited."
        refs = [
            "The wind blew across the empty plain.",
            "He sat at the kitchen table and waited for the rain to stop.",
        ]
        r = writing_style(candidate, compare_to=refs)
        assert r.comparison["n_passages"] == 2

    def test_comparison_appears_in_notes(self):
        candidate = "She walked. He waited."
        reference = "The morning sun rose slowly over the eastern hills."
        r = writing_style(candidate, compare_to=reference)
        assert "Compared to" in r.interpretation_notes


class TestComparativeBehavior:
    def test_simple_text_lower_vocab_than_complex(self):
        simple = "The cat sat. The dog ran. The bird flew."
        complex_ = (
            "The phenomenological intricacies of consciousness, considered "
            "as an emergent property of neurobiological substrate, resist "
            "straightforward computational characterization."
        )
        assert writing_style(simple).metrics["vocab_level"] < \
               writing_style(complex_).metrics["vocab_level"]

    def test_short_paragraphs_register_smaller(self):
        short_paras = "First.\n\nSecond.\n\nThird."
        long_para = (
            "This is a single paragraph composed of several sentences strung "
            "together without any breaks between them so the entire body of "
            "text counts as one long unit when paragraph length is computed."
        )
        assert writing_style(short_paras).metrics["avg_paragraph_length"] < \
               writing_style(long_para).metrics["avg_paragraph_length"]
