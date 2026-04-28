"""Tests for the stylometry primitives in owtn.tools.stylometry."""

import pytest

from owtn.tools.stylometry import (
    FUNCTION_POS,
    aggregate_function_word_distribution,
    burstiness,
    compute_signals,
    function_word_cosine_distance,
    function_word_distribution,
    mattr,
    near_default,
)
from owtn.judging.tier_a.preprocessing import get_nlp


@pytest.fixture(scope="module")
def nlp():
    return get_nlp()


# ─── Burstiness ───────────────────────────────────────────────────────────

class TestBurstiness:
    def test_uniform_sentence_lengths_low_cv(self, nlp):
        text = " ".join(["The cat sat on the mat."] * 8)
        assert burstiness(nlp(text)) == pytest.approx(0.0, abs=0.05)

    def test_varied_sentence_lengths_higher_cv(self, nlp):
        text = (
            "She left. "
            "He stayed at the kitchen table for a long time and watched the snow accumulate on the sill, "
            "considering what had been said and what had not been said and what he might have done differently. "
            "Yes."
        )
        assert burstiness(nlp(text)) > 0.6

    def test_too_few_sentences_returns_zero(self, nlp):
        assert burstiness(nlp("Just one sentence here.")) == 0.0


# ─── MATTR ────────────────────────────────────────────────────────────────

class TestMATTR:
    def test_repetitive_low_mattr(self, nlp):
        text = " ".join(["the cat sat on the mat"] * 50)
        assert mattr(nlp(text), window=20) < 0.5

    def test_diverse_higher_mattr(self, nlp):
        text = (
            "Diverse vocabulary multiplies stylistic possibilities while complicating "
            "computational analysis. Lexical breadth correlates loosely with apparent "
            "sophistication, though authorship attribution depends on subtler closed-class signals."
        )
        assert mattr(nlp(text), window=10) > 0.7

    def test_short_text_falls_back_to_ttr(self, nlp):
        text = "the cat sat on a mat"
        m = mattr(nlp(text), window=500)
        # 6 unique words / 6 total = 1.0
        assert m == pytest.approx(1.0)


# ─── Function-word distribution ───────────────────────────────────────────

class TestFunctionWordDistribution:
    def test_returns_only_function_words(self, nlp):
        # POS-based: "the" is DET (function word), "cat" is NOUN (not), "sat" is VERB (not)
        d = function_word_distribution(nlp("the cat sat"))
        assert "the" in d
        assert "cat" not in d
        assert "sat" not in d

    def test_frequencies_normalize_by_total(self, nlp):
        # 9 non-punct tokens; "the" appears 2x → 2/9
        text = "the cat sat on the mat with a dog"
        d = function_word_distribution(nlp(text))
        assert d["the"] == pytest.approx(2 / 9)
        # "on" and "with" are ADP (function words), "a" is DET
        assert "on" in d and "with" in d and "a" in d

    def test_empty_text_returns_empty_dict(self, nlp):
        d = function_word_distribution(nlp(""))
        assert d == {}


class TestFunctionWordCosineDistance:
    def test_identical_distributions_zero_distance(self, nlp):
        d = function_word_distribution(nlp("the cat sat on the mat"))
        assert function_word_cosine_distance(d, d) == pytest.approx(0.0, abs=1e-6)

    def test_disjoint_keys_orthogonal(self):
        # Keys don't overlap → cosine of vector with all zeros in opposite slots = 1
        d1 = {"the": 1.0}
        d2 = {"a": 1.0}
        assert function_word_cosine_distance(d1, d2) == pytest.approx(1.0)

    def test_one_empty_returns_one(self):
        d_full = {"the": 0.5, "of": 0.5}
        assert function_word_cosine_distance(d_full, {}) == 1.0


class TestAggregate:
    def test_aggregate_takes_union_of_keys(self, nlp):
        d1 = function_word_distribution(nlp("the the the cat"))
        d2 = function_word_distribution(nlp("a a a dog"))
        d3 = function_word_distribution(nlp("the a dog cat"))
        agg = aggregate_function_word_distribution([d1, d2, d3])
        # Both "the" and "a" should appear in aggregate
        assert "the" in agg
        assert "a" in agg
        # "the" mean: (3/4 + 0 + 1/4) / 3 (d2 doesn't contain "the")
        assert agg["the"] == pytest.approx((0.75 + 0.0 + 0.25) / 3)

    def test_empty_input_returns_empty(self):
        agg = aggregate_function_word_distribution([])
        assert agg == {}


# ─── compute_signals integration ──────────────────────────────────────────

class TestComputeSignals:
    def test_returns_signals_dataclass(self):
        # 6 words, not 8 — periods are punct, "The cat sat" + "The dog slept" = 6 tokens
        sig = compute_signals("The cat sat. The dog slept.")
        assert sig.word_count == 6
        assert sig.sentence_count == 2
        assert sig.burstiness >= 0.0
        assert 0.0 <= sig.mattr <= 1.0
        # POS-based: only "the" appears as function word ("cat", "dog", "sat", "slept" are content)
        assert "the" in sig.fw_distribution
        assert "cat" not in sig.fw_distribution


# ─── near_default heuristic ───────────────────────────────────────────────

class TestNearDefault:
    def test_low_burstiness_low_distance_flagged(self):
        sig = compute_signals(" ".join(["The cat sat on the mat."] * 6))
        assert near_default(sig, fw_distance_from_baseline=0.02) is True

    def test_high_burstiness_not_flagged(self, nlp):
        # Varied prose; high burstiness alone passes the floor
        text = (
            "She left. "
            "He sat at the table for a long time afterward, watching the snow gather on the sill. "
            "Yes."
        )
        sig = compute_signals(text)
        assert near_default(sig, fw_distance_from_baseline=0.02) is False
