"""Tests for owtn.tools.slop_score — port of EQ-Bench's slop-score."""

import json

import pytest

from owtn.tools import SlopScoreReport, slop_score
from owtn.tools.slop_score import (
    SLOP_THRESHOLD,
    STAGE1_REGEXES,
    STAGE2_REGEXES,
    _build_verb_stream,
    _load_slop_bigrams,
    _load_slop_trigrams,
    _load_slop_words,
    _normalize_text,
    _tokenize_words,
)
from owtn.judging.tier_a.preprocessing import get_nlp


# ─── Data files / loaders ─────────────────────────────────────────────────

class TestSlopListLoading:
    def test_word_list_loaded(self):
        words = _load_slop_words()
        assert len(words) > 1000
        # Spot-check a few stable entries from the ported list
        assert "delve" in words
        assert "shadows" in words

    def test_bigram_list_loaded(self):
        bis = _load_slop_bigrams()
        assert len(bis) > 100
        assert "deep breath" in bis

    def test_trigram_list_loaded(self):
        tris = _load_slop_trigrams()
        assert len(tris) > 300
        assert "voice barely whisper" in tris


# ─── Tokenization / normalization ─────────────────────────────────────────

class TestNormalization:
    def test_curly_quotes_to_straight(self):
        assert _normalize_text("“hello”") == '"hello"'
        assert _normalize_text("don’t") == "don't"

    def test_em_and_en_dashes_to_hyphen(self):
        assert _normalize_text("yes—no") == "yes-no"
        assert _normalize_text("a–b") == "a-b"


class TestTokenizer:
    def test_lowercases_and_extracts(self):
        toks = _tokenize_words("The Cat Sat.")
        assert toks == ["the", "cat", "sat"]

    def test_strips_apostrophes_at_edges(self):
        # "'twas" → "twas"; "don't" preserved; trailing "'" stripped
        toks = _tokenize_words("'Twas don't end'")
        assert toks == ["twas", "don't", "end"]


# ─── Stage 1 regex sanity ─────────────────────────────────────────────────

class TestStage1Patterns:
    def test_not_but_matches(self):
        rx = STAGE1_REGEXES["RE_NOT_BUT"]
        # Canonical "not X, but Y" pattern
        assert rx.search("It was not just warm, but alive and pulsing.")

    def test_no_longer_matches(self):
        rx = STAGE1_REGEXES["RE_NO_LONGER"]
        assert rx.search("It was no longer winter. It was spring.")

    def test_not_but_skips_excluded_continuations(self):
        rx = STAGE1_REGEXES["RE_NOT_BUT"]
        # "but when/if/that/..." follow-ons are excluded by the lookahead
        assert not rx.search("She did not understand, but when she tried again it worked.")


# ─── Stage 2 stream + regex sanity ────────────────────────────────────────

class TestVerbStream:
    def test_replaces_verbs_only(self):
        nlp = get_nlp()
        doc = nlp("She walked. The cat is here.")
        stream = _build_verb_stream(doc)
        # Verbs replaced; non-verbs preserved
        assert "VERB" in stream
        assert "She" in stream
        assert "cat" in stream

    def test_stream_round_trips_non_verb_text(self):
        nlp = get_nlp()
        original = "The chair sat empty in the corner."
        doc = nlp(original)
        stream = _build_verb_stream(doc)
        # Length is preserved per-token (verb tokens replaced 1:1 by 'VERB');
        # non-verb words are unchanged.
        assert "chair" in stream and "corner" in stream


# ─── Tool entry point ─────────────────────────────────────────────────────

class TestSlopScore:
    def test_returns_report_shape(self):
        r = slop_score("She walked. He waited. The room held its breath.")
        assert isinstance(r, SlopScoreReport)
        assert 0.0 <= r.composite <= 100.0
        assert isinstance(r.is_slop, bool)
        assert isinstance(r.placement, str) and r.placement
        assert "slop_words_per_1k_words" in r.components
        assert "slop_bigrams_per_1k_words" in r.components
        assert "slop_trigrams_per_1k_words" in r.components
        assert "contrast_patterns_per_1k_chars" in r.components
        assert "match_counts" in r.components
        assert "slop_words" in r.top_hits
        assert "slop_bigrams" in r.top_hits
        assert "slop_trigrams" in r.top_hits
        assert "contrast_patterns" in r.top_hits
        # Reference distribution exposes the four calibration buckets
        for bucket in ("human_literary", "human_amateur",
                       "frontier_llm_default", "older_llm_default"):
            assert bucket in r.reference_distribution
            assert "median" in r.reference_distribution[bucket]
            assert "p90" in r.reference_distribution[bucket]
        assert r.interpretation_notes

    def test_empty_passage(self):
        r = slop_score("")
        assert r.composite == 0.0
        assert r.is_slop is False
        assert r.components["word_count"] == 0
        assert r.top_hits["slop_words"] == []

    def test_placement_human_for_clean_text(self):
        r = slop_score(
            "The hills across the valley of the Ebro were long and white. "
            "On this side there was no shade and no trees."
        )
        assert "human" in r.placement.lower()

    def test_placement_conspicuous_for_high_score(self):
        text = (
            "She took a deep breath, her heart pounding in her chest. "
            "Her eyes widened as the shadows danced. The air was thick "
            "with anticipation."
        )
        r = slop_score(text)
        # Should reach the older-LLM/conspicuous range
        assert r.placement in (
            "within older-LLM-default range",
            "above older-LLM p90 — conspicuous slop",
        )

    def test_human_minimalist_prose_scores_low(self):
        # Hemingway, "Hills Like White Elephants" opening — the canonical
        # anti-slop human text. Should land near zero.
        text = (
            "The hills across the valley of the Ebro were long and white. "
            "On this side there was no shade and no trees. "
            "Close against the side of the station there was the warm shadow "
            "of the building and a curtain made of strings of bamboo beads "
            "hung across the open door into the bar. The American and the "
            "girl with him sat at a table in the shade outside the building."
        )
        r = slop_score(text)
        assert r.composite < SLOP_THRESHOLD
        assert r.is_slop is False

    def test_ai_default_text_flagged(self):
        # Synthetic excerpt loaded with slop-score targets: contrast pattern,
        # banned vocabulary, fiction trigrams.
        text = (
            "She took a deep breath. Her heart pounded in her chest as the "
            "shadows danced across the floor. It wasn't just warm — it was "
            "alive. A shiver ran down her spine. She felt a flicker of "
            "something ancient stirring in the dust motes that danced in the "
            "afternoon light. The air was thick with anticipation."
        )
        r = slop_score(text)
        assert r.composite > SLOP_THRESHOLD
        assert r.is_slop is True
        # At least one component should fire concretely
        assert r.components["match_counts"]["slop_words"] > 0

    def test_score_is_deterministic(self):
        text = "She took a deep breath. The shadows danced."
        a = slop_score(text)
        b = slop_score(text)
        assert a.composite == b.composite
        assert a.components == b.components

    def test_to_dict_is_json_serializable(self):
        r = slop_score("She walked into the dimly lit room.")
        d = r.to_dict()
        as_json = json.dumps(d)
        # 4kB target for agent-facing tool reports
        assert len(as_json.encode("utf-8")) < 4096

    def test_top_hits_are_capped(self):
        # Trigger many slop-word hits via a slop-rich passage
        text = (
            "She took a deep breath, her heart pounding in her chest. The "
            "shadows danced across the dimly lit room. A shiver ran down her "
            "spine as her eyes widened in fear. The air hung thick with the "
            "scent of damp earth. She glanced at the rusted figure emerging "
            "from the shadows. Her brow furrowed in confusion. She felt a "
            "flicker of recognition. Tears streamed down her face as the "
            "knuckles turning white on the doorknob. Voice barely a whisper, "
            "she asked the question. "
        ) * 3
        r = slop_score(text)
        assert len(r.top_hits["slop_words"]) <= 10
        assert len(r.top_hits["slop_bigrams"]) <= 10
        assert len(r.top_hits["slop_trigrams"]) <= 10
        assert len(r.top_hits["contrast_patterns"]) <= 8

    def test_top_hits_sorted_descending(self):
        text = (
            "Shadows shadows shadows. The shadows danced. Her brow furrowed. "
            "Brow furrowed in confusion. Voice barely audible."
        )
        r = slop_score(text)
        counts = [c for _, c in r.top_hits["slop_words"]]
        assert counts == sorted(counts, reverse=True)


class TestCompareTo:
    def test_no_comparison_by_default(self):
        r = slop_score("She walked into the room.")
        assert r.comparison is None

    def test_single_reference_passage(self):
        candidate = "She took a deep breath, her heart pounding in her chest."
        reference = "The hills were long and white. The station stood empty."
        r = slop_score(candidate, compare_to=reference)
        assert r.comparison is not None
        assert r.comparison["n_passages"] == 1
        assert "reference_composite" in r.comparison
        assert "delta_composite" in r.comparison
        # Candidate is sloppier than reference, so delta should be positive
        assert r.comparison["delta_composite"] > 0

    def test_multiple_reference_passages_averaged(self):
        candidate = "She took a deep breath."
        refs = [
            "The wind blew across the empty plain.",
            "He sat at the kitchen table and waited.",
        ]
        r = slop_score(candidate, compare_to=refs)
        assert r.comparison["n_passages"] == 2

    def test_comparison_appears_in_notes(self):
        candidate = "She took a deep breath, her heart pounding in her chest."
        reference = "The hills were long and white."
        r = slop_score(candidate, compare_to=reference)
        assert "Compared to" in r.interpretation_notes


class TestBigrams:
    def test_bigram_hits_counted(self):
        # "deep breath" and "took deep" are both in the bigram list
        text = "She took a deep breath. He took a deep breath as well."
        r = slop_score(text)
        assert r.components["match_counts"]["slop_bigrams"] >= 2
        # The specific bigram should appear in top hits
        bg_phrases = [b[0] for b in r.top_hits["slop_bigrams"]]
        assert "deep breath" in bg_phrases

    def test_bigrams_contribute_to_composite(self):
        # Same prose with vs without slop bigrams should differ in score
        clean = "She walked into the room. The window was open. Outside, it rained."
        sloppy = " ".join(["She took a deep breath."] * 5)
        assert slop_score(sloppy).composite > slop_score(clean).composite


class TestComponents:
    def test_word_rate_per_1k_calculation(self):
        # Constructed to control word rate exactly: 50 words, 1 slop word
        text = ("The cat sat on the mat. " * 10).rstrip()  # 60 words
        # Add one known slop word
        text = text + " She felt a delve."  # adds 4 words, 1 slop word
        r = slop_score(text)
        # Should have ~1 slop word in ~64 words → ~15.6 per 1k
        assert r.components["match_counts"]["slop_words"] >= 1
        assert r.components["slop_words_per_1k_words"] > 10

    def test_contrast_per_1k_chars_units(self):
        # Verify rate is per 1k *characters*, not words
        text = "It was not just warm, but alive."  # ~32 chars, 1 contrast hit
        r = slop_score(text)
        assert r.components["match_counts"]["contrast_patterns"] >= 1
        # 1 hit / 32 chars * 1000 ≈ 31 per 1k chars
        assert r.components["contrast_patterns_per_1k_chars"] > 10
