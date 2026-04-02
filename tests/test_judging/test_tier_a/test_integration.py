"""Integration tests for owtn.judging.tier_a — the full analyze() pipeline."""

from owtn.judging.tier_a import DEFAULT_THRESHOLD, TierAResult, analyze


class TestAnalyze:
    def test_clean_literary_prose(self):
        text = (
            "The morning he left, the fields were still wet. Fog "
            "clung to the hedgerows and the lane was empty. He carried "
            "nothing. The door stayed open behind him.\n\n"
            "At the station, a woman sold him a ticket without looking up. "
            "The train was late. He sat on the bench and watched "
            "sparrows fight over a crust. When the train came, it came "
            "quietly, almost apologetically.\n\n"
            "He found a seat by the window. The countryside moved past "
            "like something he'd already forgotten. A child across the "
            "aisle stared at him. He stared back until the child looked "
            "away. Then he closed his eyes."
        )
        result = analyze(text)
        assert isinstance(result, TierAResult)
        assert result.passed is True
        assert result.composite_score < DEFAULT_THRESHOLD

    def test_slop_heavy_prose(self):
        text = (
            "She delved into the multifaceted tapestry of the realm, "
            "embarking on a profound endeavor. It's worth noting that "
            "the paradigm was truly nuanced. Moreover, the holistic "
            "landscape illuminated the intricate synergy.\n\n"
            "She delved into the multifaceted tapestry of the realm, "
            "embarking on a profound endeavor. Furthermore, the testament "
            "to comprehensive camaraderie was indeed captivating.\n\n"
            "She delved into the multifaceted tapestry of the realm, "
            "embarking on a profound endeavor. Nonetheless, the myriad "
            "plethora of catalyzed juxtapositions was pivotal."
        )
        result = analyze(text)
        assert result.composite_score > 0.05
        assert len(result.flagged_items) > 0

    def test_empty_text(self):
        result = analyze("")
        assert result.passed is True
        assert result.composite_score == 0.0

    def test_result_structure(self):
        result = analyze("A simple sentence for testing purposes.")
        assert hasattr(result, "composite_score")
        assert hasattr(result, "passed")
        assert hasattr(result, "filter_scores")
        assert hasattr(result, "flagged_items")
        assert isinstance(result.filter_scores, dict)
        assert all(
            key in result.filter_scores
            for key in [
                "banned_vocabulary", "construction_patterns", "burstiness",
                "mattr", "pronoun_density", "structural_patterns",
                "slop_ngrams", "fiction_anti_patterns",
            ]
        )

    def test_custom_threshold(self):
        text = "A normal sentence."
        strict = analyze(text, threshold=0.01)
        lenient = analyze(text, threshold=0.99)
        assert strict.composite_score == lenient.composite_score

    def test_all_filter_scores_bounded(self):
        text = (
            "She delved into the tapestry. "
            "Not just a house, but a monument. "
            "His heart pounded in his chest. "
            "She took deep breath."
        ) * 5
        result = analyze(text)
        for name, score in result.filter_scores.items():
            assert 0.0 <= score <= 1.0, f"{name} out of bounds: {score}"
