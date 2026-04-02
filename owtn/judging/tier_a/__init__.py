"""Tier A Anti-Slop Filter — deterministic NLP pipeline.

No LLM calls. Input is raw text; output is a composite slop score with
per-filter breakdown. Runs on every candidate at every stage (where enabled)
as a cheap pre-filter before expensive Tier B evaluation.

Usage:
    from owtn.judging.tier_a import analyze, TierAResult

    result = analyze(text)
    if not result.passed:
        print(f"Rejected: slop score {result.composite_score:.2f}")
"""

from dataclasses import dataclass, field

from .anti_patterns import score_fiction_anti_patterns
from .construction import score_construction_patterns
from .preprocessing import extract_dialogue, get_nlp, get_paragraphs
from .statistical import score_mattr, score_participial_openings, score_pronoun_density
from .structural import score_burstiness, score_structural_patterns
from .ngrams import score_slop_ngrams
from .vocabulary import score_banned_vocabulary

# ---------------------------------------------------------------------------
# Composite weights — from tier-a-anti-slop.md §Composite Score
# ---------------------------------------------------------------------------

COMPOSITE_WEIGHTS = {
    "banned_vocabulary": 0.20,
    "construction_patterns": 0.20,
    "burstiness": 0.15,
    "mattr": 0.15,
    "pronoun_density": 0.10,
    "structural_patterns": 0.10,
    "slop_ngrams": 0.05,
    "fiction_anti_patterns": 0.05,
}

DEFAULT_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class TierAResult:
    composite_score: float
    passed: bool
    filter_scores: dict[str, float]
    flagged_items: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze(text: str, threshold: float = DEFAULT_THRESHOLD) -> TierAResult:
    """Run the full Tier A anti-slop pipeline on input text.

    Single spaCy pass, then dispatch to all filters.
    Returns TierAResult with composite score, pass/fail, and per-filter breakdown.
    """
    nlp = get_nlp()
    doc = nlp(text)

    word_count = len([t for t in doc if not t.is_punct and not t.is_space])
    if word_count == 0:
        return TierAResult(
            composite_score=0.0, passed=True,
            filter_scores={}, flagged_items=[],
        )

    paragraphs = get_paragraphs(text)
    dialogues, _narrative = extract_dialogue(text)

    # Filter 1: Banned vocabulary
    banned_score, flagged = score_banned_vocabulary(doc, word_count)

    # Filter 2: Construction patterns
    construction_score = score_construction_patterns(text, doc, word_count)

    # Filter 3: Burstiness (standalone weight) + structural composite
    burstiness = score_burstiness(doc)
    structural = score_structural_patterns(doc, text, word_count, paragraphs)

    # Filter 4: Statistical
    mattr_score = score_mattr(doc)
    pronoun_score = score_pronoun_density(doc, word_count)

    # Filter 6: Slop n-grams
    ngram_score = score_slop_ngrams(text, word_count)

    # Filter 5: Fiction anti-patterns (most expensive — runs last)
    anti_pattern_score = score_fiction_anti_patterns(doc, text, word_count, dialogues, paragraphs)

    filter_scores = {
        "banned_vocabulary": banned_score,
        "construction_patterns": construction_score,
        "burstiness": burstiness,
        "mattr": mattr_score,
        "pronoun_density": pronoun_score,
        "structural_patterns": structural,
        "slop_ngrams": ngram_score,
        "fiction_anti_patterns": anti_pattern_score,
    }

    composite = sum(
        filter_scores.get(name, 0.0) * weight
        for name, weight in COMPOSITE_WEIGHTS.items()
    )

    return TierAResult(
        composite_score=composite,
        passed=composite < threshold,
        filter_scores=filter_scores,
        flagged_items=flagged,
    )
