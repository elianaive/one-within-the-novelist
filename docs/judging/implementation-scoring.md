# Judging Implementation: Scoring, Calibration & Integration

Shared mathematical utilities, calibration procedures, and stage integration
patterns for prose-stage evaluation. Stage 1 concept evaluation uses pairwise
comparison instead of the pointwise scoring described here — see
`owtn/evaluation/pairwise.py` and `docs/CHANGELOG.md`.

---

## 1. Holder Mean

The aggregation function for combining per-dimension scores within a single
judge's evaluation. Acts as a **soft minimum**: weaknesses drag the score down
more than strengths compensate.

### Formula

```
holder_mean(scores, p) = (mean(score_i ^ p)) ^ (1/p)
```

Where `scores` is a list of dimension scores (0-5) and `p` is the penalty
parameter.

**Behavior by p value:**
- p = 1.0 → arithmetic mean (no penalty for unevenness)
- p = 0.5 → moderate soft minimum (Lechmazur benchmark default)
- p = 0.3 → strong soft minimum (heavy penalty for weak dimensions)
- p → -inf → strict minimum (weakest score determines everything)

### Parameter Selection

**Default: p = 0.4**

Rationale:
- Lechmazur benchmark uses p = 0.5 with inter-grader agreement r = 0.93
- Our dimensions are broader (resonance, not craft checkpoints), so slightly
  stronger penalty (p = 0.4) better captures how readers experience stories —
  a single broken dimension (e.g., incoherent plot) ruins the whole experience
  regardless of other strengths
- p = 0.4 means a story scoring [5, 5, 5, 5, 1] gets ~3.9, while arithmetic
  mean would give 4.2 — the weak dimension pulls the score down, though not
  enough to invert ranking against [3, 3, 3, 3, 3] = 3.0. For that level of
  harshness, use negative p values.

**When to adjust:**
- Lower p (0.2-0.3) for Stage 5 refinement — at this point, *any* weakness
  should be caught and fixed
- Higher p (0.5-0.6) for Stage 1 concepts — concepts with one brilliant
  dimension and several unknowns should be preserved for exploration

Each stage's evaluate.py can configure p. The judging library accepts it as a
parameter.

### Implementation

```python
import numpy as np

def holder_mean(scores: list[float], p: float = 0.4) -> float:
    """Compute Holder mean (generalized/power mean) of scores.

    Args:
        scores: List of scores (0-5 scale). Zeros are replaced with
                epsilon to avoid math errors at negative p.
        p: Penalty parameter. Lower = more penalty for weak scores.

    Returns:
        Holder mean as float.
    """
    scores = np.array(scores, dtype=float)
    scores = np.maximum(scores, 1e-6)  # avoid log(0) at negative p
    return float(np.power(np.mean(np.power(scores, p)), 1.0 / p))
```

### Application Points

- **Within a single judge:** Holder mean across all scored dimensions for one
  story. Produces the judge's "holistic score" for that story.
- **NOT across judges.** Cross-judge aggregation uses mean + variance (see
  Disagreement Signal below). Do not Holder-mean across judges — the variance
  is critical information.

---

## 2. Disagreement Signal

The most counterintuitive mechanism: high inter-judge variance is a feature,
not a bug.

### Metrics

For each story, compute across the judge panel:

```python
judge_mean = np.mean(judge_scores)      # quality signal
judge_variance = np.var(judge_scores)    # disagreement signal
judge_std = np.std(judge_scores)         # for diversity bonus
```

### Interpretation

| Mean | Variance | Interpretation |
|------|----------|----------------|
| High | Low | Broadly competent. Potentially safe/generic. |
| High | High | Bold. Some judges love it, some hate it. Doing something interesting. |
| Low | High | Flawed in ways some judges forgive. Worth investigating. |
| Low | Low | Consensus: doesn't work. Eliminate. |

### Diversity Bonus

Stories with high mean AND high variance receive a selection bonus. This
explicitly protects bold, polarizing work from being ground down to consensus
mediocrity.

**Formula:**

```python
def selection_score(judge_mean: float, judge_std: float,
                    diversity_weight: float = 0.15,
                    std_threshold: float = 0.8) -> float:
    """Compute selection score with diversity bonus.

    The bonus activates only when variance exceeds threshold AND mean
    is above median — we don't reward low-quality polarizing work.

    Args:
        judge_mean: Mean score across judges (0-5).
        judge_std: Standard deviation across judges.
        diversity_weight: Weight of the diversity bonus (0-1).
        std_threshold: Minimum std to activate bonus.

    Returns:
        Selection score (higher = more likely to advance).
    """
    diversity_bonus = max(0, judge_std - std_threshold) * diversity_weight
    return judge_mean + diversity_bonus
```

**Default parameters:**
- `diversity_weight = 0.15`: A story with std = 1.8 (very polarizing) gets a
  +0.15 bonus. Enough to break ties in favor of bold work, not enough to
  advance a low-quality story.
- `std_threshold = 0.8`: Bonus only activates when judges genuinely disagree.
  Normal variation (std < 0.8) gets no bonus.

### Primary vs. Contrarian Split

When the judge panel includes primary and contrarian judges (see
implementation-tier-b.md), the disagreement signal is particularly informative
along this axis:

- **Primary judges love it, contrarian judges hate it:** Well-targeted for the
  intended audience. Good signal.
- **Contrarian judges love it, primary judges hate it:** Genuinely interesting
  but wrong for the target. Archive for a different context.
- **Split within primary judges:** The most informative case — this work divides
  even the target audience. Strong candidate for diversity preservation.

---

## 3. Coevolving Judges

Goodhart's Law: optimizing stories against fixed judges will eventually game
them. The judge system must evolve alongside the stories.

### Schedule

| Mechanism | Frequency | Trigger |
|-----------|-----------|---------|
| **Rubric regeneration** | Every 5 generations | Timer-based |
| **Human anchor check** | Every 10 generations | Timer-based |
| **Model rotation** | Every 10 generations | Timer-based (stagger with anchor check) |
| **Escalating standards** | When median combined_score > 3.5 | Quality-triggered |
| **Adversarial judge** | Always active | Permanent panel member |

### Rubric Regeneration

Every 5 generations, regenerate dynamic rubric criteria (the story-specific 3-5
criteria overlaid on the baseline 10). This prevents the generator from locking
onto fixed targets.

**Regeneration procedure:**
1. Collect the top 10 stories from the current archive
2. For each, generate new dynamic criteria using the standard dynamic rubric
   prompt (see implementation-tier-b.md) but with an added instruction:
   "Generate criteria that are DIFFERENT from the previous round's criteria"
3. Previous round's criteria are provided as context for avoidance

### Human Anchor Check

Every 10 generations, evaluate a random sample of 10 current archive stories
against a held-out human preference dataset (HANNA or LitBench):

1. Select 10 stories from the current archive (stratified by MAP-Elites cell)
2. Run each through the judge panel
3. Compare panel rankings against human ground truth rankings
4. If Kendall's tau < 0.5 (panel rankings poorly correlated with human
   preferences), trigger recalibration:
   - Increase harshness distribution (more "demanding" judges)
   - Regenerate rubrics
   - Check for mode collapse (are all stories converging?)

### Model Rotation

Every 10 generations, rotate which LLM families serve as judges:
- Keep at least 1 model constant for longitudinal comparison
- Swap 1-2 models for fresh bias profiles
- A new model resists the gaming strategies that evolved against the old model

### Escalating Standards

When the population improves, tighten evaluation:
- **Trigger:** Median combined_score across the archive exceeds 3.5/5
- **Response:**
  - Shift panel composition: increase "demanding" judges from 30% to 50%
  - Tighten Tier A thresholds by 10% (e.g., burstiness threshold from 0.4 to 0.36)
  - Raise the minimum combined_score for archive entry

### Adversarial Judge

One permanent panel member is an adversarial meta-judge. Its prompt:

> "You are evaluating whether this story feels like it was written to please
> an LLM judge rather than a human reader. Look for: formulaic structure
> that hits evaluation criteria without genuine craft, vocabulary chosen to
> score well on diversity metrics rather than serve the story, emotional
> beats that feel calculated rather than felt, endings designed to satisfy
> a rubric rather than conclude a narrative."

This judge's score is tracked in `private_metrics` (not visible to the mutation
LLM) and used as a gaming detection signal. If the adversarial judge's scores
trend upward across generations, the system is being gamed — trigger
recalibration.

---

## 4. Calibration Procedure

### Datasets

| Dataset | Size | What It Provides | License |
|---------|------|-----------------|---------|
| **HANNA** | 1,056 stories, 19,008 annotations | 6-dimension human ratings (Relevance, Coherence, Empathy, Surprise, Engagement, Complexity) | MIT |
| **LitBench** | 43,827 pairs, 2,480 debiased test | Pairwise human preferences (which story is better?) | Research |
| **Reagan arcs** | 1,327 stories | Emotional arc shape classification (6 shapes) | CC BY 4.0 |
| **Lechmazur** | ~2,800 stories per model | 18-question multi-judge scores | Public |

### Validation Procedure

**Initial calibration (before first run):**

1. Run the judge panel on 100 HANNA stories (random sample, stratified by
   quality)
2. For each story, compare panel dimension scores against HANNA's 6 human
   dimensions:
   - Our "Transportation" ↔ HANNA's "Engagement"
   - Our "Emotional Depth" ↔ HANNA's "Empathy"
   - Our "Surprise + Post-dictability" ↔ HANNA's "Surprise"
   - Our "Causal Coherence" ↔ HANNA's "Coherence"
3. Compute Pearson correlation per dimension pair. Target: r > 0.6
4. If any dimension falls below r = 0.5, revise that dimension's rubric anchors

**Pairwise calibration:**

1. Run pairwise tournament on 100 LitBench story pairs (from debiased test set)
2. Compare panel preferences against human preferences
3. Compute agreement rate. Target: > 70% (LitBench SOTA is 78%)
4. If agreement < 65%, investigate: which biases are driving disagreement?

**Ongoing calibration (every 10 generations):**

The human anchor check (see Coevolving Judges above) serves as ongoing
calibration. Kendall's tau < 0.5 triggers full recalibration.

### Drift Detection

Track these metrics across generations:

- **Mean combined_score trend:** Steadily increasing may indicate inflation,
  not genuine improvement. Compare against held-out human scores.
- **Score distribution entropy:** If the distribution of scores narrows across
  generations, the panel may be converging on a formula.
- **Adversarial judge trend:** If the gaming-detection score increases, the
  system is optimizing for the judges rather than genuine quality.

---

## 5. Cost Model

### Per-Story Evaluation Costs

| Component | Input Tokens | Output Tokens | Notes |
|-----------|-------------|---------------|-------|
| **Tier A anti-slop** | 0 | 0 | Local NLP processing, no LLM calls |
| **Pointwise: 1 judge** | ~1,200 | ~800 | Story (~800 tokens) + rubric (~400) → reasoning + scores |
| **Pointwise: 5 judges** | ~6,000 | ~4,000 | 5 independent evaluations |
| **Dynamic rubric generation** | ~1,000 | ~300 | Generate 3-5 story-specific criteria |
| **Quality gate check** | ~200 | ~100 | Validate judge output (can batch) |
| **Total pointwise (5 judges)** | ~7,200 | ~4,400 | **~11,600 tokens per story** |

| Component | Input Tokens | Output Tokens | Notes |
|-----------|-------------|---------------|-------|
| **Pairwise: 1 comparison, 1 judge** | ~2,000 | ~600 | Two stories + comparison prompt |
| **Pairwise: 1 comparison, 5 judges** | ~10,000 | ~3,000 | 5 judges × 2 orderings (position bias mitigation) |
| **Swiss tournament: 32 stories, 5 rounds** | ~800,000 | ~240,000 | 16 comparisons × 5 rounds × 5 judges × 2 orderings |

### Per-Stage Cost Estimates

**Stage 1 (Concept, pointwise only):**
- 40 new concepts × 7,120 tokens (from Stage 1 implementation.md) = ~285K/gen
- No Tier A (concepts aren't prose)
- 20 generations: ~5.7M tokens

**Stage 4 (Prose, pointwise + Tier A + final pairwise):**
- 20 stories per generation × Tier A (free) + pointwise (~11.6K) = ~232K/gen
- 10 generations: ~2.3M tokens for pointwise
- Final pairwise on top 20: ~1.04M tokens for tournament
- Total: ~3.3M tokens

### Cost Optimization

- **Tier A first:** Free NLP filtering eliminates ~30-50% of candidates before
  expensive LLM evaluation
- **Cheap model for dynamic rubric generation:** Classification, not evaluation
  — Haiku-class sufficient
- **Panel size is the main lever:** 3 judges instead of 5 cuts Tier B cost by
  40%. Trade-off: less disagreement signal, less bias cancellation
- **Pairwise only for final selection:** Tournament is expensive (O(N log N)
  comparisons). Reserve for the last ~20-40 candidates per stage.
- **Batch evaluations:** Judge calls within a generation can be parallelized
  across the panel

---

## 6. Stage Integration Guide

The judging library is consumed by each stage's `evaluate.py`. This section
shows the configuration pattern — how a stage selects modules, dimensions, and
parameters.

### Configuration Schema

Each stage provides a configuration dict:

```python
stage_config = {
    # Tier A
    "tier_a_enabled": bool,          # False for concept stage, True for prose
    "tier_a_threshold": float,       # Composite score below which = reject
    "tier_a_filters": list[str],     # Which filters to run (all, or a subset)

    # Tier B: Pointwise
    "dimensions": list[str],         # Which resonance dimensions to score
    "dimension_weights": dict,       # Optional per-dimension weight override
    "extra_dimensions": list[dict],  # Stage-specific dimensions (name + anchors)
    "holder_p": float,               # Holder mean parameter (default 0.4)

    # Tier B: Panel
    "panel_size": int,               # Number of judges (3-10)
    "harshness_distribution": dict,  # e.g., {"lenient": 0.2, "moderate": 0.5, "demanding": 0.3}
    "model_families": list[str],     # LLM families for judging
    "generation_model": str,         # Model family used for generation (for self-preference avoidance)

    # Tier B: Dynamic rubrics
    "dynamic_rubrics_enabled": bool,
    "num_dynamic_criteria": int,     # 3-5

    # Tier B: Pairwise
    "pairwise_enabled": bool,        # Usually only for final selection
    "pairwise_rounds": int,          # Swiss-system rounds (log2(N))

    # Selection
    "diversity_weight": float,       # Disagreement bonus weight
    "std_threshold": float,          # Minimum std to activate bonus

    # Coevolution
    "rubric_regen_interval": int,    # Generations between rubric regeneration
    "anchor_check_interval": int,    # Generations between human anchor checks
}
```

### Example: Stage 1 (Concept Evolution)

Stage 1 uses **pairwise comparison**, not the pointwise configuration schema
above. Each new concept is compared head-to-head against its island's champion
across 9 dimensions (novelty, grip, tension_architecture, emotional_depth,
thematic_resonance, concept_coherence, generative_fertility, scope_calibration,
indelibility). Selection is binary: the winner becomes champion. Score = win
percentage. See `owtn/evaluation/pairwise.py` for implementation.
```

### Example: Stage 4 (Prose Evolution)

```python
stage_4_config = {
    "tier_a_enabled": True,
    "tier_a_threshold": 0.35,  # Reject if composite anti-slop score > 0.35
    "tier_a_filters": "all",
    "dimensions": [
        "transportation", "suspense", "curiosity", "emotional_depth",
        "emotional_arc", "causal_coherence", "surprise", "ending_quality",
        "flow", "memorability",
    ],
    "extra_dimensions": [
        {"name": "voice_adherence", "anchors": {...}},
        {"name": "show_vs_tell", "anchors": {...}},
    ],
    "holder_p": 0.4,
    "panel_size": 5,
    "dynamic_rubrics_enabled": True,
    "num_dynamic_criteria": 4,
    "pairwise_enabled": True,   # For final selection before Stage 5
    "pairwise_rounds": 5,       # log2(~30 candidates)
    "diversity_weight": 0.15,
}
```

### Library Call Pattern

```python
from judging import TierA, JudgePanel, PairwiseTournament, scoring

# Stage's evaluate.py:
def evaluate(story_text, config):
    # 1. Tier A (if enabled)
    if config["tier_a_enabled"]:
        slop_result = TierA.analyze(story_text, filters=config["tier_a_filters"])
        if slop_result.composite_score > config["tier_a_threshold"]:
            return {"correct": False, "error": "Failed Tier A anti-slop filter"}

    # 2. Pointwise evaluation
    panel = JudgePanel(
        size=config["panel_size"],
        harshness=config["harshness_distribution"],
        models=config["model_families"],
        exclude_model=config["generation_model"],
    )
    dimensions = config["dimensions"] + [d["name"] for d in config["extra_dimensions"]]
    rubric = build_rubric(dimensions, config.get("extra_dimensions", []))

    judge_results = panel.evaluate_pointwise(story_text, rubric)

    # 3. Aggregate
    for judge in judge_results:
        judge.holder_score = scoring.holder_mean(judge.dimension_scores, p=config["holder_p"])

    holder = np.mean([j.holder_score for j in judge_results])
    variance = np.std([j.holder_score for j in judge_results])

    combined = scoring.selection_score(
        judge_mean=holder,
        judge_std=variance,
        diversity_weight=config["diversity_weight"],
    )

    # holder_score: raw Hölder mean across judges. Used for MAP-Elites cell
    # replacement — within-cell competition should be pure quality, not
    # influenced by judge disagreement.
    #
    # combined_score: selection_score (holder + diversity bonus). Used for
    # parent selection and stage advancement.
    return {
        "combined_score": combined,
        "holder_score": holder,
        "public": {dim: avg_score for dim, avg_score in dimension_averages},
        "private": {"judge_variance": variance, "slop_score": slop_result},
        "text_feedback": "\n\n".join(j.reasoning for j in judge_results),
    }
```

This pattern is the same across all stages — only the configuration differs.
