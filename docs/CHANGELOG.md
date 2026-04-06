# Changelog

## 2026-04-05: Pairwise Selection

Replaced pointwise scoring with pairwise comparison. Absolute LLM scoring compressed all concepts into a 0.3-point band regardless of rubric design — the leniency bias is structural (RLHF), not fixable by prompt engineering. Pairwise comparison discriminates where scoring cannot.

**Selection**: Per-criteria pairwise voting (9 dims x 3 judges x 2 orderings) replaces Holder mean + diversity bonus. Each judge compares two concepts on all 9 dimensions independently, picks a winner per dimension. Position bias mitigated via dual orderings — if a judge picks different winners in different orderings, that vote is discarded as a tie.

**Evaluation**: Inline via `eval_function` on `JobConfig` (no subprocess). Validation + pairwise comparison in one blocking step, so ShinkaEvolve sees real scores before selecting next parent.

**Islands**: Champion-based succession. Each island maintains one champion. New concepts challenge the champion; winner takes over. Equal island scheduling (`EqualIslandSampler`) ensures balanced allocation.

**Archive**: Fitness-based (was MAP-Elites). MAP-Elites barely fired at current scale (7-9 cells occupied out of 36 with 12 concepts per run).

**Tournament**: Swiss-system tournament ranks island champions after evolution completes. Buchholz tiebreaking.

**Score**: Win percentage `(dim_wins + 0.5 * ties) / 9`, not Holder mean. Champions start at 0.5. Tournament applies final ranking.

**Removed**: Pointwise scoring functions (`holder_mean`, `aggregate_judge_scores`), subprocess evaluation entry point (`evaluation/__main__.py`), old pointwise prompts (`judge_system.txt`, `judge_user.txt`), harshness instructions, classification LLM call, dead config fields (`tier_a_enabled`, `pairwise_enabled`, `dynamic_rubrics_enabled`).

Issue: `lab/issues/2026-04-05-pairwise-selection.md`

---

## 2026-04-04: Rubric Anchor Redesign

Restructured evaluation dimensions from 9 impression-based to 9 observable sub-criteria-based, grounded in deeper cognitive science analysis. Research finding: criteria decomposition is the highest-leverage structural fix for score discrimination (+17.7 accuracy points in the literature).

**Dimensions changed**:
- Merged Originality + Anti-Cliche → **Novelty** (3 sub-criteria: domain crossing, convergence distance, generative surprise)
- Renamed Transportation Potential → **Grip** (3 sub-criteria: the thing you can't look away from, emotional stakes, sensory seed)
- Renamed Narrative Tension → **Tension Architecture** (3 sub-criteria: suspense, information architecture with resolvable vs permanent gaps, reframing potential)
- Added **Emotional Depth** split from Transportation (4 sub-criteria: recognition, complexity, source, reader implication)
- Added **Indelibility** from prose-stage Memorability (3 sub-criteria: indelible image, irreducible remainder, silhouette)
- Removed Over-Explanation Resistance (absorbed into other dimensions)

**Format**: Endpoint-only anchors (scores 1 and 5 described; 2-4 are interpolations). "Name it" prompting — each sub-criterion requires the judge to cite specific evidence from the concept text.

**Six deep patterns** identified from analysis of stories that survive across decades: the invisible made visible, permanent withholding, reader complicity, structural irony, recognition, inevitability. These are distributed across the sub-criteria.

**All examples from published fiction**, not AI-generated concepts: Jackson, Kafka, Chiang, Borges, Hemingway, Nabokov, O'Connor, Le Guin, Carver, Joyce, Ishiguro, Keyes, Melville, Gilman, Oates, Machado, Swift.

Issue: `lab/issues/2026-04-04-rubric-anchor-redesign.md`

---

## 2026-04-03: Evolution Score Debugging

Diagnosed why evolutionary search wasn't improving scores across generations. The best score (4.530) was set in Gen 0 and never surpassed across 19 generations.

**Root causes found**:
- Mutation prompts drowning the model in 4,000-6,000 tokens of raw judge feedback
- Genesis calls contaminated by eval_history prepended unconditionally
- Score compression: all 29 concepts between 4.08-4.53 (0.45-point spread on 0-5 scale)
- Feedback not actionable through most operator instructions (only 5 of 11 reference it)
- Seed reuse across generations

**Fixes**: Compressed judge feedback for mutations via `build_mutation_feedback()` (26% token reduction). Added judge scoring calibration instruction (insufficient alone — led to rubric redesign). Discovered Gemini 2.5 Flash truncates reasoning in structured output mode.

Issue: `lab/issues/2026-04-03-evolution-not-improving-scores.md`

---

## 2026-04-03: Prompt Register Rewrite

Rewrote all prompts from instructional to invitational register based on research showing instructional framing suppresses creative diversity.

**Research basis**: "Price of Format" (EMNLP 2025) — structured templates collapse diversity independent of content. Regulatory Focus Theory — "avoid mistakes" framing activates convergent processing; "approach ideals" activates divergent. CreativeDC — 63.5% novelty improvement from separating divergent/convergent phases.

**Results** (13 concepts vs 28 baseline): Self-BLEU 48.88→33.38 (↓32%), distinct-1 0.198→0.310 (↑57%), earnest tone 100%→76.9% (first non-earnest concepts), cosine similarity 0.656→0.520 (↓21%).

Issue: `lab/issues/2026-04-03-prompt-register-rewrite.md`

---

## 2026-04-02: Convergence Monoculture Fix

Identified 6 root causes of concept convergence — every concept converging to literary psychological realism about morally ambiguous intimacy.

**Causes**: (1) Anti-convergence checklist acting as convergence funnel, (2) single example anchoring register, (3) judge monoculture (3 judges share same aesthetic), (4) evolution amplifying convergence (MAP-Elites cell key missing tonal register), (5) temperature too low (2/3 calls ≤0.5), (6) strong seeds dominating.

**Fixes**: Tonal steering injection (16 affective registers x 18 literary modes = 288 combinations), temperature annealing schedule (high early for exploration, low late for refinement), genesis ratio annealing, softened anti-convergence checklist.

Issue: `lab/issues/2026-04-02-convergence-monoculture.md`
