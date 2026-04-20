# Stage 1: Concept Evaluation & Anti-Cliche Filtering

Evaluating concepts is fundamentally different from evaluating prose. You're not
assessing execution — you're assessing *potential*. The question is "could a great
story be written from this?" not "is this well-expressed?"

This means the evaluation must look forward: does this concept have the structural
properties that predict good stories? Does it avoid the patterns that predict bad
ones? Does it have enough originality, tension, and emotional charge to sustain
the downstream pipeline?

All concept evaluation runs inside ShinkaEvolve's pluggable `evaluate.py`. The
judge panel, scoring mechanics, and anti-cliche filtering are implemented there.

---

## Evaluation Pipeline

Each concept passes through three gates, in order:

### Gate 1: Validation (Fast, No LLM Calls)

Basic structural checks before any expensive evaluation:

- **Required fields present:** core premise and target emotional effect must exist
  and be non-empty
- **Genome parseable:** valid JSON structure
- **Non-trivial content:** not placeholder text, not a rephrasing of the prompt,
  not a meta-commentary on story generation
- **Minimum specificity:** premise must be concrete enough to distinguish from
  other premises (a single sentence like "a person faces a challenge" fails)

If any check fails: `correct: false`, skip evaluation, save error message. The
concept is discarded without consuming evaluation budget.

### Gate 2: Anti-Cliche Pre-Check (Cheap, Pattern-Matching)

Before running the expensive LLM judge panel, check the concept against known
convergence patterns. This is a pre-filter, not a replacement for evaluation.

**How it works:**

1. Embed the concept's core premise
2. Compare against embeddings of known convergence patterns (see anti-cliche
   list below)
3. If cosine similarity to any pattern exceeds threshold: flag the concept and
   raise its required originality score

Flagged concepts are NOT rejected. They advance to Gate 3 (judge panel evaluation)
but carry a flag in `private_metrics.anti_cliche_flag`. The judge panel sees the
concept without knowing it's been flagged (blind evaluation). After scoring, the
originality threshold for flagged concepts is raised — they need to score
significantly higher on originality to advance than unflagged concepts.

This implements the "higher bar for cliches" principle: known convergence patterns
can advance if they demonstrate substantial novelty in their specific execution,
but the bar is raised.

### Gate 3: LLM Judge Panel Evaluation (Expensive, High-Signal)

The full judge panel evaluates the concept across multiple dimensions. This is
where the real selection pressure lives.

---

## Evaluation Dimensions

Nine dimensions compared via pairwise voting. Each has 2-4 observable
sub-criteria that require judges to cite specific evidence from the concept
text. Full sub-criteria with literary exemplars are in
`owtn/prompts/stage_1/rubric_anchors.txt`.

### 1. Novelty

*Is this premise genuinely new, and does it resist AI convergence patterns?*

Merges the former Originality and Anti-Cliche dimensions. Sub-criteria: **domain
crossing** (does the concept combine conventionally unpaired domains?),
**convergence distance** (does it avoid or subvert the 10 known AI convergence
patterns?), **generative surprise** (would different writers produce different
stories from this premise?).

Grounded in: Sui Generis score (Xu et al., PNAS 2025), AI uncertainty gap (Sui,
ICML 2026), Boden's creativity framework.

### 2. Grip

*Does this concept contain the engine to pull a reader in and not let go?*

Sub-criteria: **the thing you can't look away from** (a concretization — something
abstract made visible and undeniable, like Kafka's insect or Jackson's stones),
**emotional stakes** (something specific at risk), **sensory seed** (at least one
concrete, visible scene).

Grounded in: Green & Brock's transportation theory (2000), Kahneman's peak-end
rule, Gardner's "vivid and continuous dream."

### 3. Tension Architecture

*Does the concept create inherent pull-forward force?*

Sub-criteria: **suspense** (uncertain outcome for someone the reader invests in),
**information architecture** (withheld information — distinguishing resolvable
gaps that the story will answer from permanent gaps the story structurally refuses
to resolve), **reframing potential** (structural room for a revelation that
changes how the reader understands prior events).

Grounded in: Brewer & Lichtenstein (1982), Loewenstein's information-gap theory,
Schulz et al. (2024) narrative entropy, Zillmann's excitation transfer.

### 4. Emotional Depth

*Does this concept have the capacity for genuine, complex feeling?*

Separated from Grip because they measure different things — a thriller can grip
without depth; a literary meditation can have depth without grip.

Sub-criteria: **recognition** (does the target effect name a feeling the reader
will recognize — "oh, that's what that is"?), **emotional complexity**
(contradictory or layered feelings), **emotional source** (where does the feeling
come from — character, situation, voice, structural constraint — and is it
load-bearing?), **implication** (does reading the story mirror something within
the story, so the reader is participating, not just observing?).

Grounded in: paradox of fiction (Radford 1975), Psychological Depth Scale
(EMNLP 2024).

### 5. Thematic Resonance

*Does this connect to something universal that resists easy answers?*

Sub-criteria: **question vs. message** (does the concept pose a genuine dilemma
where both sides have weight, or deliver a position?), **embeddedness** (is the
thematic concern inseparable from the premise, or could you swap in a different
theme without changing the story?).

### 6. Concept Coherence

*Do the genome's elements work as a system?*

Sub-criteria: **load-bearing elements** (would removing each major element make
the concept generic or broken?), **surface/depth architecture** (the highest form
of productive tension is structural irony — the surface tells one story while the
depth tells another, and the reader holds both).

Grounded in: Chen & Bornstein (Trends in Cognitive Sciences, 2024) — causally
central elements are rated more important and better recalled.

### 7. Generative Fertility

*Does this concept suggest multiple possible stories, or only one obvious
execution?*

Sub-criteria: **execution diversity** (can you sketch 3 meaningfully different
stories from this premise?), **generative principle vs. situation** (does the
concept contain an engine that produces possibilities, or just a single
scenario?).

### 8. Scope Calibration

*Can this produce a satisfying story in ~1,000-8,000 words?*

Sub-criteria: **natural size** (count the characters, locations, time periods,
complexity layers — does the quantity fit?), **constraint as compression** (do the
concept's constraints help scope by forcing economy, or hurt it by requiring
elaborate setup?).

### 9. Indelibility

*Would this concept stick in a reader's mind?*

Sub-criteria: **indelible image** (at least one scene vivid and emotionally
charged enough to persist in memory), **the irreducible remainder** (something
the mind cannot metabolize or file away — a permanent gap, an unresolvable
tension), **silhouette** (does the concept have an irreducible shape that
survives memory's compression? Would you recognize it years later if something
reminded you of it?).

Grounded in: story superiority effect (Mar et al., N>33,000), emotional peaks
drive memory (Leong et al., Nature Human Behaviour 2025), Kahneman's peak-end
rule.

### Known AI Convergence Patterns

These 10 patterns are checked in the Novelty dimension's convergence distance
sub-criterion. A concept mapping to a known pattern can still score well on
Novelty if it subverts the pattern with enough specificity.

| Pattern | What Makes It Cliche |
|---------|---------------------|
| Reconciliation arc | Most overrepresented LLM plot shape |
| Grief meditation | LLMs default to grief as "literary" emotion |
| The chosen one | Inherited from training on genre fiction |
| AI consciousness | Extreme self-reference frequency in LLMs |
| Sanitized conflict | RLHF safety training suppresses darkness |
| Epistolary revelation | Convenient exposition disguised as narrative |
| Time loop lesson | Mechanistic structure easily optimized |
| Magical realism metaphor | LLMs map emotion→metaphor reflexively |
| Moral clarity | RLHF preferences for prosocial outcomes |
| Small-town secret | AI version lacks specificity that makes Jackson's "Lottery" work |

---

## Selection Mechanics

> **Note:** This section was rewritten in April 2026 to reflect the shift from
> pointwise scoring to pairwise comparison. See `docs/CHANGELOG.md` for context.

### Pairwise Comparison, Not Pointwise Scoring

Concepts are not scored individually. Each new concept is compared head-to-head
against its island's current champion via a 3-judge panel. Each judge evaluates
both concepts on all 9 dimensions independently, picking a winner per dimension.
Position bias is mitigated by running each comparison in both orderings — if a
judge picks a different winner when the order is swapped, that vote is discarded.

The overall winner is the concept that wins the most dimensions across all judges.
Score = win percentage: `(dimension_wins + 0.5 * ties) / 9`.

**Why not pointwise?** Absolute LLM scoring compresses all AI concepts into a
0.3-point band (4.5-4.8 on a 0-5 scale) regardless of rubric design, calibration
instructions, or harshness settings. The leniency bias is structural (RLHF), not
fixable by prompt engineering. Pairwise comparison discriminates where scoring
cannot — EQ-Bench switched to pairwise for exactly this reason.

### Mutation Feedback

The pairwise comparison reasoning is the mutation feedback. The mutation model
sees which dimensions the champion beat the challenger on and why — directly
actionable for the next generation. No per-dimension scores, just comparative
judgment: "A wins on novelty because it crosses marine biology with courtroom
procedure, while B stays within familiar literary territory."

### Rubric Sub-Criteria

Each dimension has 2-4 observable sub-criteria that structure the judge's
comparison. Endpoint-only anchors (scores 1 and 5 described; 2-4 are
interpolations). Every sub-criterion requires the judge to "name it" — cite
specific evidence from the concept text. Full sub-criteria in
`owtn/prompts/stage_1/rubric_anchors.txt`.

---

## Judge Panel at Concept Stage

The same panel architecture as prose evaluation (see docs/judging/overview.md)
operates at the concept stage, but evaluating potential rather than execution:

### Panel Composition
- 40% target audience personas
- 30% adjacent audience personas
- 20% random/diverse personas
- 10% expert/craft personas

### Key Rules

- **Blind to the run prompt.** Judges do not receive the run's `prompt` field.
  They evaluate concept quality on its own merits, never on adherence to
  creative direction. This prevents the system from optimizing for
  prompt-parroting rather than genuine quality.

- **Different model families** from whichever models are doing mutation. If
  Claude is generating concepts, GPT-4 and Gemini judge them (and vice versa).
  Self-preference bias (models rate own outputs ~0.52 higher) is eliminated by
  model diversity.

- **Single-turn independent evaluations.** No multi-judge debate (causes
  conformity, confabulation, impersonation). Each judge evaluates in isolation.

- **0-5 grading scale** (ICC = 0.853 vs 0.805 for 0-10). Reliable for the
  subjective, open-ended task of concept evaluation.

### Pairwise Voting as Disagreement Signal

With pairwise per-criteria voting, the disagreement signal is built in. A concept
that one judge loves and two hate will lose (majority vote). But tied dimensions
(where position bias mitigation catches a flip) preserve uncertainty — the
dimension is genuinely close. A match with many ties (e.g., 1-0-8) means the
concepts are hard to distinguish; a match with few ties (e.g., 8-1-0) means
there's a clear winner.

---

## Anti-Cliche Threshold Mechanics

For flagged concepts (those matching known convergence patterns in Gate 2):

1. Compare normally via pairwise (blind — judges don't know the concept is
   flagged)
2. The flag is stored in `private_metrics` for post-run analysis
3. A flagged concept that wins its pairwise comparison advances normally — the
   pairwise comparison inherently tests whether the concept is compelling enough
   to beat the champion despite mapping to a known pattern

Jackson's "The Lottery" would win its pairwise because its specificity and
execution angle make it genuinely compelling. A generic "idyllic town hides dark
secret" would lose.

---

## Gate 2: Embedding Specification

### Embedding Model

Use ShinkaEvolve's embedding infrastructure (the same model used for novelty
rejection in `AsyncNoveltyJudge`). This avoids a second embedding dependency and
ensures concept similarity is measured on the same scale across novelty rejection
and anti-cliche detection.

### Convergence Pattern Reference Set

The 10 known convergence patterns (listed above in the anti-cliche section) are
pre-embedded as a reference set. Each pattern's embedding is computed from its
description text (the "Description" column in the table, not just the pattern
name).

The reference embeddings should be computed once and cached. They only need
recomputation if patterns are added/removed or the embedding model changes.

### Threshold Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `base_similarity_threshold` | 0.85 | Cosine similarity above this flags the concept for elevated scrutiny. Calibrated to flag concepts that are clearly in the territory of a known pattern without catching distant thematic echoes. |
| `elevated_novelty_threshold` | 3.5 | Reserved for future use — with pairwise selection, flagged concepts are evaluated via head-to-head comparison, not absolute scoring. The threshold may be used if pointwise scoring is reintroduced for prose stages. |

### Calibration Notes

- The `base_similarity_threshold` of 0.85 is a starting value. It should be
  calibrated by embedding 50-100 known concept descriptions (from HANNA and
  LitBench story premises) and verifying that clearly convergent concepts are
  flagged while merely thematically related ones are not.
- Store threshold values in `private_metrics` for post-run tuning:
  `anti_cliche_flag`, `anti_cliche_similarity`, `anti_cliche_matched_pattern`.
