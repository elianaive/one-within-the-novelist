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

## Core Evaluation Dimensions

Nine dimensions, each grounded in research on what predicts successful stories.
Not all dimensions are equally relevant to every concept — the dynamic rubric
system (inherited from the judging architecture) adjusts weights based on concept
type.

### 1. Originality (Novelty Assessment)

*Is this a premise we haven't seen? Would an LLM naturally converge on this?*

**Assessment methods:**

- **Sui Generis scoring** (computable pre-check): Generate 5-10 quick plot
  sketches from the concept using different LLMs. Measure convergence — how
  many sketches land on the same basic plot? Human concepts produce diverse
  sketches (Sui Generis ~13+); LLM-convergent concepts produce uniform sketches
  (Sui Generis ~6-7). High convergence = the concept is "too obvious."
  (Xu et al., PNAS 2025)

- **Embedding distance** from existing archive members. ShinkaEvolve's novelty
  system provides this — concepts that are too similar to existing archive
  occupants are less valuable even if individually strong.

- **Judge panel assessment:** Each judge is asked directly: "Have you encountered
  this premise before? In published fiction? In AI-generated fiction? What
  specifically is different here?" Forced articulation of novelty (or lack
  thereof) produces better calibrated scores than a simple 0-5 rating.

**Grounded in:** Echoes-in-AI (Xu et al., Microsoft, PNAS 2025) — LLM plots
converge at 6-7x the rate of human plots. AI uncertainty gap (Sui, ICML 2026) —
human writing is 2-4x more informationally surprising.

### 2. Emotional Potential (Transportation Potential)

*Does this concept contain the energy for a story that grips the reader?*

**Assessment using Green & Brock's three components:**

- **Cognitive absorption potential:** Is there enough complexity, mystery, or
  intellectual interest to sustain focused attention? A concept that can be fully
  understood in one sentence may lack the depth to sustain transportation.

- **Affective involvement potential:** Are there emotional stakes — something to
  care about, fear, hope for, grieve? The stakes don't have to be dramatic; the
  ache of a small loss can be as transporting as apocalyptic danger.

- **Vivid imagery potential:** Can the reader *see* this? Does the concept
  suggest concrete, sensory scenes? Abstract concepts need grounding in specific
  images to produce transportation.

**Target effect clarity:** Is the declared unity of effect specific enough to be
useful? "Sadness" is too vague. "The ache of knowing something beautiful is
temporary" is specific enough to guide all downstream decisions. Judges evaluate
whether the target effect is achievable and whether the premise naturally builds
toward it.

**Peak-end potential:** Does the concept have a natural climax moment? A natural
ending? Kahneman's peak-end rule means stories are remembered by their peaks and
endings — a concept that suggests a powerful peak-end structure has higher
transportation potential.

**Grounded in:** Transportation theory (Green & Brock, 2000; Green & Appel,
2024). Peak-end rule (Kahneman et al., 1993). Emotional peaks and memory (Leong
et al., Nature Human Behaviour 2025).

### 3. Narrative Tension (Suspense / Curiosity / Surprise Decomposition)

*Does the concept create inherent tension that pulls the reader forward?*

The information theory research (Schulz et al., 2024) shows that suspense,
curiosity, and surprise are computationally distinguishable — they arise from
different relationships between the reader's knowledge and the story's events.
A concept should support at least one:

- **Suspense potential:** Does the concept involve uncertain outcomes for entities
  the reader might care about? "A woman defuses a bomb" has suspense. "A woman
  remembers defusing a bomb" has less. Suspense requires uncertainty about what
  will happen next.

- **Curiosity potential:** Does the concept create information gaps — things the
  reader wants to know? "A town's annual tradition" creates curiosity about what
  the tradition is. Curiosity is epistemic (understanding), not teleological
  (outcome). It's the "page-turner" quality.

- **Surprise potential:** Does the concept have room for revelations that reframe
  prior events? "The lottery winner is actually being sentenced to death" is a
  surprise that recontextualizes everything. The ideal: low predictability + high
  post-dictability (unexpected but, in retrospect, inevitable).

A concept doesn't need all three. A voice-constraint piece ("Hills Like White
Elephants") runs almost entirely on curiosity. A thriller runs on suspense. A
reveal story runs on surprise. But it should strongly support at least one.

**Grounded in:** Information-theoretic model of narrative (Schulz et al., 2024).
Bayesian surprise (Kumar et al., Cognitive Science 2023; Chen & Bornstein, Trends
in Cognitive Sciences, 2024). Narrative surprise operationalization (Bissell et
al., WNU @ ACL 2025).

### 4. Thematic Resonance

*Does this concept connect to something that matters?*

- Does the thematic tension (if specified) connect to something universal — a
  dilemma that real people face, a question that doesn't have an easy answer?
- Does the concept resist simple resolution? The best themes are tensions, not
  answers. "Freedom vs. security" is a theme; "freedom is good" is a message.
- Is the thematic concern embedded in the premise rather than bolted on? A
  concept where the theme emerges naturally from the situation is stronger than
  one where the theme is stated separately.

### 5. Scope Calibration (Feasibility)

*Can this concept produce a satisfying story in the target word count?*

More specific than generic "feasibility":

- A concept requiring extensive worldbuilding may be infeasible at 1,000 words
  but ideal at 8,000
- A concept that's essentially one moment may be perfect at 500 words but
  unsustainable at 5,000
- A concept with multiple characters, locations, and time periods may need 8,000+
  words to breathe

The evaluation considers the concept's *natural scope* — the word count range
where it could work — and compares it to the target. Mismatches aren't fatal
(compression and expansion are possible) but are a risk signal.

### 6. Anti-Cliche Score

*Does this concept avoid known AI convergence patterns?*

This dimension reflects the Gate 2 pre-check but adds the judge panel's
qualitative assessment. Judges evaluate:

- Does this premise feel like something an AI would generate? (Not just "is it
  a known pattern?" but "does it have the *flavor* of AI fiction?")
- If it maps to a known pattern, does it subvert it meaningfully?
- Does it avoid the AI fiction failure modes: sanitized conflict, moral clarity,
  neat resolution, performed emotion?

Stored in `private_metrics` — hidden from the mutation LLM to prevent gaming.
The system shouldn't learn to *describe* novelty without *being* novel.

**Known convergence patterns:**

| Pattern | Description | What Makes It Cliche |
|---------|-------------|---------------------|
| Reconciliation arc | Protagonist returns to hometown/family, confronts past, reconciles | Most overrepresented LLM plot shape |
| Grief meditation | Dead loved one (often spouse), protagonist processes loss through metaphorical journey | LLMs default to grief as "literary" emotion |
| The chosen one | Protagonist discovers special abilities/destiny | Inherited from training on genre fiction |
| AI consciousness | AI becomes sentient, questions existence | Extreme self-reference frequency in LLMs |
| Sanitized conflict | Conflict with no real stakes, no one gets hurt, clean resolution | RLHF safety training suppresses darkness |
| Epistolary revelation | Found letters/messages/recordings reveal hidden truth | Convenient exposition disguised as narrative |
| Time loop lesson | Protagonist repeats day/event, learns to be a better person | Mechanistic structure easily optimized |
| Magical realism metaphor | Literal manifestation of emotional state (grief becomes weather, loneliness becomes invisibility) | LLMs map emotion→metaphor reflexively |
| Moral clarity | Good and evil are obvious, virtue is rewarded, evil is punished | RLHF preferences for prosocial outcomes |
| Small-town secret | Idyllic community hides dark truth | AI version lacks the specificity that makes Jackson's "Lottery" work |

These aren't bad premises. Jackson's "The Lottery" IS a small-town secret.
Chiang's "Story of Your Life" IS a grief meditation. The difference is
specificity, subversion, and execution. The anti-cliche score measures whether
a concept that maps to a known pattern brings enough novelty to justify the
familiar territory.

### 7. Concept Coherence

*Do the genome's elements work together?*

- A comedic premise + devastating target effect = potentially brilliant dark
  comedy, or potentially incoherent. Judges evaluate which.
- A spare constraint + rich character seeds = possibly inspired tension, or
  possibly confused about what it's trying to be
- Multiple incompatible style hints = might be avant-garde or might be a mess

The key question: do the elements create *productive* tension or *contradictory*
signals? Productive tension is a feature — the genome's internal friction can
drive the story. Contradiction is a weakness — the elements fight each other
rather than creating something.

### 8. Generative Fertility

*Does this concept suggest multiple possible stories, or only one obvious
execution?*

High-fertility concepts are better evolutionary starting points because they give
Stage 2 more room to explore structure variants. A concept that implies a single
obvious plot is lower-value than one that could go in several surprising
directions.

**Assessment method:** Generate 3 brief plot sketches from the concept. If all
three converge on the same basic plot, fertility is low. If they diverge into
genuinely different stories, fertility is high.

This is related to but distinct from originality. A concept can be highly original
(no one has done this before) but low fertility (there's only one way to do it).
The ideal: original AND fertile.

### 9. Resistance to Over-Explanation

*Does this concept inherently invite or resist exposition?*

The #1 fiction anti-pattern identified by Nous Research is over-explanation: the
narrator explains what a scene already showed. Some concepts inherently invite
this problem:

- Complex speculative premises that need exposition to make sense
- Multiple layers of backstory required for the situation to work
- High concept density that tempts the writer to explain rather than dramatize

Other concepts inherently resist it:

- "Two people at a train station" — nothing to explain, only subtext
- "A woman walks into a room and everything changes" — the change must be shown
- Constraint-driven concepts where the constraint forbids explanation

This dimension is a signal, not a hard filter. Some high-exposition concepts are
worth the risk (Chiang makes it work brilliantly). But concepts that resist
over-explanation tend to produce better prose downstream because the LLM can't
fall back on its strongest bad habit.

---

## Scoring Mechanics

### Holder Mean Within Judge

Each judge scores the concept across applicable dimensions (0-5 scale). The
within-judge aggregate uses the Holder mean with parameter p ~ 0.3-0.5:

```
holder_mean = (sum(score_i^p) / n)^(1/p)
```

This acts as a **soft minimum**: weaknesses drag the score down more than
strengths compensate. A concept that scores 5/5 on originality but 1/5 on
coherence will score much lower than a concept that scores 3/5 on everything.
This matches how concepts actually work — a fatal flaw in one dimension (e.g.,
the premise makes no sense) isn't rescued by strength in another.

The Holder mean is stored as `combined_score` in ShinkaEvolve — the primary
fitness signal for selection.

### Per-Dimension Scores

Individual dimension scores are stored in `public_metrics` — visible to the
mutation LLM in future generations. This gives operators targeted feedback: "this
concept scored high on originality but low on feasibility" tells the mutation
operator exactly what to fix.

### Judge Reasoning Chains

The most valuable output of concept evaluation is the reasoning chain — the
judge's natural-language explanation of why the concept works or doesn't. These
are stored in `text_feedback` and fed to mutation operators as context.

Reasoning chains serve multiple purposes:
- Guide mutation: "The premise is strong but the target effect is too vague" →
  the operator knows to sharpen the target effect
- Inform Stage 2: "This concept's strength is its reveal potential" → structure
  evolution emphasizes disclosure edges
- Build episodic memory: chains that explain failures feed the compost heap and
  anti-cliche system
- Calibrate the system: reasoning chains are human-readable audit trails

### Chain-of-Thought Before Scoring

Judges must articulate their reasoning *before* assigning scores. This increases
reliability (Prometheus 2 achieves 0.897 Pearson correlation with humans using
rubric + CoT) and produces the reasoning chains that downstream stages consume.

### Rubric Anchors

Each dimension has explicit per-score descriptions (what a 1 looks like, what a 3
looks like, what a 5 looks like). These reduce arbitrary variance between judges
and across evaluation rounds:

**Example for Originality:**
- **1/5:** Premise is a well-known trope executed without subversion. "A detective
  solves a murder." "A young person discovers they have magical powers."
- **3/5:** Premise has a fresh angle on familiar territory, or combines known
  elements in an unusual way. The core idea isn't new, but the specific execution
  suggests novelty.
- **5/5:** Premise surprises. The evaluator hasn't seen this specific combination
  of elements, or the subversion of expectations is genuinely unexpected. Reading
  this concept changes how the evaluator thinks about what's possible.

(Full rubric anchors for all dimensions to be developed during calibration against
HANNA and LitBench datasets.)

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

- **Blind to steering.** Judges do not receive the run's steering prompt. They
  evaluate concept quality on its own merits, never on adherence to creative
  direction. This prevents the system from optimizing for prompt-parroting
  rather than genuine quality.

- **Different model families** from whichever models are doing mutation. If
  Claude is generating concepts, GPT-4 and Gemini judge them (and vice versa).
  Self-preference bias (models rate own outputs ~0.52 higher) is eliminated by
  model diversity.

- **Single-turn independent evaluations.** No multi-judge debate (causes
  conformity, confabulation, impersonation). Each judge evaluates in isolation.

- **0-5 grading scale** (ICC = 0.853 vs 0.805 for 0-10). Reliable for the
  subjective, open-ended task of concept evaluation.

### The Disagreement Signal

Track both mean score and variance across the panel:

- **High mean + low variance:** Broadly competent concept. Possibly safe/generic.
- **High mean + high variance:** Some judges love it, some hate it. This concept
  is doing something *bold*. It gets a diversity bonus in selection.
- **Low mean + high variance:** Might be flawed in ways some judges forgive.
  Worth investigating, not automatically advancing.
- **Low mean + low variance:** Consensus: this doesn't work. Eliminate.

The diversity bonus for high-variance concepts explicitly protects bold,
polarizing ideas from being ground down to consensus mediocrity. This is critical
at the concept stage — safe, broadly-appealing concepts produce safe,
broadly-appealing stories.

---

## Anti-Cliche Threshold Mechanics

For flagged concepts (those matching known convergence patterns in Gate 2):

1. Evaluate normally through the full judge panel (blind — judges don't know it's
   flagged)
2. After scoring, apply the originality threshold multiplier:
   - Unflagged concepts: advance if `combined_score > base_threshold`
   - Flagged concepts: advance if `combined_score > base_threshold` AND
     `originality_score > elevated_threshold`
3. The elevated threshold should be calibrated so that ~20-30% of flagged
   concepts can still advance (we want genuinely subversive takes on familiar
   patterns, not a blanket ban)
4. Store the flag and the original/elevated thresholds in `private_metrics` for
   post-run analysis

This means a flagged concept needs to demonstrate *specific novelty* — not just
be generally good, but be specifically original despite mapping to a known
pattern. Jackson's "The Lottery" would pass because its specificity and execution
angle are genuinely novel. A generic "idyllic town hides dark secret" would not.

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
| `elevated_originality_threshold` | 3.5 | Flagged concepts must score > 3.5/5 on the originality dimension (from the judge panel) to advance. This is above the midpoint and represents a "fresh angle" — see rubric-anchors.md score 4. |

### Calibration Notes

- The `base_similarity_threshold` of 0.85 is a starting value. It should be
  calibrated by embedding 50-100 known concept descriptions (from HANNA and
  LitBench story premises) and verifying that clearly convergent concepts are
  flagged while merely thematically related ones are not.
- The `elevated_originality_threshold` of 3.5 targets the ~20-30% advance rate
  for flagged concepts specified in the design. Adjust after the first full run
  based on actual flagging rates and judge score distributions.
- Store all threshold values in `private_metrics` for post-run tuning:
  `anti_cliche_flag`, `anti_cliche_similarity`, `anti_cliche_matched_pattern`,
  `originality_score`, `threshold_applied`.
