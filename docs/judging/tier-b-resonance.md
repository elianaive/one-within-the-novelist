# Tier B: Judge Panel Evaluation

Tier A (see `anti-slop.md`) rejects the obviously bad — cheap statistical
filters that raise the floor. Tier B identifies the **resonant** among the
survivors. This is where stories live or die in the evolutionary loop.

Tier B is expensive. It uses LLM-powered judge personas making pairwise
comparisons with explicit reasoning. It runs only on candidates that pass
Tier A. The reasoning it produces is the richest signal in the system —
feeding mutation operators, refinement guidance, and episodic memory.

---

## The Pairwise Tournament

### Why Pairwise

Absolute scoring ("rate this story 0–5") is the intuitive approach but the
wrong one. Three converging lines of evidence:

**Higher human agreement.** EvolvR (arXiv:2508.06046) found pairwise evaluation
achieves 21.9% higher agreement with human judges than pointwise scoring on
coherence metrics, with consistent improvements across all six evaluated
dimensions. This makes sense — "which is better?" is an easier question than
"how good is this on an absolute scale?"

**Aligns with the actual competition question.** A competition judge doesn't
score stories on a rubric. They read two stories and decide which one they'd
rather champion. The pairwise format mirrors this.

**Natural fit for evolutionary selection.** Tournament selection is the standard
approach in evolutionary algorithms. Each generation, candidates compete
head-to-head; winners advance. The pairwise comparison IS the selection
pressure.

**LitBench validates the format.** The strongest creative writing evaluation
benchmark (Stanford, EACL 2026, 46k human-labeled pairs) trains on pairwise
preferences. Trained reward models on this data reach 78% human agreement —
the current ceiling.

### Tournament Structure

**Swiss-system brackets** (O(log N) rounds):

In each round, stories with similar win records are paired against each other.
After log₂(N) rounds, a clear ranking emerges without requiring every story to
play every other (which would be quadratic). For a population of 32 stories,
5 rounds of pairwise comparison produce a reliable ranking.

Each matchup:
1. Both stories presented to the judge panel (all judges evaluate the same
   pair independently)
2. Each judge produces a **reasoning chain** before declaring a preference:
   "Story A is stronger because [multi-paragraph analysis]. Story B suffers
   from [specific weaknesses]."
3. The preference is recorded. The reasoning chain is stored.
4. Position bias mitigation: each pair is evaluated twice (A-first and
   B-first). If the judge disagrees with itself across orderings, the
   comparison is flagged as ambiguous and run a third time.

### Multi-Persona Reasoning

EvolvR's key innovation: generate rationales from multiple persona perspectives
for each comparison. Their five personas — Academic, Artist, Pragmatist,
Sharp-Tongued Reader, Casual Netizen — produce meaningfully different
reasoning about the same pair.

For our system, each judge persona (defined by our 6-field schema) provides
their own reasoning chain. A literary editor notices different things than a
genre enthusiast. The diversity of reasoning is itself a signal — if all
judges agree for the same reasons, the comparison is decisive. If they agree
for different reasons, the story is working on multiple levels. If they
disagree, that's the disagreement signal (see below).

---

## Dimensional Diagnostic

After the tournament selects winners, run dimensional scoring on the survivors.
This is NOT for selection — the tournament already decided who advances. This
is for **diagnosis**: identifying where a story is strong and where it's weak,
which feeds the refinement stage (Stage 5).

### The 10 Resonance Dimensions

From `judging.md`, each grounded in cognitive science:

1. **Transportation / Immersion** — does it pull you in?
2. **Suspense Architecture** — does it create and manage tension?
3. **Curiosity / Information Gaps** — does it make you want to know?
4. **Emotional Depth** — does it produce genuine feeling?
5. **Emotional Arc Coherence** — does the trajectory feel earned?
6. **Causal / Logical Coherence** — do events follow from each other?
7. **Surprise + Post-dictability** — unexpected yet retrospectively inevitable?
8. **Ending Quality** — does it land?
9. **Flow / Pacing** — does it maintain momentum?
10. **Memorability / Distinctiveness** — would you remember it tomorrow?

Each judge scores each dimension on a 0–5 scale with explicit per-level
anchors (following Prometheus 2's rubric format: criterion description +
score-level descriptions for 1, 2, 3, 4, 5).

### Dynamic Rubrics

Not all dimensions are relevant to all stories. A voice/compression piece
shouldn't be penalized for low "suspense." A concept story shouldn't be
penalized for thin characters.

WritingBench (Alibaba, NeurIPS 2025) found that dynamic per-instance
evaluation criteria achieve 84% human alignment vs 67% for static rubrics.

**Implementation:** Before dimensional scoring, generate 3–5 story-specific
criteria based on what the story is trying to do — informed by its concept
(Stage 1), structure (Stage 2), and voice (Stage 3) specs:

- A reveal story gets: "reveal impact," "dramatic irony quality," "mundane
  detail as misdirection"
- A concept story gets: "conceptual surprise," "implication chain integrity,"
  "emotional grounding of abstract ideas"
- A voice piece gets: "subtext density," "constraint adherence," "compression
  quality"

These dynamic criteria are scored alongside the baseline 10 dimensions.

### Scoring Aggregation

**Within a judge:** Hölder mean (p ≈ 0.3–0.5) across all scored dimensions.
This soft minimum means weaknesses drag the score down more than strengths
compensate. A story brilliant on prose but incoherent on logic gets punished
more than rewarded.

**Across judges:** Track both the mean (overall quality signal) and variance
(disagreement signal). Don't Hölder-mean across judges — variance is data.

---

## Modeling Judges From Real People

The judge panel's power comes from modeling specific real people whose
preferences you want to target. Two primary source types, same output format.

### From Established Literary Critics

Rich, formal signal sources:

**Published criticism.** Reviews, essays, introductions to anthologies
they've edited, craft books, award commentary. This is the highest-quality
signal. Look for:
- What specific language do they use to praise? ("luminous," "precise,"
  "urgent," "restrained" — the vocabulary reveals their evaluative ontology)
- What do they critique? ("overwrought," "sentimental," "clever but cold"
  — reveals what they're allergic to)
- What patterns emerge across their selections? (Consistent preference for
  compression? For voice? For structural ambition?)

**Interviews about craft.** The Paris Review "Art of Fiction" series,
podcast appearances, panel discussions, MFA lectures. These often contain
direct statements of values: "I care most about..." / "What I look for
is..." / "The thing I can't forgive in fiction is..."

**Past judging decisions.** If they've judged competitions: what won? What
did they say about winners? What lost in the finals and why? The delta
between the winner and the runner-up is especially informative.

**Their own writing.** A critic who writes spare, compressed fiction values
different things than one who writes expansive, maximalist prose. What they
produce reveals what they value by practice, not just by declaration.

**The PREFINE process** (arXiv:2510.21721, 83% win rate over generic
personas): Feed their published evaluations into an LLM prompted to extract
"personality traits, cognitive style, emotional tendencies, and psychological
motivations." This generates 3–5 judge-specific evaluation criteria that
reflect their actual preferences rather than generic quality metrics.

### From Internet Personalities

Different signal sources, same goal: predict what would resonate.

**Their reactions to specific works.** What fiction have they publicly praised,
mocked, quote-tweeted, or recommended? A thread about a story they loved is
more informative than any demographic parameter. A dismissive subtweet about
a story they hated reveals their allergies.

**Their own creative output.** Blog posts, fiction, poetry, shitposts — all
reveal aesthetic values through practice. Someone who writes with deliberate
sincerity values different things than someone who writes with layered irony.

**Their communication style.** Sentence length, vocabulary, humor type,
reference density, relationship to earnestness vs. detachment. These are
stylistic fingerprints that indicate what prose registers they'd find
natural vs. grating.

**The taste cluster they belong to.** Online subcultures have recognizable
aesthetic tendencies. These provide a prior even when individual data is
sparse. The prior gets refined by whatever individual signal is available.

**Their general takes on art and culture.** What do they post about when
they're not talking about fiction? Philosophy, memes, politics, music? The
cultural context they operate in shapes what resonates with them.

**Adapted process:** Feed their posts/tweets/writings into an LLM acting
as a "cultural analyst" (not "psychologist" — different frame for this
population). Extract: aesthetic values, what excites vs. bores them, their
relationship to sincerity/irony/earnestness, what kind of surprise they
respond to, what registers they're fluent in.

### Minimal-Data Fallback

When you know almost nothing about a judge — a name and maybe a few posts:

1. Identify their **taste cluster** — the broadest applicable description
   of what tradition they operate in. This is the prior.
2. Add 1–2 **distinguishing specifics** — anything that differentiates them
   from the cluster center.
3. Populate the schema with this and let the system fill gaps with the
   cluster default.

Examples:
- "Iowa workshop tradition. Values compression and epiphany. Probably
  allergic to sentimentality."
- "tpot-adjacent. Leans toward sincerity over irony. Has praised Borges
  and Chiang. Skeptical of MFA workshop realism."
- "Genre fiction editor. Prioritizes pacing and emotional payoff over
  prose craft. Forgives stylistic roughness if the story delivers."

This is explicitly an approximation. The goal is to get the selection
pressure *roughly* pointed at the right target — and then let the
evolutionary loop do the fine-tuning through the disagreement signal.

### Common Principles

Regardless of source type:

- **Don't fake demographic precision.** Gupta et al. (2026) showed that
  simple demographic prompting often *degrades* alignment with real human
  preferences. "34-year-old woman from Brooklyn" adds noise, not signal.
  What matters is what they *value*, not who they *are*.

- **Approximation, not simulation.** The goal is to predict what would
  *resonate* with this judge — not to perfectly replicate their internal
  experience. The model will be wrong sometimes. That's expected and
  accounted for by the panel architecture.

- **Same 6-field output.** Regardless of whether the input was a Paris
  Review interview or a Twitter thread, the output is always the same
  schema: identity, values, exemplars, harshness, priority, model.

---

## Judge Output Quality Gates

LLM judges can produce slop too. Bad evaluations inject noise into the
evolutionary signal, misdirecting the search. The Synthetic Reader Panels
paper (arXiv:2602.14433) implements a 5-check system for catching degenerate
evaluations:

### 1. Score Clustering

If a judge gives the same score across all dimensions (SD < 0.3), the
evaluation is degenerate — the judge didn't actually differentiate. Flag
for regeneration.

### 2. Circular Reasoning

If the judge's reasoning mostly restates the story rather than analyzing
it (high word overlap between reasoning and story text, low novelty ratio),
the evaluation is empty. Measure via: overlap = shared 4-grams / total
4-grams in reasoning. High overlap + low unique assertions = circular.

### 3. Persona Mismatch

A casual genre reader using academic literary theory vocabulary. A demanding
literary editor producing only generic praise. If the judge's output doesn't
match their persona definition, the persona conditioning has failed.
Detect via vocabulary register analysis against the persona's expected register.

### 4. Generic Framing

The judge's reasoning uses LLM filler rather than specific critique: "This
story demonstrates a masterful command of narrative tension..." instead of
"The reveal in the third paragraph works because the preceding details about
the lottery tickets were innocuous enough to pass unnoticed."

Detect via: presence of Tier 1/2 slop words in the reasoning chain,
specificity score (does the reasoning reference specific passages, sentences,
or choices in the story?).

### 5. Cross-Story Repetition

If the same judge produces near-identical reasoning for different stories,
something is wrong. Measure via embedding similarity of reasoning chains
across stories for the same judge persona. High similarity = the judge is
on autopilot.

**Aggregation:** Evaluations that fail any gate are regenerated (up to 2
retries). If still failing, the judge is excluded from that comparison and
the result is flagged as reduced-confidence.

---

## The Reasoning Chain as Data

Every pairwise comparison produces explicit reasoning — "A beats B because..."
This is not just a preference signal. It's the richest data the system
generates.

### Feeding Mutation Operators

"Story A's ending felt rushed — the resolution came within two sentences
of the climax." → This becomes a targeted mutation instruction: extend the
denouement, add a beat between climax and resolution.

"Story B's voice broke in the dialogue scenes — characters sounded the same."
→ Mutation: regenerate dialogue with explicit voice differentiation constraints.

"Neither story surprised me. Both followed the same 'ordinary person
encounters the extraordinary' template." → Mutation: apply structural
inversion, start with the extraordinary and ground it.

### Episodic Memory

Across generations, the reasoning chains accumulate into a log of what
worked and what didn't:

- "Stories with this structure type keep losing in round 2"
- "The judges consistently reward stories that withhold the key information
  until the final third"
- "Voice pieces win against plot-driven stories when the voice is truly
  distinctive, but lose when the voice is generic-literary"

This memory is injected as context into future generation stages (Stage 1
concept evolution, Stage 3 voice evolution) so the system learns from its
failures across runs.

### Calibration Data

Periodically, run the pairwise tournament on stories with known human
preferences (from LitBench or HANNA) to check whether the judge panel's
rankings correlate with human rankings. If they diverge, the reasoning
chains explain *how* — which judges are drifting and in what direction.

---

## Disagreement Mechanics

The disagreement signal is tracked throughout the tournament:

### Within a Pairwise Comparison

When judges split on a pair:

- **5-0 agreement:** Decisive. One story is clearly stronger across all
  perspectives. High confidence in the result.
- **4-1 split:** Strong preference with one dissenter. The dissent reasoning
  is logged — it may identify something the majority missed.
- **3-2 split:** Genuinely contested. The losing story isn't eliminated —
  it's flagged as **polarizing** and receives a diversity bonus in the
  archive. The split reasoning (why the 3 preferred A, why the 2 preferred B)
  is the most informative data the system produces.

### Across the Tournament

Track each story's **win profile** — not just wins and losses but the
margin and consistency of those wins:

- **Consensus champion:** Wins most matchups by wide margins across all
  judge types. The safe pick. Probably excellent. Possibly generic.
- **Polarizing contender:** Wins some matchups decisively, loses others
  badly. The judge panel is split. This story is doing something bold
  that some judges love and others hate. Preserve it.
- **Narrow winner:** Wins most matchups by thin margins. Competent but not
  exciting. May advance but doesn't get a diversity bonus.

### Primary vs. Contrarian Splits

When primary judges (modeling the competition's actual judges) and contrarian
judges (deliberately opposed perspectives) disagree, the signal is
especially informative:

- **Primary loves it, contrarian hates it:** The story is well-targeted for
  the competition. Advance with confidence.
- **Contrarian loves it, primary hates it:** The story is interesting but
  wrong for this audience. Archive it for a different context.
- **Both love it:** Broad appeal. Rare and valuable.
- **Both hate it:** Genuine failure. Eliminate.

### Competition Mode

In competition targeting, **primary judge agreement drives selection.** The
diversity bonus from disagreement applies to the archive (keeping the
evolutionary population diverse) but final story selection for submission
weights primary judges most heavily.

The contrarian judges serve two purposes: (1) generating diversity pressure
that prevents the population from collapsing to a single judge-pleasing
formula, and (2) identifying stories with broader appeal that might succeed
in contexts beyond the immediate competition.

---

## Sources

- EvolvR — arXiv:2508.06046 (pairwise evaluation, multi-persona rationales)
- LitBench — arXiv:2507.00769 (pairwise preference training, 78% ceiling)
- GenRM — arXiv:2601.07149 (reasoning chains for story evaluation)
- Prometheus 2 — arXiv:2405.01535 (rubric format, 0.897 Pearson)
- WritingBench — arXiv:2503.05244 (dynamic criteria, 84% alignment)
- CritiCS — arXiv:2410.02428 (multi-persona critics, leader-referee)
- PREFINE — arXiv:2510.21721 (persona extraction from preferences, 83% win rate)
- Synthetic Reader Panels — arXiv:2602.14433 (5-check quality gates, tournament format)
- PoLL — arXiv:2404.18796 (panel > single judge, κ=0.763)
- Gupta et al. — arXiv:2602.18462 (demographic prompting degrades accuracy)
- Li et al. — ACL Findings 2025 (lay vs expert reader clusters)
- Lechmazur benchmark — github.com/lechmazur/writing (Hölder mean scoring)
