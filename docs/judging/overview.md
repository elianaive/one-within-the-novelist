# Judging & Evaluation System

The judge system is the selection pressure that drives evolution. Get it wrong
and the system optimizes for the wrong thing — polished mediocrity, judge-pleasing
formula, or outright reward hacking. Get it right and the system produces stories
that *resonate*.

This document covers the general philosophy and architecture. Stage-specific
evaluation criteria are covered in separate documents.

---

## Philosophy: Resonance, Not Polish

The north star question is not "is this story well-written?" It's **"did this
story grip you?"**

These are different questions. A technically flawless story can leave a reader
cold. A rough, imperfect story can haunt someone for years. The research is
clear on what drives the difference:

**Transportation** (Green & Brock, 2000; 24-year follow-up Green & Appel 2024)
is the master mechanism — a state of total cognitive, affective, and imagery-based
absorption where the reader loses awareness of anything outside the story.
Transportation doesn't require perfect prose. It requires vivid imagery,
emotional stakes, and the absence of "false notes" that break the dream.

**The peak-end rule** (Kahneman et al., 1993) means a story with one devastating
moment and a strong ending will be evaluated more favorably than an evenly
competent story. Duration is largely neglected in retrospective evaluation — a
reader forgives a slow middle if the climax and ending deliver.

**Emotional peaks drive memory** (Leong et al., Nature Human Behaviour 2025).
Emotionally intense narrative moments are associated with brain-wide network
integration that enhances recall fidelity. Stories with emotional peaks are
literally *remembered better* at the neural level. The meta-analysis by Mar et al.
(2021, N>33,000) confirms stories are better recalled and comprehended than
expository texts — the "story superiority effect."

**Optimal surprise** (Kumar et al., Cognitive Science 2023): Bayesian surprise
predicts where readers mark event boundaries in stories. The brain is a
prediction machine, and stories engage it by managing the balance between
predictability (enough to build a model) and surprise (enough to keep updating
it). Too little surprise is boring; too much is incoherent.

The implication for the judge system: craft metrics (lexical diversity, sentence
rhythm, structural completeness) are in service of resonance, not the other way
around. A story that scores perfectly on craft but poorly on resonance has
failed. A story with rough edges that scores high on resonance has succeeded.

---

## The Two-Tier System

The research reveals an asymmetry: we can detect *bad* much more reliably than
we can detect *great*. Some AI slop patterns appear 1,000x more frequently than
in human writing (Paech et al., ICLR 2026). Sentence burstiness, lexical
diversity, pronoun density — all cheap, reliable negative signals. But the best
automated judge achieves only 73-78% agreement with human preferences on what's
*good* (LitBench, EACL 2026).

This suggests a two-tier architecture:

### Tier A: Hard Constraint Filters

Fast, cheap, regex/stats-based checks that run on **every candidate at every
stage**. Purpose: eliminate clearly AI-sounding output before spending money on
expensive LLM evaluation.

| Filter | Metric | Reject If | Why |
|---|---|---|---|
| Slop words | Tier 1 word frequency per 1k words | >2x human baseline | "delve," "tapestry," "nuanced," etc. are 100-1000x overrepresented in LLM output |
| Burstiness | stdev(sentence_lengths) / mean | <0.4 | Human prose mixes short and long sentences; AI stays in a narrow band |
| Lexical diversity | MATTR-500 | <0.7 | LLMs reuse vocabulary more than humans |
| Pronoun density | Personal pronouns per 1k words | Significantly below human baseline | Fewer pronouns → less transportation (Voss et al., Nature 2025) |
| Em-dash overuse | Count per page | >2 per page | Documented AI writing signature |
| "Not-X-But-Y" | Pattern count per 1k chars | >0.5 | The single most overused LLM rhetorical pattern |
| Nominalization rate | -tion/-ment/-ness/-ity suffixes | >2x human baseline | AI prose is noun-heavy and informationally dense (Reinhart et al., PNAS 2025) |

These are gates, not scores. A candidate either passes or is rejected/flagged
for regeneration. The thresholds should be calibrated against a human fiction
baseline (Fiction-1B or Gutenberg3 DPO provide reference distributions).

### Tier B: Aspirational Scoring

Expensive, nuanced LLM judge panel evaluations that run only on candidates
that pass Tier A. This is where the real selection pressure lives — the
evaluation that distinguishes "competent" from "resonant."

---

## The Judge Panel

### Architecture

A panel of 5-10 parameterized reader personas, each evaluating stories
independently.

**Key implementation rules** (from PoLL, LitBench, bias research):

- **Different model families for generation vs. evaluation.** Self-preference
  bias is real: GPT-4 rates its own outputs ~0.52 higher on the Equal Opportunity
  metric (Wataoka et al., ICLR 2025). A model cannot judge its own work fairly.
  If generating with Claude, evaluate with GPT-4, Gemini, and/or open-weight
  models.

- **Single-turn independent evaluations.** Multi-agent debate causes conformity
  (opinions shift), confabulation (15.59% of reflected opinions came from
  neither original position nor debate, Chen et al. 2024), and impersonation
  (3.12% of messages). Each judge evaluates in isolation.

- **0-5 grading scale.** Achieves ICC = 0.853 vs 0.805 for 0-10 (Scale AI,
  arXiv:2601.03444). The advantage is especially pronounced on subjective
  open-ended tasks like fiction evaluation.

- **Chain-of-thought reasoning before scoring.** The judge must articulate its
  reasoning before assigning a number. This increases reliability (Prometheus 2
  achieves 0.897 Pearson correlation with humans using rubric + CoT).

- **Rubric with criterion + per-score descriptions.** Don't just say "rate
  emotional depth 0-5." Define what a 1 looks like, what a 3 looks like, what a
  5 looks like. Explicit anchors reduce arbitrary variance.

### Why Not a Single Judge?

The PoLL framework (Verga et al., 2024) showed that a panel of diverse models
achieves Cohen's κ = 0.763 vs a single GPT-4 judge at κ = 0.627 — while
costing approximately one-seventh as much. The Dynamic Jury approach (Tan et al.,
2025) further improves on static panels by weighting judges by their predicted
reliability for each specific data point.

But the deeper reason is that reader heterogeneity is a fundamental feature of
creative writing, not noise. Li et al. (ACL Findings 2025) analyzed 1,471
stories with 101 annotators and found two stable reader clusters:

- **Lay/surface readers**: prioritize sentence complexity, syntactic depth,
  lexical diversity, readability — *accessible, elaborate prose*
- **Expert/holistic readers**: prioritize topic entropy, sentence rhythm,
  rhetorical variety, sentiment dynamics — *thematic depth and craft*

AI-generated texts scored 57.6% preference among lay readers but only 2-13%
among experts. Both groups were right *relative to their criteria*. A single
judge collapses this distinction; a panel preserves it.

---

## Judge Parameters

Each judge is defined by 6 fields, designed to be fillable even with limited
information:

### 1. Identity

A name and one-paragraph description of who this judge is and how they approach
fiction. This is the core persona prompt.

Can range from a generic archetype:

> *"Elena Vasquez — A demanding literary editor with 20 years of experience at
> a mid-tier New York press. Values compression, subtext, and emotional honesty.
> Has little patience for exposition dumps or characters who explain their
> feelings. Reads 200+ manuscripts a year and rejects 95% of them."*

To a persona modeled on a real person:

> *"Modeled on [specific judge]. Values [inferred from their published
> criticism]. Tends to praise [patterns from their past selections]. Known to
> penalize [patterns they've criticized]."*

### 2. Values

A ranked list of 3-5 priorities — what this judge cares about most when
evaluating fiction. The ranking creates implicit dimension weighting without
needing numeric vectors. The LLM will naturally emphasize what's listed first.

Example:
1. Emotional honesty — does this feel *felt*, not performed?
2. Originality of premise — is there a genuine idea here?
3. Prose compression — is every word earning its place?
4. Doesn't care much about plot mechanics
5. Actively hostile toward sentimentality

The values field is where the real differentiation happens. Two judges with
identical demographic profiles but different values will evaluate very
differently — which is exactly what we want.

### 3. Exemplars (optional)

1-3 examples of this judge's taste: stories they've praised, specific quotes
about what they value in fiction, or sample evaluations they've written.

When modeling real judges, this is where their actual words go:

> *"'The best stories make me forget I'm reading. If I notice the prose, the
> prose has failed.' — from their 2024 interview in [Magazine]"*

> *"Awarded first place to [Story Title] in [Year], citing 'its refusal to
> explain itself' and 'the devastating restraint of the final page.'"*

When exemplars are available, they are the strongest signal for calibrating
the judge's behavior — stronger than any combination of demographic or
psychographic parameters. When unavailable, omit this field and rely on
identity + values.

### 4. Harshness

A simple calibration: **lenient / moderate / demanding**.

- **Lenient**: focuses on what works; generous interpretation of ambiguity; a
  3/5 from this judge means significant problems
- **Moderate**: balanced; identifies both strengths and weaknesses
- **Demanding**: focuses on what doesn't work; high bar; a 4/5 from this judge
  is exceptional praise

This exists because LLM judges have a documented sycophancy/leniency bias
(sentiment bias, compassion-fade). Explicitly setting harshness level —
especially including "demanding" judges — counteracts this. A panel of all
lenient judges produces inflated scores with no selection pressure.

Recommended panel composition: at least 30% of judges should be "demanding."

### 5. Priority

**Primary / secondary / contrarian** — determines weight in fitness calculation.

- **Primary judges** carry the highest weight in selection. In competition mode,
  these model the actual target judges. Their preferences drive the main
  evolutionary pressure.
- **Secondary judges** carry moderate weight. They represent general reader
  quality — the "would a thoughtful reader enjoy this?" check. They prevent
  over-optimization for primary judges at the expense of broader appeal.
- **Contrarian judges** are tracked but carry zero or minimal weight in
  selection. They represent deliberately opposed perspectives — the judge who
  hates what the primary judges love. Their scores are preserved for diversity
  analysis and for the disagreement signal (see below). They prevent the system
  from converging into an echo chamber.

### 6. Model

Which LLM family runs this judge. Must differ from the generation model. Can
be assigned permanently or rotated across generations.

Model diversity in the panel is one of the most reliable bias-reduction
mechanisms: a model family cannot self-preference its own outputs when other
families are in the panel (PoLL finding). Use at least 2-3 different model
families across the panel.

---

## Modeling Judges After Real People

When targeting a specific competition, audience, or editorial preference, the
system can construct judge personas modeled on real individuals.

**The approach** (validated by PREFINE, arXiv:2510.21721, which achieves 83%
win rate over generic personas when deriving evaluation criteria from observed
preferences):

1. **Gather their words.** Published criticism, interviews, essays about craft,
   introductions to anthologies they've edited, acceptance/rejection commentary.
   What specific language do they use to praise? To critique?

2. **Study their decisions.** What have they selected as winners? What patterns
   emerge across their selections — genre, tone, structure, theme? What do the
   winning stories have in common that the runners-up lack?

3. **Extract their evaluative vocabulary.** Do they talk about "voice" or
   "language"? "Plot" or "situation"? "Characters" or "people"? The words a
   judge uses reveal what ontology they evaluate within.

4. **Populate the schema.** Identity from their biography and public persona.
   Values from their stated and revealed preferences. Exemplars from their
   actual words and past selections. Harshness from the tenor of their
   criticism.

5. **Acknowledge the approximation.** This is modeling, not simulation. The
   goal is to predict what would *resonate* with this judge, not to perfectly
   replicate their internal experience. The model will be wrong sometimes —
   that's expected. The evolutionary pressure should be "stories this judge
   would probably find compelling," not "stories that perfectly match this
   judge's historical patterns."

For cases where limited information is available (a new judge, or someone with
little public criticism), start with identity + a rough values guess and
iterate: generate stories, evaluate whether they feel like the kind of thing
this person would appreciate, adjust.

---

## Resonance Dimensions

Ten evaluation dimensions grounded in cognitive science of narrative engagement.
These are **resonance dimensions**, not craft checklist items — they measure
whether the story *works on the reader*, not whether it follows rules.

### 1. Transportation / Immersion
*Does the story pull the reader into its world?*

Grounded in Green & Brock's transportation theory (2000). Full transportation
requires three components simultaneously: cognitive absorption (focused
attention), affective involvement (emotional engagement), and vivid imagery
(mental simulation of scenes). A story that delivers all three creates the
state where the reader "forgets they're reading."

Textual proxies: imagery word density, affective word density, sensory
language, scene-to-exposition ratio, absence of dream-breaking elements
(inconsistencies, wrong tone, meta-awareness — Gardner's "vivid and continuous
dream").

### 2. Suspense Architecture
*Does the story create and manage tension effectively?*

Brewer & Lichtenstein (1982) distinguish three narrative affects: suspense
(uncertain outcome, protagonist at risk), surprise (unexpected revelation),
and curiosity (information gap about past events). All three arise from
discourse structure — the temporal relationship between events and their
disclosure — not from content alone.

Zillmann's excitation transfer (1996) explains why post-tension emotional
peaks feel amplified: residual arousal from unresolved suspense transfers to
the resolution moment. A story that builds tension before releasing it
produces a more intense emotional peak than one that delivers the same
resolution without prior tension.

### 3. Curiosity / Information Gaps
*Does the story make the reader want to know something specific?*

Loewenstein's information-gap theory: curiosity arises from awareness of a
gap between current and desired knowledge. In fiction: mysteries, unexplained
backstories, hidden motivations, unanswered questions. Curiosity differs from
suspense — curiosity is epistemic (understanding), suspense is teleological
(outcome).

The pull-forward effect: a well-placed information gap at a scene break or
chapter ending creates the "page-turner" quality — the reader *must* continue.

### 4. Emotional Depth
*Does the story produce genuine feeling, not performed emotion?*

Deliberately broader than "character depth" — emotional depth can come from
a concept (the vertigo of Borges' infinite library), a situation (the horror
of Jackson's lottery), a voice (the ache of Hemingway's omissions), or a
character (the grace of O'Connor's grandmother). The mechanism varies; the
effect — real emotional response in the reader — is the constant.

The paradox of fiction (Radford 1975): readers feel genuine emotions for
characters they know don't exist, because the brain's emotion systems don't
reliably gate on ontological status. Emotional engagement with fiction is
real and intense.

### 5. Emotional Arc Coherence
*Does the emotional trajectory feel earned?*

Reagan et al. (EPJ Data Science, 2016) identified six core emotional arc
shapes across 1,327 stories: rags-to-riches (rise), tragedy (fall),
man-in-a-hole (fall-rise), Icarus (rise-fall), Cinderella (rise-fall-rise),
Oedipus (fall-rise-fall). "Man in a hole" and "Cinderella" shapes correlate
with higher story popularity.

The key word is *earned*: each emotional shift must feel like a consequence
of what preceded it. An arc that jumps from despair to triumph without
adequate cause feels manipulative; one that descends into tragedy without
establishing stakes feels arbitrary.

### 6. Causal / Logical Coherence
*Do events follow from each other, or do they feel random?*

Chen & Bornstein (Trends in Cognitive Sciences, 2024) showed that causally
central events (those with more causal connections to other events) are
rated as more important and are better recalled. The hippocampus shows
elevated activity for causally central events. A tight causal chain creates
a sense of inevitability; a loose one creates confusion or boredom.

For non-plot-driven stories, "causal" extends to logical, implicational,
or thematic coherence — does each part connect to the others in a way
that feels meaningful?

### 7. Surprise + Post-dictability
*Are there genuine surprises that feel inevitable in retrospect?*

Bissell et al. (WNU @ ACL 2025) operationalized narrative surprise across
six criteria: predictability (inverse), post-dictability (retrospective
coherence), initiatoriness (does the surprise explain earlier events?),
immutability violation (does it break story-world rules?), importance
(protagonist impact), and valence.

The ideal: low predictability + high post-dictability. The reader doesn't
see it coming, but looking back, it was the only possible outcome. This is
what O'Connor's "grace through violence" achieves, what Jackson's reveal
achieves, what Chiang's final realization achieves.

### 8. Ending Quality
*Does the ending land?*

Kahneman's peak-end rule means endings disproportionately determine overall
story evaluation. A weak ending can sink an otherwise excellent story; a
powerful ending can redeem a shaky middle.

What makes an ending work: emotional resolution (the dominant tension is
concluded), thematic resonance (the ending speaks to what the story is
*about*), and the "click" — the Gestalt sense that the pattern is complete.
Ambiguous endings can be powerful if they provide interpretive richness
rather than mere incompleteness.

### 9. Flow / Pacing
*Does the story maintain momentum?*

Thissen et al. (2018, 2021) showed that flow — effortless processing,
loss of time awareness, intrinsic enjoyment — is the **strongest predictor
of reading pleasure**, more predictive than suspense, identification, or
cognitive involvement independently.

Flow requires a balance between challenge and skill (Csikszentmihalyi):
prose too simple produces boredom; too complex produces frustration. Event
density, sentence length variation, chapter hooks, and absence of cognitive
interruption (inconsistencies, jargon, exposition dumps) are the levers.

### 10. Memorability / Distinctiveness
*Would the reader recall this story days later?*

The story superiority effect (Mar et al., N>33,000): stories are remembered
better than facts. Within stories, emotional peaks drive memory (Leong et al.,
Nature 2025). A story with at least one "indelible image" — a scene vivid
and emotionally charged enough to persist in memory — has achieved something
most stories don't.

Distinctiveness also matters at the concept level: is this story *about*
something that hasn't been said this way before? The Sui Generis score
(Microsoft, PNAS 2025) measures exactly this — how unlikely is this story's
plot relative to the space of stories an LLM would generate?

### A Note on Dimension Relevance

Not all dimensions are relevant to all stories. A voice/compression piece
("Hills Like White Elephants") might score low on "suspense architecture"
and that's perfectly fine — the story isn't trying to create suspense. A
concept story ("Tlön, Uqbar, Orbis Tertius") might score low on "emotional
arc" because the arc is intellectual, not emotional.

The dynamic rubric system (see below) addresses this: evaluation criteria are
generated per story based on what the story is trying to do. The 10 dimensions
serve as a baseline vocabulary, not a mandatory checklist.

---

## Dynamic Rubrics

WritingBench (Alibaba, NeurIPS 2025) found that dynamic per-instance
evaluation criteria achieve 84% human alignment vs 67% for static rubrics.
The reason is obvious in retrospect: a ghost story and a literary meditation
about grief need different success criteria. Evaluating both against the same
fixed rubric penalizes one of them.

**How it works in our system:**

1. After a story passes Tier A filters, generate 3-5 story-specific evaluation
   criteria based on what the story is trying to do. These are informed by the
   story's concept (from Stage 1), structure (from Stage 2), and voice (from
   Stage 3).

2. Each criterion has a name, a description, and explicit score anchors
   (what does a 1 look like? a 3? a 5?).

3. The 10 resonance dimensions serve as the baseline. The dynamic criteria
   are an overlay — additional dimensions specific to this story's ambitions.

4. Example: for a story structured primarily around disclosure edges
   (a reveal story), the dynamic criteria might include "reveal impact" and
   "dramatic irony quality" — dimensions that aren't in the baseline 10 but
   are critical for this specific story.

5. Example: for a minimalist voice/constraint story, the dynamic criteria
   might include "subtext density" and "constraint adherence" — measuring
   whether the story achieves its effects through what it *doesn't* say.

---

## Scoring: Hölder Mean

The standard approach — weighted average across dimensions — has a critical
flaw: a story that scores 5/5 on prose and 1/5 on coherence averages to the
same score as a story that scores 3/5 on everything. But the first story is
broken and the second is merely mediocre. Readers don't average across
dimensions; they're derailed by the weakest link.

**The Hölder mean** (used by the Lechmazur benchmark, the most rigorous
community writing evaluation) with parameter p ≈ 0.3-0.5 acts as a
**soft minimum**: weaknesses drag the score down more than strengths can
compensate. At p → −∞ it becomes the strict minimum (weakest link
determines everything); at p = 1 it's the arithmetic mean. The sweet spot
(p ≈ 0.3-0.5) penalizes unevenness while still rewarding genuinely strong
dimensions.

**Application:**

- **Within a single judge**: Hölder mean across all scored dimensions for
  that story. This means a story must be *well-rounded* to score highly —
  it can't exploit one strong dimension to mask failures elsewhere.

- **Across judges**: Track both the **average** score (overall quality signal)
  and the **variance** (disagreement signal). Don't Hölder-mean across judges —
  the variance is information (see next section).

---

## The Disagreement Signal

The most counterintuitive idea in the judge system: **high inter-judge
variance is a feature, not a bug.**

The individual-collective creativity paradox (Doshi & Hauser, Science
Advances 2024): optimizing individual story quality via LLM selection
increases quality but *decreases diversity* — "an increase in individual
creativity at the risk of losing collective novelty." Selecting for
consensus produces homogeneous excellence. Selecting for polarization
preserves genuine range.

**Implementation:**

Track two metrics for each story across the judge panel:

- **Mean score** — the overall quality signal
- **Score variance** — the disagreement signal

A story with high mean + low variance → competent, broadly appealing,
potentially safe/generic.

A story with high mean + high variance → some judges love it, some hate it.
This story is doing something bold, interesting, or genuinely different.

A story with low mean + high variance → might be flawed in ways some judges
forgive. Worth investigating but not automatically advancing.

A story with low mean + low variance → consensus: this doesn't work. Eliminate.

**In evolutionary selection:** stories with high mean AND high variance
receive a **diversity bonus** — they are more likely to survive selection than
an equally-scored story with low variance. This explicitly protects bold,
polarizing work from being ground down to consensus mediocrity.

**Primary vs. contrarian judges:** The disagreement signal is particularly
informative when it splits along the primary/contrarian axis. A story that
primary judges love but contrarian judges hate is well-targeted. A story that
contrarian judges love but primary judges hate might be genuinely interesting
but wrong for the target audience — worth archiving for a different context.

---

## Coevolving Judges

Goodhart's Law: "When a measure becomes a target, it ceases to be a good
measure." Optimizing stories against fixed judges will eventually game them.

The evidence is specific. Documented failure modes of LLM judge optimization
(Gao et al., ICML 2023; Lilian Weng survey, 2024):

- **Length inflation**: stories get longer to satisfy judges that conflate
  length with quality (16-20% win rate inflation from length bias alone)
- **Sycophancy patterns**: stories adopt whatever tone the judge model
  prefers (optimistic, inclusive, carefully balanced)
- **Template exploitation**: stories converge on whatever templates the judge
  LLM rates highly, reintroducing the echo problem
- **Proxy-gold divergence**: the relationship between proxy reward (judge
  score) and true quality follows a concave curve — past a threshold, higher
  judge scores correspond to *lower* actual quality

**The Red Queen solution** (from evolutionary computation):

Judges and stories evolve together in an arms race. As stories get better at
satisfying current criteria, the judges adapt:

1. **Rubric regeneration**: every N generations, regenerate the dynamic rubrics
   with fresh criteria. The evaluation dimensions shift, preventing the
   generator from locking onto fixed targets.

2. **Human anchor checks**: periodically evaluate evolved stories against
   held-out human preference data (LitBench, HANNA). If the judge panel's
   rankings diverge from human rankings, recalibrate.

3. **Model rotation**: periodically change which LLM families serve as judges.
   A new model brings new biases — but also new resistance to the old model's
   gaming strategies.

4. **Adversarial judges**: introduce judges specifically designed to detect
   gaming — a "meta-judge" that asks "does this story feel like it was written
   to please an LLM judge?" (The "Measuring AI Slop" paper, arXiv:2509.19163,
   found that LLMs under-predict slop at 0.03-0.08 vs human 0.34 — but an
   adversarial judge explicitly looking for slop can partially close this gap.)

5. **Escalating standards**: as the story population improves, tighten Tier A
   thresholds and raise the harshness distribution of the panel. What passed
   in generation 1 should be rejected in generation 50.

---

## Known Biases and Mitigations

LLM judges carry at least 12 documented systematic biases (Li et al., ICLR
2025). The panel architecture mitigates but does not eliminate them:

| Bias | What Happens | Mitigation |
|---|---|---|
| Position bias | Preference for responses at certain positions | Randomize ordering; run multiple permutations |
| Verbosity/length | Preference for longer responses (16-20% inflation) | Length normalization; explicit word count constraints |
| Self-preference | Models rate own outputs higher | Never use same model family for generation and evaluation |
| Sycophancy | Agreement with implied preferences | Separate evaluation from generation context; blind evaluation |
| Conformity | Opinion shifts in multi-agent settings | Independent single-turn evaluations; no debate |
| Sentiment bias | Skewing toward positive assessment | Include "demanding" judges; require identification of weaknesses |
| Authority bias | Deference to attributed expertise | Blind evaluation; strip author metadata |
| Compassion-fade | Less critical of later items in batches | Evaluate in randomized, bounded batches |
| Halo effect | Strong performance on one dimension inflates others | Evaluate one dimension per call (dimension-specific prompts) |
| Familiarity bias | Preference for recognizable patterns/genres | Explicit novelty dimension in evaluation |
| Cultural bias | Western/educated/English-language perspective | Include non-Western reader personas; cultural awareness in rubrics |
| Homogeneity bias | All models converge on similar evaluations | Use 2-3+ model families; track inter-model variance |

The most important single mitigation: **use different model families for
generation and evaluation.** This alone eliminates self-preference bias and
reduces several correlated biases.

---

## Calibration

The best off-the-shelf LLM judge achieves 73% agreement with human story
preferences (Claude 3.7 Sonnet on LitBench). Trained reward models achieve
78%. Human-human agreement on creative writing quality is typically 80-90%
for clear quality differences but drops for close comparisons.

This means the judge system will be wrong approximately 22-27% of the time.
For an evolutionary system, this noise compounds across generations —
incorrect selections in early generations misdirect the entire search.

**Mitigation:**

- **Panel aggregation** reduces individual judge error (PoLL: variance 2.2
  vs 6.1 for a single judge).
- **Periodic human-in-the-loop checkpoints**: every N generations, a human
  reviews the top-performing stories and recalibrates if the system has drifted.
- **Calibration datasets**: HANNA (1,056 stories, 6 dimensions, MIT license),
  CR4-NarrEmote (200k+ emotion annotations, CC0), LitBench (46k preference
  pairs), Reagan arcs (6 arc shapes, CC BY 4.0). These provide ground truth
  for validating that the judge panel's rankings correlate with established
  human judgments.
- **Accept the ceiling**: 78% is the current state of the art for automated
  creative writing evaluation. Design the system to be robust to ~25% noise
  rather than trying to eliminate it — which is why population-based evolution
  with diversity maintenance is preferable to single-trajectory optimization.
