# Stage 2: Judges

Stage 2 reuses Stage 1's judge panel — the four contest-specific personas that currently evaluate concepts. This document covers the reuse decision, the specialist-vs-generalist question, and the persona-adaptation issues specific to evaluating structural DAGs through contest-submission lenses.

Judge persona specifications live in `configs/judges/*.yaml` (same files used by Stage 1). Stage 2 does not redefine personas.

---

## Reuse Decision: Same Four Judges

For v1, Stage 2 uses the same four contest personas as Stage 1 (per `configs/stage_1/*.yaml` → `judges.panel`):

- **Gwern Branwen** — practitioner-turned-judge; adversarial taste trained on his own kept-vs-killed line inventories. Catches slop-in-costume: antithesis bloat, list-negation, schmaltz dressed as restraint, "view from nowhere" dressed as universal resonance. Bar is *"this could change how I think about something,"* not *"this is technically well-made."*
- **Roon** — shape-rotator aesthetic; *beauty equals truth under the iron law.* Values phrase-level surprise, compression over exposition, vibes carrying load-bearing content. Public-figure authenticity detector tuned against flattery-engineering.
- **Alexander Wales** — analytical systems-eye trained on 1.6M words of his own *Worth the Candle.* Articulates *why* something works rather than declaring preference. Rational fiction (logic and consequence foundational), mechanical payoff, second-order effects. Credits craft-ambition even when execution trails it.
- **Jamie Wahls** — contest organizer; his taste functionally set the rubric. Diagnoses AI fiction's structural failure as *sloploaf* — homogenized output without tension arcs, narrative loops, or knowledge of its own ending. Scale-meets-intimate voice; ironic-sincere register; alignment-stakes literate (MIRI background).

These are the Un-Slop Prize judges. Stage 2 inherits them as-is because the contest frame *is* the selection pressure we want at every stage — concept, structure, voice, and prose all get evaluated by the same readers who'd receive a finished submission.

### Why reuse rather than design Stage-2-specific judges

Three reasons:

1. **Cross-stage consistency.** A concept traverses Stages 1–5 under the same judging eye. When Wales praises a concept's mechanical payoff at Stage 1 and then Stage 2's structure commits to that payoff, we've closed a loop. Different judges per stage would fragment this signal.

2. **The contest frame is the target.** The contest's rejection criterion is "unpleasant, or mid" (`lab/references/unslop-contest/unslop-aesthetic.md`). Mid is a *whole-submission* verdict — a concept, structure, voice, or prose piece can be mid independent of the others. Evaluating structure through the contest lens means asking "does this structure produce a non-mid submission," which is the question Stage 4's prose needs a structural answer to.

3. **Research backing.** `lab/references/poll-framework/` establishes that panels with diversity across model families achieve higher κ than single judges (0.763 vs 0.627 for 3-judge panels in the cited study). The 4-judge contest panel inherits and extends this; cross-family routing per judge is already in place in the Stage 1 configs.

### When to reconsider

Retire or augment the panel if:
- A specific Stage 2 dimension systematically produces uninformative scores across all four judges (e.g., all four always tie on Edge Logic).
- Judge reasoning for a dimension is vague or reused across different DAGs (sign of prompt/persona mismatch with the task).
- The contest frame proves a poor match for structural evaluation — e.g., Gwern's prose-craft-focused slop detection doesn't translate to structure and his judgments become generic.

None of these are currently observed — v1 reuses the panel as-is and watches.

---

## What 4 Judges Changes Mechanically

Moving from 3 judges to 4 judges produces a subtle but real shift in the per-dimension vote distribution:

- **3-judge dimension majority** always resolves to a decision (one judge breaks any tie): 2-1 or 3-0 in either direction.
- **4-judge dimension majority** can split 2-2, which without further tiebreaking would be a tie on that dimension.

**Mitigation shipped (2026-04-24): magnitude-differential tiebreaker at the dim level.** `_aggregate` in `owtn/evaluation/pairwise.py` handles 2-2 splits by comparing mean magnitudes on each side. If the gap is ≥ 0.25 (the minimum distance between adjacent magnitude buckets {narrow 0.5, clear 0.75, decisive 1.0}), the dim flips to the higher-magnitude side. Same-bucket splits (e.g. 2 clear + 2 clear) stay as ties. This recovers signal that a 3-judge panel would have captured naturally via majority, without changing Option E's architecture. See `lab/issues/2026-04-24-aggregate-magnitude-tiebreaker.md` and the 4-judge panel aggregation research (`lab/deep-research/runs/20260424_031135_judge-panel-aggregation-4judge/`, Finding F9).

Residual 2-2 splits (symmetric magnitudes) still pass through as dimension ties and push weight onto the **weighted aggregate + asymmetric tiebreaker** mechanism in `_select_winner`. This is correct behavior — the tiebreaker is designed for close contests, and symmetric-magnitude 2-2 splits are genuinely close.

**Pilot watch:** track three rates across runs — (a) 2-2 dim-level raw split rate (before magnitude tiebreaker); (b) fraction of 2-2 splits resolved by magnitude gap ≥ 0.25; (c) residual tie rate on close-contest weighted aggregates. If (c) exceeds ~30% of comparisons after the magnitude tiebreaker, the aggregation is noise-limited and we should consider (i) adding a 5th judge, (ii) introducing a dim-specific primary-judge tiebreaker, or (iii) weighting judges asymmetrically (e.g., Wales's vote worth 1.5× on Edge Logic and Motivational Coherence).

---

## Generalist vs Specialist: Deferred

All four judges evaluate all 8 dimensions. This matches Stage 1's pattern.

### Arguments for specialists

- **Prompt economy**: a judge evaluating only 3 dimensions has a shorter rubric prompt, cheaper per call.
- **Focused attention**: a judge asked to look only at causal soundness may spot issues a generalist misses.
- **Natural division of labor**: logical structure (Edge Logic, Motivational Coherence) maps naturally to Wales's rational-fiction sensibility; beat specificity (Beat Quality) maps to Gwern's kept-vs-killed line inventory; concept fidelity maps to Wahls's contest-organizer "did we get what we were looking for" frame; phrase-level load-bearing-ness (inside Beat Quality's sub-criteria) maps to Roon's compression aesthetic.

### Arguments for generalists

- **Simplicity**: fewer moving parts, easier to calibrate.
- **Integration across dimensions**: Beat Quality and Concept Fidelity aren't fully independent; a judge reading all dimensions can cross-reference.
- **Persona integrity**: each persona is a whole aesthetic commitment, not decomposable into per-dimension specialists. Gwern's "view from nowhere" detector fires equally on a concept's target effect and a DAG's beat sketches — specializing him to only one dimension would amputate the detector.

### v1 decision

All four judges evaluate all 8 dimensions. Monitor for three things:
1. **Per-dimension consistency within a judge**: does Wales's reasoning on Edge Logic look like Wales's voice (systems-eye, *"what does this constrain, what does it allow"*), or does it look generic?
2. **Cross-dimension correlation**: do judges score Edge Logic and Motivational Coherence together (suggesting they're evaluating "logical coherence" holistically rather than the decomposed dimensions)?
3. **Dimension skip rate**: how often do judges default to "tie" on a specific dimension? Systematic abstention on a dimension flags either prompt mismatch or genuine dimension irrelevance for that persona.

If monitoring surfaces strong evidence for specialization, v2 can split. Not before.

---

## Judge Prompt Adaptation for Stage 2

The judge prompt structure is:

```
{JUDGE_PERSONA}                  (from configs/judges/<id>.yaml, unchanged)

{STAGE_2_BASE_SYSTEM}            (typed-edge taxonomy, genome schema, phase context)

{DIMENSION_RUBRIC}               (from rubric-anchors.md, the anchors + sub-criteria)

CONCEPT:
{CONCEPT_JSON}                   (from the Stage 1 winner, including anchor_scene)

STRUCTURE A:
{INCIDENT_ENCODED_DAG_A}

STRUCTURE B:
{INCIDENT_ENCODED_DAG_B}

TASK: Evaluate which structure is stronger on {DIMENSION_NAME} for this concept.

{OUTPUT_FORMAT}                  (reasoning + winner in JSON)
```

### What's new vs Stage 1

- **Stage 2 base system message** replaces the Stage 1 base message. It introduces the typed-edge taxonomy, the genome schema, and the phase context.
- **Stage 2 rubric anchors** replace Stage 1's (different 8 dimensions).
- **Incident-encoded DAG rendering** replaces the concept JSON that Stage 1 judges read.
- **Concept context included**: Stage 2 judges need to see the concept (including `anchor_scene`) to evaluate structure relative to concept intent. Stage 1 judges read the concept alone.

### What's reused

- Persona definitions from YAML files
- Output schema (reasoning + winner)
- Tie semantics (explicit tie permitted)
- Dual-ordering mitigation
- Majority aggregation across judges (now 4; see §What 4 Judges Changes)

---

## Character Consistency

A recurring risk in pairwise prompting: when two DAGs are similar in most ways, a judge may be unable to discriminate and falls back on position bias. Research (`lab/references/judging-judges-position-bias/`) established that quality gap drives position bias; narrow quality differences amplify it.

Mitigations:

1. **Dual ordering** (always on). Catches judges whose preference flips with position.
2. **Tie permission** (explicit in rubric). A judge who can't discriminate should return tie, not force a guess.
3. **Reasoning required**. Every judgment requires a 2–4 sentence reasoning string that cites specific beats and edges. Lazy "A seems stronger" judgments are filtered at parse time.
4. **Score 3 exemplar in rubric**. Per `lab/references/reference-answer-bias/`, providing a score-3 exemplar along with endpoints (scores 1 and 5) improves ρ from 0.62 to 0.76. Stage 2 rubric-anchors.md includes score-3 exemplars.

---

## Cost Per Pairwise Comparison

Per `evaluation.md` §Number of LLM Calls:

- **Cheap judge (rollout reward signal)**: 1 LLM call per terminal, evaluating all 8 dimensions in one structured response. At ~$0.001–$0.002 per call. Fires ~200 times per tree.
- **Full panel (commitment events)**: 4 judges × 2 orderings = 8 LLM calls per pairwise comparison. Each call evaluates all 8 dimensions. At mid-tier routing (~$0.005–$0.01 per call): **$0.04–$0.08 per full pairwise comparison**. Fires at champion-promotion gates, within-concept tournament, and QD archive insertions.

The tiered design (see `mcts.md` §Reward Function §Tiered judge design) is the primary latency lever — 200 cheap calls at ~1s each versus 200 eight-call panels at ~3-5s each is the difference between a ~3-minute concept and a ~12-minute concept.

**Change from the earlier 3-judge design:** full-panel cost up ~33% per comparison. Since full-panel events are the dominant cost at light.yaml (~$2.50–$3.30 per concept depending on promotion-gate frequency), this is a real line-item bump, not negligible. Monitor in the first pilot.

### Cost control levers

- **Dimension skipping** for obvious losers: when an overall winner is clear after 6 dimensions (6-1 or 5-0-2 splits), skip the remaining 2. Saves a fraction of comparison cost.
- **Champion-challenger pre-filter** (in `evaluation.md` §Validation Before Evaluation, Gate 3): skip pairwise for near-identical DAGs in the same QD cell.
- **Judge model routing**: the four judges can route to progressively cheaper models within the family-diversity constraint. Current setup routes each judge to a different family in Stage 1; Stage 2 inherits.

---

## Persona-Adaptation Concerns Specific to Stage 2

Each contest judge's aesthetic is tuned for submission-grade prose. Stage 2 evaluates structural DAGs — sketch-level nodes and typed edges, not prose. The translation is natural in some cases and strained in others. Stage 2's first pilot is where each translation gets validated.

### Does Gwern's slop-detector fire on structure?

Gwern's red-flag list (antithesis bloat, list-negation, schmaltz, "view from nowhere") is prose-pattern-based. *"View from nowhere"* has a direct structural analog: a DAG whose beats could attach to any concept — generic conflict → generic escalation → generic resolution. That's the structural shape of a view-from-nowhere concept. **Risk:** the other red flags (antithesis bloat, list-negation) don't obviously translate to structure, so Gwern may over-fire on "view from nowhere" and under-fire elsewhere, producing a less rich signal than his concept-stage evaluation. **Pilot check:** does Gwern's reasoning on structure cite specific beats and edges, or does it collapse to "this DAG is generic"?

### Does Roon read DAGs through compression?

Roon's *"a phrase is an operating system, not decoration"* is sentence-level. At structure granularity, his analog is beats-as-operating-systems: each beat should *do* something (plant, reframe, install motivation) rather than *describe* something. The "phrase-level surprise" he values on prose translates to "edge-payload surprise" at structure — a disclosure edge whose `withheld` field is unexpected yet inevitable. **Risk:** if Roon's persona is too sentence-focused to engage structurally, his per-dimension votes collapse to generic "A's beats feel more load-bearing" without specifics. **Pilot check:** does Roon's reasoning name specific edge payloads as load-bearing or decorative?

### Does Wales translate naturally?

Yes — this is the most direct translation of any judge. Wales's rational-fiction frame *is* structural evaluation. Mechanical payoff, second-order effects, "what does it constrain, what does it allow" — all of these read native on a typed-edge DAG. Edge Logic and Motivational Coherence are his home territory. **Risk inverted:** Wales may over-index on structural dimensions and under-weight Beat Quality or Concept Fidelity (Thematic). **Pilot check:** does Wales's vote distribution across the 8 dimensions look flat, or is he silent on non-structural dimensions?

### Does Wahls's "would the contest accept this" work at structural granularity?

Wahls's diagnosis of AI fiction is structural: *sloploaf* is the absence of tension arcs, narrative loops, knowledge of ending. This is almost by definition a structural evaluation — "does this DAG have a tension arc? does it know its ending?" are Stage 2 questions. His frame translates cleanly. **Risk:** Wahls's published work emphasizes voice (chatlog/DM format, ironic-sincere register) — at structure granularity, voice isn't available yet, so one of his signal sources is muted. **Pilot check:** does Wahls's reasoning reach for voice indicators (quoting sketch text as if it were prose), or does it stay structural?

---

## Open Questions Surfaced in Judges Drafting

1. **Panel harshness calibration (inherited Stage 1 gap).** The `harshness: moderate|demanding` field on each judge persona was declared in `configs/judges/*.yaml` but never wired in Stage 1's pairwise prompt assembly. Stage 2 inherits this unwired. **Decision: don't block Stage 2 on it.** Wiring harshness affects both stages identically (both use the same judge personas and pairwise protocol), so it's a cross-stage refactor rather than a Stage 2 concern. Same logic applies to the `priority: primary|secondary|contrarian` field. Track as a separate issue alongside Stage 2 implementation; both stages benefit when it's wired.

2. **Cross-stage judge drift.** If Stage 2 judges develop implicit calibration to structural evaluation, their Stage 1 calibration may drift. Monitoring: periodically re-evaluate a fixed Stage 1 concept set and watch for score drift.

3. **The `priority` field (primary/secondary/contrarian) in judge configs is not wired.** Should Stage 2 use it? If yes, how? Default: not used; the four judges weight equally. If we want a contrarian whose vote counts differently on close comparisons, specify. Potential Stage-2-specific answer: Wales as `primary` on structural dimensions (Edge Logic, Motivational Coherence, Tension/Info Arch, Structural Coherence), Gwern as `primary` on Beat Quality, Wahls as `primary` on Concept Fidelity (Thematic), Roon as `contrarian` on Post-dictability. Don't implement until pilot shows flat weighting is insufficient.

4. **(Resolved — 2026-04-20.)** Judge reasoning from prior comparisons feeds the next expansion prompt via a tree-level lazy summarizer producing a `ChampionBrief` (see `mcts.md` §Champion Brief Feedback Loop and `lab/issues/2026-04-20-stage-2-expansion-feedback-summarizer.md`). Note on subject: the brief is about *the tree's exploration*, not about any specific champion — Stage 2 champion churn is too fast for the per-subject Stage 1 pattern.

5. **Judge availability during bidirectional phases.** The phase transition (forward → backward) reshapes the search space. Should the judges get told which phase produced a terminal? Probably yes — it helps them contextualize the DAG (a forward-phase terminal has skeletal opening, full resolution; a backward-phase terminal has both full). Include phase metadata in the base system message.

6. **Even-panel dim-level ties.** 4-judge panels produce 2-2 splits at dimension level that 3-judge panels never did. Push more weight onto the weighted-aggregate tiebreaker in `_select_winner`. Tracked in §What 4 Judges Changes Mechanically; pilot threshold is ~30% dim-level tie rate before we revisit panel composition.
