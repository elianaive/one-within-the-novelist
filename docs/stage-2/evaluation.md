# Stage 2: Evaluation

Stage 2 evaluation answers two questions:

1. **During MCTS**: which candidate partial DAGs should be expanded next? This requires a reward signal at every rollout's terminal state.
2. **After MCTS**: which of the 4 pacing-preset winners should advance to Stage 3?

Both use per-criterion pairwise comparison across 8 evaluation dimensions, with tiered judging — a single cheap judge running both orderings in parallel (no added latency) for the rollout reward signal, and the full 3-judge × 2-ordering panel for commitment events (champion promotion, within-concept tournament, QD archive insertion). See `mcts.md` §Reward Function §Tiered judge design for the full rationale. This doc specifies the DAG rendering format, the judge prompt structure, the scoring protocol, the within-concept tournament, and top-K advancement.

Rubric anchors per dimension — the content judges read — live in `rubric-anchors.md`. Judge persona design is in `judges.md`. This doc covers the mechanics of evaluation.

---

## Evaluation Flow

Evaluation fires in two contexts:

### Context 1: Rollout terminal evaluation (during MCTS)

Every MCTS rollout produces a terminal DAG. The cheap judge compares it against the tree's running champion on all 8 dimensions — two calls in parallel, one ordering each — and per-dimension votes are collapsed using Stage 1's dual-ordering rule: a dimension counts as a win only if both orderings name the same winner; disagreement collapses to tie. Rollout score = (dimension_wins + 0.5 × ties) / 8 from the challenger's perspective on the collapsed votes. This score is backpropagated up the tree.

If the cheap judge declares the challenger wins overall (after dual-ordering collapse), the full 3-judge × 2-ordering panel runs to verify before the champion is promoted. On rejection, the score backpropagated is **0.5** (the cheap-judge's own all-tie value) — not the cheap-judge's original win-score — to correct the false-positive signal without mixing full-panel statistics into the tree's UCB. See `mcts.md` §Reward Function for the full rationale. Cheap-vs-full agreement is logged at every promotion gate for drift monitoring.

Narrative forecasting is dropped from v1 entirely (decided 2026-04-19 second review pass). Tournament tiebreakers fall back to dimension-level wins.

### Context 2: Within-concept tournament (after MCTS)

When all 4 MCTS trees have completed for a concept, the 4 preset-winner DAGs compete in a Swiss-style tournament using the same per-criterion pairwise protocol. The tournament ranks them 1st/2nd/3rd/4th. The top-K (per config, see scoping §11) advance to Stage 3.

---

## DAG Rendering: Incident-Encoded Outline

Both contexts render DAGs for judges using the incident-encoded outline format from `lab/references/talk-like-a-graph/summary.md`. The format was chosen because incident encoding beats flat adjacency by 34 accuracy points for relational reasoning tasks (Fatemi et al., ICLR 2024), and application-domain edge labels beat abstract labels by 18 points.

### Format specification

```
STORY STRUCTURE {A|B}
======================

Beats listed in causal order. Each beat shows what it causes or reveals,
with load-bearing details in parentheses.

[<node_id>] <beat_sketch>
  → causes [<target_id>] <target_sketch_snippet>
      (realizes: <realizes_field>)
  → motivates [<target_id>] <target_sketch_snippet>
      (agent: <agent>; goal: <goal>; stakes: <stakes>)
  ...

[<next_node_id>] <beat_sketch>
  ...

STRUCTURAL TENSIONS (long-range edges)
  [<src_id>] → discloses to [<dst_id>]
      (reframes: <reframes_field>; withheld: <withheld_field>)
  [<src_id>] → motivates [<dst_id>]
      (agent: ...; goal: ...)
```

### Rendering rules

1. **Topological order.** Nodes are listed in a topological order derived from the DAG. All edges in our DAG point temporally forward (src earlier than dst); topological listing preserves that. Short-range edges — those connecting adjacent or near-adjacent nodes — are rendered inline under the source node. Long-range edges — those that span distant nodes, most commonly disclosure edges (where the opening establishes what the climax will reframe) and motivates edges that anchor a character arc across many beats — are excerpted to the STRUCTURAL TENSIONS appendix so they don't get buried in the main body's flow.

2. **Node labels.** Each node renders with its ID (short descriptive label like `Gathering` or `Stoning`, not `n1`/`n2`) followed by its full sketch. If the node has a `role`, append it in square brackets: `[Stoning, role=climax]`.

3. **Edge verbs.** Use application-domain verbs, not type codes:
   - `causal` → "causes"
   - `disclosure` → "discloses to"
   - `implication` → "entails"
   - `constraint` → "prevents"
   - `motivates` → "motivates"

4. **Edge payload formatting.** Payload fields appear as parenthetical annotations on the line below the edge header. Format: `(field1: value1; field2: value2)`. Keep to one line per edge when possible; multi-line for very long payloads.

5. **Non-sequential edges → STRUCTURAL TENSIONS.** Any edge whose source and target are more than one topological position apart, OR any disclosure / motivates edge regardless of distance, is repeated in the appendix. This surfaces long-range structural relationships that would otherwise be buried.

6. **Consistent format across compared DAGs.** When rendering two DAGs for pairwise comparison, use identical formatting rules. Any formatting difference between STRUCTURE A and STRUCTURE B is a confound for position-bias measurement.

### Example rendering

From Jackson's "The Lottery" (5-node fragment):

```
STORY STRUCTURE A
==================

Beats listed in causal order. Each beat shows what it causes or reveals,
with load-bearing details in parentheses.

[Gathering] Warm summer morning. Villagers gather in the town square; children
stack stones at the edge.
  → causes [Summers] Mr. Summers arrives with the battered black box
      (realizes: the lottery is a standing annual institution with a designated
       official and a ritualized object)
  → motivates [Tessie] The Hutchinsons redraw
      (agent: villagers; goal: uphold communal tradition without exception;
       stakes: social cohesion vs individual survival)

[Summers] Mr. Summers arrives with the battered black box; heads of household
are called by name.
  → causes [Hutchinson] Bill Hutchinson draws the slip with the black dot
      (realizes: procedural authority — officiousness and cheerfulness —
       legitimizes random lethal selection)

[Hutchinson] Bill Hutchinson draws the slip with the black dot; the family
is selected for the next round.
  → causes [Tessie] The Hutchinsons redraw; Tessie gets the black dot
      (realizes: the family is the first selection stage; individual
       accountability follows group selection)

[Tessie] The Hutchinsons redraw. Tessie protests; she draws her own slip and
finds the black dot.
  → causes [Stoning, role=climax] Villagers stone Tessie
      (realizes: individual selection activates undivided communal execution;
       Tessie's resistance changes nothing)

[Stoning, role=climax] Villagers close in. Stones are picked up. Tessie
screams that it isn't fair. They stone her.

STRUCTURAL TENSIONS (long-range edges)
  [Gathering] → discloses to [Stoning]
      (reframes: cheerful gathering → prelude to stoning;
       withheld: the lottery's outcome is death)
  [Summers] → discloses to [Hutchinson]
      (reframes: innocent-seeming box → weapon-selection lottery;
       withheld: the purpose of the ritual)
```

---

## Judge Prompt Structure

Each per-criterion pairwise comparison is one LLM call to one judge. The prompt has five sections:

1. **Judge persona** — from `configs/judges/{judge_id}.yaml`, same as Stage 1.
2. **Base Stage 2 system message** — the typed-edge taxonomy and genome schema (shared with operators; see `operators.md` §Base System Message).
3. **Dimension rubric** — the anchors and sub-criteria for the dimension being judged (from `rubric-anchors.md`).
4. **The two DAGs** — both rendered in incident-encoded outline format, labeled STRUCTURE A and STRUCTURE B.
5. **Task** — "Comparing these two structures on {DIMENSION}, which is stronger?"

### Output schema

```json
{
  "reasoning": "2-4 sentences explaining the comparison on this dimension, citing specific beats and edges",
  "winner": "A" | "B" | "tie"
}
```

**Tie is explicitly permitted.** Research (`ties-matter/`, `statistical-framework-llm-ranking/`) establishes that forcing a winner when two candidates are genuinely equivalent generates noise. A judge who can't discriminate on a dimension returns tie with reasoning.

### Dual ordering

Each pairwise comparison is run twice: once with DAG X as A and DAG Y as B, once with the orderings swapped. If the judge's pick is consistent across orderings, the judgment counts. If the judge flips its pick (says "A wins" both times — meaning it always prefers position A regardless of content), the judgment is recorded as **tie** (the vote is discarded). This is Stage 1's standard dual-ordering mitigation, reused unchanged.

### Aggregation across judges

Three judges each produce a per-dimension judgment (winner or tie) with dual-ordering collapse. The dimension-winner is decided by majority of non-tie votes:

- 3 non-tie votes agreeing: winner is that DAG
- 2 non-tie votes for the same DAG, 1 tie: winner is that DAG
- 2 non-tie votes split, 1 tie: tie
- Any other configuration with ties ≥ 2: tie
- All 3 votes tie: tie

### Overall pairwise winner across dimensions

Across all 8 dimensions: count dimension-wins and dimension-losses for each DAG, plus ties.

- DAG A wins overall if `dim_wins(A) > dim_wins(B)`.
- DAG B wins overall if `dim_wins(B) > dim_wins(A)`.
- Tied overall if `dim_wins(A) == dim_wins(B)`.
- The numeric pairwise score for the challenger side is `(dim_wins + 0.5 × ties) / 8`.

---

## Number of LLM Calls per Pairwise Comparison

- 3 judges × 2 orderings = **6 LLM calls per pairwise comparison**

Each call is a judge-model call. The judge evaluates all 8 dimensions at once within the call, returning a structured response with per-dimension votes (`a`, `b`, or `tie`) and reasoning. The 8 dimension votes are not separate LLM calls; they are fields of one structured response. This follows Stage 1's implementation in `owtn/evaluation/pairwise.py` (`_judge_one_ordering` makes one call per judge per ordering, with all dimensions evaluated in the same `PairwiseJudgment` output).

For rollout reward signal, the tiered design runs just the cheap judge × 2 orderings in parallel (2 LLM calls, no added latency) instead of the full 6-call panel. Each call returns the same per-dimension structured response; per-dimension votes are collapsed across the two orderings (dual-ordering flip rule); aggregation across multiple judges is skipped.

At ~$0.005–$0.01 per call depending on judge model, each full pairwise comparison costs ~**$0.03–$0.06**.

### Frequency during MCTS

Cheap-judge comparison fires on every rollout (~200 per tree across forward + backward + Phase 3). Full-panel verification fires only when the cheap judge declares a challenger wins — roughly 10-20 promotion gates per phase. See `mcts.md` §Reward Function for the tiered design; see §Budget Management there for preliminary cost notes. Actual per-concept costs are being measured in the pilot harness.

### Frequency in within-concept tournament

4 presets → C(4,2) = 6 pairwise comparisons in round-robin format. At ~$0.05 each: ~$0.30 per concept in tournament cost.

---

## Within-Concept Tournament

After all 4 pacing-preset MCTS trees complete, their preset-winner DAGs compete.

### Tournament format

**Round-robin for 3–4 entries.** Swiss tournaments are designed for larger pools where round-robin would be expensive. With only 4 entries, round-robin is only 6 comparisons and produces a full ranking without ambiguity.

### Ranking mechanism

For each pair of preset winners, run the full per-criterion pairwise comparison. Record win/loss/tie at the DAG level.

Ranking tiebreakers:
1. Most DAG-level wins.
2. Most dimension-level wins across all comparisons.
3. Higher mean judge-reasoning-length as a proxy for judge engagement (weak tiebreaker).

### Output

A `tournament.json` per concept:

```json
{
  "concept_id": "c_a8f12e",
  "presets": ["cassandra_ish", "phoebe_ish", "randy_ish", "winston_ish"],
  "matches": [
    {
      "a": "cassandra_ish",
      "b": "phoebe_ish",
      "winner": "cassandra_ish",
      "dimension_wins": {"causal_soundness": "a", "motivational_coherence": "tie", ...},
      "score": "6-2-1"
    },
    ...
  ],
  "final_ranking": [
    {"preset": "cassandra_ish", "rank": 1, "wins": 3, "losses": 0, "ties": 0},
    {"preset": "winston_ish", "rank": 2, "wins": 2, "losses": 1, "ties": 0},
    ...
  ]
}
```

---

## Top-K Advancement

The top-K preset winners (per config) advance to Stage 3:

- `light.yaml`: K=1
- `medium.yaml`: K=2
- `heavy.yaml`: K=all

Non-advancing DAGs are archived in the QD grid (see `qd-archive.md`) and sent to compost with metadata about why they ranked lower.

### Edge case: near-tie at the K boundary

If the K-th and (K+1)-th ranked DAGs are within 1 dimension-win of each other, both advance. This prevents a narrow Swiss-tournament loss from discarding a valuable structural alternative. Document as `near_tie_promoted: true` in the handoff manifest.

### Edge case: all presets tie

Rare but possible. In this case, the tiebreakers from §Ranking mechanism above are used. If all tiebreakers also fail, advance the Cassandra-ish preset by default (most commonly-expected narrative shape).

---

## What Judges Read: Stage-2-Specific Adaptations

Stage 1 judges read concept JSON. Stage 2 judges read incident-encoded DAG outlines. The persona files at `configs/judges/*.yaml` stay the same (same personalities, same aesthetic commitments), but the prompt scaffolding changes:

- The `{CONTEXT}` section of the judge prompt includes the Stage 1 concept (premise, target effect, character seeds, etc.) so the judge can evaluate structure relative to concept intent.
- The `{CRITERION}` section names the Stage 2 dimension being judged (not the Stage 1 dimension, since they differ).
- The `{CANDIDATES}` section shows the two DAG renderings.

See `judges.md` for the full adaptation discussion and the open question about judge specialization.

---

## Scoring Protocol Reuse From Stage 1

Stage 2 reuses the Stage 1 pairwise infrastructure directly:

- `owtn/evaluation/pairwise.py` — dual-ordering, per-dimension voting, majority aggregation.
- `owtn/stage_1/tournament.py` — tournament mechanics (we'll use its round-robin mode for the within-concept tournament).
- `owtn/evaluation/prompts.py` — judge prompt assembly, adapted for Stage 2's rubric and base system message.

The only Stage-2-specific additions are:
- The incident-encoded DAG renderer
- The Stage 2 rubric anchors
- The Stage 2 base system message
- The 8 Stage 2 dimensions (different from Stage 1's 9)

---

## Validation Before Evaluation

Not every terminal DAG gets judged. A **three-tier consistency layer** runs before the expensive pairwise judge, catching structurally-broken and internally-contradictory DAGs at near-zero cost. Judge-only evaluation does not hold at 3–18 node DAGs: LLM judges fail to detect plot holes at better-than-chance (FlawedFictions, Ahuja/Sclar/Tsvetkov, COLM 2025), prioritize style over factuality (Feuer et al., ICLR 2025), and disagree ~25% of the time on difficult structural cases (Sage benchmark, 2025). Internal consistency is an **absolute** property — pairwise comparison cannot catch it because a genome with causal violations can out-read a conceptually weaker but coherent one. The consistency layer is the complement that closes this gap.

Research grounding: `lab/deep-research/runs/20260424_031220_narrative-dag-consistency-checking/final_report.md`. Most directly applicable published systems: FACTTRACK (Lyu/Yang/Kong/Klein, NAACL 2025) for time-aware atomic-fact contradiction detection on outlines; PLOTTER (Gu et al., arXiv:2604.21253, 2026) for DAG validity + connectivity constraints; ConStory-Checker (Li et al., arXiv:2603.05890, 2026) for the error-taxonomy basis.

### Tier 0: Deterministic structural checks (0 LLM calls, microseconds)

Hard constraints. Any violation rejects the DAG outright; the MCTS rollout that produced it backpropagates reward 0 and no judge fires. Reuses and extends `overview.md` §Validation Protocol:

- **DAG validity**: topological sort must succeed; reject if cycle detected (catches temporal paradoxes).
- **Connectivity**: every beat reachable from the opening beat (in-degree-0 node) through DAG edges; no orphans.
- **Node count bounds**: 3 ≤ nodes ≤ 18.
- **Required payload field presence**: each edge's typed payload fields non-empty and substantive (>3 tokens) — catches decorative empty fields that the prose stage would ignore.
- **Role cardinality**: exactly one node has a non-null `role` matching `concept.anchor_scene.role`; `reveal` and `pivot` anchors must have ≥1 incoming edge (cannot be orphan opening).

### Tier 1: Entity state consistency (~1 LLM call per genome, ~$0.001–$0.002)

One structured call reads the full genome (beats + edges + `character_arcs` + `story_constraints` in topological order) and extracts per-beat entity state changes, adapted from FACTTRACK (NAACL 2025):

1. Identify all entities (characters, objects, locations) mentioned anywhere in the DAG.
2. For each beat in topological order: extract the state delta — what entities are introduced, what state changes are established, what knowledge is revealed or concealed.
3. Check each beat against the accumulated state: does any beat assume a state not yet established, or treat an entity as unknown after introduction?
4. **Character-arc consistency**: each `character_arcs[].agent` must resolve to a single character across (a) beat sketches where the character appears, (b) all `motivates[].agent` references to it, (c) other arcs' references to it. An arc agent who never appears in any beat sketch — or a `motivates` edge whose agent name conflicts with the same-named arc — flags here.
5. **Story-constraint consistency**: for each `story_constraints[]` entry, check every beat with topological index less than `lifts_at` (or every beat if `lifts_at` is null) — no beat may enact the prohibited thing before the rule lifts. A Hemingway beat where a character directly says "abortion" would flag against the not-naming constraint.
6. **Per-node motif consistency**: every `Node.motifs[].motif` must be a verbatim string match against the genome's `motif_threads` list. Typos or drift (`"the stones"` vs `"stones"`) flag here. Each `mode` must be one of the six values defined in the **mode glossary** (`docs/stage-2/overview.md` §Per-node motifs §Mode glossary): `introduced | embodied | performed | agent | echoed | inverted`. **Return-mode temporal sanity**: a motif tagged `echoed` or `inverted` at node N must have at least one topologically-earlier node where it's tagged with a non-return mode (`introduced`, `embodied`, `performed`, or `agent`). Returning without any prior appearance is a temporal inconsistency. Same verbatim-match rule for `character_arcs[].agent` matching against `motivates[].agent`.
7. **Disclosure audience consistency**: for each disclosure edge, every non-`"reader"` entry in `disclosed_to` must name a character who appears in the target beat's sketch (the audience character must be present at the moment of realization). An audience named for a disclosure but absent from the target beat flags here.

Reject or penalize DAGs where state, arc, constraint, or motif inconsistencies are flagged. Calibrated recall: ~55% of character/factual consistency errors against injected benchmarks (extrapolated from ConStory-Checker's Character Consistency F1=0.742). Expected false-positive rate: 10–15% at outline scale.

### Tier 2: Edge payload plausibility (~1 batched LLM call, ~$0.002–$0.005)

One batched call verifies every edge's payload is semantically compatible with the source beat and references only entities/events established in or before the source beat. Per-edge-type specifics (one prompt per edge type, batched):

| Edge type | Plausibility check |
|---|---|
| `causal` (`realizes`) | Entity/event named in `realizes` appears in or is clearly inferable from source beat; no forward references to downstream beats. |
| `disclosure` (`reframes` + `withheld`) | `withheld` is plantable in source beat (not overtly stated in any prior beat); target beat topologically follows source; reframing is substantive, not gestural. |
| `implication` (`entails`) | `entails` names a specific proposition that is an **in-world logical consequence** of the source beat — a character could reason from A to B. Reject thematic-rhyme misuse: if the `entails` reads as the story *arguing* A and B resonate (authorial claim), not as state B logically following from state A, flag as miscoded. The check prompt is explicit: "Is this entailment something a character in the story could reason out, or is it a thematic parallel the author is drawing? Only the former is an implication edge." |
| `constraint` (`prohibits`) | Prohibition names a concrete capacity; downstream beats must not violate (string/entity match for common cases). |
| `motivates` (`agent` + `goal` + optional `stakes`) | `agent` appears in source beat or prior beat; `goal` is concrete (not target-effect-level); target beat shows agent acting toward goal. |

**Disclosure and motivates edges are the most dangerous for judge-only evaluation.** CFPG (arXiv:2601.07033, 2026) documents that LLMs "frequently fail to bridge long-range narrative dependencies, leaving 'Chekhov's guns' unfired." CoG 2025 benchmark shows ~53% failure rate on intentionality planning (motivates edges) for frontier models. Tier 2 checks are most critical for these two edge types.

### Tier 3: Concept-demand fidelity (~1 LLM call per preset terminal, ~$0.005)

Tiers 1–2 are about internal coherence — does the genome contradict itself, does each edge's payload hold up. Tier 3 is about *concept fidelity* — does the genome realize the concept's non-negotiable structural mechanism. This is the silent-failure mode the cheap-judge pairwise reward cannot detect: two DAGs that *both* miss the concept's central structural move (reader-address, form-as-device, deliberate irresolution, dialetheic structure) will compare cleanly on the 8 dimensions while being collectively inadequate.

**Inputs.** The DAG's `concept_demands` list (set once at seed time by `seed_root` — see `operators.md` §seed_root and `overview.md` §Concept demands). Demands are derived from the concept by an LLM call alongside motif extraction; they are *not* a Stage 1 handoff field. If `concept_demands` is empty (the common case — most concepts' mechanisms are fully schema-expressible), Tier 3 is skipped without a warning. If the seed-time extraction call failed (logged), Tier 3 is skipped with a warning.

**When it fires.** Once per preset's *final* terminal DAG (Phase 3 winner, before within-concept tournament). Not per-rollout — too expensive and not load-bearing on UCB. ~1 LLM call × 4 presets per concept = 4 calls total. Negligible cost.

**The check.** One structured LLM call passes:
- The full final DAG (incident-encoded outline + `character_arcs` + `story_constraints`)
- The concept (premise, target_effect, thematic_engine, anchor_scene, constraints)
- The `concept_demands` list

The LLM evaluates each demand independently as `satisfied` / `partial` / `failed` with a one-sentence rationale. The classifier model is the same third-family model used for the champion-brief summarizer (cross-family discipline; not the cheap judge or expansion model).

**How results are used.**
- **Tournament priority.** Within-concept tournament ranks DAGs first by concept-demand satisfaction, then by pairwise dimension-wins. A DAG satisfying all demands ranks above any DAG with at least one `failed` demand, regardless of how many pairwise dimensions it wins. Within tier (all-satisfied vs. all-with-same-failure-pattern), existing pairwise tournament decides.
- **Handoff manifest.** `Stage2Output.concept_demand_results: list[{demand, verdict, rationale}]` is included in the handoff manifest. Stage 3 sees the per-demand verdicts and can prioritize voice choices that compensate for `partial` satisfactions.
- **Run metric.** `concept_demand_failure_rate` per run is exported. A high failure rate across runs flags either (a) Stage 1 is generating concepts whose demands the operators can't realize, or (b) the demands themselves are over-specified.

**Why not gate MCTS exploration with this signal.** Firing on every rollout (~200 per tree) would cost ~$1/tree × 4 = $4/concept just for Tier 3, on top of cheap-judge cost. The signal-to-cost ratio doesn't justify it: most rollout-terminal DAGs are intermediate explorations, not final candidates, and a single LLM-judged "demand satisfied" rating per intermediate DAG is high-variance noise that UCB doesn't need.

### Total cost

Tier 0 (deterministic): 0 LLM calls per genome. Tier 1: ~$0.002 per genome × ~50–200 genomes per run = $0.10–$0.40. Tier 2: ~$0.003 per genome × same = $0.15–$0.60. Tier 3: ~$0.005 × 4 presets per concept × N concepts. At 5 concepts: $0.10. **Total consistency layer: ~$0.35–$1.10 per medium.yaml run.** Negligible relative to full-panel pairwise cost (~$3.30/concept in the example walkthrough).

### Calibration before trust

Tier 1's ~55% recall is extrapolated from ConStory-Checker's F1=0.742 on a different dataset; Tier 2 has no recall numbers. Before letting either tier reject DAGs, run the calibration mini-pilot in `implementation.md` §Tier 1/2 consistency-check calibration — inject 30–50 defects into the canonical DAGs, measure precision and recall per defect class. If precision < 80%, the tier rejects too many good DAGs; if recall < 50% on a defect class, the tier isn't catching that class and should degrade to "flag into judge context" for that class rather than reject.

### Action on tier failures

| Failure | Response |
|---|---|
| Tier 0 (deterministic) | Reject; reward 0 backpropagated. |
| Tier 1 (high-confidence entity violation) | Reject OR apply 0.5× multiplier to MCTS reward (configurable — calibrate in pilot). |
| Tier 2 (high-confidence payload violation) | Reject OR apply 0.5× multiplier. |
| Tier 1/2 (low-confidence flag) | Pass to pairwise judge; include the flag in the judge-prompt context so the rubric's Structural Coherence dimension can weigh it. |
| Tier 3 (concept demand `failed`) | DAG advances within preset (Tier 3 fires post-MCTS), but ranks below all-satisfied DAGs in within-concept tournament regardless of pairwise wins. |
| Tier 3 (concept demand `partial`) | Logged in handoff manifest; Stage 3 sees the partial verdict; tournament tiebreaker between all-partial-but-different-demands DAGs falls back to pairwise. |

The consistency layer is a **pre-filter and signal augmenter**, not a replacement for pairwise judging. Holistic quality dimensions (Indelibility, Grip, Novelty, Emotional Depth, Thematic Resonance, Generative Fertility, Scope Calibration) remain judge-evaluated — no automated check can replicate those. What the consistency layer does is ensure judges compare candidates that are at minimum internally coherent.

### Symbolic formalism deliberately rejected

Full PDDL / event calculus / classical storylet preconditions do not scale to LLM-generated natural-language content (GPT-4o PDDL prediction accuracy 34–45% at best). The Tier 2 approach adopted here follows Drama Llama (Sun et al., 2025): NL precondition evaluation via a cheap LLM call, not formal solver.

### What stays the same

Judge-budget protection in v1 still relies on the tiered-judge design (cheap judge at rollout, full panel at commitment events). The consistency layer is additive — cheap-judge continues to fire on terminals that pass Tier 0–2, and full-panel continues to fire on promotion gates / tournament / archive insertions as before.

---

## Open Questions Surfaced in Evaluation Drafting

1. **Rendering of `role` flags.** The current format appends `[role=climax]` to the node label. This may visually dominate or get ignored depending on judge attention. Consider: whether to put role on a separate line, or integrate it into the sketch rendering. Resolve empirically.

2. **Near-tie definition for Top-K advancement.** "Within 1 dimension-win" is a first-guess threshold. Could be tighter (0 dimension wins but more ties) or looser (within 2). Calibrate with real tournament data.

3. **Contest judge personas for Stage 2 dimensions.** The panel (Gwern, Roon, Alexander Wales, Jamie Wahls) is tuned for contest-submission evaluation — finished prose, not structural sketches. Do their aesthetic priorities translate to structural judgment? Wales and Wahls translate cleanly (both are explicitly structural thinkers); Gwern's slop-detector mostly surfaces as "view from nowhere" at structure granularity; Roon's phrase-level aesthetic is the least direct translation. Per-judge pilot checks are in `judges.md` §Persona-Adaptation Concerns Specific to Stage 2. Monitor dimensions where one judge systematically abstains — flags persona/dimension mismatch.

4. **(Resolved.)** Gate 2 / Gate 3 both dropped in v1 since forecasting is no longer computed.

5. **Round-robin vs Swiss for 3 entries.** When seeding fails for one preset, we run 3 MCTS trees, not 4. Tournament has C(3,2) = 3 comparisons in round-robin. Swiss with 3 is awkward. Just use round-robin for 3-entry cases.

6. **(Resolved — 2026-04-20.)** Judge reasoning IS preserved as feedback into subsequent expansion prompts via a Stage-1-style lazy summarizer adapted for Stage 2 (`lab/issues/2026-04-20-stage-2-expansion-feedback-summarizer.md`). Key shifts from the Stage 1 pattern:
   - **Subject is the tree**, not a specific champion — champion churn (1-5 iterations before replacement) is too fast to accumulate history per-champion; the tree accumulates continuously across champion changes.
   - **Corpus is full-panel critiques only** (promotion gates + tournament, ~10-30 per tree), not cheap-judge rollout critiques. Matches Stage 1's ~5-15 match corpus size; signal quality matches.
   - **Rendered output is a structured `ChampionBrief`** (4 fields paralleling Stage 1's `ParentBrief`: established structural weaknesses, contested structural choices, structural attractor signature, structural divergence directions) positioned in the expansion prompt after the DAG rendering and before the action request.
   - Reuses Stage 1's lazy-cache + third-family classifier + raw-fallback pattern from `owtn/evaluation/feedback.py`.
