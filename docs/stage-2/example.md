# Stage 2: Example Run (Walkthrough)

This document walks through a single Stage 2 run for one concept, showing what prompts fire, what metrics accumulate, and how state evolves. It's a **thinking-through-the-process** artifact — the goal is to make the pipeline's coherence inspectable at prompt/metric granularity, not to specify exact prompt text. Placeholders mark content too long to inline.

Example concept is fabricated; dimensions and edge types are per the canonical rubric.

---

## 0. Input from Stage 1

Stage 2's runner reads `results/run_<stage_1_ts>/stage_1/champions/island_3.json` and parses out:

```json
{
  "program_id": "p_a8f12e",
  "genome": {
    "premise": "An accountant with perfect autobiographical memory is asked by her dying mother to be her will's executor.",
    "anchor_scene": {
      "sketch": "Ines sits by her mother's bed and is asked, for the first time, to tell a story from a year her mother cannot remember — and realizes that what her mother needs is for Ines to invent the memory, not recall it.",
      "role": "pivot"
    },
    "target_effect": "The weight of unforgettable care — love as the thing that cannot be put down even when it becomes unbearable.",
    "character_seeds": [
      {"name": "Ines", "role": "protagonist", "sketch": "Ines, mid-40s, perfect recall of every moment; her interiority is over-indexed."},
      {"name": "mother", "role": "antagonist-by-fact-of-dying", "sketch": "Her mother, 74, late-stage cancer, afraid of being forgotten but more afraid of not being forgotten."}
    ],
    "thematic_tension": "Perfect memory as burden and as refusal; caring-for as a skill trained by forgetting.",
    "constraints": [
      "No flashback scenes; memory is always present-tense in the head.",
      "Single location (a hospice room) for primary action."
    ],
    "style_hint": "Close third on Ines; Chiang-mode thought-experiment with literary restraint"
  },
  "combined_score": 0.68,
  "tournament_rank": 2,
  "tournament_dimension_wins": [
    {"match_id": "...", "opponent": "p_c19a8f", "wins": {"novelty": "a", "grip": "tie", "tension": "a", ...}, "outcome": "won"},
    ...
  ],
  "affective_register": "GRIEF_OF_CARE",
  "literary_mode": "CHIANG_THOUGHT_EXPERIMENT",
  "patch_type": "collision"
}
```

The Stage 2 runner wraps this as a `Stage1Winner` and spawns one concept task for it.

---

## 1. Seed Phase (preset-agnostic, one shared anchor)

Anchor is consumed from `concept.anchor_scene` — already evolved and tournament-selected at Stage 1. The only LLM work here is motif extraction.

### 1a. Anchor wrap (no LLM call)

```python
anchor_node = Node(
    id="anchor",
    sketch=concept.anchor_scene.sketch,
    role=concept.anchor_scene.role,  # "pivot" for this concept
)
```

The `sketch` and `role` are already validated (Stage 1 Gate 1: min_length 40, placeholder regex, visualizability classifier, role enum). Stage 2 does not re-validate.

**Cost:** 0.

### 1b. Motif extraction

**Prompt:** `owtn/prompts/stage_2/seed_motif.txt`, filled with concept + anchor sketch + role.

```
{BASE_SYSTEM with {CONCEPT_CONTEXT} = concept_json}

TASK: Identify 2-3 motif threads for this concept — concrete recurring
elements (imagery, objects, phrases, character obsessions) that the DAG's
beats will thread through.

CONCEPT:
{concept_json}

ANCHOR SCENE:
Ines sits by her mother's bed and is asked, for the first time, to tell
a story from a year her mother cannot remember — and realizes that what
her mother needs is for Ines to invent the memory, not recall it.
(role: pivot)

Motifs are physical and specific, not abstract. "Stones" and "the black
box" are motifs; "mortality" and "tradition" are themes.

Output as JSON:
{"motif_threads": ["...", "...", "..."]}
```

**Output:**

```json
{
  "motif_threads": [
    "the plastic water cup with a straw, being refilled",
    "the mother's hand turning a ring she no longer fills",
    "the specific quality of late-afternoon hospice light"
  ]
}
```

Motif threads pass concrete-not-abstract check (all three name physical objects/images).

**Cost:** 1 call at `expansion_model` (~$0.005).

### 1c. Seed state committed

```python
shared_seed = DAG(
    concept_id="p_a8f12e",
    preset=None,  # preset-agnostic
    nodes=[anchor_node],  # role="pivot"
    edges=[],
    target_node_count=8,  # from scoping §7: 3K word concept → 5-8 nodes
)
motif_threads = [...]
```

**Seed cost total:** ~$0.005 per concept. (Earlier design had tension extraction + anchor generation + classifier validation at ~$0.03/concept; the anchor's signal now rides on Stage 1's tournament rather than on a pointwise classifier call here.)

---

## 2. MCTS Phase 1 — Forward (Cassandra-ish tree)

All 4 preset trees share `shared_seed` as root. Each runs 50 iterations concurrently; for compactness we follow just the Cassandra-ish tree.

### 2.1 Iteration 1 — detailed walkthrough

**State at start of iteration:**
- Tree has one node: root, wrapping the 1-node seed DAG
- `tree.full_panel_critiques = []`
- `tree.champion = None`
- `tree.champion_brief = None`
- `tree.iteration = 0`

**Step 1: Selection.** Root is a leaf. Since root has no cached candidates, expand immediately.

**Step 2: Expansion LLM call.**

Prompt (`expansion.txt` filled):

```
{BASE_SYSTEM}
{CONCEPT_CONTEXT: full concept_json dump}
{MOTIF_THREADS: the three motifs}
{PHASE_CONTEXT: "forward phase, anchor → resolution direction,
  permitted edge types: causal, implication"}
{PACING_HINT: Cassandra-ish — "Favor actions that establish
  rise-then-relief patterns..."}

CURRENT DAG:
[anchor, role=pivot] Ines sits by her mother's bed and is asked,
for the first time, to tell a story from a year her mother cannot
remember — and realizes that what her mother needs is for Ines to
invent the memory, not recall it.

STRUCTURAL TENSIONS:
(none yet)

WHAT THE JUDGES HAVE BEEN LOOKING AT IN THIS TREE:
(no feedback yet — tree has not accumulated full-panel critiques)

TASK: Propose up to 4 candidate actions that would extend this DAG
toward a complete structure for this concept. Each action is one of:
- add_beat ...
- add_edge ...
- rewrite_beat ...

Permitted edge types in this phase: causal, implication

<output schema>

Before outputting, list 3 structural gaps you see in the current DAG.
Then produce actions that address them.
```

**Expansion LLM output:**

```json
{
  "gaps": [
    "No downstream beats after the pivot — the reconciliation (ketsu) is absent.",
    "The mother's response to Ines's decision is not shown.",
    "Ines's internal transformation isn't anchored to a concrete post-pivot action."
  ],
  "actions": [
    {
      "action": "add_beat",
      "target": {
        "parent_id": "anchor",
        "edge_type": "causal",
        "beat_text": "Ines begins telling a story she is making up: a summer trip to the coast, a dog named for her mother's father. Her mother closes her eyes and breathes more easily.",
        "edge_payload": {"realizes": "Ines's decision to invent activates her memory's generative capacity — the skill she has never practiced is now load-bearing."}
      },
      "reasoning": "First beat after the pivot — concretizes the decision by showing her actually do it."
    },
    {
      "action": "add_beat",
      "target": {
        "parent_id": "anchor",
        "edge_type": "implication",
        "beat_text": "The mother asks Ines, later, to tell her about the day she was born.",
        "edge_payload": {"entails": "If Ines's invention satisfies real need, the mother will ask for more — and the need will now include stories Ines was present for but whose surface the mother cannot reach."}
      },
      "reasoning": "Second beat establishes the engine of the ketsu: invention becomes the default mode of care."
    },
    {
      "action": "add_beat",
      "target": {
        "parent_id": "anchor",
        "edge_type": "causal",
        "beat_text": "Ines calls her brother and does not tell him what she is doing.",
        "edge_payload": {"realizes": "Ines's new practice is private — it cannot survive being named to a witness who would call it fabrication."}
      },
      "reasoning": "Establishes the isolation of the new practice; introduces the brother as an off-stage reality-principle."
    },
    {
      "action": "add_beat",
      "target": {
        "parent_id": "anchor",
        "edge_type": "causal",
        "beat_text": "Her mother dies during a story Ines is telling about a birthday that never happened.",
        "edge_payload": {"realizes": "The invented memory becomes the last thing between them — the care does not outlive its fabrication."}
      },
      "reasoning": "Candidate resolution beat; closes the arc with the ending as continuation of the invention."
    }
  ]
}
```

**Step 3: Validation.** All four pass schema + phase + cycle checks. Cached on root as `cached_candidate_actions = [a1, a2, a3, a4]`.

**Step 4: Instantiate first candidate (a1).** A new MCTS child is created, wrapping the 2-node DAG `[anchor, a1_beat]`. Child is now the current leaf.

**Step 5: Simulation.** `SIMULATE(child, d_max=8)`:
- Call rollout LLM (cheap-model) once with DAG + "propose one-step forward extension": cheap-judge picks between the extension's reward-against-current-champion.
- No champion yet (first rollout), so the reward is 0.5 by convention.
- `s_max=3` extensions try, but since there's no champion baseline the rollout's value backpropagates as 0.5.

**Step 6: Install first champion.** This terminal becomes `tree.champion = child.dag`.

**Step 7: Backprop.** visits = 1, cumulative_return = 0.5, up to root.

**Iteration 1 cost:**
- Expansion: 1 × ~$0.01 ≈ $0.01 (deepseek-v4-pro, ~5k input + 1k output)
- Rollout extension: ~3 × ~$0.0004 ≈ $0.001 (deepseek-v4-flash, ~2k input + 500 output)
- Cheap judge: first rollout, no comparison, $0
- Total: ~$0.011

---

### 2.2 Iterations 2–14 (condensed)

UCB picks cached candidates a2, a3, a4 at root (K=4 exhausted by iter 5), then descends into the children and expands them.

State progression (champion updates shown):

| Iter | Event | `tree.champion.dag.nodes` | Full-panel events | Notes |
|---|---|---|---|---|
| 1 | First rollout, no comparison | [anchor, a1_beat] | 0 | Auto-install |
| 2 | a2 rollout vs champion: cheap judge 6 dim-wins → challenger wins | [anchor, a2_beat] | 1 (accepted) | Full panel verifies, champion swaps |
| 3 | a3 vs new champion: cheap judge 3-3-2 → tie | (unchanged) | 0 | Backprop 0.5 |
| 4 | a4 vs champion: cheap judge 2-5-1 → champion wins | (unchanged) | 0 | Backprop ~0.31 |
| 5 | Descend into a2 child, expand; rollout → cheap judge close, 4-4 → tie | (unchanged) | 0 | Backprop 0.5 |
| 6 | ... | ... | ... | ... |
| 14 | Rollout scores cheap-judge 5-1-2 win | [5-node DAG] | 2 (1 accepted, 1 rejected) | — |

At iter 14, `tree.full_panel_critiques` has 2 entries; brief cache key = 2, under threshold (N=3). No summarizer fires yet.

---

### 2.3 Iteration 15 — First champion-promotion rejection (full-panel event)

**Setup:** descent into a recent subtree yields a new terminal, `candidate_dag` (6 nodes). Cheap judge runs:

**Cheap judge prompt** (same for both orderings):

```
<insert cheap_judge_system prompt with 8 dimensions>

CONCEPT:
{concept_json}

STRUCTURE A:
{candidate_dag rendered incident-encoded}

STRUCTURE B:
{current champion rendered incident-encoded}

<insert per-dimension comparison request + output schema>
```

**Orderings run in parallel.** Outputs collapsed:

```
Dimension votes (collapsed): edge_logic: A, motivational_coherence: A,
tension_information_arch: A, post_dictability: A, arc_integrity_ending: tie,
structural_coherence: A, beat_quality: A, concept_fidelity_thematic: A
```

Overall: 7 A-wins, 0 losses, 1 tie → cheap judge declares challenger wins.

**Full panel fires** (4 judges × 2 orderings = 8 calls). Each uses:

```
{JUDGE_PERSONA from configs/judges/<judge_id>.yaml}
{STAGE_2_BASE_SYSTEM}
{DIMENSION_RUBRIC for all 8 dimensions}
<concept + STRUCTURE A + STRUCTURE B, same rendering>
<per-dimension task + output schema>
```

**Full panel outputs** (abbreviated):

- Gwern (collapsed across orderings): 3-3-2 — cites A's opening beat as "view from nowhere" on the first read but partially recants on the flip; flags B's motivates edge goal ("to care for her mother") as target-effect-level schmaltz dressed as motivation. Neither DAG fully earns the anchor.
- Roon (collapsed): 2-3-3 in B's favor. "A's disclosure-heavy middle substitutes edges for operating-system beats — the payload fields read as decoration. B's compression is better: fewer beats doing more work per beat."
- Wales (collapsed): 5-2-1 in A's favor. Cites A's causal chain as more specifically mechanical ("the realizes field on beat 3 names an actual state change; B's generic 'leads to the moment of choice' doesn't"); credits B's structural-mirroring ambition but notes execution trails it.
- Wahls (collapsed): 2-2-4. Both DAGs "know their ending" in Wahls-terms; tension arc is comparable. Heavy tie rate reflects that at structural granularity without voice, his signal sources are muted.

Aggregated per-dimension across 4 judges: A wins 3, B wins 2, ties 3. The ties here are 2-2 splits where both sides' mean magnitudes are close (gap < 0.25); if A had voted 2-decisive + 2-narrow against B's 2-clear + 2-clear, the magnitude tiebreaker in `_aggregate` would flip the dim to A (see `lab/issues/2026-04-24-aggregate-magnitude-tiebreaker.md` and `judges.md` §What 4 Judges Changes Mechanically).

Overall: cheap judge said A wins decisively; full panel says A wins narrowly with an unusual tie count (3-2-3). The weighted-aggregate tiebreaker in `_select_winner` is the arbiter at this margin. On the indexed dim_weights for this concept (thematically loaded toward Concept Fidelity), A's tiebreaker fails — **the full panel REJECTS the promotion**.

**Effect:**
- `tree.champion` unchanged.
- Backpropagated reward = 0.5 (full-panel rejection backprop per `mcts.md` §Reward Function). NOT the cheap judge's 0.875.
- `tree.full_panel_critiques.append({...})` — critique object stored:

```python
FullPanelCritique(
    iteration=15,
    challenger_dag=candidate_dag,
    champion_dag=current_champion,
    cheap_judge_verdict="A wins (7-0-1)",
    full_panel_verdict="A wins (3-2-3) — rejected via weighted-aggregate tiebreaker",
    cheap_full_agreed=False,
    judge_reasonings={
        "gwern": {"AB_reasoning": "...", "BA_reasoning": "..."},
        "roon": {...},
        "alexander-wales": {...},
        "jamie-wahls": {...},
    },
    per_dimension_panel_votes={...},
)
```

**Drift monitoring:** cheap-full agreement rate now 1/3 (was 1/2 before this event — the iter-14 rejection was the first disagreement). Below the 70% alert threshold; log continues without alerting yet.

---

### 2.4 Iterations 16–29

Two more full-panel events occur (one accepted, one rejected). At iter 22, `tree.full_panel_critiques` has 4 entries. At iter 29, it has 5 entries.

`tree.champion_brief_cache` has never been computed.

---

### 2.5 Iteration 30 — First ChampionBrief summarizer fires

Threshold met: 5 full-panel critiques ≥ N=3, and no cached brief exists. The expansion prompt assembler calls `get_or_compute_brief(tree)`. Cache miss → invoke summarizer.

**Summarizer prompt** (`champion_brief.txt`, adapted from `stage_1/parent_brief.txt`):

```
You are distilling evaluator feedback on the MCTS exploration of story
structures for a specific concept. The tree has run 5 full-panel pairwise
comparisons between candidate structures. Four judges analyzed each
comparison across 8 structural dimensions. Your job is to extract a
structured critique of THIS TREE's exploration — patterns challengers
keep failing at, structural choices judges split on, patterns this tree
keeps producing, and what the next expansion should try differently.

<rules about "established" meaning recurring across comparisons>

<A/B label disambiguation per comparison>

Respond with a single JSON object matching the ChampionBrief schema
with four fields: established_structural_weaknesses,
contested_structural_choices, structural_attractor_signature,
structural_divergence_directions. Each is a list of strings.
```

**Summarizer user message** (rendered from the 5 critiques):

```
## Comparison 1 — accepted (challenger promoted)
<critique rendered with both DAGs, all 4 judges' reasoning>

## Comparison 2 — rejected
<critique rendered ...>

... Comparisons 3, 4, 5 ...
```

**Summarizer output:**

```json
{
  "established_structural_weaknesses": [
    "Disclosure edges in rejected challengers concentrate at late beats with 'withheld' fields that are not planted in earlier beats — Gwern flags this as 'view from nowhere' structurally (3 of 5 comparisons); Wales cites it as a mechanical-payoff failure.",
    "Rejected challengers' motivates edges name 'Ines' with goals phrased at target-effect level ('wants to care for her mother') — Gwern flags as schmaltz-in-costume, Wales flags as underspecified rational-fiction motivation. Roon notes the prose compression wouldn't survive it."
  ],
  "contested_structural_choices": [
    "Whether the brother beat should carry a motivates edge (Wales yes — reality-principle function is a motivational anchor; Wahls no — the beat functions structurally without needing a named intention; Gwern and Roon split). Dimension-level vote on Motivational Coherence sees 2-2 splits on these comparisons; when the mean-magnitude gap is < 0.25 they propagate as ties to the weighted-aggregate tiebreaker."
  ],
  "structural_attractor_signature": [
    "Linear causal chain with two disclosure edges landing at the pivot anchor — this is the LLM default shape under Cassandra hints applied to pivot-anchored concepts; both accepted and rejected structures show it (Wahls: matches the 'sloploaf' attractor he's been diagnosing).",
    "Motifs (water cup, ring, light) appear at opening and at the pivot but not middle beats — bimodal motif distribution. Roon notes this as a compression failure at the structural level."
  ],
  "structural_divergence_directions": [
    "Do not propose another disclosure edge whose `withheld` content is not visibly planted in an earlier beat's sketch.",
    "Successor actions must ground motivates edges in beat-specific intentions, not target-effect-level wants.",
    "Try proposing motif-carrying beats in the middle of the DAG, not only at opening/pivot."
  ]
}
```

**Cost:** 1 summarizer call at `classifier_model` (~$0.015). Cache written:

```python
tree.champion_brief_cache = {
    "critique_count": 5,
    "brief": ChampionBrief(...),
}
```

**Rendered into subsequent expansion prompts:**

```
WHAT THE JUDGES HAVE BEEN LOOKING AT IN THIS TREE:
This tree has been evaluated across 5 full-panel comparisons
(3 accepted promotions, 2 rejections).

## Established structural weaknesses
- Disclosure edges in rejected challengers concentrate at late beats
  with `withheld` fields not planted in earlier beats — judges repeatedly
  flag them as 'unearned'.
- Motivates edges name 'Ines' with target-effect-level goals rather
  than beat-level specifics.

## Contested structural choices
- Whether the brother beat should carry a motivates edge.

## Attractor patterns this tree keeps producing
- Linear causal chain with two disclosure edges landing at the pivot anchor.
- Motifs appear at opening and at the pivot but not middle beats.

## Divergence directions for next expansion
- Do not propose another disclosure edge whose `withheld` content is
  not visibly planted in an earlier beat.
- Ground motivates edges in beat-specific intentions.
- Try motif-carrying beats in middle of DAG, not only at opening/pivot.
```

---

### 2.6 Iterations 31–50 (brief-aware)

Expansion LLM now sees the ChampionBrief in every prompt. Observable changes in proposed actions:

- Disclosure edges: LLM begins proposing edges with `withheld` fields that explicitly reference an earlier beat's sketch content (e.g., `"withheld": "the mother's earlier sentence — 'I don't remember the summer we went north' — is the trigger this pivot retroactively explains"`).
- Motivates edges: `goal` fields shift from "wants to care" to "wants to avoid saying the word 'invent' out loud in case her mother hears it as deceit."
- Motif distribution: proposals begin suggesting water-cup or ring-turning beats in positions 3-4 of the DAG.

Quality of proposals improves measurably: cheap-judge win rate on challengers climbs from 12% (iterations 1-29) to 24% (iterations 31-50). Full-panel rejection rate drops from 40% (2/5) to 25% (1/4 in the final 20 iterations).

**Rechallenge fires at iterations 25 and 50** (per `rechallenge_interval=25`). At each rechallenge, the top 10% of terminals by stored W/N are re-scored by the cheap judge against the CURRENT champion and their W updated in place (visit counts unchanged). This refreshes stale rewards against a consistent baseline, so UCB sees the drop when a previously-high-scoring branch no longer beats the stronger current champion. In this tree: ~5 terminals re-scored per firing, 80 cheap-judge calls total across both rechallenges; observed rechallenge delta mean is -0.03 (see §8 metrics) — mild stale inflation corrected, no pathological collapse.

**Phase 1 ends at iteration 50.** `tree.champion` is a 7-node DAG with:
- 4 causal edges
- 2 implication edges
- 1 disclosure edge (long-range: middle ketsu beat → pivot anchor, `withheld` planted at the middle ketsu beat)

`tree.full_panel_critiques = [9 entries]` at end of Phase 1.
`tree.champion_brief_cache` has been recomputed twice (at 5, 8 critiques).

**Phase 1 cost (Cassandra-ish tree):**
- 50 expansion calls × ~$0.01 (deepseek-v4-pro) = $0.50
- ~150 rollout extensions × ~$0.0004 (deepseek-v4-flash) = $0.06
- 50 cheap-judge calls × 2 orderings × ~$0.002 (gpt-5.4-mini) = $0.20
- 9 full-panel events × 8 calls × ~$0.008 (mixed xAI / Qwen / Z-ai / OpenAI) = $0.58
- 2 summarizer calls × ~$0.015 (gpt-4.1-mini) = $0.03
- **Phase 1 total: ~$1.37 per preset tree**

---

## 3. MCTS Phase 2 — Backward

**Transition:** Phase 1 winner (7-node DAG, pivot anchor + downstream ketsu beats) wraps as new root. MCTS tree is reset (visit counts + cumulative returns discarded); the DAG is preserved.

**Critically, `tree.full_panel_critiques` and `tree.champion_brief_cache` carry over to Phase 2** — the tree-subject summarizer sees Phase 1's exploration history. The expansion LLM entering Phase 2 gets full feedback from the start, including insights about which disclosure-edge placements have worked.

**Phase 2 loop** mirrors Phase 1 structure but with upstream expansion. Permitted edge types: causal, constraint, disclosure, motivates. The expansion LLM's first action at iteration 51:

```json
{
  "action": "add_beat",
  "target": {
    "child_id": "anchor",
    "edge_type": "motivates",
    "beat_text": "Three weeks earlier: Ines and her mother in the hospice kitchen, her mother asks Ines to remind her of the summer they went north, and Ines — startled — begins to recall it accurately. Her mother looks disappointed.",
    "edge_payload": {
      "agent": "Ines",
      "goal": "to match her mother's emotional need in specific moments, not to answer questions factually",
      "stakes": "if she continues to recall accurately, the mother's loneliness with her memory-failure becomes visible and unbearable"
    }
  },
  "reasoning": "Plants the moment where Ines notices accuracy is the wrong currency — sets up the pivot's decision, and carries the motivates-edge specificity the brief said judges penalized its absence of."
}
```

The beat-specific `goal` language ("to match her mother's emotional need in specific moments") is a direct response to the brief's divergence direction about beat-level specificity. The LLM has been told where it was failing.

Phase 2 runs 50 iterations. Tree ends with 10-node DAG: opening + ki/sho setup strands + pivot anchor + downstream ketsu beats.

---

## 4. MCTS Phase 3 — Cross-phase refinement

5 iterations, `add_edge` only. Permitted edge types: all 5 (including long-range disclosure and motivates spanning the anchor).

Expansion proposes (among 4 candidates): a long-range disclosure edge from a late ketsu beat back to the opening's "accurate memory" beat, with `withheld` = "Ines's willingness to invent had already been latent in her recall reflex."

Cheap judge rewards +0.06 over Phase 2 champion. Full panel confirms. Phase 3 improvement rate for this tree: 1/5 = 20% (within the "moderate" band per mcts.md §Monitoring Phase 3; Phase 3 doing its job).

Final Cassandra-ish tree output: 10-node DAG with 12 edges.

---

## 5. Within-concept Tournament

All 4 preset trees have produced winners. The 4 DAGs compete round-robin: C(4, 2) = 6 full-panel pairwise comparisons.

Per comparison (same as §2.3's full panel): 8 LLM calls (4 judges × 2 orderings), all 8 dimensions.

```
Match 1: Cassandra_winner vs Phoebe_winner
  Cassandra: 5 wins, Phoebe: 2 wins, 1 tie → Cassandra wins match

Match 2: Cassandra_winner vs Randy_winner
  Cassandra: 4 wins, Randy: 3 wins, 1 tie → Cassandra wins match

Match 3: Cassandra_winner vs Winston_winner
  Cassandra: 3 wins, Winston: 4 wins, 1 tie → Winston wins match

Match 4: Phoebe_winner vs Randy_winner
  Phoebe: 3 wins, Randy: 4 wins, 1 tie → Randy wins match

Match 5: Phoebe_winner vs Winston_winner
  Phoebe: 2 wins, Winston: 5 wins, 1 tie → Winston wins match

Match 6: Randy_winner vs Winston_winner
  Randy: 4 wins, Winston: 3 wins, 1 tie → Randy wins match
```

**Tournament ranking:**

| Preset | Wins | Losses | Ties | Rank |
|---|---|---|---|---|
| Cassandra | 2 | 1 | 0 | 2 |
| Winston | 2 | 1 | 0 | 1 (tiebreaker on dim-wins) |
| Randy | 2 | 1 | 0 | 3 (fewer dim-wins) |
| Phoebe | 0 | 3 | 0 | 4 |

Near-tie check: Cassandra vs Winston within 1 dim-win across their common comparisons → **both promoted under near-tie rule** (config flag on).

**Tournament cost:** 6 pairs × 8 calls = 48 calls × ~$0.008 = $0.38.

---

## 6. Archive Write

Each of the 4 preset winners is computed for QD cell coordinates:

```python
Cassandra_winner: disclosure_ratio = 3/12 = 0.25 → bin 1 (Light)
                  structural_density = 12/10 = 1.2 → bin 1 (Simple)
                  cell = (1, 1)

Winston_winner:   cell = (2, 2)  # Balanced × Moderate
Randy_winner:     cell = (0, 0)  # None × Skeletal
Phoebe_winner:    cell = (1, 2)  # Light × Moderate
```

All four written to `tree.archive.cells`. Non-advancing (Randy, Phoebe) stored with tournament rank and judge reasoning excerpts.

Serialized to `results/run_<ts>/stage_2/qd_archive.json` at run end.

---

## 7. Handoff Manifest

`K=2` in `medium.yaml`; Winston (rank 1) and Cassandra (rank 2) advance directly. Near-tie rule is not invoked — it would only apply if K=1 had been configured and two trees were within 1 dim-win across their comparisons.

```json
{
  "run_id": "run_20260501_...",
  "concept_id": "p_a8f12e",
  "winners": [
    {
      "preset": "cassandra_ish",
      "tournament_rank": 2,
      "qd_cell": [1, 1],
      "genome_path": "by_concept/p_a8f12e/winners/cassandra.json",
      "stage_1_concept_path": "by_concept/p_a8f12e/stage_1_concept.json",
      "mcts_reward": 0.68,
      "adaptation_permissions": ["prose_discovers_turn", "state_contradiction", "dead_scene"]
    },
    {
      "preset": "winston_ish",
      "tournament_rank": 1,
      "qd_cell": [2, 2],
      ...
    }
  ]
}
```

---

## 8. Metrics Exported

`results/run_<ts>/stage_2/metrics.json`:

```json
{
  "run_id": "run_20260501_...",
  "config": "medium.yaml",
  "concepts_processed": 1,
  "presets_run": ["cassandra_ish", "phoebe_ish", "randy_ish", "winston_ish"],

  "totals": {
    "llm_calls": 1352,
    "cost_usd": 8.02,
    "wall_time_sec": 1820
  },

  "by_call_kind": {
    "expansion": {"calls": 200, "cost": 2.00, "mean_latency_sec": 3.2},
    "rollout_extension": {"calls": 612, "cost": 0.25, "mean_latency_sec": 1.4},
    "cheap_judge_rollout": {"calls": 408, "cost": 0.82, "mean_latency_sec": 1.1},
    "full_panel": {"calls": 416, "cost": 3.33, "mean_latency_sec": 3.6},
    "summarizer": {"calls": 8, "cost": 0.12, "mean_latency_sec": 4.1},
    "seed": {"calls": 1, "cost": 0.005, "mean_latency_sec": 1.8},
    "tournament": {"calls": 48, "cost": 0.38, "mean_latency_sec": 3.4}
  },

  "per_tree_summary": {
    "cassandra_ish": {
      "phase_1_iterations": 50,
      "phase_2_iterations": 50,
      "phase_3_iterations": 5,
      "phase_3_improved": true,
      "full_panel_events": 11,
      "promotions_accepted": 5,
      "promotions_rejected": 6,
      "summarizer_runs": 2,
      "final_node_count": 10,
      "final_edge_count": 12
    },
    "phoebe_ish": {...},
    "randy_ish": {...},
    "winston_ish": {...}
  },

  "phase_3_improvement_rate": 0.75,

  "preset_divergence": {
    "edge_type_histogram_L1_cassandra_vs_randy": 0.41,
    "disclosure_ratio_spread": 0.18,
    "node_count_spread": 3
  },

  "cheap_full_agreement_rate": 0.72,

  "per_dimension_vote_distribution": {
    "edge_logic": {"a_wins": 0.32, "b_wins": 0.30, "ties": 0.38},
    "motivational_coherence": {...},
    "tension_information_arch": {...},
    "post_dictability": {...},
    "arc_integrity_ending": {...},
    "structural_coherence": {...},
    "beat_quality": {"a_wins": 0.28, "b_wins": 0.26, "ties": 0.46},
    "concept_fidelity_thematic": {...}
  },

  "inter_dimension_co_vote_rates": {
    "edge_logic_vs_motivational_coherence": 0.71,
    "tension_information_arch_vs_post_dictability": 0.84,
    "beat_quality_vs_concept_fidelity_thematic": 0.63,
    "... (all C(8,2) = 28 pairs) ...": "..."
  },

  "invalid_id_rate": 0.04,
  "rechallenge_delta_mean": -0.03,
  "rechallenge_delta_std": 0.11,
  "disclosure_density_correlation": 0.22,

  "baseline_head_to_head": {
    "stage_2_wins": 1,
    "baseline_wins": 0,
    "ties": 0,
    "per_dimension_wins_stage_2": {...}
  }
}
```

**Signal read from this metrics.json:**

- Phase 3 improvement rate 75% → higher than expected "moderate" band (5-30%); suggests 2-phase edge-type restrictions are too strict. Flag for v1.5 redesign.
- `cheap_full_agreement_rate 0.72` → just above the 0.70 alert threshold; cheap judge signal is usable but drifting; consider swapping the cheap-judge model at next run.
- `tension_information_arch_vs_post_dictability co-vote rate 0.84` → right at flag threshold (0.80 per `implementation.md`); likely a collapse candidate in v2.
- `beat_quality ties 46%` → under 60% threshold, discriminating adequately.
- `preset_divergence edge-type L1 0.41` → above 0.25 gate; semantic presets are producing structurally distinct trees. Scaling to medium/heavy is cleared.
- Baseline head-to-head: Stage 2 wins 1-0 on the one pilot concept. Not significant; need full 3-concept pilot to call it.

---

## 9. What This Walkthrough Makes Visible

Three things become inspectable at this granularity that weren't obvious from the design docs alone:

**1. The ChampionBrief's role in improving proposal quality.** Without the brief (iterations 1-29), the LLM is structurally amnesiac — every expansion proposes locally-greedy actions biased by the partial DAG alone. After the brief fires at iter 30, proposal quality measurably climbs. The feedback loop is what converts MCTS from "many independent LLM samples with UCB routing" to "a search that learns from its own critiques." This is load-bearing.

**2. The cost structure is judge-dominated, not expansion-dominated.** Rollouts fire 3x as often as expansions but cost less per call; full-panel events fire rarely but are expensive per event. Total cost per concept at medium.yaml is ~$8, of which ~$3.30 (~41%) is full-panel alone (4 judges × 2 orderings × ~50 events). The tiered-judge design is earning its complexity — if every rollout fired the full panel, per-concept cost would be ~$20+. Two forces concentrated cost onto the full panel: (a) moving to 4 contest judges bumped per-event cost by ~33% vs. a 3-judge panel; (b) switching expansion to deepseek-v4-pro and rollout to deepseek-v4-flash cut generation cost substantially. Net effect: full-panel is now the dominant line item by a wider margin than before.

**3. Phase transitions preserve the right thing.** Phase 2 benefits immediately from Phase 1's full-panel critique accumulation via the carried-forward ChampionBrief, even though the MCTS tree itself is reset. The tree's exploration *memory* is the brief + critiques, not the UCB statistics. This is what makes Phase 2's first-iteration expansion already better-informed than Phase 1's was.

---

## 10. What This Walkthrough Does NOT Show

Flagged for future docs or pilot findings:
- **What happens if motif extraction fails.** `seed_root` now only extracts motifs — no anchor generation can fail. On motif-extraction failure (3 retries exhausted), `motif_threads` is empty and expansion proceeds without the motif bias (non-fatal). The anchor itself cannot fail at this stage because Stage 1's Gate 1 validated it before the concept ever reached Stage 2.
- **Budget exceeded mid-run.** `per_concept_time_budget_minutes=30` enforcement path is not traced here.
- **Pathological expansion where all K=4 candidates are invalid.** The `fully_expanded` mark and the rollout fallback.
- **Multi-concept concurrency interactions.** This run covers 1 concept; interleaving across concepts adds asyncio complexity not shown.
- **What the Stage 3 runner does with the handoff manifest.** Out of Stage 2 scope; flagged for Stage 3 design.
