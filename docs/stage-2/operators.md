# Stage 2: Operators

Operators are the atomic mutations that MCTS composes into structures. Each operator produces a specific transformation of the partial DAG. This document specifies each operator's interface, prompt template structure, validation rules, and failure handling.

Stage 2 has four operators in v1:

| Operator | Purpose | When invoked |
|---|---|---|
| `seed_root` | Wrap the Stage 1 concept's `anchor_scene` as the single-node root DAG; extract motif threads | Once per concept, before MCTS begins |
| `add_beat` | Insert a new beat attached via a typed edge | MCTS expansion action |
| `add_edge` | Add a typed edge between two existing beats | MCTS expansion action |
| `rewrite_beat` | Revise an existing beat's sketch text | MCTS expansion action |

A fifth operator, `post_hoc_rationalize`, is deferred to v1.5 (see §post_hoc_rationalize below for the rationale and the v1 substitute).

Operators are deterministic in their validation logic (same inputs → same validation result) and stochastic in their LLM generation (different temperatures produce different content). Retries use temperature ramping.

---

## Operator Philosophy

### Each operator is a single LLM call

With rare exceptions (see `post_hoc_rationalize`), each operator corresponds to exactly one LLM call. This keeps the accounting simple for cost estimation and keeps MCTS timing predictable.

### Operators propose; MCTS selects

`add_beat`, `add_edge`, and `rewrite_beat` generate proposals that MCTS then evaluates via its selection/expansion/rollout loop. A proposed action is accepted into the tree only if it passes validation and is selected by UCB. Invalid proposals are discarded silently; MCTS moves on.

### Prompts follow the prompting guide

All Stage 2 prompts adhere to `docs/prompting-guide.md` principles:
- Order = causality (instructions precede context precede task)
- Decision chains, not pattern + exemption
- Additive context (operator prompt continues from a stable base system message)
- Understanding over reinforcement (explain *why* edges matter, not just format)
- Fix-plan commitment (require operators to state what they'll do before producing it, when the transformation is nontrivial)

---

## Base System Message

All operator prompts share a base system message that establishes the operator voice, the typed-edge taxonomy, the genome schema, and the phase context. The base message is shared with the judges (see `judges.md`) to ensure everyone reads the DAG the same way.

Outline of the base system message (~500 tokens):

```
You are a structural architect for short fiction. You work with a specific
representation: a typed-edge directed acyclic graph (DAG) where each node is
a story beat and each edge expresses WHY one beat follows another.

Five edge types exist:
- CAUSAL: A's event causes B's. The `realizes` field names what is caused.
- DISCLOSURE: B reveals information that reframes A. `reframes` names what
  is recontextualized; `withheld` names what was hidden between A and B.
- IMPLICATION: B logically follows from A's premises. `entails` names the
  proposition B must embody.
- CONSTRAINT: A forecloses an option at B. `prohibits` names what A prevents.
- MOTIVATES: A installs a character intention anchoring B's actions. `agent`
  names the character; `goal` the intention; `stakes` (optional) the risk.

Beat sketches are 1-2 sentences. They should be specific and evocative.
Generic placeholders ("she confronts him") fail; specific beats ("she asks
whether the elephants looked like clouds") plant material for the prose
stage to build on.

Every edge's payload fields must be populated. Empty fields produce
decorative edges that the prose stage will ignore.

{PHASE_CONTEXT}
{CONCEPT_CONTEXT}
{PACING_PRESET}
```

`{PHASE_CONTEXT}` specifies whether this is the forward or backward phase and what edge types are permitted. `{CONCEPT_CONTEXT}` injects the Stage 1 concept (premise, target effect, optional fields, judge reasoning). `{PACING_PRESET}` names the preset and its character ("Cassandra-ish: escalating peaks with guaranteed relief").

---

## seed_root

**Purpose:** wrap the Stage 1 concept's `anchor_scene` as a single-node root DAG and extract two derived fields: `motif_threads` (recurring elements that bias expansion) and `concept_demands` (predicates Tier 3 will check at terminal time — see `overview.md` §Concept demands). The anchor itself is consumed from the genome — Stage 1 has already evolved, tournament-selected, and validated it (see `lab/issues/2026-04-22-anchor-scene-in-stage-1-genome.md`). MCTS still builds around a single-node seed, preserving the BiT-MCTS (arXiv:2603.14410) bidirectional-from-anchor search structure; what's removed is the in-operator LLM generation of that anchor.

**One seed per concept, shared across all preset trees.** The anchor is the story's central moment — its content does not depend on pacing philosophy. Preset divergence happens via expansion priors (see `mcts.md` §Selection), not at the seed.

### Interface

```python
def seed_root(concept: Stage1Concept) -> DAG:
    """Wrap concept.anchor_scene as a single-node root DAG and extract motif threads.
    The resulting 1-node DAG is shared as the root of all pacing-preset MCTS trees
    for this concept."""
```

Note: no `preset` parameter. A concept has exactly one anchor; all preset trees start there.

### Procedure

1. **Wrap the anchor.** Construct a single node from `concept.anchor_scene`: `Node(id="anchor", sketch=concept.anchor_scene.sketch, role=[concept.anchor_scene.role], motifs=[])`. Stage 1 evolves one canonical role (single string); Stage 2 wraps it as a one-element list. Additional roles can be added during MCTS expansion via `rewrite_beat` (e.g., a climax node that structural analysis reveals is also a pivot gets `role: ["climax", "pivot"]`). No LLM call.
2. **Motif + demand extraction (one merged call).** A single structured LLM call reads the concept + anchor and produces both `motif_threads` (2–3 concrete recurring elements — imagery, objects, phrases) and `concept_demands` (zero or more predicates — see `overview.md` §Concept demands). Merging the two into one call costs the same as motif extraction alone, prevents the two derivations from being inconsistent (e.g., a motif-extraction interpretation of the concept that disagrees with the demand-extraction interpretation), and reduces the seed step's overall LLM budget. Both outputs are stored on the genome as top-level fields and shared across all 4 preset trees for the concept. The anchor node's per-node `motifs` field remains empty at seed time — per-node motif attachment happens during MCTS expansion when surrounding context exists.

### Prompt template (step 2 — merged motif + demand extraction)

```
{BASE_SYSTEM}

TASK: For this concept, produce two outputs:

1. motif_threads — 2-3 concrete recurring elements (imagery, objects, phrases,
   character obsessions) that the DAG's beats will thread through. Motifs are
   physical and specific, not abstract: "stones" and "the black box" are
   motifs; "mortality" and "tradition" are themes. Draw from the premise,
   character seeds, setting seeds, and anchor scene.

2. concept_demands — zero or more one-sentence predicates that any DAG must
   satisfy to realize this concept. ONLY emit a demand when the concept's
   central structural mechanism is something that:
     - cannot be expressed as a node role (climax/reveal/pivot)
     - cannot be expressed as an edge type (causal/disclosure/implication/
       constraint/motivates)
     - cannot be expressed as a motif mode or a story-wide constraint

   Examples of demands worth emitting (only when the concept calls for it):
     "the DAG must include a beat that addresses the reader directly,
      breaking the diegetic third-person frame"
     "the form's recursion must be enacted structurally — the closing beat
      must mirror or invert the opening's premise as a structural rhyme"
     "the structure must preserve a structural element that is simultaneously
      true and false (dialetheic structure)"

   Most concepts will produce ZERO demands — the standard schema (roles, edge
   types, motifs, story_constraints, character_arcs) is sufficient. If you
   cannot articulate a demand that goes beyond what those carry, output an
   empty list. False positives are worse than false negatives here: a
   spuriously-emitted demand will be checked by Tier 3 and may rank good
   DAGs below worse ones for failing a demand the concept didn't really have.

CONCEPT:
{concept_json}

ANCHOR SCENE:
{anchor_sketch}
(role: {anchor_role})

Output as JSON:
{
  "motif_threads": ["...", "...", "..."],
  "concept_demands": []  // or list of one-sentence predicates
}
```

### Validation

1. The anchor node's `sketch` and `role` were already validated at Stage 1's Gate 1 (min_length=40, placeholder regex, visualizability classifier, role enum). Stage 2 does not re-validate them.
2. No edges in the seed (the seed is a single disconnected node; edges emerge during MCTS expansion).
3. **Motif threads.** 2–3 concrete (not abstract) elements. Concrete = physical object, specific image, or literal phrase. A motif like "stones" or "the black box" passes; "mortality" or "tradition" fails as too abstract — these would not thread through prose as recognizable recurrences.
4. **Concept demands.** Each demand is one sentence (≤ ~30 words), names a structural element (not an emotional outcome — those belong in `target_effect`), and is *operational* (a Tier 3 LLM check can read the demand and the DAG and verify satisfaction). Reject demands that paraphrase `target_effect` or `thematic_engine` — those are emotional/thematic, not structural. Empty list is valid output.

### Failure handling

If extraction fails (3 retries with temperature ramp 0.7 → 0.9 → 1.1), seed with empty `motif_threads` and empty `concept_demands` and log the failure. Both fields are bias-and-check, not load-bearing — missing motifs degrade to pre-motif expansion behavior; missing demands skip Tier 3 with a logged warning. The concept still advances to MCTS. (Contrast with the earlier design, where seeding failure blocked the entire concept because anchor generation itself could fail. That failure mode now belongs to Stage 1.)

### Cost

One merged extraction LLM call at ~$0.005–$0.008 per concept (slightly higher than motif-only because of the longer combined prompt). With retries, worst case ~$0.020. Demand-extraction calibration against the canonical concepts is part of Phase 0; the prompt's tendency to emit spurious demands or miss real ones is measured before scaling.

### Why the operator still exists

Three points worth naming, since the operator shrank to a single motif call:

- **Still a seed.** MCTS needs a single-node root DAG to start; the operator constructs it. That it reads from `concept.anchor_scene` rather than generating the anchor is a narrowing of responsibility, not an elimination of the operator's role.
- **Motif extraction is Stage-2-local.** Motifs are consumed by every expansion prompt and have no Stage 1 consumer, so moving them into Stage 1 would add a genome field Stage 1 never reads. Keep them here.
- **Role is the Stage 1 enum, not a new one.** Stage 2 accepts `"climax" | "reveal" | "pivot"` directly from the genome. No re-classification, no remapping. Phase edge-type restrictions consult this role as authoritative; see `mcts.md` §Bidirectional Phases for how it's used.

---

## add_beat

**Purpose:** insert a new beat into the partial DAG, connected via a typed edge to an existing beat.

### Interface

```python
def add_beat(
    dag: DAG,
    anchor_id: str,          # existing node the new beat connects to
    anchor_role: Literal["parent", "child"],  # determined by phase:
                              #   forward phase: anchor is parent (new beat is downstream)
                              #   backward phase: anchor is child (new beat is upstream)
    edge_type: EdgeType,
    phase: Phase,
) -> AddBeatProposal:
    """Propose a new beat + edge. Returns a proposal (may be invalid).

    In forward phase: anchor_id is the source of the new edge (parent);
    the new beat is the target.
    In backward phase: anchor_id is the target of the new edge (child);
    the new beat is the source, meaning it's an upstream beat.
    """
```

### When invoked

During MCTS expansion. The expansion call returns up to K=4 proposed actions cached on the leaf; some of those are `add_beat` instances. The `add_beat` is not invoked standalone — it emerges from the expansion LLM call.

### Prompt (invoked as part of expansion)

```
{BASE_SYSTEM}

CURRENT DAG:
{incident_encoded_outline}

STRUCTURAL TENSIONS:
{long_range_edges}

WHAT THE JUDGES HAVE BEEN LOOKING AT IN THIS TREE:
{champion_brief}

TASK: Propose up to {k_candidates_per_expansion} actions that would extend
this DAG toward a complete structure for this concept. Each action is one of:
- add_beat: insert a new beat attached to an existing node via a typed edge
- add_edge: add a typed edge between two existing nodes
- rewrite_beat: revise an existing node's sketch text

Pacing: {preset_name} — {preset_character}

Permitted edge types in this phase: {permitted_types}

For each action, output:
{
  "action": "add_beat" | "add_edge" | "rewrite_beat",
  "target": {...},  # varies by action type
  "reasoning": "One sentence: why this action? What does it add or fix?"
}

Before outputting, list 3 structural gaps you see in the current DAG. Then
produce actions that address them.
```

`{champion_brief}` is a rendered `ChampionBrief` summarizing full-panel judge critiques from this tree's prior comparisons (established weaknesses, contested choices, attractor signature, divergence directions). Empty placeholder until the tree accumulates enough full-panel events for the summarizer to fire. See `mcts.md` §Champion Brief Feedback Loop.

### Validation

For an `add_beat` action specifically:
- `beat_text` passes specificity check (length, non-placeholder).
- `edge_type` is permitted for the current phase.
- `edge_payload` has all required fields for the edge type, non-empty.
- The new beat + edge do not create a cycle.
- `motifs` (if provided) is a list of `{motif, mode}` pairs. `motif` must be a verbatim match in the genome's `motif_threads` (typos and drift are rejected). `mode` must be one of the six values defined in the **mode glossary** at `docs/stage-2/overview.md` §Per-node motifs §Mode glossary (authoritative): `introduced | embodied | performed | agent | echoed | inverted`. Do not invent additional modes — if a beat resists tagging, argue the case in the design doc before extending the taxonomy.
- `role` (if provided) is a list of permitted values (`climax`, `reveal`, `pivot`) with no duplicates; judges do not see untyped roles.
- For disclosure edges: `disclosed_to` (if provided) is a list containing `"reader"` and/or character names. If omitted, defaults to `["reader"]` (authorial-only disclosure). If a character name is listed, that character must appear in the target node's sketch (the disclosure triggers their realization *at* the target beat); otherwise reject.
- **Phase direction is respected**:
  - Forward phase: the action specifies a `parent_id` (source of the new edge pointing at the new beat). `parent_id` must be the climax or a descendant. The new beat is the target; it sits downstream of the climax.
  - Backward phase: the action specifies a `child_id` (target of the new edge from the new beat). `child_id` must be the climax or an ancestor of the climax. The new beat is the source; it sits upstream of the climax.
- In both phases, the referenced existing node must exist in the current DAG.

### Failure handling

Invalid proposals are dropped silently. If every proposed action from an expansion is invalid, expansion retries once with higher temperature (0.7 → 1.0). If still no valid proposals, the expansion's parent MCTS node is marked `fully_expanded`.

### Expansion prompt variants (pilot A/B — deferred)

The prompt above is the v1 flat-propose design. An alternative propose→critique→revise→commit variant is specified in `lab/issues/2026-04-19-stage-2-third-review-followups.md` Item 10 as a pilot A/B arm. Summary of the variant:

```
Step 1 — PROPOSE. List {n_initial_proposals} candidates (default 6).
Step 2 — CRITIQUE. For each, articulate in one phrase each:
  - Structural job: what does it do that the DAG doesn't already have?
  - Reader effect: what does it do to the reader — curiosity opened,
    tension raised, emotional state shifted, information revealed?
  Flag any proposal that can't answer both.
Step 3 — REVISE (up to {max_revision_rounds} rounds per flagged proposal).
  - Rewrite to address the critique; re-critique; drop after cap.
Step 4 — COMMIT surviving top {k_candidates_per_expansion} as JSON.
```

Config keys live under `stage_2.expansion` in the run YAML (`n_initial_proposals: 6`, `max_revision_rounds: 2`, `revision_rate_alert: 0.50`). The variant ships only if pilot A/B shows it outperforms the flat-propose prompt at no latency cost. See the issue's Item 10 for full rationale, risks, and monitoring plan.

### Cost

Part of the expansion call. No standalone cost.

---

## add_edge

**Purpose:** add a typed edge between two existing beats without creating a new beat.

### Interface

```python
def add_edge(
    dag: DAG,
    src_id: str,
    dst_id: str,
    edge_type: EdgeType,
    edge_payload: dict,
    phase: Phase,
) -> AddEdgeProposal:
    """Propose a new edge between existing nodes."""
```

### When invoked

During MCTS expansion, as one possible action type. Particularly useful for discovering disclosure edges (which often span existing nodes) and motivates edges (which anchor existing character behavior after the fact).

### Prompt

(Same expansion prompt as `add_beat`; the LLM chooses which action type to propose.)

### Validation

- `src_id` and `dst_id` both exist in the current DAG.
- The edge does not duplicate an existing edge between the same nodes of the same type.
- `edge_type` is permitted for the current phase.
- `edge_payload` has all required fields, non-empty.
- The new edge does not create a cycle in the topological ordering.
- For disclosure edges: `src_id` must topologically precede `dst_id` (disclosure is revealed at dst, withheld at src), AND the `withheld` field must reference content that is not present in the src node's sketch (otherwise the disclosure is trivially satisfied).
- For implication edges: the `entails` field must name an **in-world logical consequence** — given the source beat's state, the target beat follows by necessity that a character could reason about. Reject entailments that read as thematic rhyme ("the heptapod script entails simultaneous cognition" is authorial argument, not logical consequence). Tier 2 plausibility checks this semantically (see `evaluation.md` §Tier 2); the operator validator enforces the weaker surface rule that `entails` starts with a proposition about state (an "if/then" rephrasing should be natural), not a claim about theme or resonance.
- For motivates edges: the `agent` must be consistent with other motivates edges in the DAG **and with `character_arcs[].agent`**. Validation: when a new motivates edge's agent string does not exactly match any existing motivates edge's agent or any `character_arcs[].agent`, check string similarity (normalized Levenshtein ≥ 0.7 OR token-set overlap ≥ 0.5) against existing agents. If a near-match is found, reject the candidate and let the next cached candidate from Π(v) be tried. This catches the "grandmother" vs "Mrs. Crater" failure mode where the same character is named inconsistently. Stage 4 can still introduce specific names during prose generation; the agent field just has to be consistent within the Stage 2 genome.
  - **Threshold validation.** The Levenshtein 0.7 / token-set 0.5 cutoffs are heuristic; the "grandmother" vs "Mrs. Crater" case is anecdotal. Short strings and common-token targets can produce false positives ("the villagers" vs "the children" share "the"). Measure false-positive rate against the 4 canonical DAGs at implementation time — if precision < 80%, swap the heuristic for a single LLM-call consistency check ("Are these two agent names the same character? yes/no").
- For motivates edges: **local scope**. `src_id` and `dst_id` must be within 2 topological positions of each other. A motivates edge spanning more than 2 positions represents a whole-story trajectory that belongs in `character_arcs`, not an edge — reject the proposal. MCTS should learn to propose `character_arc` augmentations (via `rewrite_beat` on a node plus an arc update, or via a future dedicated operator) rather than over-long motivates edges. This rule prevents the canonical failure mode where the expansion LLM encodes "the grandmother wants to be a lady across the whole story" as a single edge from opening to climax.

### Cost

Part of the expansion call. No standalone cost.

---

## rewrite_beat

**Purpose:** revise an existing beat's sketch text. Used when downstream edge additions require a subtly different setup at an earlier beat.

### Interface

```python
def rewrite_beat(
    dag: DAG,
    node_id: str,
    new_text: str,
) -> RewriteBeatProposal:
    """Propose a new sketch for an existing beat."""
```

### When invoked

During MCTS expansion. Often proposed after disclosure edges are added — the source beat may need its sketch adjusted to plant the information that will be reframed later.

### Prompt

(Part of the expansion prompt.)

### Validation

- `node_id` exists in the current DAG.
- `new_text` passes specificity check.
- `new_text` does not contradict existing edges' payload fields (e.g., a rewrite that removes a character mentioned in a motivates edge's `agent` field is rejected).
- `new_text` preserves the beat's `role` if any (a climax rewrite must still be climactic, etc.).

### Contradiction detection

The contradiction check is a cheap LLM call: "Does this rewrite of beat X contradict any of these edges: {edges}? Answer yes or no with one-sentence reasoning."

If contradictions are detected, the rewrite is rejected.

### Cost

Part of the expansion call. Contradiction check: ~$0.005 per proposed rewrite.

---

## post_hoc_rationalize — deferred to v1.5

**Deferred pre-implementation (2026-04-19).** The operator was designed to fabricate missing `realizes` fields for causal edges via a follow-up LLM call (Caves of Qud post-hoc rationalization). Dropped from v1 because:

1. It's preemptive compensation for a failure mode we haven't observed.
2. The expansion prompt can be made to require specific, populated `realizes` fields directly; validation rejects generic payloads.
3. Each causal edge added triggered an extra ~1–2 LLM calls; at 8 causal edges per tree the cost was non-trivial.
4. The fabricated state is discarded (run-log only), so the operator's output is hidden from downstream stages anyway.

**v1 substitute:** the expansion prompt explicitly requires a specific mechanism in `realizes`; the operator validator rejects generic payloads (`>15 chars, not "things happen" / "A enables B"`). If empirical data shows the expansion LLM can't produce specific mechanisms directly, that's evidence to re-introduce `post_hoc_rationalize` in v1.5.

The Caves of Qud rationale and algorithm are preserved below for v1.5 reference.

---

**Purpose (v1.5):** fill a causal edge's `realizes` field when it's missing or underspecified, using the Caves of Qud post-hoc rationalization technique.

### Interface

```python
def post_hoc_rationalize(
    dag: DAG,
    edge_id: str,
) -> PostHocRationalizationResult:
    """Fill a causal edge's realizes field, possibly fabricating entity state."""
```

### When invoked

**Not** an MCTS action — this is an automatic follow-up operator. After any `add_beat` or `add_edge` action that produces a causal edge with an empty or generic `realizes` field, `post_hoc_rationalize` is invoked to fill it.

It is also bounded to causal edges only. Disclosure, constraint, implication, and motivates edges carry their own semantic logic and are not subject to post-hoc rationalization.

### Algorithm

Adapted from `lab/references/caves-of-qud-posthoc/summary.md`:

```
1. GATHER CONTEXT for the edge (src, dst):
   - Character traits and world facts established by nodes topologically
     preceding src
   - Motif / thematic threads active in src's sketch
   - Other edges connected to src or dst and their payload fields

2. CHECK IF CAUSE IS ALREADY EXPLICIT
   - If src's sketch or an earlier edge's payload names a reason for dst's event,
     use that as the `realizes` value and return.

3. ATTEMPT CAUSE FROM EXISTING STATE
   - Prompt the LLM: "Given {established_context}, what existing character
     trait or world fact could plausibly cause {dst_event} to follow from
     {src_event}? Name one cause in one sentence."
   - If the response is grounded in existing state, return it as `realizes`.

4. FABRICATE CAUSE IF NONE EXISTS  [the Caves of Qud inversion]
   - Prompt the LLM: "Invent one character trait or world fact that would
     make {dst_event} follow naturally from {src_event}. Keep it consistent
     with the concept's motif threads: {motifs}."
   - The fabricated trait is recorded as new entity state (for future edges
     to reference) and used to fill `realizes`.

5. PROPAGATE
   - The fabricated or derived state is made available to downstream edges
     in the same MCTS tree.
```

### State log

Post-hoc rationalization maintains a per-MCTS-tree state log: fabricated traits, established character facts, motif threads. This log is tree-local working memory, passed to later rationalization calls in the same tree.

**The log is NOT persisted on the genome.** When MCTS produces a winning DAG, the rationalization log is written to `results/run_<timestamp>/stage_2/by_concept/<concept_id>/rationalization_log.json` as a run-log side-channel for debugging and post-hoc analysis. Stage 3 and Stage 4 read only the DAG (nodes, edges, payload fields) — they don't see the log. If a fabricated trait was load-bearing, it's visible in the edge's `realizes` field (or whichever payload field pushed the rationalization). If it didn't make it into the payload, it wasn't load-bearing. This matches Stage 1's discipline: judge reasoning is in run output, not in the concept JSON.

The earlier proposal to compress the state log into a `tree_rationalization_notes` field on the genome was rejected pre-implementation (2026-04-19) because it pollutes the structural artifact with debugging metadata that has no defined role downstream.

### Validation

- The proposed `realizes` field is non-empty and specific (>15 chars, not generic like "things happen" or "it makes sense").
- A consistency check: "Does `{realizes}` contradict any established fact in the state log?"

### Failure handling

- If 2 retries can't produce a valid `realizes`: the edge is marked `rationalization_failed` and its `realizes` field defaults to a minimal "A enables B by the logic of the story." The MCTS reward for this tree path is penalized (×0.9) to disincentivize failing rationalizations without fully blocking the path.

### Cost

1–2 LLM calls per rationalization (depending on whether fabrication is needed). At ~$0.005 per call, typically $0.01 per causal edge. For a DAG with 8 causal edges, total rationalization cost ~$0.08 — negligible.

---

## Operator Invocation Summary

| Operator | Invoked by | Cost per call | Retries |
|---|---|---|---|
| `seed_root` | Tree setup, pre-MCTS | ~$0.005 (motif extraction only) | Up to 3, temp ramp (motif call only) |
| `add_beat` | MCTS expansion | shared with expansion | N/A (expansion retries once) |
| `add_edge` | MCTS expansion | shared with expansion | N/A |
| `rewrite_beat` | MCTS expansion | shared + $0.005 contradiction check | N/A |
| `post_hoc_rationalize` | *Deferred to v1.5* | — | — |

---

## Validation Layer

All operators share validation utilities:

```python
class OperatorValidator:
    def specificity_ok(self, text: str) -> bool:
        """Check minimum length, non-placeholder text."""

    def field_completeness(self, edge: Edge) -> bool:
        """Check all required fields for edge type are populated."""

    def acyclic(self, dag: DAG, new_edge: Edge) -> bool:
        """Check the edge doesn't create a cycle."""

    def phase_permits(self, edge_type: EdgeType, phase: Phase) -> bool:
        """Check edge type is permitted for current phase."""

    def character_consistency(self, dag: DAG, edge: Edge) -> bool:
        """For motivates edges: check agent matches other motivates edges."""

    def contradicts_existing(self, dag: DAG, rewrite: RewriteBeatProposal) -> bool:
        """For rewrites: LLM check that new text doesn't contradict existing edges."""
```

These are implementation concerns; full signatures appear in `implementation.md`.

---

## Open Questions Surfaced in Operators Drafting

1. **Character-name resolution.** Motivates edges' `agent` field refers to a character by name or role. Stage 1 concepts may name characters ("Tessie") or may only sketch them ("a grandmother who thinks she's a lady"). When post-hoc rationalization fabricates a trait, what name does it use for a not-yet-named character? Proposal: use the concept's character seed sketch as a canonical reference until the DAG itself introduces a specific name. Needs prompt-level specification.

2. **State log persistence.** Post-hoc rationalization maintains tree-local working state. Whether to persist this into the final genome is undecided. Persisting it helps Stage 4 (prose has context); not persisting it keeps the genome minimal. Current plan to compress into `tree_rationalization_notes` is a compromise — flag for real decision during implementation.

3. **`rewrite_beat` retroactive effects.** A rewrite of node X may invalidate edges downstream that depended on the original text. The contradiction check catches obvious cases but may miss subtler issues. Proposal: rewrites that affect high-degree nodes trigger a full DAG re-validation. Needs implementation detail.

4. **Action diversity in expansion proposals.** Currently the expansion LLM chooses which action types to propose. If it always proposes `add_beat` (the most common action), `add_edge` and `rewrite_beat` never surface. Possible fix: explicitly prompt for at least one non-add_beat action when the DAG has >5 nodes. Defer to calibration with real data.

5. **`post_hoc_rationalize` budget cap.** A pathological MCTS tree with many causal edges could invoke rationalization hundreds of times, blowing the per-tree budget. Current estimate (~$0.08 per tree) assumes typical causal-edge density. Flag for monitoring; add a per-tree cap if empirical cost exceeds estimate.

6. **(Closed 2026-04-22.) Anchor role misclassification.** The earlier concern — `seed_root` picking the wrong role for a concept, silently misshaping the whole subsequent search — is now resolved structurally. The anchor role is produced by Stage 1 and subjected to pairwise tournament selection across judges evaluating Indelibility, Grip, and Tension Architecture against the rendered scene. That's strictly stronger selection pressure than any pointwise classifier gate. See `lab/issues/2026-04-22-anchor-scene-in-stage-1-genome.md` §"Stage 2 Handoff" for the reasoning.
