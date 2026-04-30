# Stage 2: Structure Evolution — Overview

Stage 2 is the plan behind the prose. A Stage 1 concept gives us a premise, a target emotional effect, and (usually) a sketch of what kind of story this wants to be. What it does not give us is a structure — a sequence of beats, with typed relationships between them, that an actual writer could use to draft the story.

Stage 2 produces that structure. The output is a typed-edge directed acyclic graph (DAG) whose nodes are story beats and whose edges articulate why one beat follows another: causal, disclosure, implication, constraint, or motivational. Multiple structures can serve the same concept — a reveal story and a character-collision story can arise from the same premise depending on how the structure shapes information flow and character intentionality. Stage 2 evolves that shape.

This document covers the philosophy, the genome specification, the pacing preset system, how Stage 2 differs from Stage 1 architecturally, and the handoffs on both ends. MCTS algorithm details, operators, evaluation, rubric anchors, judge design, the QD archive, and implementation specifics each live in their own docs.

---

## Philosophy: Structure Is Load-Bearing

### Why not skip straight to prose?

Eight years of research (ACL 2018 through NAACL 2025) consistently shows staged story generation — concept → outline → prose — outperforms end-to-end generation on coherence, plot quality, and human preference [Fan 2018; Plan-and-Write 2019; Re3 2022; DOC 2023; DOME 2025]. The biggest gains come from the concept→outline transition, not the outline→prose transition. Whatever we do in Stage 4 (prose), it is bounded by what Stage 2 hands forward.

LLMs asked to jump from concept to prose without an intermediate plan hallucinate structure. They invent plot convergences that don't serve the concept, or default to three-act Freytag arcs regardless of whether the concept calls for one. Stage 2 fixes this by making structure a first-class object that can be evaluated, iterated, and selected against.

### Why typed edges?

Every LLM-era narrative graph system reviewed (GraphPlan 2021, NGEP 2022, STORYTELLER 2025, Narrative Studio 2025, BiT-MCTS 2026) uses a single edge type: temporal adjacency. The only published narrative planner with typed edges is Riedl & Young's IPOCL (JAIR 2010), which uses two: causal and motivational. Both cognitive-psychology research (Trabasso & van den Broek 1985; Graesser et al. 1991, 1994) and recent LLM benchmarks (CoG 2025, 53% failure rate on motivational planning when implicit) confirm that different types of relationship between story events produce genuinely different comprehension and require different generative constraints.

Our 5-edge taxonomy names those relationships explicitly: causal for world-state change, disclosure for reader-information reframing, implication for logical entailment, constraint for what a prior node forecloses, motivates for character intentionality. The taxonomy is grounded in prior art (Riedl & Young for causal + motivational; our extensions for disclosure, implication, and constraint each defended in the typed-edges research report).

### Why a DAG and not a sequence?

A sequence collapses every relationship into "followed by." A story like Jackson's "The Lottery" has disclosure edges that point from the climax backward to the opening (the final stoning reframes the cheerful gathering); a story like Chiang's "Story of Your Life" has implication edges connecting every premise-beat to every consequence. Linear sequences can't represent these; DAGs can. The DAG structure is also what makes Stage 2's fitness function sharper than Stage 1's — we can count disclosure-edge density, identify convergent motivational arcs, detect unearned reveals, and compare structures on multiple orthogonal axes that a flat ordering can't expose.

### What Stage 2 is NOT for

- **Prose generation.** Beat sketches are 1–2 sentences, not full paragraphs. Stage 4 writes the prose.
- **Voice specification.** The voice is evolved in Stage 3 and conditions prose in Stage 4. Stage 2 does not choose voice.
- **Final concept revision.** The concept is locked when it enters Stage 2. Stage 2 may surface that a concept is structurally infeasible (no structure produces an acceptable reward) — that signal propagates back to the compost heap, not forward as a concept edit.

### The DAG is a plot substrate, not a prose specification

The typed-edge DAG captures causal sequencing, intentional arcs, information architecture, and diegetic constraints. What it does *not* capture — by design — is voice, register, prosody, or sentence-level craft. A faithful execution of Hemingway's "Hills" DAG can be written in Faulknerian maximalism and still satisfy every node, edge, and motif. Fidelity to a story's canonical tonal shape is Stage 3's (voice) and Stage 4's (prose) responsibility, conditioned on the concept's `style_hint` and the structure handed off from Stage 2.

This is the intentional consequence of the staged pipeline: pushing voice into a separate search keeps each genome's evaluable surface narrow enough to judge rigorously. The cost is that Stage 2 alone cannot guarantee the prose will sound like the canonical work — Stages 3 and 4 close that gap.

---

## The Genome: A Typed-Edge DAG

A Stage 2 genome is the structured representation of a story's structural plan. It's serialized as JSON and evaluated, mutated, and compared as an atomic unit.

### Responsibility map: what lives where

Readers opening the genome for the first time may expect it to fully characterize a story. It doesn't — by design. Responsibilities are split across stages and genome fields so each level carries what it can evaluate rigorously. Before reading the schema, know what belongs to what:

| Concern | Where it lives |
|---|---|
| Plot structure (beats, typed relationships, anchors) | Stage 2 DAG: nodes + edges |
| Character intention (spanning) | Stage 2 DAG: `character_arcs` |
| Character intention (local installation) | Stage 2 DAG: `motivates` edges |
| Diegetic rules (story-scoped) | Stage 2 DAG: `story_constraints` |
| Diegetic rules (local, A forecloses B) | Stage 2 DAG: `constraint` edges |
| Thematic recurrence | Stage 2 DAG: `motif_threads` + per-node `motifs` |
| Information architecture (what reader knows vs. characters) | Stage 2 DAG: `disclosure` edges, with `disclosed_to` distinguishing reader vs. character audience |
| Pacing philosophy | Stage 2 preset (expansion-prompt hint; not a genome field) |
| Voice, register, prosody, tonal register | Stage 3 (separate search) |
| Sentence-level craft, paragraph rhythm | Stage 4 (prose generation) |

A faithful execution of a Stage 2 genome can be written in many voices — Hemingway's "Hills" DAG can be rendered in Faulknerian maximalism and satisfy every node, edge, and motif. Fidelity to a story's canonical tonal shape is Stage 3 + 4's job, not Stage 2's. If you're reading a Stage 2 genome and noticing that register and voice are absent, that's correct.

### Structure

```yaml
genome:
  concept_id: "c_a8f12e"        # reference to the Stage 1 concept this plans
  preset: "cassandra_ish"        # which pacing preset generated this genome
  motif_threads:                 # 2-3 recurring elements that weave through the DAG
    - "stones"                   # children stacking them, the marked slip, the killing
    - "the black box"            # brought out, held, folded away
    - "Old Man Warner's mutter"  # voice of tradition, spoken three times
  character_arcs:                # whole-story character trajectories; 0-N entries
    - agent: "Tessie Hutchinson"
      goal: "to be part of the community in good standing"
      stakes: "her membership in the only moral order she knows"
  story_constraints:             # diegetic rules that hold across the whole story (not pairwise)
    - prohibits: "any villager speaking the lottery's lethal purpose aloud; naming it would rupture the ritual's performance"
      lifts_at: "stoning"        # node where the rule breaks; null = never lifts
  concept_demands: []            # extracted by seed_root from the concept; empty when the concept's
                                 # structural mechanism is fully expressible in {role, edge type,
                                 # motif mode, story_constraint, character_arc}. Non-empty for concepts
                                 # whose central move is reader-address, form-as-device, deliberate
                                 # irresolution, or other moves the schema can't directly carry.
  nodes:
    - id: "opening"
      sketch: "Warm summer morning. Villagers gather in the town square; children stack stones at the edge."
      role: null                 # list[climax|reveal|pivot] | null; first entry is primary (drives MCTS phase dispatch)
      motifs:                    # list[{motif, mode}] — which motifs surface, and how
        - {motif: "stones", mode: "introduced"}
    - id: "summers"
      sketch: "Mr. Summers arrives with the battered black box; heads of household are called by name."
      role: null
      motifs:
        - {motif: "the black box", mode: "introduced"}
    - id: "hutchinson"
      sketch: "Bill Hutchinson draws the slip with the black dot; the family is selected for the next round."
      role: null
      motifs:
        - {motif: "the black box", mode: "echoed"}
    - id: "tessie"
      sketch: "The Hutchinsons redraw. Tessie protests; she draws her own slip and finds the black dot."
      role: null
      motifs:
        - {motif: "the black box", mode: "echoed"}
    - id: "stoning"
      sketch: "Villagers close in. Stones are picked up. Tessie screams that it isn't fair. They stone her."
      role: ["climax"]           # single-role nodes use a 1-element list; multi-role uses ["climax", "pivot"] etc.
                                 # First entry drives MCTS phase dispatch, Stage 1 handoff surface slot, and
                                 # Stage 4's primary-role craft expectations. Subsequent entries are attended
                                 # to by judges but do not change MCTS behavior. Order by load-bearingness.
      motifs:
        - {motif: "stones", mode: "performed"}       # villagers enact communal violence by wielding stones
        - {motif: "Old Man Warner's mutter", mode: "echoed"}
  edges:
    - src: "opening"
      dst: "summers"
      type: "causal"
      realizes: "the lottery is a standing annual institution with a designated official and a ritualized object"
    - src: "summers"
      dst: "hutchinson"
      type: "causal"
      realizes: "procedural authority — officiousness and cheerfulness — legitimizes random lethal selection"
    - src: "hutchinson"
      dst: "tessie"
      type: "causal"
      realizes: "the family is the first selection stage; individual accountability follows group selection"
    - src: "tessie"
      dst: "stoning"
      type: "causal"
      realizes: "individual selection activates undivided communal execution; Tessie's resistance changes nothing"
    - src: "opening"
      dst: "stoning"
      type: "disclosure"
      reframes: "cheerful gathering → prelude to stoning"
      withheld: "the lottery's outcome is death"
      disclosed_to: ["reader"]   # who learns; "reader" for authorial reveal, character names for diegetic recognition
    - src: "summers"
      dst: "hutchinson"
      type: "disclosure"
      reframes: "innocent-seeming box → weapon-selection lottery"
      withheld: "the purpose of the ritual"
      disclosed_to: ["reader"]
```

### State tracking across the DAG

Short answer: the DAG rendering is the state tracker. Every MCTS expansion prompt renders the full partial DAG in incident-encoded format (per `evaluation.md`), so the LLM sees all prior beats when proposing the next. No separate entity-log or world-state field is carried in the genome.

This is viable because Stage 2 DAGs are small (3–18 nodes) — the rendered outline fits comfortably in context. For longer-form stages downstream (Stage 4 prose), state tracking may need to become explicit. At the structural granularity Stage 2 operates at, the LLM's context window is sufficient.

Thematic continuity is handled by motif threads (below). Entity-level consistency (who met whom, what's been said) emerges from the DAG rendering and is checked by judges during pairwise comparison via the Structural Coherence dimension. If empirical data shows beats systematically contradicting earlier ones, explicit entity extraction is a v2 addition.

### Motif threads

A small set (2-3) of recurring elements — imagery, objects, phrases, character obsessions — extracted from the concept at seed time. Motif threads provide thematic continuity across beats without requiring full entity/world state tracking. Adapted from Caves of Qud's domain-threading mechanism (Grinblat & Bucklew 2017): shared properties recurring across events produce apparent thematic unity without requiring planned narrative arcs.

Without explicit motif threading, LLMs generating beats in isolation reliably drop thematic continuity — the opening mentions stones, the later beats forget they exist. With motif_threads in the genome and referenced in the expansion prompt (the prompt instructs the LLM to favor actions that reference existing threads where natural), the LLM keeps the concept's load-bearing imagery present across the DAG. Thematic recurrence becomes a structural feature, not an emergent accident. No UCB bonus is applied — the prompt-side injection is the whole mechanism in v1 (see `mcts.md` §Motif Threads in MCTS Expansion for the rationale).

Extracted by a cheap LLM call at seed time alongside the climax:
- 2-3 specific, concrete threads (not abstract themes like "mortality")
- Drawn from the concept's premise, character seeds, and target effect
- Grounded in physical or linguistic specifics the LLM can actually thread through prose

For the Lottery concept: stones, the black box, Old Man Warner's muttering. For Hills Like White Elephants: the hills, the drinks, the train whistle, the thing-not-named.

Motif threads are informational, not prescriptive. A DAG may reference all threads, some, or occasionally invent new recurring elements the judges recognize as structural. They bias the search without forcing it.

### Character arcs

A whole-story character trajectory — agent, goal, and stakes. Character arcs are the genome's place for spanning intentions that can't be cleanly attached to a single source/target node pair.

```yaml
character_arcs:
  - agent: "the grandmother"
    goal: "to be seen as a lady"
    stakes: "her self-understanding; the terms on which her life has been lived"
```

**Why arcs are separate from motivates edges.** A `motivates` edge encodes a *local* intention installation — a specific prior event installs a specific goal that anchors an immediately-downstream action sequence. A character arc encodes a *spanning* trajectory — a goal present across many beats, not caused at one node and resolved at another. Forcing a spanning arc into an edge pair required choosing arbitrary endpoints; the arc object removes the arbitrariness.

**No `touches` list.** An earlier draft carried a `touches: [node_ids]` field enumerating where the arc was active. Dropped because it's almost always tautological — when the protagonist's arc touches ~N of N nodes, the field carries no information. Where a character is active is derivable from beat sketches (Tier 1 entity extraction already does this) and from `motivates[].agent` references; no need to duplicate it on the arc.

**Scope guideline.** If an intention arcs across ≥3 non-adjacent nodes, use `character_arcs`. If it installs at node A and anchors action through nodes B, C (adjacent to A), use a `motivates` edge. Local installations and whole-story arcs can coexist on the same character — the grandmother has a lifetime goal of being seen as a lady (arc) and also adopts a local goal of persuading Bailey to take the dirt road (edge).

**Validation (Tier 1).** Every `agent` in `character_arcs` must match exactly one canonical form across the genome (same name or description in beat sketches where the character appears, same form in all `motivates` edges that reference them). Goals must be specific (not "she had feelings") — same bar as motivates.

Character arcs are optional — many concepts don't have strong individual trajectories (Jackson's "The Lottery" is largely ensemble; a thought-experiment concept may have no character arcs at all).

### Concept demands

A small list of free-text predicates the DAG must satisfy — one sentence each, naming a non-negotiable structural element of the concept that the existing schema (roles, edge types, motif modes, character_arcs, story_constraints) can't directly express.

```yaml
concept_demands:
  - "the DAG must include at least one beat that addresses the reader directly, breaking the diegetic third-person frame"
  - "the form's recursion must be enacted structurally — the closing beat must mirror or invert the opening's premise as a structural rhyme, not only as a thematic claim"
```

**Why this field exists.** Some concepts hinge on a structural move that doesn't map onto {role, edge type, motif mode}: reader-address (the text implicates the reader as participant), form-as-device (the story's form *is* its meaning), ambiguity-as-design (deliberate irresolution), dialetheic structure (something simultaneously true and false). Without an explicit predicate, the MCTS reward signal evaluates 8 pairwise dimensions but cannot detect that the concept's central mechanism is missing entirely — two DAGs that *both* miss the demanded element compare cleanly while being collectively inadequate. Concept demands are checked at Tier 3 (see `evaluation.md`).

**Where they come from.** Extracted by `seed_root` (see `operators.md`) from the concept genome at the start of MCTS, in the same LLM call that produces `motif_threads`. Stage 1 is unchanged; the field is computed once per concept at MCTS-seeding time and shared across all 4 preset trees for that concept. If the extraction call fails or the prompt determines the concept's mechanism is fully schema-expressible, `concept_demands: []` (empty) is the correct output and Tier 3 is skipped for that concept.

**Why not evolved in Stage 1.** Three reasons. (1) Motif threads use the same derive-once-from-concept pattern and work fine — adding a parallel mechanism for demands keeps Stage 1's surface area stable. (2) Demands evolved by Stage 1 would risk the LLM generating demands it thinks it can satisfy, biasing toward schema-expressible structures. (3) If extraction proves unreliable in pilot, the right escalation is moving demands to Stage 1 evolution (v1.5) — not adding intermediate post-processing steps.

**When this field is empty.** Most stories whose mechanisms are conventional (climax-driven, character-arc-driven, structurally legible) won't need demands — the existing schema carries the load. Of the rev-6 canonical DAGs, only Chiang plausibly needs a demand (the second-person frame narrative is structurally load-bearing in a way disclosure edges only partially encode). The other three canonicals have empty demands. Empty is informative — it means "the schema is sufficient for this concept."

**Validation (Tier 3).** Each demand is evaluated as `satisfied | partial | failed` once per preset's final terminal DAG. See `evaluation.md` §Tier 3.

### Story constraints

Diegetic rules that hold across the whole story, parallel to character_arcs for motivations. Constraint edges encode local forecloseures (A specifically prevents a capacity at adjacent B); story constraints encode the rule that holds story-wide.

```yaml
story_constraints:
  - prohibits: "naming the pregnancy or the operation in direct terms; the register set by the opening's figurative substitution holds across the entire exchange"
    lifts_at: null               # rule never lifts; null = holds to story end
  - prohibits: "any villager speaking the lottery's lethal purpose aloud; naming it would rupture the ritual's performance"
    lifts_at: "stoning"          # rule collapses at this node
```

**Why constraints need a story-scoped form.** A `constraint` edge points from an installing node A to a constrained node B, with the rule holding between them. For a rule that holds across the entire story (Hemingway's not-naming, Jackson's ritual silence), the edge endpoints would be arbitrary — why `arrival → fracture` and not `arrival → departure`? This is the same scope-mismatch we solved for character motivations: some rules are story-scoped, not pairwise. Story constraints let the genome record the rule once with its scope (whole story until `lifts_at`, or permanently) rather than manufacturing endpoints.

**`lifts_at` semantics.** The node ID at which the rule breaks — often the climax or a late-story beat where the constraint's rupture is itself the dramatic event. `null` means the rule holds to the end (Hemingway: the not-naming never lifts; the final "I feel fine" is a refusal of naming rather than a breaking of silence).

**When to use edge vs. story constraint.** An edge-level `constraint` applies when the prohibition is installed at a specific beat and operates on an adjacent one — e.g., the opening of an Agatha Christie drawing-room establishes "the room is locked," which constrains the next beat's action. A story constraint applies when the rule holds wholesale — e.g., Hemingway's characters never name the thing, from first line to last.

**Validation (Tier 1).** Each story constraint's `lifts_at` must be a real node ID or null. The prohibition must be specific ("no naming the pregnancy," not "characters avoid difficult topics"). Tier 2 plausibility checks that no beat before `lifts_at` violates the rule — a beat where a character *does* name the forbidden thing (before the rule lifts) flags as an inconsistency.

### Nodes

A node is a story beat. It carries a minimal free-text sketch of 1–2 sentences describing what happens in that beat. Sketches are *specific and evocative* (the `Beat Quality` evaluation dimension judges this), not generic placeholders. An LLM default sketch like "she confronts him" fails; a sketch like "she asks whether the elephants looked like clouds" plants material Stage 4 can build on.

The optional `role` field (a list, not a single value) captures narrative functions that have no forward-pointing edge analog. Stage 2 accepts the same three roles Stage 1 evolves:

- `climax` — the node carrying the story's peak emotional intensity (the anchor for most conflict-driven concepts; required for bidirectional MCTS when present, see `mcts.md`)
- `reveal` — the beat that delivers the withheld information the structure has been managing around (the anchor for reveal-driven stories; also a structural marker for Stage 4 to treat with care)
- `pivot` — the kishotenketsu "ten" beat that recontextualizes preceding beats (Japanese/East Asian non-conflict structure)
- `null` — most beats carry no explicit role; their function is expressed by their incoming and outgoing edges

**Multi-role nodes.** `role` is a list to allow a single beat to carry more than one narrative function. The canonical case is O'Connor's grace moment — it's both the climax (peak dramatic intensity) and the pivot (the touch recontextualizes the grandmother's whole life retroactively). Forcing a single role drops one of the two structural facts. A list of `["climax", "pivot"]` records both.

**Primary role.** The first entry in the list is the *primary* role — used for MCTS phase dispatch (which role's edge-type restrictions apply in the forward phase), for Stage 4 prose treatment (which role's craft expectations to honor most), and as the surface slot in the Stage 1 handoff. Secondary roles (subsequent list entries) are read by judges during rubric evaluation but do not change MCTS behavior. Convention: order roles by how load-bearing they are to the concept — climax for conflict-driven O'Connor goes first; pivot (the reframing) second.

**Stage 1 handoff.** Stage 1's `concept.anchor_scene.role` is a single string (Stage 1 evolves one canonical role). `seed_root` wraps it as a one-element list (`[concept.anchor_scene.role]`) when setting the anchor node's role. Additional roles, if warranted, are added during MCTS expansion via `rewrite_beat`. See `lab/issues/2026-04-22-anchor-scene-in-stage-1-genome.md` for the anchor-from-Stage-1 rationale.

### Per-node motifs

Each node carries a `motifs` field listing which motif threads surface at this beat *and how*. Each entry is a `{motif, mode}` pair: `motif` is a string that must appear in the genome's top-level `motif_threads`; `mode` is one of the six values defined in the **mode glossary** below: `introduced | embodied | performed | agent | echoed | inverted`.

```yaml
nodes:
  - id: "pivot_perception"
    sketch: "Louise, immersed in Heptapod B, begins to experience time as the heptapods do..."
    role: ["pivot"]
    motifs:
      - {motif: "the heptapod script, drawn all at once", mode: "embodied"}
      - {motif: "the variational principle (Fermat's path of least time)", mode: "embodied"}
```

#### Mode glossary (authoritative)

The six modes below are the complete vocabulary. **This is the source of truth** — operators, judges, validators, and canonical fixtures must use these strings exactly and these definitions exactly. If the vocabulary needs to change, change it here first, then propagate.

| Mode | Definition (one sentence) | Discriminating example | Discriminator vs. nearest neighbor |
|---|---|---|---|
| `introduced` | The motif first appears, named or shown directly; the reader meets it here. | "Mr. Summers arrives with the battered black box." | vs. `embodied`: introduced uses the motif as something the text *names or points to*; embodied means the motif's shape is present *without* being named. |
| `embodied` | The motif's *shape* is present in the beat's structure, register, or character interior, without being named, handled, or causally invoked. | Louise's cognition at `pivot_perception` has the variational-principle shape; the prose doesn't say "she thinks variationally," she just does. | vs. `introduced`: embodied is unnamed shape; introduced is named or shown. vs. `performed`: embodied has no character handling; performed requires gesture. |
| `performed` | A character handles, wears, gestures with, or physically executes something with the motif-object. The interaction is gestural; the object's *meaning* may or may not be activated. | Misfit cleaning his dark glasses at `the_statement` (gesture without activating opacity-meaning); grandmother dressed in her lace collar at `departure` (gestural use of the motif as status-argument). | vs. `agent`: performed is what a *character* does *with* the motif; agent is what the *motif itself* does to events. |
| `agent` | The motif (object, shape, or phrase) plays a *causal* role in the beat — it's the proximate cause of what happens. | Pitty Sing's startle causes the crash at `crash` — the cat-motif is the agent of the accident. | vs. `performed`: agent's causal force comes from the motif itself; performed's force comes from the character handling it. The train arriving is causal *of the train*; the divergence-geometry-motif being performed is what the American does by crossing tracks. |
| `echoed` | The motif returns in a later beat with its valence *preserved* — reinforces, quotes, varies. | "You're a good man, I know you are" (`grandmother_pleads`) echoes "a good man is hard to find" (`diner`) — same valence, the title-phrase reused as personal claim. | vs. `inverted`: echoed keeps the motif's truth-valence intact; inverted reverses it. |
| `inverted` | The motif returns with its valence *reversed* — same object, phrase, or shape, but negated, counterfactual, or stripped of its earlier meaning. | "She would of been a good woman" (`the_statement`) inverts the title-phrase via counterfactual. "The old lady" (`no_pleasure`) strips the lady-self-presentation the lace-collar motif has been building. | vs. `echoed`: inverted reverses what echoed preserves. |

**Coverage strings should be derived from tags, not hand-written.** When describing a DAG's motif coverage, list modes by reading off the actual `Node.motifs[].mode` values, not by paraphrasing memory of the genome. Mismatches between commentary and tags are silent drift bugs.

**The taxonomy is closed at six.** Earlier drafts used a single `enacted` label that conflated three jobs (gestural, causal, embodying-shape); rev 5 split it. If a beat resists tagging under all six, the right move is to argue the case before adding a seventh mode — not silently invent one.

**Return-mode temporal sanity.** A motif tagged `echoed` or `inverted` at node N must have at least one earlier-in-topological-order node where it's tagged `introduced`, `embodied`, `performed`, or `agent`. Returning without a prior appearance is a temporal inconsistency flagged at Tier 1.

**The distinctions are structural, not aesthetic.** `performed` vs. `agent` is about what the motif is *doing* in the beat — being handled, or causing — not about the prose's tone. `embodied` vs. `introduced` is about whether the motif is named or merely shaped, not about how pretty the sentence is. Judges score the structural fact; prose craft is Stage 4's concern.

**Why typed and not a flat list.** A flat `motifs: ["stones"]` tag tells the generator the motif should be *present* but not *how*. Without the mode, generators default to `introduced` (naming) at every appearance, which flattens the story — every beat re-introduces the motif. Typed mentions let the generator distinguish "show the stones one last time" (`echoed`) from "the villagers are picking up stones to throw" (`performed`). This distinction is also what judges need to score motif work: a DAG that marks a motif as `performed` but the beat sketch only names it has a structural-surface mismatch worth flagging.

**Why per-node and not a top-level mapping.** Motifs-as-strings in the top-level `motif_threads` list declare the inventory; per-node typed mentions declare the surfacings with mode. This co-locates the structural fact (which motifs are present at this beat, and how) with the beat it modifies, keeps the genome traversable left-to-right, and avoids a separate `motif_appearances` mapping that duplicates signal across two places.

**Why per-node motifs carry thematic rhyme.** When thematic rhyme between two beats isn't captured by an edge (because it isn't an in-world logical consequence — that would be `implication` — and isn't a causal relation — that would be `causal`), it's carried by shared motif presence. For Chiang: `first_script` (`embodied`) and `physicist` (`introduced`) both touch "the variational principle" motif under different modes — the physicist introduces the principle by name, the script has already been embodying it. The rhyme is the shared motif plus the mode-shift, not an edge.

**Validation.** Every `motif` string must exactly match an entry in the genome's `motif_threads` (no typos, no drift). Every `mode` must be one of the four permitted values. Nodes may have zero motifs (many do); there's no requirement that every motif appears in every beat, nor that every beat carries at least one motif.

The kishotenketsu "ketsu" (resolution that reveals the connection made by the pivot) is intentionally not a role in v1. The pivot anchor is the role-marked node; the ketsu emerges as a downstream beat whose semantics are carried by the `reframes`/`entails` edge payloads flowing into it, not by a flag. If pilot data shows pivot-anchored DAGs need a distinct "payoff-after-the-twist" marker for judges to evaluate well, add it in v1.5 with a clear assignment mechanism.

Why not put state effects or preconditions on nodes (storylet-style)? Because the typed-edge DAG already encodes state transitions at a higher granularity. A `causal` edge with its `realizes` field says what state change occurred. A `disclosure` edge's `withheld` field says what epistemic state shifted. Duplicating this information on nodes risks conflicts between node-state and edge-semantics; letting edges carry it keeps one source of truth. The edge-driven decision is documented in the scoping issue §3; the short version is that this aligns with our typed-edge taxonomy philosophy and avoids over-specifying nodes the way NGEP's dependency-parsed argument schemas did.

### Edges

An edge is a typed relationship between two nodes. Each edge carries a type and a payload of fields specific to that type. **Every edge must have its payload fields populated** — empty fields cause edges to become decorative labels that the downstream prose generator will ignore (a failure mode documented in the typed-edges research report).

| Type | When to use | Required fields |
|---|---|---|
| `causal` | A's world-state change enables B | `realizes`: what specifically is caused or enabled |
| `disclosure` | B reveals information that reframes A. Audience can be the reader, a specific character, or both | `reframes`: what about A is recontextualized; `withheld`: what was hidden; `disclosed_to`: list of recipients — `"reader"` for authorial reveal, character names for diegetic recognition (defaults to `["reader"]` if omitted) |
| `implication` | B follows from A by **in-world logical entailment** — given A's premises, B must hold. Not authorial thematic rhyme (that's motif-carried) | `entails`: the specific proposition B must embody as a logical consequence of A |
| `constraint` | A forecloses an option or capacity at B **in the world** (diegetic prohibition; what characters/physics/social rules prevent) | `prohibits`: the specific thing A prevents B from doing |
| `motivates` | A installs a character intention that anchors B's action sequence. **Local scope**: A and B should be near-adjacent. Whole-story arcs go in `character_arcs`, not here. | `agent`: which character; `goal`: the intention adopted; `stakes` (optional): what is risked |

**disclosure vs. constraint.** Disclosure is an *audience-facing reframe* — what someone (reader, character, or both) learns that changes how prior beats are understood. Constraint is an in-world prohibition — what the *diegesis* (characters, physics, social rules) prevents. These often coincide (Jackson's villagers don't name the lottery's purpose — a social constraint — and this also keeps it withheld from the reader — a disclosure) but should be encoded separately: the constraint edge names the in-world rule, the disclosure edge names the reframe (use `disclosed_to` to distinguish reader vs. character audiences). Encoding both is correct, not redundant.

**disclosure audience matters.** A disclosure where `disclosed_to: ["reader"]` is authorial shape — the text's architecture of what the reader is trusted to hold. A disclosure where `disclosed_to: ["the grandmother"]` is diegetic — a moment of realization that drives the character's next action. Both-at-once is common and load-bearing (O'Connor's `opening → arrival`: the reader learns the Misfit has arrived AND the grandmother recognizes him from the newspaper — and her recognition is what seals the family's fate). Downstream prose treats character-disclosures differently from reader-disclosures: the first must be dramatized (a visible moment of realization), the second can be purely authorial (the reader's understanding shifts without any character noticing). Single-audience disclosures silently collapse this structural distinction; use `disclosed_to` to keep it.

**implication vs. motif rhyme.** `implication` is strict in-world entailment: a character could reason from A to B. Thematic rhyme — the story arguing that A and B resonate — is *not* implication; it's carried by shared motif presence (`motif_threads`). If two beats both touch the same motif, the rhyme is in the rendering, not an edge.

Edges are directional — `src → dst`. The DAG property is enforced at validation: edges cannot form cycles in the topological ordering. Disclosure and motivates edges often skip adjacency (e.g., disclosure edge from the opening directly to the climax), which is legal as long as it doesn't create a cycle.

### Why this schema resists downstream bias

The node/edge schema instills specific priors that shape what Stage 3 and Stage 4 can produce:

- **Edge fields are load-bearing** → Stage 4 must honor them in prose; a `realizes` field is a constraint the scene must actually enact. Crucially, "honor" means *enact*, not *state*: if Stage 4 receives `entails: X` verbatim, Claude will write *"X happened"* as narration. The Stage 4 prompt builder must wrap every payload per `docs/prompting-guide.md` §"Goals Cannot Ride in Plaintext." The Stage 4 *Payload enactment (not gesture)* evaluation criterion (in `docs/stages.md` §"Stage 4") catches scenes that only gesture at a payload.
- **Role flags on nodes are explicit** → Stage 4 treats `climax` and `reveal` differently from ordinary beats (prose rhythm, sentence-level pacing).
- **Beat sketches are specific** → Stage 4 has concrete anchors, not abstract slots; specific beats seed specific prose.
- **DAG structure is the plan** → Stage 4 follows topological order by default but may re-enter Stage 2 for adaptation (see §Handoff to Stage 4).

See `evaluation.md` for how these priors are enforced at judging time.

---

## Pacing Presets

A single concept generates 3–4 parallel MCTS trees in Stage 2, one per pacing preset. Each preset is a pacing philosophy expressed as an expansion-prompt hint; the four hints bias four parallel MCTS trees toward distinct pacing shapes.

The presets are inspired by RimWorld's AI Storyteller system (Cassandra Classic / Randy Random / Phoebe Chillax) and extended with a discrete-wave variant (Winston Waves) that the community mod ecosystem suggests is more appropriate for short-form narrative. Crucially, these presets are **not fundamental dramatic philosophies** — research into RimWorld's system found that the three storytellers share the same underlying wealth-to-threat formula and differ only in scheduling parameters. Our presets inherit that insight: they are configurations of four primitives, not hand-designed curves.

### Where presets enter the pipeline (v1: semantic presets)

Since the anchor is preset-agnostic (see §Bidirectional Anchor-First Expansion), presets influence MCTS in **one place** in v1: the expansion prompt. The LLM proposing candidate actions receives a preset-specific pacing hint, so the candidate pool is already tilted toward the preset's philosophy. Cache consumption, UCB selection, and validation are all preset-agnostic — standard UCB1 with c=0.5 on pairwise reward.

The 4 trees share a root; they diverge entirely through what the expansion LLM proposes under each preset's hint. An earlier design added a tension-inferred `preset_prior` bonus to UCB that read numerically from each preset's primitives; that mechanism was deferred to v1.5 in favor of semantic presets to avoid building tension-inference infrastructure before confirming it's needed (see `lab/issues/2026-04-19-stage-2-third-review-followups.md` Item 2).

### Pacing primitives (descriptive, not numerical in v1)

Each preset's philosophy is characterized by four parameters. In v1 these inform the expansion-hint language and the character column — they are **not read numerically by the search algorithm**. v1.5 may restore parametric priors if semantic divergence proves insufficient.

| Parameter | Meaning |
|---|---|
| `min_rest_beats` | Minimum low-tension beats required between high-tension beat clusters |
| `max_flat_beats` | Maximum consecutive low-tension beats before escalation |
| `intensity_variance` | Distribution width (in `{tight, wide}`) for the tension level of high-tension beats |
| `recovery_required` | Whether the DAG includes an explicit recovery beat after the climax |

### Preset values (descriptive) and expansion hints

| Preset | `min_rest_beats` | `max_flat_beats` | `intensity_variance` | `recovery_required` | Character |
|---|---|---|---|---|---|
| Cassandra-ish | 1 | 4 | tight | true | Escalating peaks with guaranteed relief; high regularity |
| Phoebe-ish | 3 | 6 | tight (soft peaks) | true | Long recoveries; softer peaks; "benevolent" alternation |
| Randy-ish | 0 | 8 | wide | false | Stochastic, no forced recovery, high variance in peak intensity |
| Winston-ish | 2 | 3 | tight | true (explicit reward) | Discrete numbered waves; each climax followed by a clear reward beat |

Randy is *not* uniform random — its underlying RimWorld implementation uses weighted category selection. Our Randy-ish preset inherits that: no forced escalation cadence, but coherence-biased.

Each preset's expansion-prompt hint (substituted via `{PACING_HINT}` in `operators.md` §Base System Message):

- **Cassandra-ish:** "Favor actions that establish rise-then-relief patterns. After each high-tension beat, the next move should offer breathing room before the story escalates again. Escalating peaks, guaranteed recovery."
- **Phoebe-ish:** "Favor long recoveries and softer peaks. After high-tension beats, insert at least one contemplative beat before escalating. Prefer accumulated dread and slow intensification to sharp spikes."
- **Randy-ish:** "Tolerate variance. High-tension peaks may follow each other without recovery; low-intensity stretches can be long. Avoid a forced escalation cadence; let the story's logic set the rhythm."
- **Winston-ish:** "Structure tension as discrete waves. Each complication should resolve with an explicit reward beat before the next complication begins. Numbered stakes rather than continuous escalation."

These hints are the entire divergence mechanism in v1: same anchor, same DAG state, same validator, different prompt → different expansion cache → different tree.

### Why not commit to just one preset per run?

Different pacing shapes expose different structural possibilities. A Cassandra-ish search over a Chiang-mode concept may produce a disclosure-heavy escalation toward the conceptual payoff; a Phoebe-ish search over the same concept may produce a contemplative implication-chain. Both might be valid structures for the same concept. Running 3–4 presets per concept preserves this diversity at the cost of budget. Budget control lives at Stage 1's island count (§13 of scoping), not at Stage 2's preset count.

### Reward signal: tiered pairwise judging

MCTS rollout rewards come from pairwise comparison against the tree's running champion. To keep latency manageable across hundreds of terminal evaluations per concept, we use a **tiered judge** design: a single cheap judge running **both orderings in parallel** (no added latency) produces the rollout reward signal; the full 4-judge × 2-ordering panel only fires when the cheap judge declares a challenger wins, verifying before the champion is actually promoted. The full panel also runs for the within-concept tournament and QD archive insertions. Cheap judge must be from a different model family than the expansion model (cross-family discipline).

Dual-ordering on the cheap judge addresses the position-bias caveat: a challenger "wins" only if both orderings agree; disagreement collapses to tie. Position bias can no longer cleanly direct UCB down a wrong branch — it shows up as increased tie rate on close calls, which UCB handles as noise, not misdirection.

Narrative forecasting (adapted from *Spoiler Alert*, arXiv:2604.09854) is retained as optional metadata — a structural-unpredictability measure useful as a tiebreaker or for diagnostic logs — but no longer participates as a rollout reward component. The earlier 0.9/0.1 hybrid weighting was removed pre-implementation (2026-04-19) because forecasting-only rollout signal systematically discriminated against constraint-heavy low-tension architectures (Hemingway-mode), which the rubric explicitly praises. Full details in `mcts.md`.

---

## How Stage 2 Differs From Stage 1

Stage 2 runs on a different engine than Stage 1, produces a different genome, and has different selection mechanics. What carries over is the judging infrastructure and the pairwise-comparison philosophy.

**On engine heterogeneity across the pipeline.** The "6-stage evolutionary pipeline" framing describes the *shape* of selection pressure, not a single evolutionary substrate. Stage 1 uses ShinkaEvolve (async population, islands, champion succession). Stage 2 uses per-concept MCTS. Stages 3–6 are likely to introduce further engines. The binding infrastructure is the judge panel + pairwise protocol + compost heap — not a shared search algorithm. This is intentional: each stage's genome (concept vs. DAG vs. voice vs. prose) has different search topology, and forcing one engine across them would mean the wrong tool at 4 out of 6 stages. The engineering cost is real (more modules, more mental models) and should be weighed explicitly when designing Stages 3+.

| Aspect | Stage 1 | Stage 2 |
|---|---|---|
| Engine | ShinkaEvolve (async evolution with islands) | MCTS (per-concept, one tree per pacing preset) |
| Population | Population of concepts across islands | Per-concept, per-preset MCTS tree + QD archive |
| Selection | Pairwise champion succession within islands + Swiss tournament | UCB1 inside each tree; within-concept tournament across preset winners |
| Genome | Concept JSON (premise, target effect, optional fields) | Typed-edge DAG JSON (nodes + edges with payload fields) |
| Seed bank | Yes — curated seeds drive initial concepts | No — the Stage 1 concept IS the seed |
| Islands | Yes — drive diversity | No — diversity comes from presets + QD archive |
| Compost heap | Accumulation archive across runs | Shared with Stage 1 (structural fragments feed the same compost) |
| Classifier | MAP-Elites classifier disabled; tonal register + literary mode as metadata only | Structural classifier (disclosure-ratio × structural-density) for QD archive |
| Evaluation | 9-dimension per-criterion pairwise | 8-dimension per-criterion pairwise (different dimensions) |
| Judges | 4 contest personas, cross-family models | Same 4 personas (for v1), same cross-family setup |
| Typical cost | $5–$10 per light run | TBD (pilot); provisional ceiling $30–$48 / $50–$80 / $80–$192 for light/medium/heavy on top of Stage 1 |

### What carries over

- **LLM client infrastructure** (`owtn/llm/`): bandit ensemble, caching, pricing, prompt-caching machinery
- **Judge personas** (`configs/judges/*.yaml`): same 4 contest judges (Gwern, Roon, Alexander Wales, Jamie Wahls) — the Un-Slop Prize panel active in Stage 1; their aesthetics are submission-focused, not stage-specific. Translation concerns per judge are in `judges.md` §Persona-Adaptation Concerns Specific to Stage 2.
- **Pairwise protocol** (`owtn/evaluation/pairwise.py`): 4 judges × 2 orderings × per-criterion voting is the same mechanism
- **Swiss tournament** (`owtn/stage_1/tournament.py`): reused for within-concept tournaments in Stage 2
- **Compost heap**: structural fragments from Stage 2 (interesting beat sketches, productive edge patterns, failed graphs with a spark) feed the same `compost` SQLite table that Stage 1 writes to

### What does not carry over

- **ShinkaEvolve**. The async evolution engine is Stage 1's. Stage 2 uses a different engine because MCTS's tree search structure doesn't map onto ShinkaEvolve's population-based loop.
- **Islands**. Stage 2's per-concept runs are the analog; pacing presets provide the analog of island diversity.
- **Seed bank**. Stage 2 has no seed bank — the concept provides the seed.
- **MAP-Elites concept-type classifier**. Stage 2 has its own QD axes (disclosure-ratio × structural-density) computed from the finished DAG, not classified by LLM.

---

## Handoff From Stage 1

Stage 2 receives for each advancing concept:

1. **Complete concept genome** — all Stage 1 fields: premise, `anchor_scene` (sketch + role, required), thematic engine, target emotional effect, character seeds, setting seeds, constraints, style hint.
2. **Anchor scene** — `concept.anchor_scene.sketch` is the structural anchor (consumed verbatim by `seed_root` as the root node); `concept.anchor_scene.role` is `"climax" | "reveal" | "pivot"` and governs which phase's edge-type restrictions apply (see `mcts.md` §Bidirectional Phases). The anchor was evolved and tournament-selected in Stage 1 rather than generated by an operator here — see `lab/issues/2026-04-22-anchor-scene-in-stage-1-genome.md`.
3. **Judge panel scores** — per-dimension scores from Stage 1's final evaluation.
4. **Judge reasoning chains** — natural-language explanations of what makes this concept work and what risks it carries. These are the most valuable narrative Stage 1 signal for Stage 2: they tell MCTS what to emphasize and what to watch out for.
5. **Auto-detected structural affinities** — inferred concept type (thought experiment / situation with reveal / voice constraint / character collision / atmospheric / constraint-driven / kishotenketsu) and suggested primary edge affinity. These are *hints*, not *constraints*. A concept auto-classified as "situation with reveal" may be better served by a structure with more motivates edges than disclosure edges; Stage 2 is free to discover that.
6. **Identified risks** — "reveal is powerful but setup needs to earn it"; "concept is strong but may not sustain 3,000 words." These shape MCTS reward weighting and the pacing preset selection.
7. **Diversity metadata** — affective register, literary mode. Carried through for Stage 3 voice and archival purposes; Stage 2 doesn't condition on these directly.
**Note: `concept_demands` is *not* a Stage 1 handoff field.** Concept demands are extracted from the concept by Stage 2's `seed_root` operator alongside `motif_threads` — same pattern, same prompt locale. See §Concept demands below for the rationale and `operators.md` §seed_root for the extraction call. This puts demand derivation co-located with motif derivation and avoids a Stage 1 schema change in v1.

### How Stage 2 uses this

The concept is injected into the root MCTS state. The MCTS expansion prompt includes:
- Premise + target effect (always)
- Character seeds if present
- Thematic tension if present
- Constraints if present
- Judge reasoning chains (compressed for prompt economy; see `mcts.md`)

**Goal-field wrapping.** `target_effect` and `thematic_engine` are narrative goal fields — if passed verbatim into the expansion prompt, the LLM will literalize them into beat sketches (*"she felt dread"*, *"hope is a discipline, not a feeling"*). These fields must be wrapped with explicit instruction not to name them directly; see `docs/prompting-guide.md` §"Goals Cannot Ride in Plaintext" for the wrapping pattern. This applies to the MCTS expansion prompt and any downstream prompt that consumes these fields.

**All Stage 1 winners advance** (see scoping §13). There is no top-K filter between stages. If a Stage 1 concept is too weak for Stage 2 to find a good structure, that signal emerges from Stage 2's pairwise judges — not from a filter that discards concepts prematurely.

---

## Handoff To Stage 3

Stage 3 (voice evolution) receives for each advancing concept-structure pair:

1. **The complete structure DAG** — all nodes, edges, payload fields.
2. **Tournament ranking** — the within-concept tournament result that determined advancement (1st/2nd/… among presets).
3. **QD archive metadata** — which (disclosure_ratio, structural_density) cell the structure lives in.
4. **Pacing preset label** — which preset generated this structure (Cassandra/Phoebe/Randy/Winston-ish).
5. **Forwarded Stage 1 concept** — Stage 3 needs this to evaluate voice appropriateness.
6. **Adaptation permissions** — declarative rules for when Stage 4 may re-enter Stage 2 to request structural mutation (see below). Stage 3 (voice) does not currently need structural re-entry; all documented use-cases are Stage 4 (prose) phenomena.

### How many structures advance

Per-concept top-K, configurable:

- `light.yaml`: K=1 (one structure per concept; minimum downstream cost)
- `medium.yaml`: K=2 (preserves the largest within-concept diversity gap)
- `heavy.yaml`: K=all (maximum structural diversity to Stage 3)

Non-advancing DAGs are archived in the QD grid and may seed future runs via the compost heap. Their structural ideas are not lost.

### Adaptation permissions

**v1 status: named only.** The `adaptation_permissions` field is present on the handoff manifest for forward compatibility, but no re-entry mechanics ship in v1. Stage 4 will define the triggers and the mutation-response mechanism. This keeps the handoff schema stable while deferring design work that requires Stage 4 context.

The default contract (applies in v1): Stage 3 evolves voice for the given structure; Stage 4 generates prose per node in topological order. No structural changes.

**Re-entry cases documented for v1.5 / Stage 4:**

1. **Prose discovers a better turn.** A node's prose reveals that the following beat should be different — a character's revealed interiority doesn't match the motivates edge payload, or a disclosure's withheld fact gets resolved more naturally a beat earlier.
2. **State consistency fails.** The realized prose for a node contradicts a downstream node's expected preconditions (e.g., "opening" prose shows the characters never meeting, but a "motivates" edge assumes they know each other).
3. **Dead-scene detection.** A reader-simulation check flags a node as non-advancing (doesn't change the reader's emotional/informational state). This triggers a `rewrite_beat` or `add_edge` mutation request.

The **mechanics** of re-entry — whether it's a full MCTS restart, a local mutation, or a shallow regeneration — is deferred to Stage 4 design. What Stage 2 commits to: the DAG artifact is versioned, and re-entry produces a new version rather than mutating the handed-off artifact in place.

---

## Run Output

A completed Stage 2 run auto-exports its results:

```
results/run_<timestamp>/stage_2/
├── by_concept/
│   ├── c_a8f12e/
│   │   ├── winners/                 # top-K DAG genomes that advance
│   │   │   └── main.json
│   │   ├── tournament.json          # within-concept tournament results
│   │   ├── trees/                   # per-preset MCTS tree snapshots
│   │   │   ├── cassandra.json
│   │   │   ├── phoebe.json
│   │   │   ├── randy.json
│   │   │   └── winston.json
│   │   └── run.log
│   └── ...
├── qd_archive.json                  # global QD grid (disclosure_ratio × structural_density)
├── handoff_manifest.json            # what advances to Stage 3, with all metadata
└── evolution_run.log
```

### Winner format

Each `winners/main.json` contains the complete genome plus forwarded Stage 1 metadata:

```json
{
  "concept_id": "c_a8f12e",
  "preset": "cassandra_ish",
  "tournament_rank": 1,
  "qd_cell": {"disclosure_ratio": 2, "structural_density": 3},
  "genome": { "nodes": [...], "edges": [...] },
  "stage_1_concept": { ... },
  "stage_1_judge_reasoning": "...",
  "mcts_reward": 0.72,
  "adaptation_permissions": ["prose_discovers_turn", "state_contradiction", "dead_scene"]
}
```

### Re-export

`owtn export-structures <run_id>` re-exports Stage 2 results from the DB with optional overrides (different K, different selection criteria). The auto-export at run end uses the config defaults.

---

## Validation Protocol for Stage 2 Genomes

Every genome passes through three gates before reaching the judges:

### Gate 1: Schema validation

- `concept_id` references a valid Stage 1 winner
- `preset` is one of the known presets
- `nodes` array is non-empty; node count within the configured range for the target prose length
- All node `id` values are unique
- `edges` array has `src` and `dst` referencing existing node ids
- All edges have a valid `type` and all required fields for that type are non-empty

### Gate 2: Structural validation

- The DAG is acyclic in topological order
- Exactly one node has a non-null `role`, and it matches `concept.anchor_scene.role` (the anchor set by Stage 1 — one of `climax | reveal | pivot`)
- If `pacing_preset.recovery_required` is true, at least one beat exists downstream of the anchor
- Disclosure edges' `withheld` fields reference content not yet established at the source node
- Motivates edges' `agent` values are consistent (the same character named consistently throughout)

### Gate 3: Sketch specificity

- Node sketches exceed minimum specificity threshold (length + non-placeholder text check, same rules as Stage 1's Gate 1)

Genomes failing any gate are discarded at MCTS evaluation time; they produce a reward of 0 and do not propagate. See `operators.md` for how mutation operators handle validation failures.

**Note on edge-type diversity:** an earlier Gate 3 required "at least one edge of each pacing-preset-permitted type" to prevent all-causal degenerate graphs. Dropped 2026-04-19 (second review pass) because it misfired on legitimately-dominant-edge-type structures — a Hemingway-mode all-constraint DAG on a Hemingway concept would have been rejected despite being the right shape for the concept. Edge-type distribution is now judged semantically via the `Edge Logic` rubric dimension's Mode-match sub-criterion (see `rubric-anchors.md`), not gated at construction time.

---

## Bidirectional Anchor-First Expansion

Stage 2's MCTS draws its structural insight from BiT-MCTS (arXiv:2603.14410): the seed is a **single anchor beat**, not an opening/middle/resolution triple. Two sequential MCTS phases expand from this anchor plus a short cross-phase refinement pass; opening and resolution emerge from the search rather than being prescribed. The paper validates the bidirectional-from-climax design empirically — their ablations show removing bidirectional expansion or swapping phase order produces 97–100% loss rates. We adopt the structural design because the evidence is strong; we do not replicate the paper's specific reward function, output length, or genre focus.

**Seed**: `seed_root(concept)` wraps `concept.anchor_scene` as a single-node root DAG — the **structural anchor**, the single beat the whole story hinges on. For conflict-driven concepts this is the climax; for kishotenketsu it's the pivot; for reveal-driven stories it's the reveal beat. The role (`climax` / `reveal` / `pivot`) is carried on the genome — Stage 1 evolved and tournament-selected it; `seed_root` does not re-classify. **One seed per concept**, shared as the root of all pacing-preset MCTS trees. Preset divergence happens via expansion priors, not at the seed; the anchor is the story's central moment and its content shouldn't depend on pacing philosophy. See `operators.md` §seed_root and `lab/issues/2026-04-22-anchor-scene-in-stage-1-genome.md` for the handoff details.

Two MCTS phases run sequentially, both attached to the anchor, plus a short third refinement pass. Edges in the final DAG always point temporally forward (the topological arrow goes from earlier-in-story to later-in-story). What changes between phases is the *search direction* — where MCTS adds new nodes relative to the anchor:

- **Phase 1 (forward from anchor, 50 iterations):** the single-node seed wraps as the root MCTS state. Expansion adds new beats *downstream of the anchor* — new nodes become targets of new edges from the anchor (or from its descendants added earlier in the phase). Search direction: anchor → resolution (in story time). For climax-anchored concepts this produces the falling-action structure; for pivot-anchored kishotenketsu concepts this produces the ketsu (resolution that reveals the pivot's meaning). Permitted edge types: causal, implication.
- **Phase 2 (backward from anchor, 50 iterations):** the forward-phase winner wraps as the new MCTS root. Expansion adds new beats *upstream of the anchor* — new nodes become sources of new edges into the anchor or its ancestors added earlier in the phase. Search direction: anchor → opening (in story time). For climax-anchored concepts this produces the rising action; for pivot-anchored concepts this produces the ki/sho setup strands being juxtaposed. Permitted edge types: causal, constraint, disclosure, motivates. BiT-MCTS paper: "Generating the effect (falling action) before the cause (rising action) constrains antecedent generation and reduces incoherent setups."
- **Phase 3 (cross-phase refinement, ~5 iterations):** `add_edge`-only pass. Both endpoints may be anywhere in the DAG, subject to acyclicity and payload validation. This permits edges that span the anchor — most commonly disclosure edges from post-anchor resolution beats backward to opening beats (epilogue-reveal structures), or motivates edges that arc across the whole story. Without Phase 3, phase-local `add_edge` restrictions silently exclude these valid structural shapes. Precedent: BiT-MCTS's separate Outline Refinement stage.

The `motivates` edge type is permitted only in the backward phase (and Phase 3) because motivations must be *established* before the anchor, not added after. Disclosure edges are backward-phase or Phase-3 only.

**Opening and resolution are emergent**: after all three phases complete, the final DAG has some node with in-degree 0 (topologically first = the opening) and some node with out-degree 0 (topologically last = the resolution). These are just positional features of the finished DAG, not pre-seeded roles. A subsequent step (TBD for Stage 5) may craft explicit opening/closing beats that bookend the outline, analogous to BiT-MCTS's Outline Refinement stage.

Full algorithmic details — UCB selection, cached expansion, bounded simulation with early stopping, reward function, budget management — in `mcts.md`.

---

## Where to Look for More Detail

- **`mcts.md`** — MCTS algorithm, bidirectional expansion, action space, rollout strategy, reward function, budget.
- **`operators.md`** — `seed_root`, `add_beat`, `add_edge`, `rewrite_beat`, `post_hoc_rationalize` — prompts and validation.
- **`evaluation.md`** — DAG rendering for judges, per-criterion pairwise protocol, within-concept tournament, top-K advancement.
- **`rubric-anchors.md`** — 8 dimensions with sub-criteria, endpoint anchors, literary examples.
- **`judges.md`** — judge reuse question, persona adaptation, specialist vs. generalist debate.
- **`qd-archive.md`** — disclosure_ratio × structural_density grid, population rules, cross-run feedback.
- **`implementation.md`** — code structure, config schema, OWTN integration points.

---

## Open Questions Surfaced During Drafting

The following are flagged for resolution during implementation or in subsequent docs:

1. **State tracking across DAG traversal.** The current node schema has no explicit entity/world state. Two sequential beats could contradict each other (beat A establishes "they've never met"; beat C treats them as long-acquainted) without a shared state log catching it. Decision pending: do we add a `state_log` field to the genome, relying on the LLM's context window, or leave this as a Stage 4 concern?

2. **Motif threads.** Caves of Qud's post-hoc rationalization relies on "domain" threads — recurring properties (e.g., "ice", "cats") woven through events to produce thematic coherence without planned arcs. Should the genome carry a `motif_threads` field separate from Stage 1's `thematic_tension`? TBD in `mcts.md` where the post-hoc rationalization operator is specified.

3. **Stage 4 re-entry mechanics.** Adaptation permissions are named but the triggers and response mechanics aren't concrete. Needs co-design with Stage 4; stub until then.

4. **Tournament mechanics for 3- vs 4-preset brackets.** Swiss tournaments are designed for larger pools. With 4 entries, round-robin may be simpler and equally valid. Decide during implementation of the evaluation module.

5. **Narrative forecasting as user-facing metric.** After being removed from the rollout reward (2026-04-19), forecasting is now primarily a diagnostic / tiebreaker signal. Whether to surface it as a run-level metric (preset-by-preset scores, concept-type correlations) is open. Decide once pilot runs produce baseline distributions.
