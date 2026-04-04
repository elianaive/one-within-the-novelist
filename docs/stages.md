# Pipeline Stages

The system generates short stories (500–10,000 words) through a 6-stage
evolutionary pipeline. Each stage maintains a population of candidates, evolves
them under selection pressure from a diverse judge panel, and passes winners to
the next stage. The "genome" — what is being evolved — is different at each stage.

## Why Stages?

Eight years of research (ACL 2018 through NAACL 2025) consistently shows that
staged story generation outperforms end-to-end generation on coherence, plot
quality, and human preference. The gains come from separating *what to say* from
*how to structure it* from *how to say it* — each is a different optimization
problem with different fitness criteria.

But the standard academic pipeline (concept → outline → prose) makes a hidden
assumption: that stories are plot-driven, with characters serving the plot. Many
of the greatest short stories violate this. "The Lottery" is situation-driven.
"Story of Your Life" is concept-driven. "Hills Like White Elephants" is
voice-driven. Our pipeline must support all of these without forcing a mode.

---

## Overview

```
                    ┌──────────────────────────────────┐
                    │                                  │
                    ▼                                  │
  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
  │  1.     │   │  2.     │   │  3.     │   │  4.     │   │  5.     │   │  6.     │
  │ CONCEPT │──▶│STRUCTURE│──▶│  VOICE  │──▶│  PROSE  │──▶│ REFINE  │──▶│ SELECT  │
  │         │   │         │   │         │   │         │   │         │   │& ARCHIVE│
  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
       ▲                                                                      │
       │                                                                      │
       └──────────── archived seeds, episodic memory of failures ─────────────┘
```

Each stage runs its own evolve-evaluate loop. Winning candidates from one stage
become the starting population for the next. The judge panel operates at every
stage but weights different evaluation dimensions depending on what's being
evolved.

Cross-cutting: a co-evolving state (world, characters, voice, structure, prose)
propagates changes bidirectionally across stages 1–5, so a discovery in the prose
stage can feed back and reshape the structure.

---

## Stage 1: Concept Evolution

**What is being evolved:** Structured premises — the seed idea for a story.

**The genome:** A concept is not a template. It's a loose bundle:

- A "what if" or situation or collision or constraint
- A target emotional effect (Poe's unity of effect: name the feeling the reader
  should have at the end *before* generating anything)
- Optional character seeds (not required — concept stories may have none)
- Optional setting/world seeds
- Optional thematic concern

The key: no forced entry mode. A premise might be:

- *"What if learning an alien language changed how you perceive time?"*
  — a thought experiment (Le Guin/Chiang mode)
- *"A small town's cheerful annual tradition is actually a stoning."*
  — a situation with a reveal (Jackson mode)
- *"Two people at a train station discuss something they never name."*
  — a voice constraint (Hemingway mode)
- *"A grandmother who thinks she's a lady meets an escaped convict who thinks
  he's a philosopher."*
  — a character collision (O'Connor mode)

All of these are valid genomes. The system doesn't need to know which "type"
they are.

**Mutation operators:**

- **Collision** (King): take two premises from the population and combine them.
  "What if a telekinetic girl + high school bullying?" Both elements must
  survive; the story lives in their intersection.
- **Noun-list recombination** (Bradbury): generate 15–20 emotionally resonant
  nouns, find unexpected clusters, derive a premise from the tension between
  them. "THE LAKE. THE LOTTERY. THE SCISSORS. THE GRANDMOTHER."
- **Thought experiment** (Le Guin): take a philosophical or social premise and
  push it to its logical/emotional conclusion. "What if a utopia required one
  child to suffer?"
- **Compost recombination** (Gaiman): draw from an accumulation archive of
  fragments — images, overheard phrases, unfinished ideas from previous runs —
  and combine two that weren't originally related.
- **Crossover**: take the situation from one premise and the character seeds
  from another.
- **Inversion**: flip the premise's emotional valence, power dynamic, or
  expected outcome.

**Evaluation criteria:**

The judge panel scores concepts on:
- Originality (is this a premise we haven't seen?)
- Emotional potential (does this seed contain the energy for a powerful story?)
- Narrative tension (is there an inherent conflict, mystery, or question?)
- Thematic richness (does this connect to something that matters?)
- Feasibility (can this actually be realized in 500–10,000 words?)

**Population:** 30–60 diverse premises.

**Exit condition:** Top N premises, selected via MAP-Elites to ensure diversity
across the concept space, advance to Stage 2. Each winning concept may spawn
multiple structure variants in the next stage.

---

## Stage 2: Structure Evolution

**What is being evolved:** The story's structural plan — a typed-edge directed
acyclic graph (DAG).

**The genome:** A DAG where:

- **Nodes** are story units — a scene beat, a revelation, a conceptual step, a
  moment of constraint. Each node carries a payload describing what happens in
  that unit.
- **Edges** are typed relationships between units:

| Edge Type | Meaning | Example |
|---|---|---|
| **Causal** | A causes B | "She crashes the car" → "They're stranded on the road" |
| **Disclosure** | A reveals information that reframes B | "We see them drawing lots" → "The 'winner' is stoned" |
| **Implication** | A logically entails B | "Their writing is simultaneous" → "They perceive all time at once" |
| **Constraint** | A limits what can happen in B | "They never name the operation" → all dialogue must work around it |

Most stories use a mix of edge types. The proportions emerge naturally:

- A graph that's mostly causal edges → a plot-driven story
- A graph that's mostly disclosure edges → a reveal/twist story
- A graph that's mostly implication edges → a concept/thought-experiment story
- A graph that's mostly constraint edges → a voice/compression piece
- A graph with a rich mix → a hybrid (the best stories often are)

The system doesn't label or choose a mode. It evolves graphs, and the edge-type
proportions are an emergent property.

**Exploration methods:**

- **MCTS (Monte Carlo Tree Search):** Explore branching narrative paths. At each
  node, generate possible continuations, simulate forward (via LLM rollout),
  evaluate, backpropagate. More compute → deeper exploration of the structure
  space.
- **BiT-MCTS (climax-first):** Generate the story's most powerful moment first
  (the climax, the reveal, the conceptual punchline), then expand bidirectionally
  — backward to build rising action/setup, forward to build resolution/aftermath.
  This mirrors how many great writers actually work: start with the powerful
  moment, build around it. For a Hemingway-style story, the "climax" might be
  a single line of dialogue. For a Jackson story, it's the reveal.
- **L4D Director pacing:** Track narrative tension as a continuous variable
  across the graph. Gate which nodes can fire at each point:
  - **Build-up:** escalating stakes, introducing complications
  - **Sustained Peak:** maximum tension, the crisis moment
  - **Peak Fade:** tension easing, first signs of resolution
  - **Relaxed:** recovery, reflection, planting seeds for what's next
  Different "storyteller personalities" (Rimworld-inspired) could be population
  members: a Cassandra (escalating with guaranteed breaks), a Randy (high-variance
  surprise), a Phoebe (hardship balanced by reward).

**Evaluation criteria:**

- Causal/logical coherence (do the edges make sense?)
- Tension arc quality (does it build and release satisfyingly?)
- Surprise + post-dictability (are there genuine surprises that feel inevitable
  in retrospect?)
- Structural completeness (does the story feel whole?)
- Compression (for short stories: is every node earning its place? A 1,000-word
  story might need only 3–5 nodes)

**Exit condition:** Top N structure graphs per concept, selected for diversity
of edge-type proportions (ensuring the pipeline doesn't converge on only
plot-driven structures).

---

## Stage 3: Voice/Style Evolution

**What is being evolved:** A style specification — the *how* of the prose,
separated from the *what*.

**Why a separate stage:**

Le Guin's *Steering the Craft* argues that rhythm, syntax, and sound are the
*foundation* of prose quality — not decoration applied after the fact. The
O'Sullivan stylometry study (Nature 2025) showed that LLM outputs cluster tightly
by model: without explicit voice differentiation, every story generated by the
same model sounds the same regardless of concept or structure. And the
uncertainty gap research (Sui, ICML 2026) shows that human writing is 2–4x more
informationally surprising than LLM output — an idiosyncratic voice spec pushes
prose away from the model's generic default toward something more distinctive and
surprising.

Nous autonovel recognized this: their 5-layer system has Voice as an explicit
layer established *before* any chapter drafting begins.

**The genome:** A style specification containing:

- **Sentence rhythm:** target pattern — short/punchy (Carver), long/flowing
  (Woolf), mixed with deliberate variation (Munro), minimal (Davis)
- **Diction/register:** vocabulary level and flavor — spare and concrete
  (Hemingway), precise and clinical (Chiang), ornate and labyrinthine (Borges),
  colloquial and warm (Saunders)
- **POV and tense:** first/second/third person; past/present; omniscient/limited/
  objective. These are massive choices. Second-person present creates immediacy.
  Third-person omniscient past creates distance and authority. Each fundamentally
  shapes what the story can do.
- **Narrative distance:** close interiority (we're inside the character's head)
  vs. removed observer (documentary, almost journalistic) vs. something stranger
  (Borges' academic narrator, Kafka's flat matter-of-fact surrealism)
- **Voice personality:** the narrator's relationship to the material — ironic,
  earnest, detached, intimate, unreliable, clinical, conspiratorial
- **Constraints:** what this voice does NOT do. No adjective stacking. No
  exposition dumps. Dialogue never exceeds two sentences. No metaphors. Only
  concrete nouns. These negative constraints often define a voice more sharply
  than positive descriptions.
- **Style exemplars:** 2–3 short passages (from existing literature or generated)
  that demonstrate the target voice. A "this is what it should sound like" anchor.

**Mutation operators:**

- Shift register (more formal ↔ more colloquial)
- Swap POV or tense
- Add or remove a constraint
- Blend two voice specs (crossover: take rhythm from one, diction from another)
- Swap style exemplars
- Push toward a specific author's style then mutate away from it (use as
  launching point, not destination — avoid pastiche)

**Evaluation criteria:**

- **Distinctiveness:** does this voice sound different from generic LLM prose?
  (Measurable via stylometric distance from a baseline LLM output)
- **Internal consistency:** does the spec hold together? (A spare, Hemingway
  diction with ornate Borges sentence rhythms might clash — or might be
  interesting)
- **Appropriateness:** does this voice serve this concept and structure? (A
  clinical, detached voice for a grief story could be devastating or could be
  wrong — the judge panel evaluates this contextually)
- **Anti-slop:** does the spec actively avoid known LLM voice patterns? (Does
  it explicitly constrain away from the "assistant voice"?)

**Output:** A locked voice specification that conditions all prose generation
in Stage 4. The voice doesn't evolve further during prose generation — it's a
fixed constraint that the prose must honor.

---

## Stage 4: Prose Evolution

**What is being evolved:** The actual written text — scene by scene, following
the structure DAG, in the locked voice.

**The genome:** Prose for each node in the structure graph. This could be the
text itself, or the prompt/instructions used to generate prose for each node.
Evolving the *instructions* rather than the *text* is often more efficient: a
mutation to the instruction ("make the subtext more opaque," "cut the last
paragraph," "rewrite the dialogue with more interruptions") generates a new
text variant.

**Generalized story state:**

Instead of character-specific state tracking, the system maintains a flexible
state store that tracks whatever the story needs. What gets tracked depends on
what kind of story emerged from Stages 1–2:

- **Character state** (for character-driven stories): emotions, goals,
  relationship shifts, what they know, what they've decided
- **Reader knowledge state** (for reveal stories): what the reader knows vs.
  what is true, the dramatic irony gap, what has been disclosed and what remains
  hidden
- **Concept development state** (for thought-experiment stories): what
  implications have been explored, what remains, how the reader's understanding
  of the premise has evolved
- **Voice constraint state** (for compression pieces): what constraints have been
  honored, what tension has been built through what's unsaid

The state accumulates as scenes are generated. Each new scene has access to the
full state so far, preventing the drift and inconsistency that plagues AI
long-form generation. But the state isn't prescriptive — it doesn't force the
next scene to do anything specific. It's context, not instruction.

**Scene generation:**

For each node in the structure DAG, generation is conditioned on:
1. The structure node's payload (what this scene unit should accomplish)
2. The locked voice specification (how it should sound)
3. The accumulated story state (what has happened so far)
4. The overall concept and target effect (what this is all building toward)

Multiple prose variants are generated per node. The population at this stage
is a set of complete or near-complete drafts.

**Mutation operators:**

- Style shifts within the voice spec's range (more compressed, more expansive)
- Dialogue rewrites (add imperfection, subtext, interruption)
- Hemingway iceberg passes: generate full context/backstory for a scene, then
  cut to only what's essential — the authority of the omitted material remains
- Carver cuts: radical compression. Remove entire paragraphs. What survives?
- Detail substitution: swap generic descriptions for specific, concrete ones
- Perspective shifts: retell the same scene beat from a different focal point

**Evaluation criteria:**

- **Transportation / immersion** (does the prose pull the reader in?)
- **Flow / pacing** (does it maintain momentum without rushing or dragging?)
- **Anti-slop** (Tier 1 regex + Tier 2 NLP checks run on every candidate)
- **Voice adherence** (does the prose match the locked voice spec?)
- **Show vs. tell** (is emotion demonstrated through action and detail, not
  stated?)
- **Sentence-level craft** (burstiness, lexical diversity, rhythm variation)
- **State consistency** (does this scene contradict established state?)

**Exit condition:** Top N complete drafts advance to refinement.

---

## Stage 5: Refinement

**What is being evolved:** Full stories under editorial pressure — critique,
revision, compression.

This is not open-ended evolution. Research (Self-Refine, NeurIPS 2023; CritiCS,
EMNLP 2024) shows clear gains from iterative critique-then-revise, but
diminishing returns after 2–3 rounds. More rounds risk mode collapse toward
"safe" revisions that sand off the interesting edges.

**Round structure (2–3 rounds):**

Each round:
1. **Multi-critic evaluation:** Different critics apply different criteria
   (CritiCS model). One critic evaluates emotional resonance. Another evaluates
   structural tightness. Another hunts for anti-patterns. They don't debate —
   independent evaluations prevent conformity.
2. **Revision brief synthesis:** Critiques are synthesized into a prioritized
   revision brief (what to fix, in order of importance).
3. **Targeted revision:** The story is revised to address the brief. Only the
   flagged issues are changed — unflagged passages are preserved.

**Craft audits (run once, after rounds complete):**

- **Poe's Unity of Effect:** Is every element serving the target emotional
  response defined in Stage 1? Anything that doesn't contribute is a candidate
  for cutting.
- **Chekhov's Gun:** Is everything on the page paid off? Are there planted
  details that go nowhere, or payoffs that weren't planted?
- **Saunders' escalation:** Does every scene/beat change the story's state? If
  two consecutive beats leave the reader in the same emotional/informational
  position, one should be cut or revised.
- **Peak-end check:** Is the ending strong? Cognitive science (Kahneman's
  peak-end rule) shows endings disproportionately determine overall evaluation.
  If the ending is weak, prioritize ending revision over everything else.
- **Anti-slop deep pass:** Tier 3 (LLM judge) checks: over-explanation, all 12
  Nous Research structural anti-patterns, dialogue naturalness.

**Exit condition:** Stories that pass all craft audits and score above threshold
on the judge panel advance to selection. Stories that fail are either revised
further or discarded.

---

## Stage 6: Selection & Archive

**What is happening:** Quality-diversity selection — choosing which stories
survive, and feeding the evolutionary loop.

This is not "pick the best story." It's "maintain a diverse archive of excellent
stories across the quality-diversity landscape."

**MAP-Elites archiving:**

The archive is a grid indexed by diversity dimensions — axes that describe *how*
stories differ from each other. Dimensions might include:

- Dominant edge type in the structure graph (causal / disclosure / implication /
  constraint — this is the emergent "story mode")
- Emotional arc shape (rise, fall, fall-rise, rise-fall-rise, etc.)
- POV and tense
- Setting type
- Thematic concern
- Prose register (spare ↔ ornate)
- Story length

A new story enters the archive only if it's better than the current occupant of
its cell (the combination of its diversity dimension values). This ensures every
cell contains the best story *of its type*, not just the best story overall.

Grid dimensionality will be determined by Stage 1 experience. Stage 1 uses a
2D grid (concept_type × arc_shape = 36 cells) after review
against QDAIF's empirical findings that higher-dimensional grids cause sparse
archive problems (see `docs/issues/2026-04-01-map-elites-review.md`). Stage 6 has an
advantage: some dimensions (dominant edge type, story length, POV/tense) are
deterministically measurable from the finished story rather than LLM-classified,
making them more reliable grid axes. Full Stage 6 grid design is deferred until
Stage 1 provides empirical data on appropriate dimensionality, classifier
stability, and within-cell competition rates.

**Diversity signals:**

- **Inter-judge disagreement as positive signal:** A story that gets 5/5 from
  every judge might be safely mediocre. A story that gets 5/5 from the literary
  critic judge and 1/5 from the genre reader judge — or vice versa — is doing
  something bold. High variance in judge scores is preserved, not penalized.
- **Fitness sharing / niching:** Stories too similar to others in the population
  have their fitness penalized, forcing the evolutionary search to spread across
  story space.
- **Novelty rejection-sampling** (ShinkaEvolve): candidates that are >0.95
  cosine similar to existing archive members are rejected before expensive
  evaluation, preventing the archive from filling with near-duplicates.

**Feedback to Stage 1:**

- **Go-Explore restarts:** Periodically, evolution restarts not from the current
  best stories but from promising-but-undeveloped seeds archived from earlier
  runs. This prevents the system from getting stuck in a local optimum.
- **Episodic memory of failures:** A running log of what didn't work in previous
  generations ("the reveal was too predictable," "the voice was inconsistent,"
  "the concept was interesting but couldn't sustain 3,000 words"). This memory
  is injected as context into Stage 1's concept generation, so the system learns
  from its mistakes across runs.
- **Accumulation archive** (Gaiman's compost heap): Interesting fragments,
  images, half-formed ideas, and promising concepts that didn't win their
  generation are saved to an accumulation archive. Stage 1's compost
  recombination operator draws from this archive, giving old ideas new chances
  in new combinations.

---

## Cross-Stage Mechanisms

**Co-evolving state layers:**

Inspired by Nous autonovel's 5-layer system, but generalized beyond
character-centric storytelling. The layers are:

1. **Voice** — style spec (locked after Stage 3, but can trigger re-evaluation
   of structure if the chosen voice fundamentally conflicts with the plan)
2. **World** — setting rules, constraints, what's possible in this story's
   universe
3. **Agents** — characters, forces, systems, narrators — whatever entities act
   in the story (not necessarily human characters)
4. **Structure** — the typed-edge DAG
5. **Prose** — the actual text

Changes propagate bidirectionally: a discovery in prose ("this character's voice
naturally wants to take over the narration") can propagate up to restructure the
DAG. A structural change ("moving the reveal earlier") propagates down to
require prose revision.

**Post-hoc rationalization (Caves of Qud):**

Rather than planning every causal connection forward (computationally hard and
creatively constraining), the system can generate story events first and
construct plausible causal connections afterward. This is what LLMs are actually
good at — retrospective explanation is easier than forward prediction. A
structure DAG can be partially randomized and then rationalized into coherence.

**Dynamic outlining (DOME):**

The structure from Stage 2 is not a rigid contract. During prose generation
(Stage 4), the outline can adapt. If a scene takes an unexpected turn that's
better than what was planned, the remaining structure adjusts. The structure is
a guide, not a cage.

---

## The Judge Panel Across Stages

The same panel of diverse reader personas operates at every stage, but the
evaluation rubric shifts to match what's being evolved:

| Stage | Primary dimensions weighted | Secondary / reduced |
|---|---|---|
| **Concept** | Originality, emotional potential, tension, thematic richness | — |
| **Structure** | Causal coherence, tension arc, surprise + post-dictability | — |
| **Voice** | Distinctiveness, consistency, appropriateness | — |
| **Prose** | Transportation, flow, anti-slop, show-vs-tell, voice adherence | Character depth*, memorability |
| **Refinement** | All 10 dimensions at full weight | — |
| **Selection** | Holistic quality + diversity signals (judge disagreement) | — |

*Character depth is weighted only when the story's structure suggests character
is a primary vehicle (high proportion of causal edges involving character
decisions). For concept-driven or voice-driven stories, character depth may
be legitimately low and that's fine.*

The panel composition (40% target audience / 30% adjacent / 20% random / 10%
expert) remains constant. What changes is what they're asked to evaluate.

---

## How Different Stories Flow Through

**"The Lottery" (Jackson) — situation/reveal story:**
- Stage 1: Concept = "a small town ritual with a horrifying reveal"
- Stage 2: Structure DAG is mostly **disclosure** edges — the story is an
  information-disclosure schedule. Each node adds innocent detail that will
  become horrifying in retrospect.
- Stage 3: Voice = flat, journalistic, deliberately mundane. The horror comes
  from the contrast between tone and content.
- Stage 4: Prose is deliberately boring on the surface. The story state tracks
  *reader knowledge* — what has been disclosed, what hasn't.
- Stage 5: Refinement ensures every "innocent" detail actually serves the
  reveal. Chekhov's Gun audit is critical here — nothing extraneous.

**"Story of Your Life" (Chiang) — concept story:**
- Stage 1: Concept = "what if learning an alien language changed your perception
  of time?"
- Stage 2: Structure DAG is mostly **implication** edges. Each node explores a
  consequence of the premise. The "plot" (alien arrival, linguistic fieldwork)
  is scaffolding for the concept.
- Stage 3: Voice = precise, academic with emotional undercurrent. Chiang's
  signature: clinical language carrying devastating feeling.
- Stage 4: Prose tracks *concept development state* — what the reader understands
  about the premise's implications. Character emotion is present but serves the
  concept, not the reverse.
- Stage 5: Unity of effect check — does every scene serve the final emotional
  punch of knowing your daughter will die and choosing it anyway?

**"Hills Like White Elephants" (Hemingway) — voice/constraint story:**
- Stage 1: Concept = "two people discuss an abortion without ever naming it."
  Target effect: dread, helplessness, the weight of what's unsaid.
- Stage 2: Structure DAG is mostly **constraint** edges. The structure IS the
  constraint: one scene, two people, one conversation. Nodes might be as simple
  as "they order drinks," "he brings it up obliquely," "she deflects," "nothing
  is resolved."
- Stage 3: Voice = spare, concrete, almost nothing but dialogue and action.
  No interiority. No metaphors. Constraint: the word "abortion" never appears.
- Stage 4: Prose tracks *voice constraint state* — ensuring the prohibition is
  maintained while the subtext remains legible. The story state is about what
  ISN'T said.
- Stage 5: Compression is everything. Carver cut pass. Every word load-bearing.

**"A Good Man Is Hard to Find" (O'Connor) — character collision:**
- Stage 1: Concept = "a self-deluded grandmother meets an escaped convict." A
  collision between two worldviews.
- Stage 2: Structure DAG is mostly **causal** edges — the grandmother's vanity
  causes the detour, which causes the crash, which causes the encounter. Classic
  chain of consequences from character flaws.
- Stage 3: Voice = Southern Gothic — conversational, wryly comic, with sudden
  violence. Close third person on the grandmother.
- Stage 4: Prose tracks *character state* here — the grandmother's self-delusion,
  the Misfit's philosophy. This is where the wound-fear-lie framework activates
  naturally, because the story calls for it.
- Stage 5: The grace moment at the end — the grandmother reaching out to the
  Misfit — must feel both shocking and inevitable. Post-dictability is everything.
