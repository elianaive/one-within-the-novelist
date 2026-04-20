# Stage 1: Concept Evolution — Overview

Stage 1 is where stories begin. Before structure, before voice, before prose —
there's the concept: a seed idea with enough tension, originality, and emotional
potential to sustain a story worth reading.

This document covers the philosophy, the genome specification, concept types, the
relationship to ShinkaEvolve, and the handoff to Stage 2. Mutation operators,
evaluation criteria, and population dynamics are covered in their own documents.

---

## Philosophy: The Concept Is the Ceiling

No amount of structural elegance, voice distinctiveness, or prose polish can save
a weak concept. A story built on a cliché premise, no matter how beautifully
executed, will feel like a beautifully executed cliché. Conversely, a powerful
concept can survive imperfect execution — readers forgive rough edges when the
idea grips them.

This asymmetry means Stage 1 has outsized impact on final story quality. The
evolutionary search here should be broad, diverse, and aggressive about
originality. Every downstream stage is bounded by what emerges from this one.

### Unity of Effect

Poe's principle (1846) remains foundational: **every element of a short story
should serve a single predetermined emotional response.** The target feeling the
reader should have at the end — not a vague mood, but a specific emotional effect
— must be named *before* anything else is generated. This becomes the north star
for all downstream stages:

- Stage 2 (Structure) builds toward this effect
- Stage 3 (Voice) serves this effect
- Stage 4 (Prose) delivers this effect
- Stage 5 (Refinement) audits against this effect (Poe's Unity of Effect check)

If the target effect is vague ("make the reader feel things") or absent, the
story has no compass. This is why target emotional effect is a required field in
the concept genome, not optional.

### Mode Agnosticism

The standard academic pipeline assumes stories are plot-driven: you generate a
concept, outline events, write prose. But many of the greatest short stories
aren't plot-driven at all:

- **"The Lottery" (Jackson)** is situation-driven — the horror comes from a
  situation, not a plot
- **"Story of Your Life" (Chiang)** is concept-driven — the "plot" (alien
  contact) is scaffolding for a philosophical thought experiment
- **"Hills Like White Elephants" (Hemingway)** is voice-driven — the story IS
  the constraint of what can't be said
- **"A Good Man Is Hard to Find" (O'Connor)** is collision-driven — two
  incompatible worldviews forced into the same space

Our pipeline must support all of these without forcing a mode. The concept genome
is designed so that any of these entry points produces a valid genome. The system
never asks "what type of story is this?" — it evolves concepts, and the type
emerges from what works.

### Non-Western Structures as First-Class Citizens

The system's concept operators should be capable of generating premises that
naturally lead to structures beyond the Western conflict-driven arc:

**Kishotenketsu (Ki-Sho-Ten-Ketsu):** Japanese/East Asian four-act structure —
introduction, development, twist (an unrelated element that reframes everything),
reconciliation. No conflict required. The "Ten" (twist) element is particularly
relevant to concept generation: a premise that includes an intentionally
incongruous element can naturally lead to this structure.

**Zuihitsu:** "Following the brush" — associative, essayistic, fragment-based.
Not plot-driven. A concept genome for a zuihitsu-style story might be a
collection of resonant images and moments rather than a single premise.

**Spiral/recursive structures:** Stories that return to the same moment or image
with deepening understanding (Borges, Calvino). The concept genome would identify
the recursive element and what changes each time.

These don't need to be forced or labeled. But the mutation operators (see
operators.md) should be diverse enough to produce concepts that naturally lead to
any of these structures.

---

## The Concept Genome

A concept genome is the structured representation of a story seed — the unit that
gets evolved, evaluated, and passed to Stage 2. It's serialized as JSON and
stored in ShinkaEvolve's `Program.code` field.

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| **Core premise** | Yes | The "what if" / situation / collision / constraint. 1-3 sentences that capture the story's essential tension. This is the irreducible seed. |
| **Target emotional effect** | Yes | The single feeling the reader should have at the end. Must be specific: not "sadness" but "the ache of knowing something beautiful is temporary." Poe's unity of effect. |
| **Character seeds** | Optional | Brief sketches of key characters. For character-driven concepts, can include wound/fear/lie/want/need (see below). For concept or voice stories, may be absent entirely. |
| **Setting/world seeds** | Optional | Key constraints on what's possible in the story's universe. Not worldbuilding — just the rules that matter for the premise. |
| **Thematic tension** | Optional | Structured as a tension between two values: freedom vs. security, knowledge vs. innocence, duty vs. desire. Gives evolutionary operators something concrete to mutate. |
| **Constraints** | Optional | What the story does NOT do. "The word 'death' never appears." "Only dialogue." "Single scene, real-time." "No interiority." Negative constraints often define a story more sharply than positive descriptions. |
| **Style hint** | Optional | Tonal/atmospheric direction — "clinical precision with emotional undercurrent," "flat affect masking horror," "conspiratorial intimacy." Not a full voice spec (that's Stage 3), just enough to guide concept evaluation. |

### Design Principles

**No forced entry mode.** A valid genome can be just a core premise and a target
effect. Or it can have all seven fields filled. A Hemingway-style concept might
have only a premise, a target effect, and a constraints field. A character-driven
O'Connor concept might have premise, target effect, and rich character seeds. Both
are equally valid genomes.

**The genome is a loose bundle, not a template.** Fields don't need to be
consistent or harmonious. A comedic premise with a devastating target effect could
produce dark comedy. A spare constraint with a rich thematic tension could produce
something Kafka-esque. Productive tension between fields is a feature.

**Constraints are first-class.** The deep research on LLM writing (Sui, ICML
2026; Paech et al., ICLR 2026) consistently shows that LLMs are better at
honoring negative constraints than positive instructions. The constraint inversion
principle: telling a model what NOT to do is more reliably followed than telling
it what TO do. This makes the constraints field particularly powerful — it shapes
the story by excluding the model's default patterns.

### Character Seeds: The Wound-Fear-Lie Framework

When a concept includes character seeds (optional, but enriching for
character-driven stories), the system supports the wound-fear-lie framework that
converges across independent craft traditions (Weiland's *Creating Character
Arcs*, Truby's *Anatomy of Story*, McKee's *Story*):

| Field | Description | Example |
|-------|-------------|---------|
| **Wound** | Formative event the character hasn't resolved | Abandoned by a parent at age 8 |
| **Fear** | What the character dreads experiencing again | Being left behind, being insufficient |
| **Lie** | False belief adopted to cope with the wound | "I don't need anyone" |
| **Want** | External goal the character consciously pursues | Career success, independence, control |
| **Need** | What the character actually needs (internal) | Connection, vulnerability, trust |

The gap between want and need is where the story lives — the character pursues
what they think they want while the story gradually reveals what they actually
need. This creates natural narrative tension without requiring explicit conflict
plotting.

**Not all stories need this.** A Borges concept story may have no characters with
psychological depth. A Hemingway compression piece may have characters defined
entirely by what they do, not by internal frameworks. The wound-fear-lie fields
are available when useful, invisible when not.

---

## Run Prompt

An optional `prompt` field in the run config lets a human express creative
direction without constraining the evolutionary process.

```yaml
prompt: >
  Stories that explore the relationship between physical spaces
  and the memories they hold. Domestic settings, the uncanny,
  objects that outlast the people who used them.
```

**How it works:**

- When present, the run prompt is wrapped in the *Magnes* template
  (`owtn/prompts/stage_1/run_prompt.txt`) and injected into every operator's
  system message — both in genesis (gen 0 cold-start) and in mutation
  (every subsequent generation). The block sits between the tonal-steering
  paragraph and the base task description, so the directional pressure is in
  place before the structural contract appears.
- Operators still do their thing. Collision still collides, Compression still
  compresses. The run prompt is gravitational pull, not a hard constraint.
  If the evolutionary process finds something brilliant that's off-theme, it
  should survive.
- Seed bank selection is also filtered by tag relevance to the run prompt,
  using the existing tag-based filtering mechanism.
- **Judges never see the run prompt.** Evaluation is blind to creative
  direction. This prevents the system from optimizing for prompt-adherence
  ("this matches the theme!") rather than quality ("this is a great concept").
- When absent or empty, the run-prompt block is omitted entirely from the
  system message, and operators and seed selection behave exactly as they do
  in an unprompted run.

The run prompt is the primary mechanism behind targeted mode (see
population.md). A competition run might use the prompt to direct toward the
competition's theme. An exploratory run leaves it empty and lets the system
surprise you.

---

## Concept Types

The system auto-detects concept types from genome content. These labels are
descriptive — they inform evaluation weighting and Stage 2 exploration methods,
but never constrain evolution. A concept can be hybrid or defy categorization.

### Taxonomy

**Thought experiment** (Le Guin, Chiang, Borges)
- Core: a philosophical or speculative "what if" pushed to its emotional conclusion
- Genome signature: strong premise with implication chains, thematic tension present, character seeds secondary
- Structural affinity: implication-heavy DAGs in Stage 2
- Example: "What if learning an alien language changed how you perceive time?"

**Situation with reveal** (Jackson, Shirley, du Maurier)
- Core: a seemingly normal situation concealing something horrifying, surprising, or transformative
- Genome signature: premise contains hidden information, reader will be surprised by what's actually happening
- Structural affinity: disclosure-heavy DAGs
- Example: "A small town's cheerful annual tradition is actually a stoning."

**Voice constraint** (Hemingway, Carver, Davis)
- Core: the constraint IS the story — what can't be said, what must be omitted, what formal restriction shapes everything
- Genome signature: constraints field is the most important field, premise may be simple
- Structural affinity: constraint-heavy DAGs
- Example: "Two people at a train station discuss something they never name."

**Character collision** (O'Connor, Munro, Chekhov)
- Core: two incompatible people/worldviews forced into proximity
- Genome signature: rich character seeds (possibly with wound-fear-lie), premise centers on the encounter
- Structural affinity: causal-heavy DAGs
- Example: "A grandmother who thinks she's a lady meets an escaped convict who thinks he's a philosopher."

**Atmospheric/associative** (Murakami, Schulz, Carson)
- Core: meaning emerges from juxtaposition of images, moods, and fragments rather than from plot or argument
- Genome signature: multiple images/moments rather than a single premise, style hint is prominent
- Structural affinity: loose coupling, zuihitsu or spiral structures
- Example: "A series of encounters in an empty seaside town where time moves differently for each person."

**Constraint-driven** (Perec, Queneau, Calvino)
- Core: a formal constraint generates the story — the constraint comes first, the content follows
- Genome signature: constraints field is primary and specific, premise derived from constraint
- Structural affinity: depends on the constraint
- Example: "A story in which every paragraph contains exactly one lie, and the lies build a second story."

**Kishotenketsu**
- Core: introduction, development, an apparently unrelated element (the "ten"), reconciliation that reveals the connection
- Genome signature: premise contains or implies an incongruous element that will be reconciled
- Structural affinity: four-part structure with twist-as-recontextualization
- Example: "A retired calligrapher practices daily. A construction crew begins demolishing the building next door. Both are preparing for the same thing."

---

## ShinkaEvolve Mapping

Stage 1 runs on ShinkaEvolve's async evolution engine. The concept genome is a
"program" in ShinkaEvolve's terms — but instead of executable code, it's a
structured JSON document that gets mutated by LLM prompt templates and evaluated
by a custom judge panel.

### Field Mapping

| Our Concept | ShinkaEvolve Field | Notes |
|---|---|---|
| Concept genome (all fields) | `Program.code` | Serialized as JSON |
| Pairwise win percentage | `combined_score` | (dim_wins + 0.5 * ties) / 9 |
| Anti-cliché flags | `private_metrics` | Hidden from mutation LLM |
| Pairwise comparison reasoning | `text_feedback` | Which dimensions won/lost and why |
| Valid genome (required fields present) | `correct` | Boolean validation gate |
| Operator that created this concept | `metadata.patch_type` | Enables operator performance analysis |

### What We Use From ShinkaEvolve

- **Async evolution loop** (`ShinkaEvolveRunner`): proposal generation,
  evaluation, persistence, slot management
- **Inline evaluation** (`eval_function` on `JobConfig`): evaluation runs as a
  direct function call, not a subprocess
- **Bandit LLM selection** (`llm_dynamic_selection`): UCB/Thompson sampling
  across model families for mutation
- **Island model** (`CombinedIslandManager`): independent lineages with equal
  island scheduling and migration
- **Novelty rejection** (`AsyncNoveltyJudge`): embedding similarity check before
  evaluation
- **Parent selection** (`CombinedParentSelector`): power-law selection within
  islands
- **Cost tracking and budget management**

### Additions to ShinkaEvolve

- **`eval_function` on `JobConfig`**: inline evaluation without subprocess
  overhead. The eval function validates the concept, reads the island champion
  from disk, and runs pairwise comparison — blocking until complete.
- **`parent_id` and `island_idx` passthrough**: the scheduler passes the
  parent's island index to the eval function, enabling island-aware pairwise.
- **`get_island_champion()` and `update_program_score()`**: database methods
  for champion lookup and score updates after pairwise comparison.
- **`EqualIslandSampler`**: balanced island allocation (each generation goes
  to the island with fewest programs).

---

## Handoff to Stage 2

When Stage 1 completes, island champions compete in a **Swiss-system pairwise
tournament**. The tournament ranking determines which concepts advance to Stage 2.

Each island maintains one champion — the concept that has beaten all challengers
on that island. The tournament runs ceil(log2(N)) rounds, pairing champions with
similar records. Each pairing uses the same per-criteria pairwise protocol as
within-island comparison (3 judges, 2 orderings, majority of non-tie votes).

The top-ranked concepts advance. Each may spawn multiple structure variants — a
single premise might be explored as both a disclosure-heavy reveal story and a
causal-heavy character piece.

### What Gets Passed Forward

1. **Complete concept genome** — all fields, as evolved
2. **Judge panel scores** — per-dimension scores from the final evaluation
3. **Judge reasoning chains** — natural-language explanations of what makes this
   concept work (and what risks it carries). These are the most valuable signal
   for Stage 2: they tell structure evolution what to emphasize.
4. **Auto-detected structural affinities** — inferred from concept type and judge
   reasoning. "This concept's strength is its reveal potential" → hint toward
   disclosure-heavy DAGs. "This concept lives in the collision between two
   characters" → hint toward causal-heavy DAGs.
5. **Identified risks** — from judge reasoning: "The concept is strong but may be
   difficult to sustain past 2,000 words." "The reveal is powerful but the setup
   needs to earn it." These help Stage 2 focus its structural search.
6. **Diversity metadata** — affective register and literary mode from the tonal
   sampling system. Helps Stage 2 understand the concept's aesthetic character.

### Mapping to Stage 2 Edge Types

Concept types have natural affinities with Stage 2's typed-edge DAG system, but
these are hints, not constraints:

| Concept Type | Primary Edge Affinity | Why |
|---|---|---|
| Thought experiment | Implication edges | Each story beat explores a consequence of the premise |
| Situation with reveal | Disclosure edges | The story is an information-disclosure schedule |
| Voice constraint | Constraint edges | The structure IS the constraint |
| Character collision | Causal edges | Character decisions cause consequences |
| Atmospheric/associative | Mixed/loose | Connections are atmospheric, not causal |
| Constraint-driven | Varies | Depends on the specific constraint |
| Kishotenketsu | Mixed with twist | Four-part: setup, development, incongruity, reconciliation |

Stage 2 is free to explore structure types that don't match the "expected"
affinity — surprising structural choices can produce the most interesting stories.

---

## Run Output

A completed Stage 1 run auto-exports its results and supports re-export via CLI.

### Output Directory

```
results/run_<timestamp>/stage_1/
├── best/
│   └── main.json                  # tournament winner genome
├── tournament.json                # Swiss tournament results with rankings
├── champions/                     # island champion genomes (for pairwise)
│   ├── island_0.json
│   └── ...
├── gen_0/                         # per-generation concept files
│   ├── main.json
│   └── results/metrics.json
├── gen_1/
│   └── ...
└── evolution_run.log              # full run log with pairwise matchups
```

### Winner Format

The tournament winner's genome in `best/main.json` contains everything Stage 2
needs:

```json
{
  "premise": "...",
  "target_effect": "...",
  "character_seeds": [],
  "setting_seeds": null,
  "thematic_tension": "...",
  "constraints": [],
  "style_hint": null
}
```

The tournament results in `tournament.json` contain the full ranking with match
history:

```json
[
  {
    "rank": 1,
    "program_id": "...",
    "wins": 2,
    "losses": 0,
    "buchholz": 1,
    "matches": [
      {
        "opponent": "...",
        "result": "win",
        "dimension_wins": {"novelty": "a", "grip": "b", ...},
        "score": "6-1-2"
      }
    ]
  },
  ...
]
```

Each concept's `metrics.json` contains its pairwise result:

```json
{
  "correct": true,
  "combined_score": 0.78,
  "text_feedback": "Pairwise result: Won (7-1-1)\nDimensions to improve: ...",
  "metadata": {
    "patch_type": "thought_experiment",
    "affective_register": "DREAD",
    "literary_mode": "GOTHIC"
  }
}
```

### Re-export

`owtn export-winners <run_id>` re-exports from the DB with optional overrides
(different selection criteria, different max_concepts). The auto-export at run
end uses the config's handoff settings as defaults.
