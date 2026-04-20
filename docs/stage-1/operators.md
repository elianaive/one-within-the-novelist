# Stage 1: Mutation & Generation Operators

The mutation operators are how the system generates new concepts. They are the
creative engine — the mechanism through which the evolutionary search explores the
vast space of possible story premises.

Each operator is inspired by a documented creative method used by working writers.
The research is clear: the best human writers don't generate ideas from nothing.
They use *methods* — collision, association, inversion, constraint, compression —
and the methods shape what they find. Our operators formalize these methods as
prompt templates that ShinkaEvolve's mutation system can select and apply.

---

## How Operators Work in ShinkaEvolve

ShinkaEvolve's mutation system has three patch types:

- **`diff`**: Modify part of an existing genome (keep most, change one aspect)
- **`full`**: Generate an entirely new genome (fresh creation or radical rewrite)
- **`cross`**: Combine elements from two parent genomes

Each of our 11 operators maps to one of these patch types. The operator's logic
lives in the prompt template — the LLM receives the prompt, the parent genome(s),
any relevant context (text feedback from judges, archive inspirations), and
generates a new concept genome.

### Operator Selection

The `PromptSampler` selects which operator to use for each mutation, weighted by
configured probabilities. The initial weight distribution favors `full` (novel
generation) and `cross` (blending) over `diff` (tweaking):

```
patch_type_probs: [0.3, 0.4, 0.3]  # diff, full, cross
```

This differs from code evolution (which defaults to 60% diff) because concept
space rewards bold jumps over incremental edits. A small tweak to a premise often
doesn't meaningfully change it, while a fresh generation or a collision between
two premises can open entirely new territory.

### Run Prompt

When the run config includes a `prompt` field, its content is wrapped in the
*Magnes* template (`owtn/prompts/stage_1/run_prompt.txt`) and injected into
every operator's system message — both in genesis and in mutation. The system
message order is:

1. Operator persona ("Deep within your weights is a fulgurite...")
2. Tonal atmosphere (random affective register × literary mode)
3. **Run-prompt block** (if present — the *Magnes* prose framing the user's prompt)
4. Base task description ("You're generating story concepts...")
5. Operator-specific instructions in the user message (collision logic, compression method, etc.)
6. Output format requirements

The run prompt provides gravitational pull, not a hard constraint. Operators
should follow their own creative method — the prompt just tilts the search
space. An operator that produces something brilliant but off-theme should not
suppress it.

The injection template is a single editable file. To swap the framing prose
(e.g. to try one of the other drafts in
`lab/issues/2026-04-19-rename-steering-to-prompt.md`), edit
`owtn/prompts/stage_1/run_prompt.txt` directly. The `{prompt}` placeholder is
replaced with the user's content; the surrounding prose is the template.

### Seed Bank Integration

Most operators can draw starting material from the seed bank
(`data/seed-bank.yaml`). Each seed has a `type` that maps to one or more
operators:

| Seed Type | Operator |
|-----------|----------|
| `real_world` | Real-World Seed Injection (#11) |
| `thought_experiment` | Thought Experiment / Le Guin (#3) |
| `axiom` | Thought Experiment / Le Guin (#3) |
| `dilemma` | Thought Experiment (#3), Collision (#1), Compression (#8) |
| `constraint` | Constraint-First / Oulipo (#9) |
| `noun_cluster` | Noun-List / Bradbury (#2) |
| `image` | Discovery Mode / Murakami (#7) |
| `compression` | Compression / Borges (#8) |
| `collision_pair` | Collision / King (#1) |
| `anti_target` | Anti-Premise (#10) |

**Injection mechanism:** During prompt assembly (before the prompt reaches
ShinkaEvolve's sampler), the code:

1. Looks up the operator's matching seed type(s)
2. Queries the bank for matching seeds
3. If found, selects one (diversity-weighted — prefer types underrepresented
   in recent concepts; filtered by run-prompt tags if a prompt is set)
4. Wraps the seed in labeled delimiters and replaces the `{seed_content}`
   placeholder at the end of the operator prompt:
   ```
   ---SEED---
   {content from seed-bank.yaml}
   ---END SEED---
   ```
5. If not found, `{seed_content}` is replaced with empty string — the operator
   generates from scratch, no orphaned delimiters

**Prompt structure:** `{seed_content}` is placed **at the end** of each
operator's instructions, after the model has been told what to do. Each
operator's relevant step includes a conditional that references the seed
location: "If seed material is provided at the end of these instructions,
use it as your [axiom/image/constraint/etc]." This follows the
order-equals-causality principle from the prompting guide — instructions
before data, not data before instructions.

Seeds are enrichment, not hard dependency. See `docs/stage-1/seed-bank.md`
for the full schema and curated starter set. See
`docs/stage-1/implementation.md` section 4 for the injection implementation.

### Bandit Adaptation

ShinkaEvolve's bandit ensemble (UCB or Thompson sampling) tracks which LLM
model/operator combinations produce the highest-scoring concepts. Over
generations, the system allocates more budget to high-performing combinations
while maintaining exploration. A targeted edit adds operator-type tracking
alongside model tracking, so the system learns that (e.g.) collision operators
with Claude produce better concepts than noun-list operators with GPT, and adjusts
accordingly.

#### Operator-Level UCB1

Extend ShinkaEvolve's existing model-level UCB bandit to also track per-operator
performance. The implementation mirrors the model bandit pattern:

**Tracked per operator:**
- `attempts` — number of times this operator was selected
- `cumulative_fitness` — sum of `combined_score` of all offspring
- `ucb1_score` — upper confidence bound: `mean_fitness + c * sqrt(ln(total) / attempts)`

**Parameters:**
- Exploration constant `c = 1.0` (balances exploration vs. exploitation)
- Minimum probability floor: `0.02` per operator (prevents starvation — every
  operator gets at least 2% of selections regardless of performance)
- Warmup period: first 3 generations use the static weights from the table below.
  After warmup, UCB1 scores determine selection probabilities (softmax over UCB1
  scores, then clipped to the 0.02 floor and renormalized).

**Implementation:** Add operator tracking alongside model tracking in
ShinkaEvolve's bandit. Each mutation records both the model used and the operator
used. The bandit jointly optimizes model × operator, but operator selection and
model selection remain independent (no combinatorial explosion).

**Cross-type availability:** When the population has insufficient material for
cross-type operators (cold start, sparse archive), those operators are excluded
from the bandit's candidate set and their probability mass is redistributed
proportionally among available operators.

### Operator Metadata

Every concept records which operator created it (stored in
`Program.private_metrics.operator`). This enables post-run analysis: which
operators produced the archive's best concepts? Which produced the most diverse
concepts? Which have diminishing returns after N generations?

---

## Operator Catalog

### 1. Collision (King)

**Patch type:** `cross`

**Method:** Take two premises from different parents — ideally from different
MAP-Elites cells or different islands — and force them together. The story lives
in the interference pattern between two incompatible elements.

**Algorithm:**
1. Select two parent concept genomes (ShinkaEvolve provides parent + archive
   inspiration)
2. Identify the most *unlikely* connection point between them — the place where
   these two ideas would never naturally meet
3. Generate a new premise that *requires both elements to survive* — neither is
   subordinated to the other
4. Derive target emotional effect from the collision's natural tension
5. Carry forward any character seeds, constraints, or thematic tensions that still
   serve the new premise

**Key insight:** Collision isn't blending. Blending averages two ideas into
something safe. Collision maintains the tension between incompatible elements —
the story exists in the space where they can't be reconciled. King's formulation
(*On Writing*): "What happens when you take two completely unrelated 'what if'
scenarios and smash them together?"

**Example:** Parent A: "A linguist decoding an alien language." Parent B: "A
mother knowing her daughter will die young." Neither subordinated — the collision
produces "Story of Your Life."

**When it works best:** When the two parents are semantically distant but could
share an emotional or thematic resonance. The bandit will learn which parent
pairings produce productive collisions.

---

### 2. Noun-List Recombination (Bradbury)

**Patch type:** `full`

**Method:** Generate a list of emotionally resonant nouns, cluster them by
emotional (not semantic) resonance, and derive a premise from the tension between
clusters.

**Algorithm:**
1. Generate 15-20 nouns that carry strong emotional charge — objects, places,
   sensations, memories. Not abstract concepts; concrete things. "THE LAKE. THE
   LOTTERY. THE SCISSORS. THE GRANDMOTHER. THE FOG. THE TEETH."
2. Cluster the nouns by emotional resonance, not semantic category. "LAKE" and
   "GRANDMOTHER" might cluster (nostalgia, loss, summers past) even though
   they're unrelated semantically.
3. Find the cluster pairing that creates the most surprising juxtaposition —
   nouns that are emotionally proximate but semantically distant
4. Derive a premise from the tension between the clusters
5. Name the target emotional effect that the juxtaposition suggests

**Implementation note:** This maps naturally to an embedding-space operation.
Generate nouns, embed them in both semantic and emotional/sentiment embedding
spaces, find clusters that are far in semantic space but close in
emotional/sentiment space. The gap between these spaces is where surprising
premises live.

**Source:** Bradbury's *Zen in the Art of Writing* — his daily practice of
writing noun lists and finding the stories hiding in their connections.

---

### 3. Thought Experiment (Le Guin)

**Patch type:** `full`

**Method:** Take a social, philosophical, or scientific axiom. Invert it, extend
it, or transplant it to an unfamiliar context. Push to its logical AND emotional
conclusion.

**Algorithm:**
1. Select or generate a philosophical/social axiom. Can come from:
   - Established philosophy (what if free will doesn't exist?)
   - Social norms (what if gender were fluid?)
   - Scientific principles (what if entropy could be reversed locally?)
   - Everyday assumptions (what if everyone could hear your thoughts?)
2. Apply one transformation:
   - **Inversion:** What if the opposite were true?
   - **Extension:** What if this were taken to its extreme?
   - **Transplant:** What if this were true in a completely different context?
3. Push to both logical conclusion (what would actually happen?) and emotional
   conclusion (how would this *feel* for the people living it?)
4. The emotional conclusion becomes the target effect
5. The logical exploration becomes the premise

**Why it works:** The information theory research (Schulz et al., 2024; Kumar et
al., Cognitive Science 2023) shows this creates maximal Bayesian surprise — the
reader's model of reality gets updated in ways that are unexpected but internally
consistent. A thought experiment asks the reader to genuinely reconsider something
they took for granted.

**Source:** Le Guin's introduction to *The Left Hand of Darkness*, her essays on
the thought experiment as fiction's native mode.

---

### 4. Compost Recombination (Gaiman)

**Patch type:** `cross`

**Method:** Draw two fragments from the accumulation archive (the "compost heap"
— see population.md) and find the story hiding in their connection.

**Algorithm:**
1. Select two fragments from the compost archive, prioritizing high semantic
   distance but compatible emotional valence
2. The fragments might be: an evocative image from a previous run, a half-formed
   concept that scored high on some dimensions but failed on others, a constraint
   that never found its story, an interesting character seed
3. Prompt the LLM to find the resonance between these two fragments — the story
   that connects them
4. Generate a full concept genome from the connection

**Source:** Gaiman's metaphor (*The View from the Cheap Seats*, various
masterclass discussions): ideas are compost. You throw everything in — images,
overheard phrases, half-thoughts, things that bothered you — and years later, two
things that were never related have decayed into the same soil and something
grows.

**Requires:** A populated compost archive. Not available on cold start — this
operator activates after the system has accumulated material from previous runs.

---

### 5. Crossover

**Patch type:** `cross`

**Method:** Standard genetic crossover applied to structured concept genomes.
Take specific fields from one parent and combine with fields from another.

**Algorithm:**
1. Select two parent concept genomes
2. Choose a crossover strategy:
   - **Field swap:** Take the situation from parent A and the character seeds from
     parent B
   - **Thematic transplant:** Take the thematic tension from one and the
     setting/constraints from another
   - **Effect transfer:** Keep one parent's premise but replace the target
     emotional effect with the other parent's
3. Generate a new genome that integrates the borrowed elements coherently
4. Adjust any fields that now conflict

**When it works best:** When the parents occupy different MAP-Elites cells. A
character-driven concept's characters combined with a thought-experiment's premise
can produce something neither parent contained.

---

### 6. Inversion

**Patch type:** `diff`

**Method:** Flip one aspect of an existing concept while preserving everything
else. Produces the "shadow" of a concept — the same thematic space explored from
the opposite angle.

**Algorithm:**
1. Select a parent concept genome
2. Choose one dimension to invert:
   - **Emotional valence:** hope → dread, comfort → unease, triumph → failure
   - **Power dynamic:** victim → perpetrator, powerful → powerless, teacher →
     student
   - **Expected outcome:** the premise's obvious resolution is replaced by its
     opposite
   - **Perspective:** who the story is "about" shifts to a different character or
     viewpoint
   - **Temporal direction:** consequences → origins, ending → beginning
3. Generate the inverted genome, adjusting target effect and other fields to
   match the new orientation
4. Preserve the non-inverted elements — the story should still be recognizably
   "about" the same thing, just from the opposite angle

**Why it works:** Inversion produces concepts that share thematic DNA with their
parent but explore different territory. Two inversions of the same concept occupy
different MAP-Elites cells, maintaining archive diversity while building on proven
thematic ground.

---

### 7. Discovery Mode (Murakami)

**Patch type:** `full`

**Method:** Start with a single evocative image or situation. Generate forward
associatively — each next element follows emotional and atmospheric logic, not
causal logic. Then extract the concept retrospectively.

**Algorithm:**
1. Generate or select a single evocative starting image — a scene, a sensation,
   an atmosphere. "A man sits in an empty baseball stadium at 3 AM." "Rain on a
   window in a language school." "A letter that arrives 40 years late."
2. Generate 200-500 words of associative prose from this image. No plan, no plot,
   no structure. The LLM follows the emotional current of the image — what it
   evokes, what it connects to, where the feeling leads.
3. Read back the generated prose and extract:
   - What premise is latent in this material?
   - What emotional effect does it converge toward?
   - Are there character seeds, constraints, or thematic tensions embedded?
4. Formalize into a concept genome

**Why it works:** This inverts the normal concept-first pipeline. When an LLM
generates a concept directly, it tends to produce "concept-shaped" output —
clean, logical, and often cliché. When it generates prose associatively and the
concept is extracted afterward, the concept inherits the prose's specificity and
strangeness. The deep research found that this produces more surprising premises
because the LLM's pattern-completion works differently when not "trying to be
creative."

**Source:** Murakami's described process — sitting at a desk with no plan,
following the first image that comes, trusting that the subconscious will produce
story structure. Also aligned with the "post-hoc rationalization" principle from
Caves of Qud: generate first, find the meaning after.

---

### 8. Compression (Borges)

**Patch type:** `full`

**Method:** Don't generate a story concept — generate the *review* of an
imaginary story. What would a thoughtful critic say about a story that doesn't
exist? Then extract the concept that would produce that review.

**Algorithm:**
1. Prompt the LLM to write a brief (200-400 word) critical review of a fictional
   short story that doesn't exist. The review should discuss:
   - What the story is about (thematically, not just plot)
   - What emotional effect it achieves
   - What craft choices make it work
   - What makes it distinctive
2. Extract from the review:
   - The premise implied by the reviewer's description
   - The target emotional effect the reviewer ascribes
   - Any structural, character, or constraint details mentioned
3. Formalize into a concept genome

**Why it works:** Reviews are inherently analytical — they describe what a story
is *about*, not just what happens. Concepts born from this critical frame tend to
be thematically richer and more intentional than concepts generated directly.
Borges practiced this literally — writing reviews and prefaces for books that
didn't exist (*Ficciones*), arguing that the review was more interesting than the
imaginary book would have been.

**Source:** Borges' literary practice, particularly *The Garden of Forking Paths*
and the fictional book reviews in *Ficciones*.

---

### 9. Constraint-First Generation (Oulipo)

**Patch type:** `full`

**Method:** Start with a formal constraint and derive a concept that would make
the constraint feel *necessary* rather than arbitrary.

**Algorithm:**
1. Select or generate a formal constraint:
   - Linguistic: no adjectives, only dialogue, every sentence must contain a
     color, no letter 'e'
   - Structural: single scene, real-time, reverse chronological, alternating
     perspectives with no overlap
   - Content: a word that never appears but is always present, characters who
     never lie, a room no one leaves
   - Length: exactly 500 words, exactly 55 words, one paragraph
2. Ask: "What story would make this constraint feel inevitable? What premise
   would transform this constraint from arbitrary exercise to essential form?"
3. Generate a concept genome where the constraint field is primary and the premise
   serves the constraint
4. Name the target emotional effect that the constraint naturally produces

**Why it works:** Constraints force the LLM away from its default patterns. The
uncertainty gap research (Sui, ICML 2026) shows that human writing is 2-4x more
informationally surprising than LLM output. A strong constraint eliminates the
model's most comfortable paths, forcing it into territory that's more surprising
and distinctive. The Oulipo proved that severe formal constraints can produce
extraordinary literature (Perec wrote an entire novel without the letter 'e').

**Source:** The Oulipo movement — Perec's *A Void*, Queneau's *Exercises in
Style*, Calvino's *If on a winter's night a traveler*.

---

### 10. Anti-Premise Generation

**Patch type:** `diff`

**Method:** Take a known LLM convergence pattern — a premise type that AI systems
generate at disproportionately high rates — and deliberately subvert it.

**Algorithm:**
1. Select a convergence pattern from the anti-cliché list (see evaluation.md):
   - The reconciliation arc (protagonist returns, confronts past, reconciles)
   - The grief meditation (dead loved one, metaphorical journey)
   - The chosen one (special destiny discovered)
   - The AI consciousness story (AI becomes sentient)
   - Sanitized conflict (no real stakes, clean resolution)
   - Epistolary revelation (found letters/messages reveal hidden truth)
   - The time loop lesson (repeat day, learn to be better)
   - Magical realism metaphor (emotion literally manifests as physical phenomenon)
   - Moral clarity (good and evil are obvious, virtue rewarded)
   - Small-town secret (idyllic community hides dark truth)
2. Identify the *expectation* the pattern creates in readers familiar with AI
   fiction
3. Generate a premise that deliberately violates that expectation while remaining
   in the same thematic territory:
   - "Chosen one" → person discovered to have special destiny who *refuses* it,
     and the story explores why refusal is the right choice
   - "Small-town reconnection" → person returns to hometown and realizes they
     were *right to leave*
   - "Grief meditation" → the dead person was terrible, and the grief is
     about feeling relief instead of sadness
4. The subversion itself becomes the core tension of the new concept

**Why it works:** The Echoes-in-AI research (Xu et al., PNAS 2025) showed that
LLM-generated plots have a Sui Generis uniqueness score of 6-7 vs. 13+ for human
plots. The convergence patterns are *specific and identifiable*. Deliberately
subverting them produces concepts that are anti-correlated with the model's
default outputs — a reliable source of novelty.

**Source:** AI fiction failure mode research (Xu et al., PNAS 2025), the
anti-slop framework (Paech et al., ICLR 2026).

---

### 11. Real-World Seed Injection

**Patch type:** `full`

**Method:** Draw from real-world material — historical incidents, scientific
discoveries, cultural phenomena, overheard conversations — and transplant into a
fictional context.

**Algorithm:**
1. Draw a seed from a curated bank of real-world material:
   - Historical incidents (a specific event, not a broad era)
   - Scientific discoveries or phenomena (a specific finding, not a field)
   - Cultural observations (a specific practice or contradiction)
   - Everyday oddities (specific situations, not generic experiences)
2. Transplant the seed to a fictional context — change the setting, the era, the
   specifics, but preserve the *emotional kernel* and the *structural shape* of
   the real event
3. Generate a concept genome where the real-world seed provides specificity and
   grounding that pure imagination often lacks
4. Name the target emotional effect suggested by the real material

**Why it works:** Real-world material has specificity that LLMs struggle to
generate from scratch. A premise grounded in something that actually happened has
details that feel earned rather than constructed. The transplantation to fiction
prevents the story from becoming journalism — it's about the shape and feeling of
the real event, not the event itself.

**Note:** The seed bank should be periodically refreshed and should avoid
sensitive current events where fictionalization could be inappropriate.

---

### 12. Niche Target

**Patch type:** `full`

**Method:** Generate a concept explicitly targeting an empty or under-explored
MAP-Elites cell. Inspired by QDAIF's "targeted poems" approach (Bradley et al.,
ICLR 2024, Section 4.4), where guiding generation toward specific diversity
categories matched the best QD-scores.

**Algorithm:**
1. Receive a target cell descriptor from the convergence detection system:
   a target concept_type, arc_shape, and constraint_density (the 3 grid
   dimensions)
2. Build a generation prompt that includes:
   - The target concept_type definition (from classification.md)
   - The target arc_shape definition (from classification.md)
   - Instruction: "Generate a concept that fits this specific niche"
   - Optional: 1-2 examples of concepts from neighboring occupied cells as
     inspiration (not to copy, but to calibrate the level of specificity)
3. Generate a full concept genome targeting the niche
4. The concept goes through normal evaluation and classification — it may or may
   not actually land in the target cell (classification is independent)

**Activation:** This operator is NOT part of the standard rotation. Initial
bandit weight is 0. The convergence detection system (population.md) activates
it when:
- Diversity is declining (occupied cells decreasing for 2+ generations)
- Specific cells have been empty for 3+ generations despite having neighboring
  occupied cells (suggesting the niche is reachable but not being explored)
- Archive stagnation is detected (no cell replacements for N generations)

When activated, the convergence system sets the niche target operator's weight
to 0.10-0.20 and specifies 1-3 target cells per generation. The weight returns
to 0 when diversity stabilizes.

**Why it works:** Undirected mutation explores the grid randomly, which is
sufficient for moderate grids (108 cells) most of the time. But when the system
gets stuck in a subset of the grid, directed niche-targeting is more efficient
than hoping random mutation discovers the right combination. The key insight
from QDAIF: telling the generator what kind of output you want is cheap and
effective for categorical diversity dimensions.

**Risk:** The generated concept may be superficially targeted — fitting the
classifier's definition of the niche without being a genuinely good concept of
that type. The judge evaluation (Gate 3) catches low-quality output regardless
of its classification. This operator trades generation efficiency for evaluation
cost — most niche-targeted concepts will be evaluated and discarded, but the
ones that survive fill real gaps.

---

## Operator Distribution Across Concept Types

Different operators tend to produce different concept types, though this isn't
deterministic:

| Operator | Most Likely Concept Types |
|---|---|
| Collision | Character collision, hybrid |
| Noun-list | Atmospheric, situation |
| Thought experiment | Thought experiment (naturally) |
| Compost recombination | Any — depends on archive contents |
| Crossover | Hybrid (by definition) |
| Inversion | Shadow of parent's type |
| Discovery mode | Atmospheric, voice constraint |
| Compression | Thought experiment, literary |
| Constraint-first | Constraint-driven, voice constraint |
| Anti-premise | Inversion of cliché type |
| Real-world seed | Situation, character collision |
| Niche target | Any — explicitly targets underrepresented cell |

The bandit ensemble will discover more specific patterns over generations —
which operators produce which types, at what quality, with which LLM models.
