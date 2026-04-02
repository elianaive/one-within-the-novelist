# Stage 1: Population Dynamics & Archive Architecture

This document covers how the evolutionary population is managed: how many
concepts, how they're organized, how diversity is maintained, and how the
accumulation archive (compost heap) works across runs.

The key insight driving all numerical decisions: **concepts are not code.**
Concepts are cheap to generate (short text, no compilation), cheap to evaluate
(no execution, just LLM judge calls), naturally blend-able (crossover between
premises is semantically meaningful), and live in a vast search space. Every
parameter inherited from ShinkaEvolve's code-evolution defaults should be
reconsidered through this lens.

---

## ShinkaEvolve Configuration

### Recommended Parameters

```
EvolutionConfig:
  num_generations: 15-30
  language: "json"                        # genome is structured JSON, not code
  patch_types: ["diff", "full", "cross"]
  patch_type_probs: [0.3, 0.4, 0.3]      # favor novel generation + blending
  max_patch_resamples: 3
  llm_dynamic_selection: "ucb"            # bandit selects across model families
  evolve_prompts: true                    # co-evolve the system prompt
  meta_rec_interval: 3-5                  # frequent strategy reflection
  code_embed_sim_threshold: 0.88-0.92     # tighter than code (concepts cluster)
  max_novelty_attempts: 3

DatabaseConfig:
  num_islands: 8-12
  archive_size: 80-120                    # ignored under map_elites strategy
  archive_selection_strategy: "map_elites" # targeted edit (see below)
  migration_interval: 2-3                  # frequent cross-pollination
  migration_rate: 0.15-0.25               # concepts blend naturally
  island_elitism: true
  enable_dynamic_islands: true
  stagnation_threshold: 20-30             # lower — concept generations are fast
  parent_selection_strategy: "power_law"
  exploitation_alpha: 1.0
  exploitation_ratio: 0.3-0.4            # draw from archive more often
```

### Why These Differ from Code Defaults

**8-12 islands (vs. 2).** Each concept candidate is cheap — a few hundred tokens
to generate, a few judge calls to evaluate. With low per-candidate cost, running
more parallel sub-populations is affordable and valuable. Each island naturally
drifts toward different concept types through selection pressure, providing
geographic diversity that's harder to achieve with fewer islands. Even without
explicit specialization, 8-12 random islands will produce more diverse final
archives than 2.

**15-30 generations (vs. 50).** Concept genomes are simpler than code — fewer
interacting components, fewer ways to be "almost right." This means faster
convergence. But the concept search space is vast and sparse — genuinely novel
premises are rare — which rewards extended exploration. 15-30 generations
balances faster convergence against the value of continued search. The
meta-summarizer at intervals of 3-5 provides course correction.

**Migration every 2-3 generations at 15-25% rate (vs. every 10 at 0%).** Concept
blending is natural and productive. When you take a premise from one island and
inject it into another's population, the cross-pollination creates meaningful
variation — unlike code, where injecting foreign code often produces syntax
errors. Frequent, high-rate migration spreads discoveries across the archipelago
quickly while island-level selection pressure maintains local specialization.

**Novelty threshold 0.88-0.92 (vs. 0.99).** Concept embeddings are denser than
code embeddings — premises that are semantically different can still produce
similar embeddings because they share vocabulary and structure. A concept about
"a woman confronting her mother's legacy" and "a man confronting his father's
legacy" might have >0.95 cosine similarity despite being narratively distinct.
The tighter threshold catches genuine near-duplicates without rejecting
structurally similar but narratively different concepts.

**Patch type probs [0.3, 0.4, 0.3] (vs. [0.6, 0.3, 0.1]).** Code evolution
favors `diff` (small modifications to working programs) because code is fragile —
large changes usually break something. Concepts aren't fragile. A radically new
concept (`full`) or a collision between two concepts (`cross`) is as likely to
produce something good as a small tweak (`diff`). The rebalanced weights reflect
this: more novel generation, more blending, less incremental editing.

**Exploitation ratio 0.3-0.4 (vs. 0.2).** With the MAP-Elites archive (up to 36
cells), the archive contains diverse high-quality material — one champion per
behavioral niche. Sampling parents from the archive more often leverages this
accumulated quality. The power-law selection ensures the best archive members are
sampled most often, but the 60-70% non-archive samples maintain exploration.

**archive_size: 80-120 (ignored under MAP-Elites).** In ShinkaEvolve's
fitness-ranked mode, this caps retained programs. Under `map_elites` strategy, the
archive size is determined by the number of occupied cells (up to 108). The
parameter is retained for compatibility but has no effect — all occupied cells
keep their champion unconditionally.

---

## MAP-Elites Behavioral Dimensions

### Targeted Edit to ShinkaEvolve

ShinkaEvolve's archive is currently fitness-ranked — it keeps the N
highest-scoring programs regardless of behavioral diversity. For concept
evolution, we need true quality-diversity: the archive should maintain the *best
concept of each type*, not just the best concepts overall.

**The edit:** Add a `"map_elites"` option to `archive_selection_strategy` in
`shinka/database/dbase.py`. Implementation:

1. Each concept's evaluate.py output includes behavioral dimension classifications
   in `public_metrics` (e.g., `concept_type: "thought_experiment"`,
   `arc_shape: "fall_rise"`, `constraint_density: "moderate"`)
2. The grid axes (concept_type × arc_shape × constraint_density) define a 3D
   grid of 108 behavioral cells
3. When a new concept is evaluated, it's placed in a cell based on its grid axes
4. The concept replaces the current cell occupant only if its `holder_score`
   (raw Hölder mean, no diversity bonus) is higher
5. Empty cells represent unexplored niches — the niche target operator (#12) can
   be activated to fill them when diversity stalls

This ensures the archive contains the best thought-experiment AND the best
character collision AND the best fall-rise AND the best rise-fall — not just the
highest-scoring concepts overall (which might all be thought experiments if
that's what scores best).

### Grid Dimensions (Active)

Three behavioral dimensions form the MAP-Elites grid key, chosen for
orthogonality, measurability, and grid density. Concept generation is cheap
(~hundreds of tokens + judge calls), so with 1,200-5,400 candidates per run a
108-cell grid produces healthy within-cell competition (~10-50 candidates per
cell). This follows QDAIF's principle that the grid should be dense enough for
within-cell replacement to fire regularly (Bradley et al., ICLR 2024), while
capturing meaningfully more diversity than a 2D grid.

**1. Concept type** (auto-detected from genome content)
- Thought experiment
- Situation with reveal
- Voice constraint
- Character collision
- Atmospheric/associative
- Constraint-driven

6 values. Auto-detection based on which genome fields are most prominent and the
premise's structural properties.

**2. Emotional arc shape** (Reagan et al., EPJ Data Science 2016)
- Rise (rags to riches)
- Fall (tragedy)
- Fall-rise (man in a hole)
- Rise-fall (Icarus)
- Rise-fall-rise (Cinderella)
- Fall-rise-fall (Oedipus)

6 values. Classified by analyzing the target emotional effect and the premise's
implied trajectory. "Fall-rise" and "rise-fall-rise" correlate with higher story
popularity in Reagan's analysis of 1,327 stories.

**3. Constraint density**
- Unconstrained (no constraints field, or minimal)
- Lightly constrained (1-2 soft constraints)
- Heavily constrained (3+ constraints, or one severe constraint)

3 values. Directly measurable from the constraints field. This dimension is
deterministic — zero LLM cost, zero classification instability, zero boundary
effects. It captures something real: unconstrained, moderately constrained, and
heavily constrained concepts produce genuinely different stories and should be
preserved independently.

### Tracked Metadata (Not Grid Axes)

These dimensions are still classified and stored in `public_metrics` for analysis,
convergence detection, and potential future promotion to grid axes — but they do
not determine archive cell placement.

**4. Tonal register**
- Comedic
- Tragic
- Ironic
- Earnest
- Surreal
- Matter-of-fact

6 values. Classified from the style hint, the premise's emotional texture, and
the target effect's valence.

**5. Thematic domain**
- Interpersonal (relationships, family, love, betrayal)
- Societal (institutions, communities, power structures)
- Philosophical (epistemology, ethics, metaphysics, identity)
- Existential (mortality, meaning, isolation, transcendence)
- Mundane-elevated (everyday experience rendered extraordinary)

5 values. Classified from the thematic tension field and the premise's subject
matter.

### Promoting Metadata to Grid Axes

A metadata dimension should be promoted to a grid axis when:
- Cell occupancy consistently exceeds ~80% of the current grid
- Active within-cell competition is occurring (cells have been replaced 2+ times)
- The classifier shows >80% test-retest stability on the candidate dimension
- The resulting grid size still allows ~10+ candidates per cell given the
  generation budget

Promotion is a configuration change, not an architecture change — the classifier
already classifies all 5 dimensions.

### Grid Size

3 dimensions with 6 × 6 × 3 values = 108 possible cells. Under MAP-Elites, the
`archive_size` config parameter is ignored — all occupied cells retain their
champion. The archive naturally caps at 108 entries.

With 1,200-5,400 candidates per run, this produces ~10-50 candidates per cell on
average — dense enough for regular within-cell competition (the primary
quality-driving mechanism) while capturing meaningfully more diversity than a
smaller grid. A well-explored archive should occupy 50-108 cells.

---

## Population Structure

### Total Population

80-180 concepts across 8-12 islands (10-15 per island).

This is much larger than the 30-60 originally specified in stages.md. The
increase is justified by the cheapness of concept generation and evaluation —
the per-candidate cost is a fraction of what code evolution requires.

### Per-Island Population

10-15 concepts per island. Smaller per-island populations maintain tight
selection pressure — only the best concepts on each island survive and reproduce.
With 8-12 islands providing geographic diversity, per-island diversity isn't the
primary concern.

### Island Specialization

Islands are NOT explicitly assigned to concept types. They start with diverse
random seeding and naturally specialize through selection drift — one island might
drift toward literary thought experiments, another toward dark situational
premises, another toward constraint-driven experiments.

This emergent specialization is preferable to explicit assignment because:
- The system discovers which niches are productive, not the designer
- Islands can explore hybrid territories that don't map to predefined categories
- Migration between islands creates unexpected cross-pollination

Dynamic island spawning (triggered after `stagnation_threshold` generations
without improvement) creates new islands seeded from the archive's
best-performing or most-diverse concepts, providing fresh exploration vectors
when existing islands plateau.

---

## Seeding Strategies

### Cold Start (No Prior Runs)

Generate the initial population using at least 5 different operators. Most
operators draw starting material from the seed bank (`data/seed-bank.yaml`) when
matching seeds are available — axioms feed Thought Experiment, constraints feed
Constraint-First, etc. When no seed is available, operators fall back to pure
LLM generation. See `docs/stage-1/seed-bank.md` for the full type mapping.

| Operator | Allocation | Rationale |
|----------|-----------|-----------|
| Collision (King) | 20% | High-energy starting concepts |
| Thought experiment (Le Guin) | 20% | Conceptually rich |
| Noun-list (Bradbury) | 15% | Emotionally grounded |
| Constraint-first (Oulipo) | 15% | Structurally distinctive |
| Anti-premise | 10% | Actively avoids convergence patterns |
| Discovery mode (Murakami) | 10% | Atmospheric, surprising |
| Compression (Borges) | 5% | Thematically dense |
| Inversion | 5% | Shadows of other initial concepts |

Distribute initial concepts across islands roughly evenly. Crossover and compost
recombination operators aren't available at cold start (they need existing
population and archive material).

### Warm Start (Prior Runs Available)

When the compost heap contains material from previous runs:

- **40-60% from compost:** Draw high-spark-quality fragments and undeveloped
  concepts from the accumulation archive. These are concepts that showed promise
  in previous runs but weren't fully developed.
- **40-60% fresh generation:** New concepts from the full operator suite,
  including compost recombination (which draws from the archive)

The warm start leverages accumulated learning — the system gets smarter across
runs, not just within them.

### Targeted Mode (Competition/Specific Audience)

When generating for a specific competition, audience, or theme, set the
`steering` field in the run config (see `docs/stage-1/overview.md`). The
steering prompt is injected into every operator's system message and seed
bank selection is filtered by tag relevance.

The operator allocation shifts to favor theme-relevant generation:

- **60-70% targeted:** All operators receive the steering prompt, but the seed
  bank pre-filters to theme-relevant seeds (by tag matching). Operators naturally
  pull toward the steering direction.
- **20-30% wild:** Operators that ignore or only loosely follow the steering.
  These maintain diversity and occasionally produce surprising connections to
  the theme that targeted generation wouldn't find.
- **10% anti-premise:** Deliberate subversion of expected responses to the theme.
  If the steering is about "memory," anti-premise might generate concepts about
  *forgetting* or *false memory* or *the burden of remembering everything*.

---

## Novelty Rejection

ShinkaEvolve's `AsyncNoveltyJudge` handles novelty rejection, used as-is with
tighter thresholds for concepts:

### Embedding Similarity Check

Before expensive evaluation, embed the new concept and compute max cosine
similarity against all concepts on the parent's island:

- If `max_similarity > 0.88-0.92`: flag as potential duplicate
- If flagged, optionally run LLM semantic novelty check: "Is this concept truly
  different from [most similar concept]?"
- If confirmed duplicate: reject, request regeneration
- Max 3 regeneration attempts before accepting

### Why Tighter Threshold

Code with 0.95 cosine similarity is usually genuinely different (different
algorithms, different approaches). Concepts with 0.95 cosine similarity are often
near-duplicates phrased differently. The tighter threshold (0.88-0.92) catches
these while allowing through concepts that share vocabulary but differ
narratively.

The threshold should be calibrated empirically: too tight rejects genuinely
different concepts that happen to use similar words; too loose allows the archive
to fill with variants of the same idea.

---

## Convergence Detection

Monitor three signals across generations:

### 1. Fitness Plateau

If the mean `combined_score` across all islands hasn't improved for 3+
consecutive generations, the search may have converged. Options:

- Inject fresh seeds from the compost heap
- Increase mutation rate (more `full` patches, fewer `diff`)
- Spawn new islands from diverse archive regions
- Terminate and advance the current best concepts to Stage 2

### 2. Diversity Decline

If the number of occupied MAP-Elites cells is declining (concepts are clustering
into fewer behavioral niches), the system is losing diversity. Options:

- **Activate niche target operator** (operators.md #12): set weight to 0.10-0.20,
  target 1-3 empty cells per generation that have neighboring occupied cells
- Increase `cross` patch probability (more blending between diverse parents)
- Inject anti-premise concepts targeting underrepresented niches
- Temporarily disable fitness-based selection and use pure novelty selection
- Increase migration rate to spread diversity across islands

### 3. Archive Stagnation

If archive cells haven't been replaced (improved) for N generations, the archive
is stable. This can mean:

- The search has found good concepts for occupied niches (good — advance them)
- The search has stopped exploring new niches (bad — inject fresh operators)

The meta-summarizer (running every 3-5 generations) provides strategic
recommendations based on these signals — "the search is stagnating in the
literary-ironic quadrant; consider injecting more genre-earnest concepts."

---

## The Compost Heap (Accumulation Archive)

The compost heap is a persistent archive that spans across evolutionary runs. It's
separate from ShinkaEvolve's within-run archive — it's a long-term memory of
interesting material that didn't make the cut but might be valuable later.

### Targeted Edit to ShinkaEvolve

Add an optional `compost_db_path` field to `EvolutionConfig`. When configured:

- The evaluation pipeline (`evaluate.py`) writes qualifying entries to the compost
  DB after each evaluation
- The seeding logic reads from the compost DB on warm starts
- The compost recombination operator draws fragments from the compost DB during
  evolution

### What Gets Composted

Not everything. Only material with "spark quality" — concepts that contain
something worth returning to even if they failed overall:

**Interesting failures:** Concepts that scored high on some dimensions but low on
others. A concept with 5/5 originality but 1/5 feasibility has a spark — the
premise is brilliant but can't sustain a story in the current form. Worth
saving for recombination.

**Evocative fragments:** During mutation, operators sometimes produce intermediate
material that isn't a complete concept but contains a resonant image, an
interesting constraint, or a surprising thematic tension. These fragments are
worth saving.

**Failure lessons:** When a concept fails, the judge reasoning chains explain why.
These explanations are episodic memory: "a premise about X didn't work because Y."
Future runs can draw on this memory to avoid repeating mistakes.

**Interesting collisions:** When the collision operator combines two premises and
the result doesn't work, the collision point itself (the unexpected connection
between the two elements) might be worth saving separately.

**Undeveloped seeds:** Concepts that scored well enough to be interesting but
didn't make the final cut for Stage 2. They're not failures — they're
opportunities waiting for the right evolutionary context.

### Composting Criteria

evaluate.py flags compost candidates in `private_metrics` — it does NOT write
to the compost DB directly (avoids concurrent SQLite access with ShinkaEvolve's
runner). A post-run step reads flagged programs from the DB and populates the
compost table. The flagging criteria:

- **High max, low min:** If any single dimension score is >= 4/5 but the Holder
  mean is below the advancement threshold, the concept has a salvageable spark
- **High originality, any other score:** Novel concepts are always worth saving,
  even if they're currently incoherent or infeasible
- **High judge variance:** Polarizing concepts that some judges loved and others
  hated — the spark is in what the fans saw
- **Novel operator output:** First-time products of underused operators (the
  system should accumulate material from all operator types)

### SQLite Schema

The compost heap lives in ShinkaEvolve's existing SQLite database, as two
additional tables.

**`compost` table** — interesting fragments and undeveloped concepts:

```sql
CREATE TABLE compost (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    genome_json     TEXT NOT NULL,           -- full concept genome
    spark_score     REAL NOT NULL,           -- see formula below
    tags            TEXT,                    -- JSON array of freeform tags
    embedding       BLOB,                   -- concept embedding for similarity search
    valence         TEXT CHECK(valence IN ('positive','negative','mixed','ambiguous')),
    thematic_domain TEXT,                    -- MAP-Elites thematic domain category
    top_dimensions  TEXT,                    -- JSON array: which eval dimensions scored highest
    operator        TEXT,                    -- which operator created this concept
    failure_reason  TEXT,                    -- from judge reasoning chains
    source_run      TEXT NOT NULL,           -- evolutionary run ID
    source_gen      INTEGER,                -- generation number
    created_at      TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_compost_spark ON compost(spark_score DESC);
CREATE INDEX idx_compost_domain ON compost(thematic_domain);
```

**`compost_lessons` table** — episodic memory from judge reasoning:

```sql
CREATE TABLE compost_lessons (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern                 TEXT NOT NULL,     -- "premises about X tend to fail because Y"
    source_generation       INTEGER,
    judge_reasoning_excerpt TEXT,              -- the reasoning chain that surfaced this
    source_run              TEXT NOT NULL,
    created_at              TEXT DEFAULT (datetime('now'))
);
```

### Spark Quality Score

The spark score identifies concepts with interesting fragments worth returning to
— not overall good concepts, but concepts with at least one exceptional dimension:

```
spark_score = max(dimension_scores) - mean(dimension_scores)
```

A high spark score means one dimension stands out well above the others. A concept
with scores [5, 1, 2, 1, 2, 1, 2, 1, 2] has spark 5 - 1.89 = 3.11 (very high —
the originality is brilliant but everything else is weak). A concept with scores
[3, 3, 3, 3, 3, 3, 3, 3, 3] has spark 0 (competent, nothing worth salvaging
individually).

### Composting Threshold

A concept is composted when:
- `spark_score >= 2.0` (at least one dimension is ~2 points above mean), OR
- `originality_score >= 4` (novel concepts are always worth saving), OR
- `judge_variance >= 1.5` (polarizing concepts have something worth investigating)

### Indexing

The compost DB stores each entry with:

- **Embedding:** For similarity search during compost recombination
- **Emotional valence tags:** Positive / negative / mixed / ambiguous
- **Thematic domain tags:** Matching the MAP-Elites thematic dimension categories
- **Spark quality score:** See formula above
- **Operator provenance:** Which operator created this, what were its parents
- **Failure reason:** Why it didn't advance (from judge reasoning chains)
- **Run metadata:** Which evolutionary run produced this, what generation

### Usage

**Compost recombination operator:** Draws two fragments from distant regions of
the archive (high semantic distance, compatible emotional valence) and asks the
LLM to find the story connecting them.

**Go-Explore restarts:** Seeds new evolutionary runs from undeveloped
high-potential archive entries rather than from scratch. This is the "return to
interesting places" mechanism from Go-Explore (Ecoffet et al., 2019).

**Cross-run learning:** The archive accumulates material over many runs. Early
runs produce raw material; later runs have a richer compost to draw from. The
system gets more creative over time as its compost heap grows.

**Episodic memory:** Failure lessons (judge reasoning chains explaining why
concepts failed) are injected as context into concept generation, so the system
learns from its mistakes: "In previous runs, premises about X tended to fail
because Y — avoid this pattern or address Y explicitly."

---

## Feedback Loop from Stage 6

Stage 6 (Selection & Archive) feeds information back to Stage 1, closing the
evolutionary loop:

### Go-Explore Restarts

Periodically, evolution restarts not from the current best stories but from
promising-but-undeveloped concept seeds archived from earlier runs. A concept that
was discarded in generation 3 of run 1 might be the perfect starting point for
run 5, in combination with material that didn't exist during run 1.

### Episodic Memory of Downstream Failures

When a concept makes it through Stage 1 but fails in a later stage, that failure
information propagates back:

- "This concept seemed great but couldn't sustain structure" (Stage 2 failure) →
  future concepts with similar properties get a feasibility warning
- "The voice couldn't match the concept's ambition" (Stage 3 failure) → future
  concepts with similar voice requirements get flagged
- "The prose was technically good but the story didn't transport" (Stage 4
  failure) → the concept's transportation potential estimate was wrong; adjust
  evaluation calibration

### Accumulation Archive Additions

Interesting fragments from all stages feed back into the compost:

- Structural patterns that worked well (from Stage 2)
- Voice discoveries (from Stage 3)
- Evocative prose passages (from Stage 4)
- Critique insights (from Stage 5)

These cross-stage fragments enrich the compost heap with material that concept
operators can draw on — a beautiful sentence from Stage 4 might inspire a new
concept in a future run's compost recombination.
