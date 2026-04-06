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

### Current Parameters

See `configs/stage_1/light.yaml` for the reference configuration. Key settings:

```
EvolutionConfig:
  num_generations: 10
  language: "json"
  patch_types: [collision, noun_list, thought_experiment, ...]  # 11 concept operators
  llm_dynamic_selection: "ucb"
  code_embed_sim_threshold: 0.90
  max_novelty_attempts: 3
  genesis_ratio: 0.2                     # 20% fresh concepts, 80% evolve parent
  annealing:                             # temperature schedule
    temp_early: [0.8, 0.9, 1.0]
    temp_late: [0.5, 0.6, 0.7]

DatabaseConfig:
  num_islands: 3
  archive_size: 50
  archive_selection_strategy: "fitness"
  island_selection_strategy: "equal"     # balanced allocation across islands
  migration_interval: 5
  migration_rate: 0.15
  island_elitism: true
  parent_selection_strategy: "power_law"
```

### Selection: Pairwise, Not Pointwise

Evaluation is inline (`eval_function` on `JobConfig`, no subprocess). Each new
concept is validated (gates 1 & 2), then compared head-to-head against its
island's champion via a 3-judge panel. The comparison is per-criteria: each judge
evaluates both concepts on all 9 dimensions, picking a winner per dimension with
position bias mitigation (dual orderings). Winner becomes champion;
`combined_score` = win percentage.

This replaced MAP-Elites (which barely fired at current generation counts — 7-9
cells occupied out of 36 with 12 concepts per run) and pointwise scoring (which
compressed all concepts into a 0.3-point band due to model-level leniency bias).

### Island Model

Each island is an independent lineage with its own champion. `EqualIslandSampler`
ensures each generation goes to the island with fewest programs, distributing
concepts evenly. Different islands start from different seeds with different
affective registers and literary modes, providing diversity without needing
MAP-Elites' behavioral grid.

Migration means a champion from one island challenges another island's champion
via pairwise comparison. The migrant either wins and takes over, or loses and is
discarded — concepts spread by proving they're better, not by having a higher
number.

### Why These Parameters

**3 islands.** With pairwise comparison blocking on LLM calls (~30-40s per
comparison), more islands means more parallel comparisons. 3 islands balances
diversity with the constraint that each island needs enough generations to
develop its lineage.

**10 generations.** Fast iteration for testing. Medium config uses 20.

**Fitness archive (not MAP-Elites).** MAP-Elites requires dense cell occupancy
for within-cell competition to fire. With 10-20 concepts per run and 36+ cells,
most cells have 0-1 occupants — the archive degenerates into an unfiltered list.
Fitness archive with island champions provides selection pressure that actually
works at current scale.

**Equal island scheduling.** Without this, ShinkaEvolve's uniform random island
selection can send multiple consecutive concepts to the same island, leaving
others starved. Equal scheduling guarantees balanced allocation.

---

## Diversity Maintenance

With MAP-Elites removed, diversity comes from three mechanisms:

### 1. Island Separation

Each island starts from a different seed concept with a different affective
register and literary mode (16 registers x 18 modes = 288 combinations). Islands
evolve independently — no shared selection pressure. Different starting points
naturally produce different lineages.

### 2. Tonal Steering

Each concept receives a paired affective register + literary mode that shapes its
emotional and stylistic character. These are injected into the generation prompt
and stored in metadata for inheritance by offspring. See
`owtn/prompts/affective_registers.yaml` and `owtn/prompts/literary_modes.yaml`.

### 3. Novelty Rejection

ShinkaEvolve's `AsyncNoveltyJudge` rejects concepts that are too similar to
existing archive members (cosine similarity > 0.90). This prevents near-duplicate
concepts from accumulating, pushing evolution toward genuine exploration.

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
