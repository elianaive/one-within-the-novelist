# MAP-Elites Proposal Review

**Date:** 2026-04-01
**Scope:** Stage 1 MAP-Elites archive (population.md, classification.md, implementation.md) + Stage 6 archive (stages.md)
**References:** Mouret & Clune 2015 (canonical MAP-Elites), Bradley et al. ICLR 2024 (QDAIF), project deep research reports

---

## Assessment

The core MAP-Elites algorithm is correctly implemented. The replacement rule is textbook (empty cell = insert, occupied = replace only if higher fitness). The quality-diversity framing is sound. Each behavioral dimension is individually well-motivated. The compost heap is a novel mechanism with no QDAIF analog. Anti-reward-hacking defenses (diverse judge panel, coevolving judges, Tier A filters, human anchors) are substantially stronger than QDAIF's single-model approach.

No fundamental misunderstandings of MAP-Elites.

Six changes needed before implementation.

---

## Changes

### 1. Start with 3D grid, not 5D

**Problem:** The 5D grid (6×6×6×5×3 = 3,240 cells) is 30-130x larger than anything empirically validated. QDAIF tested 1D (10-20 cells) and 2D (~100 cells) and called 2D "challenging." With expected 1.5-6% occupancy, within-cell competition — the core quality-driving mechanism of MAP-Elites — almost never fires. The archive degenerates into a diversity-filtered elitist list. Many of the 3,240 cell combinations are semantically incoherent.

**Change:** Initial grid uses concept_type (6) × arc_shape (6) × constraint_density (3) = 108 cells. Constraint density is rule-based (deterministic, zero LLM cost, zero classification instability), making it essentially free to add as a grid axis. Tonal register and thematic domain are still classified and stored in `public_metrics` as tracked metadata but are not grid axes. With 1,200-5,400 candidates per run, 108 cells produces ~10-50 candidates per cell — dense enough for regular within-cell competition. Promote metadata dimensions when occupancy consistently exceeds ~80% and the classifier shows >80% stability on the candidate dimension.

**Files:** `population.md`, `classification.md`, `implementation.md`

### 2. Add niche-targeting mutation operator

**Problem:** No directed mechanism for exploring empty cells. Canonical MAP-Elites fills the grid through random mutation, which works when the grid is small. QDAIF solved this for poetry (Section 4.4) with LMX-guided mutation that explicitly targets diversity categories. The current 11 operators are concept-type-inspired but none generates "a concept for cell X."

**Change:** Add operator #12: Niche Target. Receives a target cell descriptor (concept_type + arc_shape), generates a concept matching that niche. Triggered by convergence detection when diversity stalls or cells remain unfilled for 3+ generations. Initial weight 0 in the bandit — only activated by the convergence detection system.

**Files:** `operators.md`, convergence detection section of `population.md`

### 3. Split fitness score for different uses

**Problem:** `combined_score` is `selection_score()` which includes the diversity bonus (judge_mean + diversity_bonus). This is used for MAP-Elites cell replacement (`implementation.md` Edit 6, line 230). A polarizing concept gets a boost in within-cell competition, but being polarizing doesn't make it a better example of its niche. In canonical MAP-Elites, within-cell competition is purely fitness-based.

**Change:** Add `holder_score` (raw Hölder mean, no diversity bonus) as a named output. Use `holder_score` for MAP-Elites cell replacement. Keep `selection_score` for parent selection and stage advancement decisions.

**Files:** `implementation.md` (Edit 6), `implementation-scoring.md` (evaluate_concept pseudocode + new field)

### 4. Remove archive_size cap under MAP-Elites

**Problem:** `archive_size: 80-120` in DatabaseConfig is a hard cap from ShinkaEvolve's fitness-ranked mode. Under MAP-Elites, the archive size is the number of occupied cells (up to 108 with the 3D grid). If `archive_size` were enforced as a cap, evicting an occupied cell would violate MAP-Elites' core guarantee that every niche keeps its best.

**Change:** Under `archive_selection_strategy: "map_elites"`, the `archive_size` parameter is ignored. All occupied cells retain their champion. With a 36-cell grid this is a non-issue — the archive naturally caps at 36.

**Files:** `population.md` (config section)

### 5. Validate classifier before trusting it

**Problem:** QDAIF validated human-AI agreement on diversity labels (73% overall, 95% when annotators agreed). The OWTN classifier assigns 4 subjective dimensions in a single Haiku-class call on *concepts* (not finished stories), which is a harder classification task than QDAIF validated. No stability or reliability plan exists.

**Change:** Before the first evolutionary run, pilot with 50 concepts classified 3× each by the production model. Measure: (a) test-retest stability (% landing in same cell), (b) confidence-stability correlation. Acceptance threshold: >80% same-cell rate on the 2 grid dimensions (concept_type, arc_shape). Fallback: upgrade model or reduce to concept_type only (6 cells).

**Files:** `classification.md` (new validation section)

### 6. Defer Stage 6 grid design

**Problem:** Stage 6 lists 7 potential dimensions with no values per dimension, no classification mechanism, no grid size, and no implementation plan. It claims "5-8 dimensions are practical" — but Stage 1 experience with 5 dimensions demonstrates this is likely too many.

**Change:** Mark Stage 6 grid design as deferred until Stage 1 provides empirical data on appropriate dimensionality. Note that Stage 6 has an advantage (some dimensions are deterministically measurable from finished stories), but the grid size should be informed by Stage 1 lessons.

**Files:** `stages.md` (Stage 6 section)

---

## Monitor (No Change Yet)

These are lower-severity issues worth watching during implementation:

- **No KL divergence / distributional regularization.** The multi-layered defense is arguably sufficient. Monitor distribution entropy across generations; add regularization if convergence detection catches distributional collapse.
- **Category boundary effects.** Concepts at the boundary between categories (e.g., "ironic" vs. "comedic") may flip cells stochastically. The classifier validation pilot (Change 5) will surface this if it's significant.
- **Fitness sharing may conflict with MAP-Elites.** Fitness sharing penalizes concepts similar to others, but MAP-Elites already separates them into cells. Within a cell only the best survives anyway. Clarify which diversity mechanisms operate at which level when implementing.
- **Island-archive interaction.** The global MAP-Elites archive vs. per-island populations — how archive membership influences parent selection and migration — needs specification during implementation. Not blocking for the spec updates.
