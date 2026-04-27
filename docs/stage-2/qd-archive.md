# Stage 2: Quality-Diversity Archive

The QD archive is Stage 2's cross-concept diversity memory. It records structurally-distinct terminal DAGs in a grid indexed by measurable properties of the DAG. Unlike Stage 1's abandoned MAP-Elites design (failed because classifier reliability + cell density were both too low at our scale), Stage 2's QD archive uses **deterministically computed** axes rather than LLM classification — eliminating the classifier-stability issue that killed Stage 1's grid.

The archive serves two roles in v1:

1. **Within-run diversity preservation.** Non-advancing DAGs from a concept (those that lost the within-concept tournament) are archived rather than discarded. Their structural ideas stay accessible.
2. **Structural pattern mining (post-hoc).** Over many runs, the archive reveals which DAG shapes work for which concept types. This data shapes v2 decisions (pacing preset parameters, edge-type affinity hints, pattern templates).

Cross-run compost integration is deferred. Stage 1 does not yet ship cross-run archive infrastructure (no `compost.db`, no persisted compost tables in `owtn/`); the "compost" feature in Stage 1 today is one of the mutation operators, not a storage layer. When Stage 1 lands its cross-run archive, Stage 2 will extend it analogously. Until then, Stage 2 archives to per-run JSON only (`qd_archive.json` under the run directory).

---

## Grid Axes

The archive is a 5×5 grid with two axes, both deterministically computable from a terminal DAG:

### Axis 1: Disclosure Ratio

The fraction of the DAG's edges that are `disclosure` edges.

```python
disclosure_ratio = count(edges, type="disclosure") / total_edges
```

Binning (5 bins):

| Bin | Range | Character |
|---|---|---|
| 0 | `[0.00, 0.10)` | No reveal architecture; purely causal/implication/motivates |
| 1 | `[0.10, 0.25)` | Light disclosure — mostly surface story, some hidden information |
| 2 | `[0.25, 0.40)` | Balanced disclosure — reveals are a structural feature |
| 3 | `[0.40, 0.55)` | Heavy disclosure — the story is substantially about information revelation |
| 4 | `[0.55, 1.00]` | Disclosure-dominant — the story IS its reveals (Jackson-mode reveal stories) |

### Axis 2: Structural Density

Edges per node — how interconnected the DAG is.

```python
structural_density = total_edges / total_nodes
```

**Known confound: node scope.** `edges/nodes` is not scope-invariant because nodes aren't scope-normalized. A 4-node Hemingway with 6 edges has density 1.5; a 9-node Chiang with 9 edges has density 1.0 — but whether the Hemingway is "denser" than the Chiang depends on whether you mean "more edges per beat of equal scope" (yes) or "more overall structural complexity" (not obviously). Since node granularity is a property of the story's natural beat count, normalizing it would be either arbitrary (token-based?) or information-destroying. What density actually measures is *edge-intensity per beat*; that's a real thing even when confounded with beat count. Use the metric as a coarse proxy, not a fine discriminator. The pre-registered `hub_index` replacement axis (below) captures a different property (concentration vs. distribution) and can be swapped in if the confound makes density non-discriminating in practice.

Binning (5 bins). **All bin names describe the DAG's edge density per beat, not the story's thematic weight or narrative intensity.** Chiang's "Story of Your Life" — a thematically dense novella — would land in the `Skeletal` or `Simple` bin because its DAG has few edges per beat, not because the story is thin. Conversely, a genome-dense DAG can be a relatively thin story if the edges are generic. Use these labels to describe the *genome*, not to grade the story.

| Bin | Range | Character (of the DAG) |
|---|---|---|
| 0 | `[0.0, 1.2)` | Skeletal — near-linear chain of beats with few cross-edges |
| 1 | `[1.2, 1.8)` | Simple — mostly sequential with occasional non-adjacent edges |
| 2 | `[1.8, 2.5)` | Moderate — standard mix of causal sequence + cross-cutting relationships |
| 3 | `[2.5, 3.2)` | Dense — rich cross-beat relationships (high disclosure + motivates edge count) |
| 4 | `[3.2, ∞)` | Very dense — every beat connects to many others (risk: overcomplexity for short fiction) |

### Why these two axes

They were chosen because:

1. **Both are deterministic.** No LLM classifier, no inter-run drift. Two DAGs with the same edge list always land in the same cell.
2. **They are hypothesized to be (mostly) uncorrelated — verify post-run.** Disclosure ratio doesn't directly determine density; a high-disclosure structure can be skeletal (few beats with heavy reveals) or dense. An all-causal structure can be skeletal (linear chain) or dense (many cross-beat causal relationships). That said, typed-edge-heavy DAGs may tend toward higher density in practice (they carry more relationships per node), so empirical DAG distributions could populate diagonally. Pearson correlation between the two axes across run terminals is tracked in `implementation.md` §Metrics exported; if |ρ| > 0.5, the grid's orthogonality assumption is violated and the density axis is replaced with **hub index** (pre-registered replacement, below).

**Pre-registered replacement axis: hub index.** `hub_index = (edges incident to top-20%-by-degree nodes) / total_edges`. Measures whether structural load concentrates at a few key beats (high hub_index → hub-and-spoke; low hub_index → distributed). Computed deterministically from the DAG's degree distribution; O(n+e). Chosen over cycle-topology metrics because (a) our DAGs are acyclic by construction so cycle counts reduce to trivial signals, and (b) hub-index cleanly distinguishes the common "anchor-centric" DAG shape from distributed ones. Binning to be calibrated empirically on the same pilot data that triggers the swap — pre-registered to prevent post-hoc axis choice inviting p-hacking on the grid layout.
3. **They capture meaningfully different story modes.** Low disclosure × low density = linear plot-driven story. High disclosure × low density = Hemingway-style spare reveal. Low disclosure × high density = Chiang-style implication web. High disclosure × high density = Jackson-style where the reveal recontextualizes everything.

Other candidate axes considered and rejected:

- **Edge-type entropy** (how evenly distributed are the 5 edge types). Highly correlated with structural density; adds no new signal. Rejected.
- **Climax position** (where in topological order is the climax). Depends on bidirectional phase boundary which is a design artifact; not a meaningful dimension. Rejected.
- **Average beat specificity** (heuristic measure of beat-sketch concreteness). Would require the beat-specificity heuristic from `mcts.md` §Tension Inference to be consistent, which it isn't guaranteed to be across runs. Rejected for axis use; retained as archive metadata.

### Grid size: 25 cells

5×5 = 25 cells is deliberately coarser than Stage 1's abandoned 6×6×6×5×3 = 3,240-cell grid. The lesson from Stage 1 (`lab/issues/2026-04-01-map-elites-review.md`): archive density matters. At 25 cells with ~30 DAGs per run (3–5 concepts × 4 presets, including non-advancing), each cell averages ~1 DAG per run. This is sparse but functional. Across 10+ runs, the archive densifies enough that within-cell competition fires regularly.

---

## Population Rules (v1: write-only)

In v1, the archive is **write-only** — every terminal DAG that passes validation gets written to its cell with full metadata, and a single cell may hold multiple entries. No pairwise competition between incumbents and challengers fires at insertion time.

Rationale (decided 2026-04-19 second review pass, see `lab/issues/2026-04-19-stage-2-critical-review-followups.md` Item 3):

1. Cross-concept / cross-run archive value depends on accumulated data that v1 won't have.
2. Competitive insertion adds $0.30–$12 per run without affecting v1 output (winners go to Stage 3 via the tournament, not the archive).
3. Multiple entries per cell give v2 retrieval/analysis richer material than a single "survivor" per cell.

### What v1 writes

A new DAG is written to the archive when:

1. The DAG has passed all validation gates (`overview.md` §Validation Protocol).
2. MCTS for this DAG's tree has terminated (no mid-tree snapshots).
3. Regardless of tournament rank — non-advancing DAGs are also archived.

Cell coords are computed deterministically from `(disclosure_ratio_bin, structural_density_bin)`; the DAG is appended to that cell's entry list with its metadata.

### v1.5: competitive insertion

The previous competitive-insertion design is preserved below for v1.5 reference. Re-enable once accumulated archive data shows cell churn is a real concern.

- **Empty cell**: DAG inserted unconditionally.
- **Occupied cell**: incumbent and challenger compete via full 3-judge × 2-ordering pairwise on all 8 dimensions. Winner holds; loser goes to compost. On tie, incumbent holds.
- **Gate 3 skip**: if challenger is structurally similar to incumbent (same cell, edge-histogram L1 < 0.3, node count within 20%), skip pairwise and let incumbent hold.

---

## Archive as Cross-Concept Memory (within a run)

A concept-A DAG and a concept-B DAG may land in the same cell within a run. The v1 write-only archive simply stores both; no competitive insertion fires. Tournament-level feedback and pattern mining operate on the union of entries in the run.

### Caveat

Cross-concept comparison is not 1:1 meaningful. A sparse reveal structure for a concept about small-town violence is not directly comparable to a sparse reveal structure for a concept about grief. The pairwise judges see both concepts (included in the judge prompt) and can account for concept context. But archive cell membership blurs concept context.

**Mitigation**: archive entries store their originating concept. Future consumers (pattern mining) can filter cells by concept type when that matters.

---

## Cross-Run Persistence — Deferred

Stage 1 does not yet persist structures across runs (no `compost.db`, no SQLite "compost" or "compost_lessons" tables exist in the codebase — "compost" today names a mutation operator, not a storage layer). Stage 2 inherits that constraint.

In v1, the archive is per-run only: serialized as `qd_archive.json` in the run directory. Non-advancing DAGs are preserved within the run for tournament analysis and inspection; they do not carry forward to future runs.

When Stage 1 lands cross-run storage, Stage 2 will extend it with a structural-archive table. Candidate schema (record it here so we don't have to reinvent when the time comes):

```text
stage_2_archive row:
  dag_json, concept_id, preset,
  disclosure_bin, density_bin,
  edge_histogram (per-type counts), node_count,
  tournament_rank, source_run, created_at
```

### Future feedback to Stage 1 (deferred with cross-run storage)

Two mechanisms planned for when cross-run persistence exists:

1. **Structural pattern templates.** Mine cells with multiple high-scoring entries for common structural features; expose as exemplars to Stage 1's `compost` mutation operator.
2. **Episodic failure lessons.** Judge reasoning from bottom-ranked DAGs becomes context for future Stage 1 runs: "past stories with {concept_feature} tended to produce structures that failed at {structural_aspect}."

Both require accumulated data to be useful; both wait on Stage 1's cross-run infrastructure.

---

## Planned v2: Archive as Retrieval for Expansion

After enough runs have populated the archive, it becomes a library of proven structural patterns per concept type. A v2 enhancement: during MCTS expansion, inject 2-3 archived DAGs from the closest concept type as structural exemplars in the prompt — "Here are DAGs that worked for similar concepts; use them as inspiration, not a template."

This is retrieval-augmented generation at the DAG level, mirroring Stage 1's compost-heap philosophy (past artifacts feed future generation). Not implemented in v1 because:

1. V1 runs have an empty archive; nothing to retrieve from
2. Concept-type similarity retrieval needs calibration data to work well
3. Exemplars risk anchoring new DAGs too strongly — would need "inspiration not template" prompt framing validated empirically

Revisit after ~10 runs have populated the archive. Design considerations then:
- How to measure concept-type similarity for retrieval (Stage 1 already classifies concept types; use that)
- How many exemplars to inject (2-3 is a reasonable starting point)
- Whether to exclude the current concept's own prior DAGs to avoid self-reinforcement

## Archive Visualization

For run output (`results/run_<timestamp>/stage_2/qd_archive.json`):

```json
{
  "run_id": "run_20260418_221534",
  "grid_size": [5, 5],
  "axes": {
    "disclosure_ratio": {
      "bins": [[0.00, 0.10], [0.10, 0.25], [0.25, 0.40], [0.40, 0.55], [0.55, 1.00]],
      "labels": ["None", "Light", "Balanced", "Heavy", "Dominant"]
    },
    "structural_density": {
      "bins": [[0.0, 1.2], [1.2, 1.8], [1.8, 2.5], [2.5, 3.2], [3.2, 100.0]],
      "labels": ["Skeletal", "Simple", "Moderate", "Dense", "Very dense"]
    }
  },
  "cells": [
    {
      "disclosure_bin": 4,
      "density_bin": 0,
      "label": "Dominant × Skeletal",
      "entries": [
        {
          "dag_id": "c_a8f12e_cassandra_tournament_1",
          "concept_id": "c_a8f12e",
          "preset": "cassandra_ish",
          "tournament_rank": 1
        }
      ]
    },
    ...
  ]
}
```

A CLI tool (`owtn qd-archive-viz`) renders the grid as a text heatmap for inspection:

```
              disc=0    disc=1    disc=2    disc=3    disc=4
dens=4       .         .         c_12a     .         .
dens=3       .         c_a8f/P   c_a8f/C   .         .
dens=2       c_9b2/R   c_9b2/P   c_a8f/W   .         c_12a/?
dens=1       .         .         .         c_a8f/?   .
dens=0       .         .         .         .         c_12a/J
```

where each cell shows the concept_id and preset-letter of the current occupant (J for Jackson-style hypothetical, etc.). Empty cells are `.`. Aids debugging.

---

## Per-Run Archive Reset

Each Stage 2 run starts with an empty archive. The archive is about within-run diversity; cross-run accumulation is not implemented in v1 (see §Cross-Run Persistence — Deferred).

---

## Open Questions Surfaced in QD Archive Drafting

1. **Bin boundaries are guesses.** The disclosure_ratio and structural_density bin ranges were chosen by informal analysis of what "low/mid/high" should mean. First-run data may show the distribution of actual DAGs is skewed (e.g., most DAGs fall into bin 1–2 on density with nothing in 0 or 4). Recalibrate empirically.

2. **Density as a proxy for complexity.** Structural density is a rough measure. Two DAGs with the same density can have very different structural character (e.g., 10 causal edges linearly vs. 10 edges forming a hub-and-spoke). Consider adding a hub-index or cycle-topology metric as a third axis if initial data shows density isn't discriminating.

3. **Archive across Stage 2 phases.** Should we archive forward-phase terminals separately from backward-phase (final) terminals? Current design archives only final (backward-phase) winners. Forward-phase terminals have no full DAG yet; archiving them would confuse cell assignment. Keep current design.

4. **Per-concept archive alongside global archive?** The current design has one global 5×5 grid across all concepts in a run. For small runs this is reasonable. For heavy runs (12 concepts × 4 presets = 48 DAGs) the grid becomes dense enough that some cells may churn heavily. Consider per-concept sub-grids or a hierarchical archive. Defer to post-v1 data.

5. **Archive insertion cost cap.** Moot in v1 — write-only insertion has no pairwise cost. Relevant only if v1.5 restores competitive insertion. At that point, Gate-3-style similarity skip thresholds need tuning to keep archive maintenance bounded.

6. **Retention policy when cross-run storage lands.** Deferred with the storage layer itself (§Cross-Run Persistence — Deferred). Starting point: keep top-K per cell indexed by `created_at`.
