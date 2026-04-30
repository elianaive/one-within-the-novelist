# Stage 2: Implementation

This document covers the code-level decisions: file layout, module boundaries, config schema, integration with existing OWTN infrastructure, test structure, and cost/budget enforcement. It is the bridge from the design docs to actual code.

Stage 2 follows Stage 1's organizational conventions where applicable. Where it diverges (no ShinkaEvolve, MCTS instead of async evolution), new structure is introduced.

---

## Package Layout

Stage 2 follows Stage 1's per-concern layout: code lives alongside Stage 1's in the existing top-level packages (`owtn/evaluation/`, `owtn/models/`, `owtn/prompts/`), with Stage-2-specific modules named `stage_2` within each. New top-level packages are introduced only for concerns Stage 1 doesn't already have a home for (MCTS).

```
owtn/
├── evaluation/
│   ├── stage_1.py                # (existing)
│   ├── stage_2.py                # NEW — reward function, champion tracking, within-concept pairwise orchestration
│   ├── pairwise.py               # (shared — dual-ordering + per-criterion voting; Stage 2 calls into it)
│   ├── tournament.py             # (shared — round-robin mode used for within-concept tournament)
│   ├── models.py                 # (shared — add Stage 2 Pydantic models here or split into submodule if it grows)
│   └── ...
├── models/
│   ├── stage_1/                  # (existing — concept_genome.py, config.py, seed_bank.py, classification.py)
│   └── stage_2/                  # NEW
│       ├── __init__.py
│       ├── dag.py                # DAG, Node, Edge dataclasses + validation
│       ├── mcts_node.py          # MCTSNode (tree wrapper around DAG + bookkeeping)
│       ├── pacing.py             # Preset metadata + expansion-hint text (metadata-only in v1)
│       ├── config.py             # Stage2Config
│       └── handoff.py            # Stage2Output / handoff manifest models
├── prompts/
│   ├── stage_1/                  # (existing — base_system.txt, operators/, pairwise_*.txt, registry.py)
│   └── stage_2/                  # NEW
│       ├── __init__.py
│       ├── registry.py           # Prompt assembly, analogous to stage_1/registry.py
│       ├── base_system.txt       # Base system message (shared with judges)
│       ├── expansion.txt         # MCTS expansion prompt (returns K=4 actions, cached per leaf)
│       ├── seed_motif.txt        # Motif extraction (preset-agnostic; one call per concept — anchor itself is read from concept.anchor_scene)
│       ├── contradiction_check.txt  # rewrite_beat contradiction detection
│       ├── champion_brief.txt    # Tree-level feedback summarizer (adapts stage_1/parent_brief.txt)
│       ├── judge_system.txt      # Judge base system (extends base_system.txt)
│       ├── judge_user.txt        # Judge pairwise comparison prompt
│       └── rubric_anchors/       # One file per dimension (see §Prompt Templates)
├── stage_2/                      # NEW — MCTS algorithm code (no Stage 1 analog)
│   ├── __init__.py
│   ├── mcts.py                   # Selection, expansion, rollout, backprop
│   ├── bidirectional.py          # Forward + backward phase orchestration
│   ├── refinement.py             # Phase 3 cross-phase refinement pass (add_edge only)
│   ├── operators.py              # seed_root, add_beat, add_edge, rewrite_beat
│   ├── rendering.py              # Incident-encoded outline + STRUCTURAL TENSIONS
│   ├── archive.py                # QD archive (write-only in v1; per-run JSON)
│   └── runner.py                 # Top-level Stage 2 runner (entry point)
└── judging/                      # (shared with Stage 1 — no new judging module)
```

### What's new vs. shared

**Shared with Stage 1 (no change needed, or small additions):**
- `owtn/llm/` — reused as-is
- `owtn/judging/` — reused; verify `tier_b.py` handles the DAG rendering path without Stage-2-specific logic
- `owtn/evaluation/pairwise.py` — dual-ordering + per-criterion voting unchanged; Stage 2 calls it with different prompts/rubrics
- `owtn/stage_1/tournament.py` — reused for within-concept round-robin
- `owtn/models/judge.py` — unchanged

**Net new infrastructure:**
- `owtn/stage_2/` — MCTS algorithm (no Stage 1 analog)
- `owtn/models/stage_2/dag.py` — typed-edge DAG data structure
- `owtn/stage_2/rendering.py` — incident-encoded outline renderer
- `owtn/stage_2/archive.py` — QD archive (write-only in v1; per-run JSON)
- `owtn/evaluation/stage_2.py` — Stage-2-specific reward + champion orchestration on top of shared pairwise
- `ChampionBrief` model + tree-subject summarizer — adapts `owtn/evaluation/feedback.py`'s `ParentBrief` pattern; subject shifts from per-program to per-tree because Stage 2 champion churn is too fast for per-champion history. See `lab/issues/2026-04-20-stage-2-expansion-feedback-summarizer.md` for the mechanism; decide at implementation time whether to generalize `feedback.py` or create a Stage-2 parallel.

**Deferred to v1.5:**
- Narrative forecasting (dropped per critical-review follow-ups Item 3)
- `post_hoc_rationalize` operator (dropped per same issue)
- QD archive competitive insertion (dropped per same issue)
- QD archive cross-run persistence (depends on Stage 1 landing a cross-run store)
- Stage 4 adaptation re-entry mechanics (deferred to Stage 4 design)

---

## Data Models

### DAG

```python
RoleName = Literal["climax", "reveal", "pivot"]

# MotifMode definitions are the responsibility of the mode glossary in
# `docs/stage-2/overview.md` §Per-node motifs §Mode glossary (authoritative).
# Comments here are reminders, not the spec — keep the glossary in sync.
MotifMode = Literal[
    "introduced",   # first appearance, named or shown directly
    "embodied",     # motif's shape is in the beat's structure/interior; not named or handled
    "performed",    # a character handles, wears, or gestures with the motif-object
    "agent",        # the motif plays a causal role in the beat's events
    "echoed",       # return with valence preserved
    "inverted",     # return with valence reversed
]

@dataclass
class MotifMention:
    """A motif's appearance at a single node, with its mode.

    The six modes answer distinct questions a generator must decide at each
    beat: is the motif named (introduced), shaped (embodied), handled (performed),
    causal (agent), or recurring (echoed / inverted)? Without the distinction,
    generators default to `introduced` at every tag, flattening recurrence.
    """
    motif: str                           # must exactly match a string in DAG.motif_threads
    mode: MotifMode

@dataclass
class Node:
    id: str
    sketch: str
    role: list[RoleName] | None = None   # list to allow multi-role beats (O'Connor's grace: ["climax", "pivot"])
                                         # First entry is primary: drives MCTS phase dispatch
                                         # and Stage 1 handoff surface slot.
    motifs: list[MotifMention] = field(default_factory=list)

@dataclass
class Edge:
    src: str
    dst: str
    type: Literal["causal", "disclosure", "implication", "constraint", "motivates"]
    # Payload fields: all optional at the type level, enforced via validator
    realizes: str | None = None                 # causal
    reframes: str | None = None                 # disclosure
    withheld: str | None = None                 # disclosure
    disclosed_to: list[str] | None = None       # disclosure: "reader" or character names
                                                # default (when None or missing) is ["reader"]
                                                # distinguishes authorial reveal from diegetic recognition
    entails: str | None = None                  # implication
    prohibits: str | None = None                # constraint (local; story-scoped rules go in DAG.story_constraints)
    agent: str | None = None                    # motivates (local; whole-story arcs go in DAG.character_arcs)
    goal: str | None = None                     # motivates
    stakes: str | None = None                   # motivates (optional)

@dataclass
class CharacterArc:
    """Whole-story trajectory for a character.

    Spans ≥3 non-adjacent nodes. Contrast with a `motivates` edge, which
    installs a *local* intention at one node anchoring an adjacent action.
    See `docs/stage-2/overview.md` §Character arcs for scoping.

    No `touches` list: the nodes where a character is active are derivable
    from beat sketches (Tier 1 entity extraction) and from motivates edges.
    Duplicating that on the arc is tautological at cardinality ~N.
    """
    agent: str
    goal: str
    stakes: str | None = None

@dataclass
class StoryConstraint:
    """A diegetic rule that holds across the whole story.

    Parallel to CharacterArc for motivations. Use when a prohibition holds
    wholesale rather than between two specific beats (Hemingway's not-naming,
    Jackson's ritual-silence). Local constraints — A forecloses a capacity
    at adjacent B — stay as `constraint` edges.
    """
    prohibits: str
    lifts_at: str | None = None   # node ID where rule breaks; None = holds to story end

@dataclass
class DAG:
    concept_id: str
    preset: str
    motif_threads: list[str]                  # set once by seed_root; ~2-3 entries
    concept_demands: list[str]                # set once by seed_root alongside motifs (same LLM call);
                                              # zero or more one-sentence Tier 3 predicates.
                                              # Empty is the common case — see overview.md §Concept demands.
    nodes: list[Node]
    edges: list[Edge]
    character_arcs: list[CharacterArc]        # may be empty
    story_constraints: list[StoryConstraint]  # may be empty
    target_node_count: int
```

(`post_hoc_rationalize` is deferred to v1.5; if restored, its state log is written as a run-log side-channel, never on the genome. See `operators.md` §post_hoc_rationalize §State log.)

### MCTSNode

```python
@dataclass
class MCTSNode:
    dag: DAG                          # the partial DAG at this tree position
    action: Action | None
    parent: MCTSNode | None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    cumulative_return: float = 0.0
    terminal: bool = False
    fully_expanded: bool = False
    phase: Literal["forward", "backward"] = "forward"
    cached_candidate_actions: list[Action] | None = None
    # preset_prior_bonus deferred to v1.5 with tension inference — see mcts.md §Tension Inference
```

### Stage 1 input

Shape derived from inspecting actual Stage 1 output at `results/run_<timestamp>/stage_1/` — specifically `champions/island_{idx}.json`, `best/main.json`, and `tournament.json`. Fields reflect what Stage 1 actually emits today, not what an idealized handoff would contain.

```python
@dataclass
class Stage1Winner:
    program_id: str                    # Shinka program UUID (matches champions/island_{idx}.json.id)
    genome: ConceptGenome              # existing Stage 1 Pydantic model; parsed from the program's `code` JSON string
                                       # fields: premise, thematic_engine, target_effect, character_seeds,
                                       # setting_seeds, constraints, style_hint
    combined_score: float              # from programs.sqlite / champions JSON
    tournament_rank: int               # from tournament.json
    tournament_dimension_wins: list[dict]  # per-match dimension votes (a/b/tie) from tournament.json
                                           # nested per-program → per-match → per-dim;
                                           # Stage 2 uses as judge-signal context, not aggregated
    affective_register: str | None     # from metadata (e.g. "JOY"); MAP-Elites classifier is disabled,
                                       # so these are metadata tags, not a source of selection pressure
    literary_mode: str | None          # from metadata (e.g. "CARNIVALESQUE")
    patch_type: str                    # which Stage 1 mutation operator produced this (e.g. "collision")
    source_run: str                    # run id, for lineage
    # NOTE: concept_demands is NOT a Stage 1 field. Demands are derived by Stage 2's seed_root
    # operator alongside motif_threads (one merged LLM call). They live on the DAG, not on this
    # handoff dataclass. See operators.md §seed_root and implementation.md DAG dataclass.
```

**Fields intentionally not on this dataclass:**
- `auto_detected_type` — MAP-Elites concept-type classifier is disabled in Stage 1 (see `CLAUDE.md`); only `affective_register` and `literary_mode` are populated and both are metadata-only. Anchor role (climax / reveal / pivot) now rides on `concept.anchor_scene.role` directly, evolved and tournament-selected at Stage 1 — no pre-classified type, no Stage-2-side role selection.
- `identified_risks` — Stage 1 does not currently produce a structured risks list per concept. Judge reasoning chains sit in `results/run_<timestamp>/stage_1/best/results/metrics.json` under `private_metrics.match_critiques` but are per-match, not per-concept-aggregated.
- `judge_panel_scores` — Stage 1 stores `combined_score` (scalar) on the program and per-match `dimension_wins` in `tournament.json`. There's no "panel score per dimension per concept" aggregate; if Stage 2 wants that, it has to aggregate the match-level votes itself.
- `judge_reasoning_chains` — available in the `metrics.json` path above but requires Stage 2 to parse and (optionally) compress. Treat as nice-to-have context in the expansion prompt; the genome itself carries most of what Stage 2 needs.

The Stage 2 runner's handoff reader (`owtn/stage_2/runner.py`) loads these fields from the Stage 1 run directory; a Stage 1-side "handoff manifest" writer is a potential cleanup but not a prerequisite for Stage 2.

### Handoff manifest

```python
@dataclass
class Stage2Output:
    concept_id: str
    preset: str
    tournament_rank: int
    qd_cell: tuple[int, int]
    genome: DAG
    stage_1_forwarded: Stage1Winner
    mcts_reward: float
    adaptation_permissions: list[str]  # v1: named only, no re-entry mechanics ship
```

---

## Config Schema

`configs/stage_2/light.yaml`, `medium.yaml`, `heavy.yaml`:

```yaml
stage_2:
  # MCTS budget
  iterations_per_phase: 50           # 25 (light) | 50 (medium) | 75 (heavy)
  phase_3_iterations: 5              # cross-phase refinement
  k_candidates_per_expansion: 4   # BiT-MCTS's reported optimum; cached per leaf, consumed one per subsequent expansion

  # UCB
  exploration_constant: 0.5
  # Progressive widening has no formula parameters — widening emerges from cache exhaustion (see mcts.md §Cached expansion).

  # Running-champion re-challenge (see mcts.md §Open Questions #2)
  rechallenge_interval: 25            # every N iterations within a phase, refresh stale high-scoring terminals
  rechallenge_top_pct: 0.10           # top fraction of terminals by W/N to re-score against current champion

  # Models — cross-family discipline: cheap_judge must differ in family from expansion/rollout
  expansion_model: "deepseek-v4-pro"      # generative; creative, tree-persistent outputs (cached K=4 candidates per leaf); matches Stage 1 generation
  rollout_model: "deepseek-v4-flash"      # generative; cheap, high-volume, ephemeral one-step extensions during simulation (same family as expansion — rollouts are one-step ephemeral checks, so cross-family with expansion doesn't buy diversity)
  cheap_judge_model: "gpt-5.4-mini"       # evaluative; rollout reward signal; different family from expansion/rollout
  full_panel_on_promotion: true           # fire full 4-judge × 2-ordering panel to verify cheap-judge wins before champion promotion
  full_panel_rejection_backprop: 0.5      # score backpropagated when cheap judge says win but full panel rejects (see mcts.md §Reward Function)
  cheap_judge_agreement_alert: 0.70       # log an alert if cheap-vs-full agreement drops below this

  # Presets per config (light runs 2 for pilot measurement; medium/heavy run 4)
  presets:
    light:
      - "cassandra_ish"
      - "randy_ish"
    medium:
      - "cassandra_ish"
      - "phoebe_ish"
      - "randy_ish"
      - "winston_ish"
    heavy:
      - "cassandra_ish"
      - "phoebe_ish"
      - "randy_ish"
      - "winston_ish"

  # Pacing preset parameters (full table — values are tentative, calibrate empirically)
  preset_params:
    cassandra_ish:
      min_rest_beats: 1
      max_flat_beats: 4
      intensity_variance: tight
      recovery_required: true
    phoebe_ish:
      min_rest_beats: 3
      max_flat_beats: 6
      intensity_variance: tight
      recovery_required: true
    randy_ish:
      min_rest_beats: 0
      max_flat_beats: 8
      intensity_variance: wide
      recovery_required: false
    winston_ish:
      min_rest_beats: 2
      max_flat_beats: 3
      intensity_variance: tight
      recovery_required: true  # with explicit reward beat

  # Handoff
  advance_from_stage_1: "all"         # "all" | "top_k"
  max_concepts_from_stage_1: null     # optional cap
  top_k_to_stage_3: 1                 # 1 (light) | 2 (medium) | null=all (heavy)
  near_tie_promoted: true             # when 1st and 2nd are within 1 dim-win, both advance

  # Evaluation (8 dimensions — Edge-Type Appropriateness merged into Causal Soundness as "Edge Logic" on 2026-04-19)
  dimensions: [
    "edge_logic",
    "motivational_coherence",
    "tension_information_arch",
    "post_dictability",
    "arc_integrity_ending",
    "structural_coherence",
    "beat_quality",
    "concept_fidelity_thematic"
  ]
  # Gates 2 and 3 (forecasting-dependent) dropped in v1 — see evaluation.md

  # Budget
  per_concept_time_budget_minutes: 30
  per_phase_time_budget_minutes: 10
  no_improvement_cutoff_iterations: 15

  # Archive
  archive_bin_boundaries:
    disclosure_ratio: [0.10, 0.25, 0.40, 0.55]
    structural_density: [1.2, 1.8, 2.5, 3.2]

  # Node count targets by prose length
  node_count_targets:
    1000: [3, 5]
    3000: [5, 8]
    5000: [7, 12]
    10000: [10, 18]
```

Stage 1's config and Stage 2's config coexist in the same run YAML at the top level (similar to how judges and LLM config coexist today):

```yaml
# configs/light.yaml
stage_1:
  ...
stage_2:
  ...
judges:
  ...
llm:
  ...
```

---

## Runner Integration

`owtn/stage_2/runner.py` is the top-level Stage 2 entry point. Invocation pattern:

```bash
uv run python -m owtn.stage_2 --config configs/stage_2/light.yaml --stage-1-results results/run_<timestamp>/stage_1/
```

The runner's handoff reader loads winners from `stage_1/champions/island_*.json` (each island's promoted champion), parses the embedded genome JSON out of `program.code`, and consults `stage_1/tournament.json` for per-match dimension votes. No separate "handoff manifest" is written by Stage 1 today; if that becomes annoying to maintain, add a `stage_1/stage_2_handoff.json` writer to the Stage 1 runner.

The Stage 2 runner:

1. Loads Stage 1's output directory — reads all island champions from `stage_1/champions/*.json`, parses the embedded genome from each champion's `code` field, and reads `stage_1/tournament.json` for per-match dimension votes. All Stage 1 island champions advance (see scoping §13).
2. For each advancing concept, spawns an async task that runs the 4 pacing presets' MCTS trees concurrently.
3. Collects each concept's tournament results and archives non-advancing DAGs.
4. Writes the handoff manifest for Stage 3.

### Concurrency

```python
async def stage_2_run(stage_1_winners: list[Stage1Winner], config: Stage2Config) -> Stage2Manifest:
    # Concept-level parallelism
    concept_tasks = [
        asyncio.create_task(run_concept(winner, config))
        for winner in stage_1_winners
    ]
    concept_results = await asyncio.gather(*concept_tasks)
    ...

async def run_concept(winner: Stage1Winner, config: Stage2Config) -> ConceptResult:
    # Seed once per concept, preset-agnostic. Failure here blocks the whole concept.
    shared_seed = await seed_root(winner.genome, config)
    if shared_seed is None:
        return ConceptResult.unseedable(winner.concept_id)

    # Preset-level parallelism — all 4 trees share the same seed as their root
    preset_tasks = [
        asyncio.create_task(run_preset_tree(winner, preset, shared_seed, config))
        for preset in config.stage_2.presets
    ]
    preset_terminals = await asyncio.gather(*preset_tasks)
    # ... (within-concept tournament, archive non-winners, build handoff)

async def run_preset_tree(winner: Stage1Winner, preset: str, config: Stage2Config) -> DAG | None:
    # seed_root is called ONCE per concept, outside run_preset_tree — see run_concept below.
    # The seed is shared across all preset trees for this concept.
    forward_champion = await mcts_phase(shared_seed, phase="forward", config=config, preset=preset)
    backward_champion = await mcts_phase(forward_champion, phase="backward", config=config, preset=preset)
    return backward_champion
```

### Budget enforcement

A `Stage2Budget` instance tracks cumulative LLM call costs and enforces the `per_concept_time_budget_minutes` and global cost cap. Each LLM call routes through a budget-aware wrapper:

```python
class BudgetTracker:
    def __init__(self, config: Stage2Config):
        self.per_concept: dict[str, ConceptBudget] = {}
        self.global_cost: float = 0.0
        self.global_cap: float | None = config.stage_2.global_cost_cap

    async def checked_call(self, concept_id: str, call: Awaitable[LLMResponse]) -> LLMResponse | BudgetExceeded:
        budget = self.per_concept[concept_id]
        if budget.should_abort():
            return BudgetExceeded(reason=budget.abort_reason)
        response = await call
        budget.record_call(response)
        return response
```

Budget exhaustion per concept terminates all its preset trees cleanly (currently-running MCTS iterations complete; no new ones start; current champions become the concept's outputs).

---

## Prompt Templates

All prompts live in `owtn/prompts/stage_2/` as `.txt` files with `{placeholder}` syntax for runtime substitution. This matches Stage 1's pattern.

### base_system.txt

The shared base system message introduced in `operators.md` §Base System Message. Used by:
- MCTS expansion calls
- motif extraction (inside `seed_root`)
- judge calls (with judge-specific persona prepended)
(Rationalization calls would be added here if `post_hoc_rationalize` returns in v1.5.)

Placeholders:
- `{PHASE_CONTEXT}` — "you are in the forward phase" / "you are in the backward phase"
- `{CONCEPT_CONTEXT}` — full Stage 1 concept dump (premise, target effect, character seeds, etc.)
- `{PACING_PRESET}` — preset name and character

### expansion.txt

The MCTS expansion prompt, per `operators.md` §add_beat. Returns JSON array of up to K=4 proposed actions, cached on the leaf.

Placeholders:
- `{DAG_RENDERING}` — incident-encoded outline of the current partial DAG
- `{PERMITTED_EDGE_TYPES}` — list of edge types allowed in the current phase
- `{PACING_HINT}` — a short preset-specific hint (e.g., "Favor actions that establish relief after peaks — Cassandra")

### seed_motif.txt

Preset-agnostic motif extraction. Returns `{"motif_threads": ["...", "...", "..."]}` — 2–3 concrete recurring elements (objects, images, phrases) to thread through expansion prompts. The anchor itself is not generated here; `seed_root(concept)` reads `concept.anchor_scene.sketch` and `.role` directly off the Stage 1 genome.

Placeholders:
- `{CONCEPT_CONTEXT}` — full Stage 1 concept JSON
- `{ANCHOR_SKETCH}` — the concept's anchor scene sketch
- `{ANCHOR_ROLE}` — `climax | reveal | pivot`

No preset parameter. One motif extraction call per concept, shared across all preset MCTS trees.

### judge_system.txt + judge_user.txt

Follow Stage 1's prompt-split pattern. The `judge_system.txt` combines the persona's persona.yaml text with the Stage 2 base system message. The `judge_user.txt` contains the DAG rendering for STRUCTURE A and STRUCTURE B, the rubric for the dimension being judged, and the task.

### rubric_anchors.txt

One file per dimension is probably cleaner; use a directory:

```
owtn/prompts/stage_2/rubric_anchors/
├── edge_logic.txt
├── motivational_coherence.txt
├── tension_information_arch.txt
├── post_dictability.txt
├── arc_integrity_ending.txt
├── structural_coherence.txt
├── beat_quality.txt
└── concept_fidelity_thematic.txt
```

Each contains the dimension's definition, sub-criteria, endpoint anchors, score-3 exemplar, and literary examples — content from `docs/stage-2/rubric-anchors.md`.

---

## Archive Persistence (v1: per-run JSON)

The archive is in-memory during the run and serialized at run end to `results/run_<timestamp>/stage_2/qd_archive.json`. No cross-run SQLite schema in v1 — Stage 1 does not yet ship a cross-run store (`compost` in Stage 1 today names a mutation operator, not a storage layer), so Stage 2 has nothing to extend. See `qd-archive.md` §Cross-Run Persistence — Deferred.

```python
# owtn/stage_2/archive.py
class Stage2Archive:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.cells: dict[tuple[int, int], list[ArchiveEntry]] = defaultdict(list)

    def write(self, dag: DAG, concept_id: str, preset: str, tournament_rank: int) -> None:
        cell = self.compute_cell(dag)
        self.cells[cell].append(ArchiveEntry(dag=dag, concept_id=concept_id,
                                              preset=preset, tournament_rank=tournament_rank))

    def flush(self) -> None:
        """Serialize the full archive to qd_archive.json at run end."""
        out = self.run_dir / "stage_2" / "qd_archive.json"
        ...
```

Competitive insertion is deferred to v1.5 (see `qd-archive.md` §v1.5: competitive insertion). When cross-run persistence lands, this module gains a DB-backed writer that mirrors the in-memory grid.

---

## Testing

Test organization follows Stage 1's convention (`tests/test_stage_2/`):

```
tests/
├── test_stage_2/
│   ├── conftest.py               # Stage 2 fixtures (test DAGs, test concepts)
│   ├── test_dag.py               # DAG data structure, validation, cycle detection
│   ├── test_operators.py         # each operator's validation paths
│   ├── test_rendering.py         # incident-encoded outline format correctness
│   ├── test_pacing.py            # preset metadata, expansion-hint assembly
│   ├── test_mcts.py              # UCB selection, progressive widening, backprop
│   ├── test_bidirectional.py     # phase transition, edge-type restrictions per phase
│   ├── test_evaluation.py        # reward function, champion tracking
│   ├── test_tournament.py        # round-robin tournament mechanics
│   ├── test_archive.py           # QD archive, cell assignment, competitive insertion
│   └── test_handoff.py           # manifest construction
```

Shared fixtures in `tests/conftest.py`:

```python
@pytest.fixture
def canonical_dag_lottery() -> DAG:
    """Jackson's 'The Lottery' as a 5-node Stage 2 DAG."""
    ...

@pytest.fixture
def canonical_dag_hemingway() -> DAG:
    """Hemingway's 'Hills Like White Elephants' as a 4-node DAG."""
    ...

@pytest.fixture
def canonical_dag_chiang() -> DAG:
    """Chiang's 'Story of Your Life' as a 10-node DAG (implication-heavy)."""
    ...

@pytest.fixture
def canonical_dag_oconnor() -> DAG:
    """O'Connor's 'A Good Man Is Hard to Find' as an 11-node DAG (causal + motivates)."""
    ...
```

These four canonical DAGs are hand-crafted to exercise different parts of the typed-edge taxonomy and serve as offline regression anchors — any change to validation or rendering should preserve their validity.

**Test principles** (from `CLAUDE.md`):
- Don't assert on LLM output content — it's non-deterministic. Assert structural validity only.
- Mark tests requiring API keys with `@pytest.mark.live_api`.
- One concern per test file.

Offline test suite (no `live_api`): ~120 tests, runs in <5 seconds.

Live test suite: exercises seed_root, one-iteration MCTS, one-dimension judge call — about $0.50 per full run. Run weekly or on change to LLM-touching code.

---

## Cost Budget Enforcement

> **Cost numbers below are preliminary. Actual per-concept cost is being measured in the pilot harness (see §Pilot harness below).** The design changed substantially on 2026-04-19 — tiered judge replaced the hybrid reward function, Phase 3 refinement was added, `light.yaml` dropped from 4 presets to 2, and the rubric reduced from 9 to 8 dimensions. Paper estimates against the prior design are unreliable. Update this section with pilot `metrics.json` numbers once first run completes.

Expected cost drivers under the new design:
- **Cheap-judge rollout signal**: 1 LLM call per terminal, ~$0.001-$0.002 per call. Dominant loop by count (~200 per tree).
- **Full-panel promotion gate**: 6 LLM calls, fires only when cheap judge declares a win (~10-20 gates per phase).
- **Full-panel tournament**: 8 calls × C(presets,2) pairs per concept.
- **Full-panel archive insertions**: 8 calls per contested cell (subject to Gate 3 similarity skip).
- (Narrative forecasting dropped from v1 — see `mcts.md` §Narrative forecasting — deferred to v1.5.)

| Config | Presets | Concepts | Per-concept (preliminary) | Total Stage 2 (preliminary) |
|---|---|---|---|---|
| light | 2 | 3 | TBD (pilot) | TBD |
| medium | 4 | 5 | TBD (pilot) | TBD |
| heavy | 4 | 8–12 | TBD (pilot) | TBD |

Full-panel pairwise comparisons follow Stage 1's pattern: 4 judges × 2 orderings = 8 LLM calls, each call evaluates all 8 dimensions in one structured response (~$0.04–$0.08 per comparison). Cheap-judge comparisons are 1 call, ~$0.001–$0.002.

### Pilot harness

Before scaling, run a small pilot with full per-call cost logging. Output: `results/run_.../stage_2/metrics.json` with per-call-type costs, latencies, totals, and the cross-arm comparison below. This run's numbers become the authoritative estimates. Build the metrics collection first; it's cheap and it's the only way the cost numbers stay honest.

**Pilot scope:** ≥6 concepts × 2 presets × 20 iterations per phase. Earlier drafts specified 3 concepts; doubled because a 3-concept head-to-head has no statistical power — "Stage 2 wins 2 of 3" is p ≈ 0.5 under the null and cannot reject or confirm anything. The go/no-go thresholds below are directional at ≤3 concepts; hold the trigger until the expanded pilot reports. At 6 concepts, "Stage 2 wins ≥5" is p ≈ 0.1 under the null — still loose, but acting-on-it defensible.

**Pilot arms (run head-to-head per concept):**

1. **Stage 2 production arm.** Full MCTS + bidirectional + Phase 3 + tiered judge + within-concept tournament, per `mcts.md`. Outputs one winner DAG per concept (top of the tournament).
2. **Direct-outline baseline (single-shot).** One LLM call per concept asking for a typed-edge DAG outline using the same schema the expansion operator uses (5–8 nodes with sketch + optional role; edges with type + populated payload fields). No MCTS, no multi-preset, no tournament. Same concept input as arm 1. Implementation: `lab/scripts/stage_2_baseline.py`, one function, no new package dependency. Uses the same expansion-target model as Stage 2 proper (for a fair head-to-head on generator capability, not model tier). Isolates *single-shot LLM vs. search*.
3. **Iterated-generation baseline (budget-matched).** Generate N=20 independent direct-outline DAGs per concept using the same prompt as arm 2, then run a round-robin tournament (same full-panel pairwise protocol as the within-concept tournament) to select the winner. No MCTS, no cached progressive widening, no bidirectional phases — just parallel sampling plus tournament selection. Budget is chosen to roughly match arm 1's expansion + full-panel cost. This is the real competitor to MCTS: arm 2 isolates "single-shot vs. search," arm 3 isolates "tree search vs. budget-matched i.i.d. sampling." If arm 3 ties arm 1, the tree-search machinery is overhead.

**Head-to-head.** For each pilot concept, run a full-panel pairwise comparison (4 judges × 2 orderings = 8 calls) between every pair of arm winners, on all 8 dimensions. Record overall winner + per-dimension scores.

**Signal interpretation (at ≥6 concepts; directional at fewer):**
- **Stage 2 wins both baselines on ≥⅔ of concepts:** the full stack is earning its cost. Scale to medium.
- **Stage 2 beats arm 2 but ties or loses to arm 3:** the tree-search machinery is not doing work beyond budget-matched sampling. Triggers a go/no-go conversation — simplify radically (N-sample tournament may be the right architecture), or identify which specific MCTS mechanism adds value via the ablation suite below.
- **Stage 2 loses to arm 2:** the stack is worse than a single LLM call. Hard go/no-go — reconsider shipping MCTS.
- **Per-dimension asymmetry:** if Stage 2 wins structurally (Edge Logic, Arc Integrity) but loses on Beat Quality or Concept Fidelity vs. iterated-generation, that identifies which mechanisms earn their cost and which are overhead.

**Cost (baseline arms):** arm 2 is ~$0.05–$0.08 per concept (1 gen + 6 judge calls). Arm 3 is ~$0.45–$0.65 per concept (20 gen + ~19 pairwise tournament calls at ~$0.02 each). Total baseline cost at 6 concepts: ~$3–$5 — negligible against Stage 2's own budget.

**Pilot prerequisite (gate):** only the canonical-DAG gate (`tests/test_stage_2/conftest.py`, dev-order step 1b) remains. Tension-inference validation was removed in the semantic-presets refactor (2026-04-20) — no heuristic to validate. Preset structural divergence is measured *during* the pilot rather than guarded before it.

**Preset divergence gate before medium/heavy scale-up (added 2026-04-20).** `medium.yaml` and `heavy.yaml` currently list 4 presets. Do not run either config against real spend until the light pilot (2 presets: Cassandra + Randy) shows **edge-type histogram L1 distance ≥ 0.25** and **disclosure-ratio spread ≥ 0.15** across the two preset trees, averaged across pilot concepts. If the two trees collapse to near-identical outputs on the most deliberately-distinct preset pair (Cassandra vs Randy), a 4-preset run pays 4× for the same collapse. In that case:
- Either restore the tension-inferred parametric preset_prior (deferred to v1.5 in `mcts.md` §Tension Inference — its return trigger names this exact pilot outcome) before scaling, or
- Ship medium/heavy with 2 presets (Cassandra + Randy) and treat the extra budget as more concepts or more iterations, not more trees.

**Noise-floor control for preset divergence.** The L1 ≥ 0.25 threshold is uninformative without knowing what same-preset run-to-run variance produces. Before interpreting preset L1, run the same preset (Cassandra) twice with different LLM seeds on 2 pilot concepts. If baseline same-preset L1 is already ≥ 0.25, the signal is drowned by sampling noise — tighten the between-preset threshold to `baseline + 0.15` or restore the parametric preset_prior before scaling. Cheap: 2 concepts × 1 extra tree = ~15% of pilot budget.

Thresholds above are starting points; calibrate against pilot distributions before committing.

**Future pilot arms (not in first pilot):** Item 10's propose→critique expansion prompt would slot in as arm 4 once the first pilot confirms Stage 2 is worth running. A/B between the current flat-propose prompt and propose→critique→revise on the same concepts.

### MCTS ablation suite

Passing the head-to-head pilot (§Pilot harness) establishes Stage 2 beats budget-matched sampling. It does not establish *which* MCTS mechanisms are doing the work. Run this ablation after the first pilot to identify load-bearing vs. ornamental machinery. Each arm runs on the same 2 pilot concepts with fixed expansion seeds; compare against the production arm via full-panel pairwise on all 8 dimensions.

| Arm | Question | Change from production |
|---|---|---|
| UCB1 (γ=1) + no rechallenge | Does either non-stationarity mechanism earn its complexity, or are both redundant? | Set γ=1.0; disable event-driven rechallenge |
| UCB1 (γ=1) + rechallenge ON | Is rechallenge alone sufficient (cleaner: event-driven, targets affected branches)? | Set γ=1.0; keep rechallenge |
| D-UCB (γ=0.93) + no rechallenge | Is D-UCB alone sufficient? | Keep γ=0.93; disable rechallenge |
| Uniform-random selection | Does UCB exploration add signal at this budget? | Replace UCB with uniform random child selection; keep cached expansion |
| Greedy-argmax (c=0) | Does the exploration term matter? | UCB with c=0 (pure exploitation) |
| Zero structural side-signal (β=0) | Is the structural side-signal load-bearing? | Set β=0 |
| β sweep {0.05, 0.2} | Is β=0.1 in a sensible zone? | Sweep β; baseline is production (β=0.1) |

**Why two non-stationarity arms.** Production combines D-UCB (γ=0.93) with event-driven rechallenge — both attack the same problem (early-tree rewards inflated against weak champions). Either should be sufficient on its own; together they may over-correct, especially since promotions happen every 1–5 iterations. The first three arms isolate the contribution of each mechanism so the production combination can be defended (or simplified) on the data.

**Decision rule.** An arm that *ties or beats* production on both concepts replaces the production setting — "less complexity for equal quality" is always the right trade at this scale. An arm that loses on both concepts validates the mechanism; a 1-1 split is inconclusive and expands to more concepts.

**Budget.** ~$15–$25 total across 5 arms × 2 concepts (each arm is a full Stage 2 run at pilot scale, ~$3–$5). Fits inside the go/no-go decision window.

### Tier 1/2 consistency-check calibration

Evaluation's Tier 1 (entity state) and Tier 2 (edge payload plausibility) claim recall via extrapolation from ConStory-Checker (F1=0.742 on a different dataset). Calibrate before trusting:

- **Injection set.** Take the 4 canonical DAGs (`tests/test_stage_2/conftest.py`) and inject 30–50 defects: introduce unknown entities, reveal withheld content early, flip agent names, drop edges' causal antecedents. Mix with unmodified canonicals.
- **Measure.** Run the Tier 1 + Tier 2 pipeline on each. Record precision and recall per tier per defect class.
- **Decision.** If precision < 80%, the tier rejects too many good DAGs — relax the threshold or fall back to judges. If recall < 50% on a defect class, the tier isn't catching that class — relax the "reject" action to "flag into judge context" for that class only.

Cost: ~$0.50 total (50 DAGs × ~$0.005 per tier call × 2 tiers). Runs offline before the main pilot.

### Summarizer faithfulness mini-pilot

The champion-brief summarizer compresses raw full-panel critiques into 4 structured fields. If the summarizer systematically drops a feedback category — for example, judges critiquing reader-address failures get normalized into "pacing" — the expansion LLM never sees the real signal even though judges produce it. Calibrate before trusting:

- **Sample.** From the first pilot run, collect 20 (rendered_brief, raw_critiques) pairs across pilot trees.
- **Manual review.** For each pair, a human reader identifies the 2–3 main thrusts of the raw critiques and checks whether the brief preserves each. Specific failure mode to watch: structural-mechanism critiques (anything not on the 8 rubric dimensions) being collapsed into the nearest dimension and losing their specificity.
- **Decision.** If ≥3 of 20 briefs systematically drop a category present in raw critiques, the summarizer prompt or model is the wrong abstraction layer — either revise the prompt to preserve the missing category or downgrade to raw critiques verbatim until the prompt is fixed.

Cost: human-review only; no additional LLM calls beyond what the pilot already runs.

### Cost levers

When a specific run is cost-constrained, pull these levers in order:

1. **Reduce `iterations_per_phase`** from 50 to 25 — halves MCTS expansion, rollout, and cheap-judge costs proportionally.
2. **Reduce `presets`** — `light.yaml` is already 2 (Cassandra + Randy); `medium`/`heavy` can drop from 4 to 2 if needed.
3. **Reduce Stage 1 island count** — fewer concepts advancing to Stage 2.

Enforce a global cost cap in config: if the run exceeds it, pending concepts are skipped and the cap is logged.

### Metrics exported

Every Stage 2 run emits:
- Total LLM calls by kind (expansion, rollout, judge)
- Total cost by kind
- Per-concept cost breakdown
- Per-tree (preset) cost breakdown
- Average cost per terminal DAG evaluation
- **Phase 3 improvement rate** — how often the Phase 3 champion differs from the Phase 2 champion (see `mcts.md` §Monitoring Phase 3). Qualitative thresholds — first-pass cutoffs are <5% / 5–30% / >30% but these are guesses pending pilot calibration.
- **Inter-dimension correlation matrix** across pairwise comparisons (see below)
- **Preset-concept-type win rates** (for eventual meta-learning data)
- **Preset structural divergence per concept** — edge-type histogram L1, node count spread, disclosure-ratio spread across the preset trees (from the pre-implementation-fixes Item 3 measurement plan)
- **Stage 2 vs direct-outline baseline** — per-concept pairwise result (overall winner + per-dimension wins) and aggregate win-rate across pilot concepts. Sub-2-of-3 for Stage 2 is the go/no-go trigger documented in §Pilot harness.
- **Per-leaf cheap-judge variance** — variance of cheap-judge scores across a leaf's children after N≥3 visits each. High variance at leaves whose children are near-equivalent flags the cheap-judge position-bias pathology concentrated in UCB's closest calls (see `mcts.md` §Tiered judge design caveat).
- **Per-dimension absolute vote distribution** — for each of the 8 dimensions, the fraction of win / tie / loss votes across all pairwise comparisons. Dimensions collapsing to tie-dominance (e.g., Beat Quality > 60% ties) have low discriminatory power and flag persona/rubric mismatch or a weak structural lever driving the underlying quality.
- **Invalid-ID rate on expansion-proposed actions** — fraction of proposed actions rejected at validation for referencing nonexistent node IDs. High rates (>15%) mean effective K is below configured K=4 and expansion-prompt rendering may need clarification.
- **Rechallenge delta distribution** — mean and spread of W-score changes across re-challenge refresh cycles (see `mcts.md` §Open Questions #2). Large negative mean deltas indicate systematically inflated early-tree rewards; tune `rechallenge_interval` / `rechallenge_top_pct` accordingly.
- **Disclosure-ratio × structural-density correlation** — Pearson correlation across all terminal DAGs in the run. The QD archive assumes these are mostly uncorrelated (see `qd-archive.md`); if |ρ| > 0.5, the grid's orthogonality assumption is violated and the density axis should be reconsidered.

Data lives in `results/run_<timestamp>/stage_2/metrics.json`.

### Inter-dimension correlation monitoring

After the rubric reduction from 9 to 8 dimensions (pre-implementation-fixes Item 7, Edge-Type Appropriateness merged into Causal Soundness as Edge Logic), Post-dictability vs. Tension & Information Architecture is the pre-flagged at-risk pair — surprise-that-feels-inevitable is related to gap management; kept separate on the bet that it's a distinct craft property. But other pairs may co-vote too; don't pre-judge the space.

Per pairwise comparison, record the 8 per-dimension winners (a/b/tie). After the run, compute pairwise Cramér's V or a simple "co-vote rate" across all C(8,2)=28 dimension pairs: for each pair, the fraction of comparisons where both dimensions voted for the same side (or both tied). Flag any pair with co-vote rate > 0.80 (tightened from 0.85 per third-review Item 9 — catches more potential merges; false positives are cheap since this is logging-only in v1).

**Multiple-comparison caveat.** 28 dimension pairs × <100 comparisons per run means 1–2 pairs will cross 0.80 by chance alone. Treat per-run flags as directional: act on a pair only if it crosses threshold across ≥3 runs, or apply a Bonferroni-style effective threshold of ~0.92 before acting on single-run evidence. Otherwise "we flagged 4 pairs" is mostly sampling noise.

Beat Quality carries a separate non-co-vote risk: judges may evaluate sketch-specificity as prose craft rather than beat-level specificity. Monitored via judge-reasoning text search, not co-vote rate.

Two outcomes after flagging:
- Redundant: collapse the pair into one dimension in v2
- Informative but correlated: keep separate, but weight votes to avoid double-counting

This is logging-only in v1. No automatic response. Review after first real run.

---

## Migration From Stage 1 Code

Components that need small additions to support Stage 2:

- `owtn/evaluation/pairwise.py`: add support for the Stage 2 rubric format if it diverges from Stage 1's. If not, no change.
- `owtn/stage_1/tournament.py`: confirm it supports round-robin mode for small pools.
- `owtn/judging/personas.py`: no change.
- `owtn/llm/client.py`: if Stage 2 introduces new model routing (e.g., cheap rollout model), add routing rules.

---

## Development Order

Implementation-time dependency order (roughly topological):

1. **`dag.py`** — data structures and validation. No dependencies.
1b. **Canonical test DAGs (gate).** Write the four hand-crafted DAGs (Lottery, Hemingway, Chiang, O'Connor) as `tests/test_stage_2/conftest.py` fixtures. All four must pass every validation gate. This is a gate on step 2 — if a canonical DAG fails validation, either the DAG is wrong or the schema/gates are wrong, and that's information to act on before building the runner. See §Testing for the fixture spec.
2. **`rendering.py`** — renderer for incident-encoded outline. Depends on `dag.py`. Render all four canonical DAGs; visually inspect for correctness before continuing.
3. **`pacing.py`** — preset metadata (name, character description, 1–2 sentence expansion-hint text per preset). Metadata-only in v1; no tension inference, no preset priors. Depends on `dag.py`.
4. **`operators.py` (seed_root, validators)** — non-MCTS parts first. Depends on `dag.py`, `pacing.py`.
5. **`mcts.py`** — tree, UCB1 (no preset augmentation), progressive widening. Depends on `dag.py`, `pacing.py`.
6. **`bidirectional.py`** — phase orchestration. Depends on mcts.
7. **`evaluation.py`** — reward, champion tracking, `full_panel_critiques` collection on the tree. Depends on rendering, pairwise (shared).
7b. **`ChampionBrief` + tree-subject summarizer** — `champion_brief.txt` prompt, Pydantic model, lazy cache. Adapt `owtn/evaluation/feedback.py` (generalize or duplicate — decide at implementation time). Wire the rendered brief into `expansion.txt` as `{champion_brief}`. Depends on evaluation. See `lab/issues/2026-04-20-stage-2-expansion-feedback-summarizer.md`.
8. **`tournament.py`** — within-concept tournament. Depends on evaluation.
9. **`archive.py`** — QD archive (write-only in v1). Depends on dag, evaluation.
10. **`handoff.py`** — manifest construction. Depends on all above.
11. **`stage_2/runner.py`** — top-level orchestration. Depends on all above.

Each layer has tests before the next layer begins. Integration tests exercise the full stack with mocked LLM calls. The canonical-DAG gate (step 1b) is a hard block, not a soft checkpoint — fix upstream before proceeding. The earlier tension-inference gate (step 3b) was removed in the 2026-04-20 semantic-presets refactor; preset divergence is measured during pilot, not gated pre-pilot.

`forecasting.py`, `post_hoc_rationalize`, and QD competitive insertion are not in v1 (see `2026-04-19-stage-2-critical-review-followups.md` Item 3).

---

## Open Questions Surfaced in Implementation Drafting

1. **Budget numbers withdrawn pending pilot.** Paper cost estimates have gone through several rounds of recalibration and the design changed substantially on 2026-04-19 (tiered judge, Phase 3, 2 presets in light, 8 dimensions). Authoritative numbers will come from the pilot harness's `metrics.json`. See §Cost Budget Enforcement and §Pilot harness above.

2. **Where does Stage 1 → Stage 2 actually invoke?** Current design assumes a command-line invocation after Stage 1 completes. Should it be a single `owtn run` command that chains stages? Or explicit per-stage CLI invocations? Defer until Stage 3 design is clearer.

3. **Shared vs per-stage config files**. Current config layout uses one YAML per run with nested `stage_1`/`stage_2`/`judges`/`llm` sections. This is clean but grows. At 6 stages, the config YAML will be very long. Consider per-stage files with a top-level manifest that includes them, once we have more stages.

4. **Test DAG canonical set**. The four canonical DAGs (Lottery, Hemingway, Chiang, O'Connor) are hand-crafted. They should be reviewed against the Stage 2 schema once implementation starts to ensure they're actually valid Stage 2 genomes (not just plausible structural descriptions).

5. **Budget tracker precision**. Current design tracks costs post-call based on LLM response metadata. For pre-flight cost estimation (to avoid starting runs that will exceed the cap), we need per-call cost estimates. Use prompt length + model pricing as an estimate. Implementation detail.

6. **MCTS tree serialization**. For debugging, we'd like to snapshot MCTS trees at key points (end of each phase, on champion changes). Tree serialization format is TBD — probably JSON with explicit parent/child references via IDs.

7. **Judge reasoning storage**. Stage 1 stores judge reasoning alongside scores in `metrics.json`. Stage 2 will generate much more judge reasoning (more pairwise comparisons). The **in-memory subset used by the expansion-feedback summarizer** (`mcts.md` §Champion Brief Feedback Loop) lives on the MCTS tree as a `full_panel_critiques` list; that's hot-path state and doesn't require persistent storage. The broader question of persisting all reasoning for post-run analysis is still open — options: append to per-concept log files, or store in SQLite (reuse compost DB). Decide early.

8. **What happens when MCTS runs out of valid actions mid-phase?** If expansion repeatedly returns all-invalid actions, the phase may end early with few iterations. Is that acceptable? Current design says yes (the phase terminates with no-more-expansions), but we should log and monitor for pathological concepts that systematically fail expansion.
