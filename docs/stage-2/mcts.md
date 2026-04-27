# Stage 2: Monte Carlo Tree Search

Stage 2 uses MCTS to evolve a typed-edge DAG from a **single-node anchor seed** into a full structural plan. The anchor (climax, reveal, or pivot) is set by Stage 1 — see `operators.md` §seed_root. The structural insights — anchor-first seeding, bidirectional phases, forward-before-backward ordering, bounded simulation with early stopping — are drawn from BiT-MCTS (arXiv:2603.14410) Section 2.3, which validates them with strong empirical ablations for the climax-anchored case (removing bidirectional expansion → 100% loss; swapping phase order → 97%; removing early stopping → 58%). The "effect constrains cause" argument behind forward-first carries to reveal anchors too; for pivot anchors it's weaker (see Open Question 8). The reward function, hyperparameter calibration, edge-typed DAG representation, multi-preset search, and generalization from a climax-specific anchor to the three-role anchor enum are OWTN design choices, not inherited. This document specifies the algorithm: state representation, action space, selection, expansion, simulation, reward, bidirectional phases, budgets, and pacing-preset integration.

The blueprint derives from `lab/deep-research/runs/20260418_034927_stage2-mcts-blueprint/final_report.md`. Where the research left choices open, this doc commits.

---

## Algorithm Overview

Each Stage 2 run creates one MCTS tree per pacing preset per concept. For a concept with 4 pacing presets (Cassandra/Phoebe/Randy/Winston-ish), 4 independent trees grow in parallel.

Each tree follows the standard MCTS loop:

1. **Selection.** Descend from root to a leaf node using UCB1 (c=0.5), biased by pacing-preset action priors.
2. **Expansion.** Generate K candidate child actions for the selected leaf via a single LLM call; instantiate them as child tree nodes.
3. **Rollout.** Simulate forward from the new child by LLM-completing the partial DAG to a terminal state; evaluate the terminal with the reward function.
4. **Backpropagation.** Update visit counts and cumulative returns up the path to the root.

Two sequential phases of MCTS per tree, both attached to the anchor. Forward first (expansion grows downstream of the anchor — the falling-action / ketsu / epilogue region — 50 iterations), then backward (expansion grows upstream of the anchor — the rising-action / ki-sho / setup region — 50 iterations). Edges in the resulting DAG always point temporally forward (opening → anchor → resolution — invariant); only the search direction differs by phase. Details in §Bidirectional Phases. v1 uses forward-first uniformly across roles; Open Question 8 flags per-role budget tuning (reveal is typically downstream-light) and retains phase order as an open question for pivot specifically.

Terminal condition: the partial DAG meets size targets for the concept's prose length AND all permitted action types have been explored for the frontier nodes. See §Termination.

---

## State Representation

An MCTS tree node wraps a partial DAG plus the metadata MCTS needs to operate:

```python
class MCTSNode:
    dag: DAG                         # the partial DAG at this tree position
    action: Action | None            # the action that produced this state (None at root)
    parent: MCTSNode | None          # parent in the tree (not in the DAG)
    children: list[MCTSNode]         # spawned children
    visits: int                      # N(v)
    cumulative_return: float         # W(v)
    terminal: bool                   # has this state been fully explored?
    fully_expanded: bool             # have all available actions been tried at this node?
    phase: Literal["forward", "backward"]
    cached_candidate_actions: list[Action] | None  # from the most recent LLM expansion call
```

The **partial DAG** includes:
- Current nodes (with sketch, role)
- Current edges (with type, payload fields)
- `frontier`: the set of nodes that are candidates for further expansion (typically the current anchor-side and resolution-side boundary in forward phase, or the current opening-side boundary in backward phase)
- `target_node_count`: derived from the concept's prose length (3–5 for 1K words, scaling per the scoping issue §7)
- `target_forward_nodes` and `target_backward_nodes`: per-phase splits keyed on anchor role. Defaults: climax → 4 forward / 4 backward; reveal → 1 forward / 5 backward (the reveal *is* the ending; minimal downstream); pivot → 3 forward / 4 backward. Stage 1 may override via concept-genome metadata when a specific concept needs unusual proportions (e.g., a reveal concept with a meaningful coda gets 2–3 forward instead of 1). Without per-role splits, reveal-anchored trees burn forward-phase budget producing LLM-padding "downstream" beats that subsequently constrain Phase 2's `withheld` fields against fictional resolution content. See Open Question 8.

An MCTS tree may contain many tree nodes that wrap DAGs differing by one action each. A partial DAG can be shared by reference only within one branch (since mutations are atomic per action); duplicates across branches are not deduplicated — MCTS tolerates redundant exploration at small scale.

---

## Action Space

Three kinds of actions are available at any MCTS expansion. A single LLM expansion call returns up to K=4 candidate actions of any mixed types (BiT-MCTS's reported optimum; cached on the leaf and consumed one per subsequent expansion).

### Action types

| Action | Effect | When available |
|---|---|---|
| `add_beat(parent_id, edge_type, beat_text, edge_payload)` | Insert a new beat attached to `parent_id` via a typed edge | Always, subject to phase edge-type restrictions |
| `add_edge(src_id, dst_id, edge_type, edge_payload)` | Insert a new typed edge between two existing nodes | Always, subject to phase restrictions and DAG acyclicity |
| `rewrite_beat(node_id, new_text)` | Revise a node's sketch text | Always; useful when downstream edges need a subtly different setup |

### Phase-restricted edge types

The bidirectional expansion imposes edge-type restrictions:

- **Forward phase** (anchor → resolution): permitted edge types are `causal` and `implication`. Disclosure, constraint, and motivates are excluded because they require setup that happens in earlier beats not yet expanded.
- **Backward phase** (expansion grows upstream from the anchor; search direction anchor → opening): permitted edge types are `causal`, `constraint`, `disclosure`, `motivates`. Implication is permitted but de-emphasized — the forward phase produced the implication chains; the backward phase adds setup and intentionality. DAG edges from the new upstream beats still point temporally forward (new beat → existing node).

### Phase-restricted action targets

The anchor (climax, reveal, or pivot, per `concept.anchor_scene.role`) is the fixed point both phases grow around. The phase determines which direction MCTS grows new beats relative to the anchor:

- **Forward phase (downstream of anchor — anchor → resolution direction)**:
  - `add_beat` adds a new beat as the *target* of a new edge from an existing node. The edge's source (parent) must be the anchor or a descendant of the anchor; the new beat sits downstream.
  - `add_edge` must have both endpoints in the anchor-or-descendants set.
- **Backward phase (upstream of anchor — anchor → opening direction)**:
  - `add_beat` adds a new beat as the *source* of a new edge into an existing node. The edge's target (child) must be the anchor or an ancestor of the anchor; the new beat sits upstream.
  - `add_edge` must have both endpoints in the anchor-or-ancestors set (which includes opening and any previously-added upstream beats).
  - Note the asymmetry with Phase 1: in Phase 2, we're growing ancestry backward from the anchor, not growing descendants forward from the opening.

### Cached expansion per leaf (progressive widening via cache consumption)

Mechanism: when a leaf is first expanded, the LLM proposes a ranked candidate list Π(v) = {a1, ..., aK} in ONE call. The candidates are cached on the leaf. Each subsequent expansion of that leaf pulls the next unused candidate from the cache, instantiating exactly one new child per MCTS iteration. When the cache is exhausted, the leaf is marked `fully_expanded`. This is a standard MCTS cache-per-node pattern; BiT-MCTS reports K=4 works for their problem and we start there.

Progressive widening emerges naturally from this mechanism: a leaf can have at most K=4 children ever, and they're added one at a time across iterations rather than all at once. No separate widening formula is needed. This is cheaper than "propose K candidates per iteration" (1 LLM call per leaf, not per expansion iteration).

Proposed candidates are ranked by the LLM's own prior. The first candidate expanded has the highest prior; subsequent expansions try progressively lower-prior candidates. Combined with UCB1 selection, this means the tree spends most budget on high-prior actions at each leaf and only explores lower-prior alternatives when UCB bonuses indicate the tree is over-confident in the best candidate.

---

## Selection: D-UCB (γ-discounted)

Standard UCB1 with a custom exploration constant, modified with an exponential discount on accumulated history:

```
W̃(v) = Σ_i γ^(t_last - t_i) · r_i                  (discounted cumulative return)
Ñ(v) = Σ_i γ^(t_last - t_i)                        (discounted visit count)

UCB(child) = W̃(child) / Ñ(child)  +  c × sqrt(ln Ñ(parent) / Ñ(child))  +  β × structural_score(child)
```

where `c = 0.5`, `γ = 0.93`, `β = 0.1`. `t_i` is the iteration at which reward `r_i` was recorded; `t_last` is the current iteration for the node being selected. `structural_score(v)` ∈ [0, 1] is a zero-LLM-cost side-signal (see §Structural Side-Signal below).

**Why `c = 0.5`.** Lower than games-standard `c = √2 ≈ 1.41`. BiT-MCTS (arXiv:2603.14410) reports this as the narrative-MCTS optimum — LLM text generation has narrow reward variance compared to game outcomes, and higher exploration just wastes budget on indistinguishable children.

**Why γ-discount (`γ = 0.93`).** Stage 2 has two layers of non-stationarity that break UCB1's stationarity assumption (see §Reward Function §Running champion). The running champion drifts monotonically upward — early-tree rewards were scored against a weaker champion, so their W/N values systematically over-represent branch quality. UCB1 can't distinguish "genuinely good branch" from "got easy grading." D-UCB (Garivier & Moulines 2011; Wei et al. 2021) applies exponential decay γ^(Δt) to older returns, giving recent observations more weight and attenuating the stale-reward inflation. The deeper issue — every MCTS node is intrinsically a non-stationary bandit, so UCT's logarithmic exploration bonus is theoretically unjustified (Shah, Xie & Xu SIGMETRICS 2020) — D-UCB partially addresses by down-weighting stale observations that no longer reflect the node's current subtree quality.

**Why `γ = 0.93`.** For our budget (~500 rollouts per tree, ~5 expected champion updates), `γ_opt ≈ 1 - √(S/T) ≈ 1 - √(5/500) ≈ 0.90` per Abbasi-Yadkori et al. (JMLR 2023), where S = expected champion-identity changes and T = total rollouts. We round up to 0.93 for slightly more history retention; the regret bound is not sensitive to this choice in the 0.9–0.95 band.

**Rechallenge becomes event-driven.** The earlier design (`rechallenge_interval=25` iterations, refresh top 10% W/N) was time-driven. Under D-UCB, the discount handles staleness inline — rechallenge is retained but fires on champion-update events rather than on a clock. When a champion is promoted, the top 10% of terminals by W̃/Ñ are re-scored against the new champion immediately (not waiting 25 iterations). This is what MuZero Reanalyze (Schrittwieser et al. NeurIPS 2021) does in practice; FreshPER (2026) is the closest published theory for interval choice.

**Preset-agnostic in v1.** Preset divergence happens entirely at expansion time via the preset-specific pacing hint in the prompt (see `overview.md` §Where presets enter the pipeline). An earlier design added a `preset_prior` bonus to UCB — deferred to v1.5 along with the tension-inference infrastructure it required (see §Tension Inference below).

Research grounding: `lab/deep-research/runs/20260424_031109_mcts-nonstationary-rewards/final_report.md`.

### Structural Side-Signal

`structural_score(v) ∈ [0, 1]` computes deterministically from v's DAG — zero LLM cost — and penalizes structurally-broken branches before expensive judge evaluation fires. Components (all reduce to numeric scores; final value is a weighted combination with all weights equal):

| Check | Signal | Cost |
|---|---|---|
| Orphan beats | 0 if any node has zero in-edges AND zero out-edges (besides root), else 1 | O(n) |
| Edge-type entropy | normalized Shannon entropy of edge-type distribution — penalizes >80% single-type | O(e) |
| Anchor reachability | 1 if anchor reachable from opening beat within ≤ `n_beats × 0.8` hops, else 0 | O(n+e) |
| Arc density | 1 if edge_count / (n × (n−1)) ∈ [0.1, 0.5], else linear penalty | O(1) |
| Payload completeness | 1 if every edge's required payload fields are non-empty, >3 tokens; else fractional | O(e) |

**Note on payload completeness.** Operator validation already rejects edges with empty or generic payloads (`>15 chars, not "things happen"`). In-tree DAGs will typically score 1.0 on this component; it's kept as a safety net for rare operator-validation bypasses, not as a primary discriminator. If post-pilot data shows every in-tree DAG scores 1.0, drop the component.

`β = 0.1` keeps structural_score as a secondary signal — it routes around clearly-broken branches without overwhelming the pairwise judge's signal on genuinely-close comparisons. Zero training required; no LLM calls; no calibration burden. The β=0.1 value is hand-set; sensitivity is tested in `implementation.md` §MCTS ablation suite (β ∈ {0, 0.05, 0.1, 0.2}).

Research grounding: (a) PRM-vs-ORM report concluded terminal-only pairwise is the right primary reward (`lab/deep-research/runs/20260424_031201_prm-vs-orm-narrative-mcts/final_report.md`); structural checks are the best near-term complement. (b) PLOTTER (Gu et al., arXiv:2604.21253, 2026) validates the deterministic-constraints approach empirically — DAG validity + connectivity constraints in an Evaluate-Plan-Revise cycle achieves 62–100% win rate vs. DOC/Dramatron. (c) The specific check list is calibrated against ConStory-Bench's (Li et al., arXiv:2603.05890, 2026) error taxonomy — "view from nowhere" / over-homogeneous structure / orphan beats are among the most frequent consistency failures.

---

## Expansion: Generating Candidate Actions

Each expansion is one LLM call that returns K=4 ranked candidate actions (cached on the leaf for use across subsequent expansions of that leaf — see §Cached expansion). The prompt is constructed from:

1. **Concept context** — premise, target effect, character seeds, thematic tension, constraints, style hint (from Stage 1).
2. **Motif threads** — the 2-3 recurring elements extracted at seed time. "Favor actions that reference these threads where natural; invented motifs are acceptable if they become recurring."
3. **Current partial DAG** — rendered using the incident-encoded outline format from `evaluation.md` (same format used by judges).
4. **Phase instructions** — "you are in the backward phase; permitted edge types are causal, constraint, disclosure, motivates."
5. **Action request** — "Propose up to 4 candidate actions that would extend this DAG toward the target. Each action is one of: add_beat (inserts a new beat attached via a typed edge), add_edge (adds a new typed edge between existing nodes), rewrite_beat (revises an existing beat's text)."
6. **Output schema** — JSON array of actions, each with required fields for its type.
7. **Pacing hint** — the preset-specific 1–2 sentence hint from `overview.md` §Preset values and expansion hints (substituted via `{PACING_HINT}`). This is the sole mechanism of preset divergence in v1.
8. **Champion brief** — a structured distillation of what full-panel judges have valued and penalized in this tree's prior comparisons, rendered from a `ChampionBrief` (established structural weaknesses, contested structural choices, structural attractor signature, structural divergence directions). Substituted via `{CHAMPION_BRIEF}`; empty placeholder until the tree has accumulated enough full-panel critiques for the summarizer to run. See §Champion Brief Feedback Loop below and `lab/issues/2026-04-20-stage-2-expansion-feedback-summarizer.md`.

The K=4 returned actions are cached on the leaf (ranked by LLM prior). Each subsequent expansion of that leaf instantiates exactly one unused candidate as a new sibling tree node. Progressive widening emerges naturally: the leaf accumulates at most K=4 children across iterations, gated by UCB's drive to revisit.

### Champion Brief Feedback Loop

Without feedback, the expansion LLM sees only the current partial DAG — the MCTS tree's alternative branches and prior evaluations are invisible at the LLM layer. UCB routes exploration but never tells the LLM *why* a branch worked or didn't. To close that loop, Stage 2 runs a lazy summarizer (adapted from Stage 1's `owtn/evaluation/feedback.py`) that distills full-panel critiques into a `ChampionBrief` rendered into the expansion prompt.

**Subject: the tree, not a champion.** Stage 1 summarizes per-concept across its tournament matches. Stage 2's champion churns too fast for that parallel (1-5 iterations before replacement gives too little history). Instead, the tree itself (`concept_id:preset`) is the subject — it accumulates full-panel critiques continuously across champion changes.

**Corpus: full-panel events only.** Promotion-gate verifications (accepted and rejected) plus within-concept tournament. Cheap-judge rollout critiques are not summarized — too noisy, and cheap-judge is the UCB signal rather than commitment-weight evaluation. ~10-30 full-panel events per tree matches Stage 1's typical 5-15 match corpus.

**Cache key: count of full-panel critiques.** Re-run the summarizer every N new events (N=3-5, calibrate in pilot). Champion churn is invisible to this cadence by default. **Cold start**: before the first summarizer fires, render the last 1-2 full-panel critiques verbatim in the expansion prompt via `render_raw_fallback` — the same fallback used on summarizer failure. A stale brief is worse than raw recent critiques during the first ~10 iterations, when the brief would otherwise be empty.

**Forced re-render on champion promotion.** When a champion is promoted (full panel confirms a challenger wins), the brief cache is invalidated and the summarizer re-fires immediately, regardless of how many new events have accumulated. Without this, rechallenge fires on promotion and updates the tree's UCB statistics to reflect the new champion *while* the expansion prompt still shows the LLM the old champion's weaknesses — for ~1-2 iterations the tree gets structurally inconsistent guidance (old brief steering the LLM, new statistics steering UCB). Forcing re-render aligns the two signals at every promotion event.

**Summarizer faithfulness — pilot validation.** The summarizer compresses raw critiques into 4 structured fields. If it systematically drops a feedback category (e.g., judges critique reader-address failures but the summarizer normalizes this into "pacing issues"), the expansion LLM never sees the real signal even though judges produce it. Calibrate before trusting: sample 20 brief / raw-critique pairs from pilot, manually verify each brief preserves the critique's main thrust. Specific failure mode to watch for: structural-mechanism critiques (anything not on the 8 rubric dimensions) being collapsed into the nearest dimension. See `implementation.md` §Pilot harness §Summarizer faithfulness.

**Fields (parallel to Stage 1's `ParentBrief`, re-scoped to structure):**
- `established_structural_weaknesses` — what challengers keep failing at across rejection critiques
- `contested_structural_choices` — structural choices where judges have split
- `structural_attractor_signature` — patterns this tree keeps producing across winners and losers (preset-default or LLM-default shapes)
- `structural_divergence_directions` — prescriptive "don't propose another X / try Y instead"

**Fallback and cost.** If summarizer fails, render last 1-2 critiques minimally (same pattern as `render_raw_fallback`). Summarizer uses `classifier_model` (third-family from generator and judges); ~$0.01-0.02 per summarizer call × ~5-10 invocations per tree = negligible vs. main MCTS cost.

### Validation of proposed actions

Each candidate action is validated before instantiation:

- **Schema validation** (per action type): required fields present, non-empty payloads.
- **Structural validation**: `add_edge` must not create a cycle; `add_beat` must attach to an existing node; `rewrite_beat` must reference an existing node.
- **Phase validation**: edge type is permitted for the current phase; action target is in the permitted subset.

Invalid actions are dropped silently. If all K proposed actions are invalid, expansion retries once with increased temperature. If the second attempt also produces no valid actions, the parent MCTS node is marked `fully_expanded: true` with its current children (or `terminal: true` if it has no children yet — this is a degenerate case that should be rare).

### Forbidden duplicates

Actions that would produce a DAG identical (up to node-id renaming) to an existing sibling are rejected. This is a cheap string-hash check on the serialized DAG and prevents the LLM from proposing essentially the same action three times.

---

## Simulation: Bounded Extension with Early Stopping

When expansion creates a new tree node, MCTS needs a value estimate to backpropagate. BiT-MCTS's Algorithm 1 specifies a bounded-depth simulation with early stopping — NOT a "complete to terminal" rollout. Complete-to-terminal rollouts were found to be unstable for LLM narrative generation. We adopt the BiT-MCTS approach.

### Simulation policy

From the new tree node `v` with partial DAG `S_cur` and cached reward `ρ(v)`, attempt up to `s_max = 3` one-step extensions:

```
procedure SIMULATE(v, d_max):
    if terminal(v) or depth(v) >= d_max: return ρ(v)
    reward_cur ← ρ(v)
    S_cur ← S(v)
    for i in 1..min(s_max, d_max - depth(v)):
        e ← G(·|S_cur, dir)                # propose one extension via LLM
        S_new ← apply(e, S_cur, dir)       # append (forward) or prepend (backward)
        if depth(S_new) > d_max: break
        reward_new ← R(S_new)              # evaluate new partial DAG
        if reward_new >= reward_cur:
            S_cur ← S_new
            reward_cur ← reward_new        # accept only if reward didn't drop
        else:
            break                          # early stop — stopped improving
    return reward_cur
```

Key properties:

1. **Bounded depth**: at most `s_max = 3` one-step extensions per simulation. Not "complete the full DAG."
2. **Early stopping**: simulation halts as soon as an extension would decrease reward. Avoids wasting budget on declining paths.
3. **Per-step evaluation**: each accepted extension triggers an evaluation of the NEW partial DAG, not just the terminal. The evaluator R is the cheap-judge pairwise score against the tree's running champion (see §Reward Function §Tiered judge design). Earlier drafts used narrative forecasting here; that was dropped along with the hybrid reward function.
4. **Simulation nodes are ephemeral**: the extensions added during simulation are discarded after the simulation returns. Only the scalar reward backpropagates; simulation beat sketches do not enter the tree.
5. **Parent termination**: the expanded MCTS node `v` is marked `terminal` only when its search depth reaches `d_max = 8`. Early stopping within simulation does not mark the node terminal.
6. **Active constraints in the proposal prompt**: the proposal distribution G(·|S_cur, dir) receives, alongside the partial DAG, an explicit list of active constraints: disclosure edges' `withheld` fields (content the simulation must not reveal) and constraint edges' `prohibits` fields (capacities foreclosed at targeted beats). This prevents rollouts from contradicting existing edge payloads — e.g., a rollout can't reveal content that a disclosure edge says is withheld. The prompt includes a line like: "The following must NOT appear in your proposed extension: {list of withheld content}."

### Starting hyperparameters (to calibrate for our shorter outputs)

These are initial values informed by BiT-MCTS's reported optima for Chinese long-form fiction (8K–58K tokens). OWTN targets 500–10K words (~650–13K tokens) with 3–18 node DAGs — substantially smaller. Early calibration runs should revisit each:

- `s_max = 3` — max simulation extension steps. BiT-MCTS found shorter is better than deeper. Reasonable starting point.
- `d_max = 8` — max search depth. **Likely too high for our target node counts**; we may want 5–6. Calibrate against first-run data.
- `K = 4` — max cached candidate expansions per leaf. Standard MCTS cache size; reasonable.
- `c = 0.5` — UCB exploration constant. Lower than games-standard √2, matching BiT-MCTS's finding that narrative reward variance is narrow. Direction is what matters; exact value is tunable.
- `50 iterations per phase` — BiT-MCTS's choice for their problem size. Likely enough for our shorter DAGs; could go lower.

BiT-MCTS Table 4 shows that early stopping in simulation is load-bearing (removing it → 58% loss). That structural choice carries over; the numeric bound (s_max=3) is a calibration value.

---

## Reward Function

A terminal state's reward is a pairwise comparison against the tree's running champion, produced by a **tiered judge** to balance signal quality against rollout latency. **Rewards are terminal-only — no per-action (process) rewards are emitted by the judge.** The reward is scalar backpropagation of a full-DAG pairwise comparison; no step-level LLM scoring fires during simulation.

### Why terminal-only (and not step-level / PRM)

The decision to stay with terminal-only pairwise rewards is grounded in the process-vs-outcome reward research (`lab/deep-research/runs/20260424_031201_prm-vs-orm-narrative-mcts/final_report.md`):

1. **PRM-guided tree search does not reliably beat Best-of-N in math** (Cinquin et al., arXiv:2510.20272) — the domain where PRMs are best validated. PRMs "poorly approximate state values" and reliability degrades with reasoning depth. Tree search amplifies PRM error across selection steps.
2. **Narrative is structurally incompatible with PRMs.** No oracle terminal labels (unlike math's answer-correctness); no local step validity (a beat's quality is arc-position-dependent); heterogeneous action types (add_beat / add_edge / rewrite_beat have different quality criteria); no formal-notation basis for step-level pattern matching. Five independent narrative-reward papers (LitBench EACL 2026, StoryAlign ICLR 2026, Gurung & Lapata COLM 2025, RLMR AAAI 2025, Retell/Reward/Repeat 2026) all use terminal/outcome rewards — empirical convergence.
3. **LLM-judge reliability at step level is likely too low for narrative.** Whole-story judges achieve 73% agreement with humans (LitBench). Step-level judges applied to a single beat in a 10-beat DAG are estimated at 60–65% reliability at 3–5× the cost.
4. **The partial-state scoring we already do is close to right.** BiT-MCTS's bounded simulation (`s_max=3` extensions, early-stop on reward non-improvement) scores the partial DAG as a whole at each step — this is ORM applied to intermediate states, not PRM. SP-PRM (Xie et al., ACL 2025) shows ORMs applied to partial sequences achieve ~57% score consistency but retain human-preference correlation: noisy but not random.

The **structural side-signal** (§Structural Side-Signal under §Selection) is the zero-LLM-cost complement that catches definitively bad branches early — taking the structural-coherence signal out of the judge and into cheap deterministic checks is where the real per-step leverage lives.

### Pilot to validate before any future reversal

A single $20 pilot would change confidence on the no-PRM decision: design 100 (partial_DAG, action) pairs, have 3 judges score action quality, measure inter-judge agreement and human-rater agreement. If agreement is ≥ 75% and consistent across action types, step-level narrative judging becomes viable and we reopen the question. Below that threshold, the terminal-only decision holds.

### Longer-term alternative: self-taught value function (v1.5+)

After ≥500 terminal evaluations accumulate from Stage 2 runs, a small (1B–7B) model can be fine-tuned to predict terminal-pairwise-score from partial DAG. Self-Taught Lookahead (Mendes & Ritter, arXiv:2503.02878) shows this approach gives +39% web-agent success at 37× cost reduction vs. frontier-LLM-as-judge in adjacent domains. One complete Stage 2 run likely produces enough labeled terminals for training. This replaces (not augments) the cheap judge for partial-state valuation.

### Tiered judge design

Pairwise judging fires at different granularities in different contexts:

| Context | Judge configuration | LLM calls per comparison |
|---|---|---|
| Rollout reward (every terminal) | 1 cheap judge × 2 orderings (in parallel — no added latency) | 2 |
| Champion promotion gate | Full 4-judge × 2-ordering panel, fires only when the cheap judge declares a win | 8 |
| Within-concept tournament | Full 4-judge × 2-ordering panel | 8 per pair |
| QD archive insertion competition | Full 4-judge × 2-ordering panel | 8 |

The cheap judge produces the rollout reward signal that UCB selects on. Dual-ordering mitigates position bias directly: both orderings run concurrently (asyncio.gather); a challenger's "win" on a dimension requires both orderings to agree, and disagreement is recorded as tie. Position bias — which scales inversely with quality gap, concentrating on exactly the close calls UCB depends on — no longer steers UCB down the wrong child. It shows up as elevated tie rate on near-equivalent children, which UCB treats as noise rather than false direction. Extra cost is ~2× per-rollout judge calls (one cheap model, doubled), negligible against expansion + tournament costs; latency is unchanged.

When the cheap judge declares a challenger wins, the full panel verifies before the champion is actually promoted. This keeps commitment events rigorous while allowing cheap-judge noise at the rollout granularity.

The cheap judge model must be from a different model family than the expansion/rollout model (cross-family discipline, same as Stage 1's panel). Config names a specific model (tentatively `gpt-5.4-mini`); finalize at config time.

### Running champion

- Each MCTS tree maintains an internal champion: the best terminal DAG accepted so far.
- Each rollout produces a terminal DAG; the cheap judge compares it against the current champion on all 8 dimensions in two parallel calls (dual-ordering). Per-dimension votes are collapsed across orderings using Stage 1's rule (`_flip_votes` in `owtn/evaluation/pairwise.py`): a dimension counts as a win only if both orderings name the same winner; disagreement collapses to tie.
- Rollout score = (dimension_wins + 0.5 × ties) / 8, from the challenger's perspective, computed on the collapsed per-dimension votes.
- If the cheap judge declares overall-win, the full panel verifies. On full-panel confirmation, the challenger becomes the new champion AND the cheap-judge score is backpropagated up the tree as normal.
- **On full-panel rejection**: the challenger does NOT become champion, and the score backpropagated is **0.5** (the cheap-judge's own all-tie value), not the cheap-judge's original win-score. This corrects the false-positive signal without mixing the full panel's score distribution into the tree's UCB statistics. The full panel is systematically stricter than the cheap judge (4 judges × dual ordering → more ties, especially 2-2 dim-level splits that the 3-judge predecessor never produced), so substituting an actual full-panel score on rejection would bias UCB against branches that hit promotion gates. Substituting the cheap-judge's own tie value keeps all backpropagated rewards in a single distribution while honestly encoding "this wasn't an improvement."
- For normal rollouts (cheap judge declares tie or challenger loses), the reward backpropagated is the cheap-judge score against the champion at the time of evaluation.

**Special case — first rollout**: no champion yet, so the first terminal DAG's pairwise score is 0.5 (neutral) and it immediately becomes the champion (no panel verification needed for the initial install).

**Why running champion rather than absolute scoring**: absolute LLM scoring compresses scores into a narrow band (documented in Stage 1 scoping). Pairwise vs. a running champion produces comparative gradients that actually discriminate.

### Monitoring cheap-judge drift

Every champion-promotion gate logs whether the full panel confirmed the cheap judge's pick. Track agreement rate per run. If it drops below ~70%, the cheap judge has drifted from the full panel's signal and should be swapped at the next run. Early-run monitoring is part of the pilot deliverable.

**Rejection-rate side effect.** The 0.5 backprop on full-panel rejection (above) substitutes the cheap judge's all-tie value for its actual win-score. This correctly prevents false-positive wins from inflating UCB, but it also *under-scores* branches whose cheap-judge per-dim scores were reliable on most dimensions and disputed on 1–2. If the full-panel rejection rate exceeds ~30% of promotion gates, this rule is creating a dead zone in the search (branches whose cheap-judge signal was mostly right get scored as ties). Pilot deliverable: rejection rate per tree and the distribution of backpropped 0.5s vs. natural ties. Above 30%, reconsider — e.g., backprop the cheap judge's per-dim score gated on per-dim full-panel agreement, rather than collapsing to 0.5.

### Narrative forecasting — deferred to v1.5

Narrative forecasting (adapted from *Spoiler Alert*, arXiv:2604.09854) was originally part of a hybrid rollout reward, then downgraded to diagnostic metadata (2026-04-19), then dropped from v1 entirely (2026-04-19 second review pass, see `lab/issues/2026-04-19-stage-2-critical-review-followups.md` Item 3). Reason: with forecasting no longer in the reward function, keeping it as "diagnostic / optional tiebreaker" costs implementation time for no load-bearing signal. Tournament tiebreakers can fall back to dimension-level wins instead.

Revisit if post-v1 data shows tournament ties are common.

### Cost notes

Actual per-concept cost is deferred to pilot measurement (see `implementation.md` §Cost Budget Enforcement). The tiered-judge design is the primary cost and latency lever: cheap judge at ~1s latency × ~200 rollouts per tree is the dominant loop, with full-panel calls firing only at ~10-20 promotion events per phase.

---

## Backpropagation

Standard: when a rollout produces reward `r`, walk up the tree from the expanded node to the root, incrementing `visits` and adding `r` to `cumulative_return` at each tree node. No discount factor — narrative reward is terminal-only and doesn't need discounting.

---

## Bidirectional Phases

Two sequential MCTS phases run per tree. Forward first, then backward. Each phase has its own 50-iteration budget. Both phases are rooted at or near the anchor; both add new nodes in opposite temporal directions (forward = downstream of anchor; backward = upstream of anchor). Edges in the resulting DAG always point temporally forward — only the search direction differs by phase.

### Phase 1: Forward from anchor (adds downstream material)

- Seed: single-node DAG containing only the anchor (climax, reveal, or pivot), from `seed_root(concept)`. The anchor is read verbatim from `concept.anchor_scene`; preset-agnostic; the same seed is shared across all preset trees for a given concept.
- Root MCTS node wraps the single-node DAG as its state.
- **Search direction**: anchor → resolution (in story time). Expansion adds new beats *downstream of the anchor*: new nodes are targets of new edges from existing anchor-side nodes (the anchor itself, or anchor-descendants added earlier in this phase). For climax-anchored concepts this is the falling action; for pivot-anchored concepts it's the ketsu; for reveal-anchored concepts it's whatever the reveal makes possible (often a brief coda — see Open Question 8 on role-scaled iteration budget).
- **Action targets** for `add_beat`: the parent (source of the new edge) must be the anchor or a descendant of the anchor. The new node is the target, representing a beat that happens after the anchor in story time.
- **Action targets** for `add_edge`: both endpoints must be in the anchor-or-descendants set.
- Permitted edge types: causal, implication. Disclosure, constraint, and motivates are not permitted — they require setup that happens earlier in story time, which this phase doesn't generate.
- Phase ends when (a) 50 iterations are completed, (b) the target downstream node count is reached, or (c) no further productive expansions are available (all leaves marked `fully_expanded`).
- Output: the highest-reward terminal DAG from this phase becomes the root DAG for Phase 2.

### Phase 2: Backward from anchor (adds upstream material)

- The starting DAG carries forward the forward phase's downstream subtree intact.
- The MCTS tree is reset — a new tree is built with the forward-phase winner as its root DAG. Only the upstream region (beats temporally before the anchor) can be expanded.
- **Search direction**: anchor → opening. Expansion adds new beats *upstream of the anchor*: new nodes are sources of new edges into existing anchor-side nodes (the anchor itself, or nodes added earlier in this phase as anchor-ancestors). For climax-anchored concepts this is the rising action; for pivot-anchored concepts it's the ki/sho setup strands being juxtaposed; for reveal-anchored concepts it's the concealment architecture that earns the reveal.
- **Action targets** for `add_beat`: the child (target of the new edge) must be the anchor or an ancestor of the anchor. The new node is the source. This is the reverse of Phase 1's direction.
- **Action targets** for `add_edge`: both endpoints must be in the anchor-or-ancestors set (which grows as the phase adds new upstream beats).
- Permitted edge types: causal, constraint, disclosure, motivates (plus implication at low prior). Motivates and disclosure are unique to this phase because they require setup that happens before the anchor — which this phase generates.
- Phase ends with the same conditions as Phase 1, scaled for the upstream region.
- Output: the highest-champion terminal DAG from this phase is the preset's final output.

### Why reset the MCTS tree between phases?

The forward-phase tree's UCB statistics are about the falling-action search space; those statistics don't transfer meaningfully to the rising-action search space. Resetting the tree forces fresh exploration of the new space. The forward winner's DAG is preserved; only the MCTS bookkeeping (visit counts, cumulative returns) is discarded.

**v1.5 investigation:** the forward-phase tree encodes more than just its winner — branches it explored but didn't pick, and its confidence about different parts of the falling action. Resetting discards that signal entirely. Worth exploring whether backward-phase expansion priors could benefit from a summary of forward-tree uncertainty (e.g., "the forward tree was confident about this beat's position but uncertain about that one — consider what setup the uncertain part needs"). Not a v1 priority; BiT-MCTS ships with full reset and we inherit that default. Flag as a research question for v1.5 if pilot data suggests phase transitions are losing useful context.

### Why not run both phases concurrently?

Two reasons: (a) the backward phase needs the forward phase's resolution structure to know what setup is needed (disclosure edges' `withheld` fields must reference actual resolution content), and (b) the UCB dynamics of two interleaved search spaces are hard to tune. Sequential is simpler and aligns with BiT-MCTS's published approach.

### Phase 3: Cross-phase refinement (adds spanning edges)

Phases 1 and 2 both restrict `add_edge` endpoints to within-phase sets (anchor-or-descendants in Phase 1, anchor-or-ancestors in Phase 2). This silently excludes edges that span the anchor — most commonly disclosure edges from post-anchor resolution beats backward to opening beats (epilogue-reveal structures like Ishiguro's "Never Let Me Go"), or motivates edges that arc across the whole story.

Phase 3 is a short post-Phase-2 pass that permits any `add_edge` action subject only to acyclicity and payload validation.

- **Starting DAG**: the Phase 2 champion (full node set with rising and falling action).
- **Action space**: `add_edge` only. No new nodes, no beat rewrites. Both `src` and `dst` may be any existing node.
- **Budget**: ~5 iterations. K=4 candidates per expansion. Uses the cheap judge for reward signal like Phases 1 and 2.
- **Expansion prompt steering.** Phase 3 exists specifically to recover edge classes the 2-phase split excluded — disclosure edges from post-anchor beats backward to opening, motivates edges arcing across the whole DAG. With ~110 candidate src/dst pairs × 5 edge types = ~550 raw candidates and only ~20 actual proposal slots (5 iterations × K=4), random expansion would sample a vast space for a small target. The Phase 3 expansion prompt explicitly steers: *"Propose only edges where `src` and `dst` sit on opposite sides of the anchor — at least one endpoint upstream of the anchor and at least one downstream. Phase 3 exists for these spanning edges; non-spanning edges are out of scope for this phase."* Edges with both endpoints on the same side of the anchor are validation-rejected at instantiation, not just deprioritized — the prompt and validator agree on the constraint.
- **Validation**: standard edge validation (payload fields populated, no cycle created, no duplicate edge between same nodes of same type, phase-agnostic on edge type — all 5 types permitted) PLUS the spanning constraint above.
- **Output**: the Phase 3 champion is the preset's final DAG.

Phase 3 is cheap because (a) the action space is narrow (only `add_edge`, not `add_beat` or `rewrite_beat`), (b) budget is small (~5 iterations vs. 50 per main phase), and (c) most rollouts don't need simulation — an `add_edge` on a fully-developed DAG either improves the structure or doesn't, measurable with one cheap-judge comparison.

If Phase 3 finds no improving edge across its 5 iterations, the Phase 2 champion is the preset's output unchanged. This is expected for many DAGs — not every structure benefits from cross-anchor edges.

Precedent: BiT-MCTS's Outline Refinement stage is a similar post-MCTS pass that fixes structural shapes MCTS's core loop can't reach.

### Monitoring Phase 3 (v1)

Phase 3 is a small shape adjustment that the 2-phase edge-type restrictions make structurally necessary. The open question is whether those restrictions are load-bearing for short-form DAGs. Track the Phase-3 improvement rate — how often the Phase 3 champion differs from the Phase 2 champion — as a run-level metric:

- **Small improvement rate:** the restrictions were fine at v1's scale; Phase 3 adds complexity for no benefit and can be dropped in v1.5.
- **Large improvement rate:** the 2-phase restrictions are rejecting too many valid structural moves; reconsider the forward/backward split, likely moving to a single anchor-first phase with soft edge-type priors.
- **Moderate improvement rate:** keep as-is; Phase 3 is doing its job.

First-pass cutoffs (<5% / 5–30% / >30%) are guesses — calibrate after the first pilot produces a distribution of improvement rates across concepts.

Tracked in `implementation.md` §Metrics exported.

---

## Budget Management

> **Cost numbers below are preliminary and are being re-measured in pilot.** The design changed substantially pre-implementation (tiered judge replaces hybrid reward, Phase 3 added, `light.yaml` drops to 2 presets, 8 dimensions not 9). Paper estimates are unreliable until first pilot `metrics.json` lands. See `implementation.md` §Cost Budget Enforcement for the measurement plan.

### Per-tree budget (preliminary)

- 50 iterations forward + 50 iterations backward + ~5 iterations Phase 3 refinement ≈ 105 iterations per tree.
- Expansion call per iteration: at most 1 LLM call per leaf (K=4 candidates cached on first expansion; later expansions consume the cache with no additional LLM calls).
- Rollout per iteration: ~1 LLM call with cheap model, up to `s_max=3` simulation extensions.
- Cheap-judge reward per terminal: 1 LLM call with structured 8-dimension response, fires every rollout.
- Full-panel verification: 6 LLM calls, fires only when the cheap judge declares a challenger wins (~10-20 promotion gates per phase).

### Per-concept budget (preliminary)

At `light.yaml` with 2 presets: ~2× per-tree cost plus ~6 pairwise comparisons for within-concept tournament. At `medium.yaml`/`heavy.yaml` with 4 presets: ~4× per-tree cost plus round-robin tournament.

The judge-call structure (1 call per judge-ordering pair with all 8 dimensions evaluated in one structured response, following Stage 1's pattern in `owtn/evaluation/pairwise.py`) keeps full-panel pairwise cost at ~$0.04–$0.08 per comparison (4 judges × 2 orderings × ~$0.005–$0.01 per call).

### Early termination triggers

- **No improvement over 15 iterations** within a phase: phase terminates early.
- **Champion DAG fails validation for 3 consecutive iterations**: the preset is flagged as not converging; MCTS terminates for this preset and the last-passing champion is the output.
- **Per-concept time budget exceeded** (configurable, default 30 minutes): all running trees checkpoint and terminate.

### Concurrency

MCTS is sequential per tree. Across a concept's 4 trees, execution is parallel (each tree is an independent asyncio task). Across concepts, execution is parallel up to the configured concurrency limit.

Within a single tree, expansion and evaluation can be batched: when multiple frontier nodes are selected for expansion in the same iteration loop, their expansion LLM calls can run concurrently and their rollouts can run concurrently. This doesn't change MCTS semantics but amortizes LLM round-trip latency.

---

## Termination

A phase terminates when any of these conditions hold:

1. **Iteration budget exhausted.** Default 50 per phase.
2. **Target node count reached** in the phase's expansion subtree, with all frontier nodes marked `fully_expanded`.
3. **No-improvement cutoff.** Champion has not changed for 15 iterations in this phase.
4. **Phase time budget exceeded.** Per-phase soft cap (default 10 minutes) to prevent runaway on pathological concepts.

At phase termination, the phase's champion is the best terminal DAG evaluated during the phase. This champion advances to the next phase (if forward) or to the within-concept tournament (if backward).

---

## Pacing-Preset Integration Summary

In v1, pacing presets enter the MCTS in **one place** (the seed is also preset-agnostic — see `operators.md` §seed_root):

1. **Expansion prompt** (this doc, §Expansion): the prompt includes a preset-specific pacing hint so the LLM proposes candidate actions already tilted toward the preset's philosophy. This shapes the cache that subsequent expansions draw from.

The 4 trees share a root and a UCB formula; they diverge entirely through the expansion LLM's response to different pacing hints. If pilot data shows the semantic hints alone produce insufficient structural divergence, a parametric `preset_prior` re-entry point (deferred to v1.5 — see §Tension Inference) can be restored.

## Motif Threads in MCTS Expansion

Motif threads (from the genome, extracted at seed time — see `overview.md` §Motif threads) are included in every expansion prompt as additive context. The expansion prompt instructs the LLM to favor actions that reference existing motif threads where natural; invented motifs are acceptable if they become recurring.

No UCB bonus is applied for motif references. The prompt-side injection is the entire mechanism: biasing what the LLM proposes is more decisive than a +0.05 nudge on a UCB score whose exploration term is typically ~0.5-1.0. If pilot data shows motif recurrence systematically drops across expanded DAGs, revisit — either strengthen the prompt language or restore a UCB bonus calibrated against observed reward variance.

---

## Tension Inference — Deferred to v1.5

An earlier design (pre-2026-04-20) inferred a tension level for each beat from keyword heuristics + a cheap LLM classifier fallback, fed it into a `preset_prior(action, parent, preset)` function, and added `λ × preset_prior` as an augmenting term on UCB. The preset's numerical primitives (`min_rest_beats`, `max_flat_beats`, etc.) were read by this function to shape action selection.

v1 drops this machinery in favor of **semantic presets** — preset divergence happens entirely through the preset-specific pacing hint in the expansion prompt (see `overview.md` §Preset values and expansion hints). The trade-off: we lose the option to numerically penalize actions that violate a preset's pacing shape at UCB selection time. We gain: one less heuristic layer to calibrate, one less validation gate before pilot, one less failure mode (tension-inference noise propagating into preset divergence).

**Return trigger (v1.5):** if pilot measurements of preset structural divergence (edge-type histogram L1, node-count spread, disclosure-ratio spread across preset trees — see `implementation.md` §Metrics exported) show the 4 trees collapsing to near-identical outputs, restore the parametric preset_prior. At that point, this section reactivates along with the hand-label validation gate (ρ ≥ 0.7 against canonical-DAG beat sketches).

See `lab/issues/2026-04-19-stage-2-third-review-followups.md` Item 2 for the decision record.

---

## Open Questions Surfaced in MCTS Drafting

1. **(Resolved — pending pilot.)** Paper cost estimates have gone through several rounds of recalibration and the design changed substantially on 2026-04-19 (tiered judge, Phase 3, 2 presets in light, 8 dimensions, semantic presets). Authoritative numbers will come from the pilot harness's `metrics.json`. Provisional ceiling for go/no-go gating: $30–$48 / $50–$80 / $80–$192 for light/medium/heavy (scoping §13 recalibration range; the highest honest number we've carried). See `implementation.md` §Cost Budget Enforcement.

2. **Running-champion non-stationarity — mitigated by periodic re-challenge (2026-04-20).** UCB cumulative returns conflate rewards measured against early-tree vs. late-tree champions (different baselines), violating UCB's stationarity assumption. The pathology isn't champion drift (champions only improve monotonically); it's that early-tree branches accumulate high W/N because they were scored against a weaker champion, while later branches earn rewards against a stronger champion. UCB can't distinguish "this branch is genuinely good" from "this branch got easy grading."

**Mitigation (v1):** periodic re-challenge. Every `rechallenge_interval=25` iterations within a phase, take the top `rechallenge_top_pct=10%` of terminals by stored W/N, re-run the cheap judge against the CURRENT champion, and backprop the delta (subtract old W, add new W; visit count unchanged). This refreshes stale rewards against a consistent current baseline so UCB sees the drop when a previously-high-scoring branch no longer beats the stronger current champion. The refresh target is the *competitive* terminals only (top p%) — non-competitive ones don't need rescoring because UCB already isn't revisiting them.

Cost: ~20 cheap-judge calls × 4 refresh cycles per phase ≈ 80 extra cheap-judge calls per tree (~$0.15). Negligible.

Monitor: distribution of refreshed-W deltas. If refreshed scores drop dramatically across the board, early-tree rewards were systematically inflated and we should increase refresh frequency (lower `rechallenge_interval`) or raise `rechallenge_top_pct` to refresh more candidates per cycle.

3. **(Resolved — dropped for v1.)** Narrative forecasting phase-boundary question is moot now that forecasting isn't in the pipeline. Kept as a note for v1.5 if/when forecasting returns.

4. **Rollout can contradict partial DAG's edge fields.** A rollout LLM may produce a continuation that doesn't honor the partial DAG's disclosure edges' `withheld` fields (e.g., reveals the withheld content during rollout). Currently unhandled; the rollout terminal will just be a plausible continuation, even if it contradicts edge semantics. Two options: (a) rollouts include the edge payload constraints in the prompt explicitly, (b) rollouts are post-validated and penalized for contradiction. Prefer (a); specify in implementation.

5. **What exactly is a "frontier node"?** Current definition is informal ("anchor-side or resolution-side boundary"). Needs precise algorithmic definition: probably nodes with fewer than K outgoing edges of permitted types, OR the root / leaf nodes of the phase-subtree. Precise definition deferred to implementation.

6. **Cross-tree contamination?** The 4 pacing-preset trees run concurrently per concept. If one tree's champion is promoted before the others finish, do we introduce a cross-tree bias? No — pairwise comparisons are within-tree only (per-tree running champion). Tournament across trees happens once all 4 finish. Confirmed clean.

7. **Running-champion vs. canon-set reference (v1.5 investigation).** Every rollout is scored pairwise against the tree's running champion, which is itself MCTS-selected from earlier noisy pairwise scores. Pairwise noise compounds: cheap-judge position bias on close calls + slow champion drift can produce local traps that D-UCB and rechallenge only partially correct. A canon-set alternative — scoring rollouts pairwise against 3–5 fixed reference DAGs (drawn from pilot winners or hand-crafted) rather than the running champion — eliminates drift-induced noise in the signal UCB consumes most. The trade is (a) losing the champion-as-moving-target exploration pressure and (b) needing a canon set that's calibrated to the concept class. Worth investigating post-pilot if UCB traps show up in tree traces. Keep running-champion in v1 because it needs no extra calibration data; flag as an explicit alternative to evaluate if tree-quality issues are traceable to reward noise.

8. **Budget allocation by anchor role.** BiT-MCTS's forward-before-backward ordering rests on "generating the effect constrains the cause" (swapping phase order → 97% loss in their ablation). For **climax** and **reveal** anchors that argument applies directly — falling action and coda are the effect; forward-first lets the backward phase see what actually got resolved before generating the setup that earns it. v1 runs forward-first for both. What varies by role is how much downstream material exists to find: reveal anchors typically have a brief coda, so 50 forward iterations may hit the no-improvement cutoff with most of the budget unspent. **Pilot plan:** log per-role iterations-to-early-termination and phase-3 improvement rate. If reveal-anchored trees consistently burn less than half the forward budget, scale `iterations_per_phase` by role in v1.1 (forward smaller for reveal, backward unchanged). Do not change phase order for climax or reveal based on this data — phase order is structurally justified by the effect/cause argument; budget is the tunable knob.

   **For pivot anchors the argument is weaker.** Kishotenketsu's pivot recontextualizes rather than causes; the ki/sho upstream and the ketsu downstream are mutually constraining rather than related as effect and cause. The BiT-MCTS justification for forward-first doesn't transfer cleanly. v1 still ships forward-first for pivot (keep one mechanism across roles until data says otherwise), but if pilot data shows pivot-anchored champion quality systematically below climax/reveal, try backward-first as a v1.1 A/B arm on pivot only.
