# Stage 4: Prose Generation — Overview

Stage 4 is where prose happens. A Stage 1 concept gives us a premise; a Stage 2 structure gives us a typed-edge plot DAG; a Stage 3 voice spec gives us the rhythm, register, narrator stance, and three rendered prose passages on adjacent scenes. What none of these gives us is the actual story — 5,000 to 10,000 words across ~8-15 scenes that honor the upstream genome, hold their voice across length, and produce the resonance the pipeline has been preparing for.

Stage 4 produces that. The output is a finished short story as a markdown manuscript file.

This document covers the philosophy, the architecture (file-based agent with critic ensemble), the three phases, the critic suite, the tool surface, the surgical-edit-dispatch mechanism, plateau detection and termination, what carries over from earlier stages, and the handoffs on both ends. Per-critic personas, prompt files, and implementation details each live in their own docs.

> **Document status (2026-04-29):** v1 architecture as scoped in `lab/issues/2026-04-29-stage-4-architecture-scoping.md`. Pilots pending. The architectural shape is committed; specific parameters (critic budgets, cycle cap, plateau thresholds, model assignments) will be tuned by pilot signal before full v1 commit.

---

## Philosophy: Prose Is Where Everything Earns Its Place

### Why Stage 4 is the load-bearing stage

Stages 1-3 each evolved a *small, discrete* genome that could be evaluated in one or two LLM calls. Stage 4's output is an order of magnitude longer. The failure modes that don't apply upstream become load-bearing here:

1. **Voice fidelity decays across length.** Stage 3's renderings are 150-300 words each; Stage 4's prose is 5-10× that per scene and N scenes per story. The model's default attractor reasserts as the cached voice exemplars drift further into context.
2. **Continuity is harder at length.** Stage 2's DAG was small; the rendered outline fits in any prompt. By scene 8 of generated prose, prior scenes exceed the verbatim window. Entity drift, established-fact violations, motif drops become possible.
3. **Payload enactment vs. statement.** Edge payloads passed verbatim become *narration* — the model writes "the disclosure was made" rather than dramatizing the disclosure. Wrapping payloads per `docs/prompting-guide.md` §"Goals Cannot Ride in Plaintext" is necessary but not sufficient; the evaluator must distinguish enactment from statement.
4. **Single-pass prose is empirically dead** for our voice/quality targets (Stage 3 style-hint executor pilot; voice persona pilot v1+v2). Iteration with concrete feedback is the lever.
5. **Round-2 critique-revise games the metric** (`project_critique_revise_round1.md`). With a single explicit metric, the model finds the cheap path. Stage 4 inherits this constraint.

If Stage 4 collapses to model defaults across length, the entire pipeline collapses regardless of upstream genome quality. Everything Stages 1-3 produced exists to condition this generation.

### What Stage 4 is NOT for

- **Concept revision.** The concept is locked at Stage 4 entry.
- **Structural revision.** The DAG is locked. Stage 4 may surface that a structure can't be executed as prose; that signal feeds the compost heap, not back into Stage 2 in v1.
- **Voice revision.** The voice spec is the operational aesthetic commitment; Stage 4 executes against it. If Phase 3 critics surface that the voice is fundamentally unworkable for this concept, that's a Stage 3 → Stage 4 handoff failure, not a Stage 4 problem.
- **Cross-candidate selection.** Stage 4 produces one prose for one (concept, struct, voice) tuple. If multiple prose candidates need to be generated and selected among, that's Stage 6's quality-diversity work in v1.

### The architectural commitment: prose lives on a file

Stage 4 differs from Stages 1-3 in one foundational way: **the manuscript is a real file on disk, not state held in conversation context.** The agent reads, writes, and edits the file via generic file tools (Claude-Code-shape). This scales beyond context limits, gives persistent state across phase boundaries, and matches how human writers work with a manuscript. It also enables the surgical-edit pattern that the surgical-edit-dispatch mechanism depends on.

---

## Architecture: One Agent, Three Phases, Tool-Callable Critics

Stage 4 runs a single Stage-4 agent through three orchestration phases. The agent's identity persists across phases (same persona-less workshop framing, same model, same system prompt). What changes per phase is the available tool surface and the task description.

```
Stage 4 architecture:

  Setup (LLM filter call + orchestration)
    │
    ├─ Pre-stage filter call returns 0-2 ExpertNeed specs
    │  (instantiates dynamic domain_expert critics) and
    │  the work's implied AudienceFraming
    ├─ Initialize manuscript file (story.md) with scene-ID
    │  heading scaffolding
    ├─ Cache voice genome + 3 renderings in system-prefix block
    └─ Spawn the Stage 4 agent

  Phase 1 — PreThink
    │  Free-form per-scene planning in natural language.
    │  Reader-state model at scene exits (3 dimensions).
    │  Tools: read_file, write_file, edit_file, think,
    │         note_to_self, finalize_pre_think
    │  Output: pre_think.md
    └─ Ends with finalize_pre_think()

  Phase 2 — DownDraft
    │  Per-scene write-prose-to-file in topological order.
    │  Read-as-reader micro-loop after each scene write.
    │  Tools: read_file, write_file, edit_file, think,
    │         note_to_self, finalize_down_draft
    │  Output: complete (rough) story.md
    └─ Ends with finalize_down_draft()

  Phase 3 — Revise (cycled sub-phases)
    │
    │  Sub-phase A — Critique gather
    │    Tools: read_file, call_critic, think, note_to_self,
    │           finalize_critique_plan
    │    Tier A critics fire mandatory; Tier B agent-invoked
    │    Ends with finalize_critique_plan(plan_summary)
    │
    │  Sub-phase B — Revise
    │    Tools: read_file, write_file, edit_file,
    │           dispatch_surgical_edit, think, note_to_self,
    │           finalize_cycle
    │    Ends with finalize_cycle() → orchestrator runs
    │    plateau check → next cycle OR exit Phase 3
    │
    │  Either:
    │    Plateau → Phase 3 exits
    │    Cycle cap (~6) → Phase 3 exits
    │    Hard call ceiling (30-50) → Phase 3 exits
    │    Agent calls finalize_stage_4() (allowed after ≥1
    │    complete cycle) → Stage 4 exits
    └─ Output: finished story.md
```

The phase shape gives orchestration logging clean partitions and lets us enforce phase-specific tool allowlists. The agent's autonomy lives within phases; the structure between phases is the orchestrator's job.

---

## The Three Phases

### Phase 1 — PreThink (free-form natural language planning)

The agent walks each scene in natural language before any prose is written. The prompt suggests what to cover but doesn't enforce structure — natural language, not structured scaffolding.

Suggested coverage (the prompt makes these visible without forcing them):

- **What this scene is doing in the story** — its role, its function, what the prose has to accomplish.
- **Which edge payloads must be enacted (not stated).** For each payload (`realizes`, `entails`, `withheld`, `reframes`, `prohibits`, `agent`/`goal`), the agent commits to *how* the prose will dramatize it — the enactment hypothesis, not just a list of things to mention.
- **Which voice moves apply.** Drawing from the voice spec's positive constraints, characteristic moves, and the renderings.
- **Which motifs surface and how** — per the per-node motifs from the DAG, in their declared modes (introduced / embodied / performed / agent / echoed / inverted).
- **The reader-state model at scene exit** — see below.
- **Continuity facts inherited from prior scenes** — what's been established, who knows what, what's been said.

Output: `pre_think.md` (one file with per-scene headings). The agent reads its own pre-think during the DownDraft phase.

#### The reader-state model

The reader-state model is the load-bearing target the agent writes toward — not just a payload list, but a specific cognitive/affective state the prose engineers in the reader by scene end. Per the deep-research run on this question (`lab/deep-research/runs/2026-04-29-reader-state-modeling-prompts/`), the formulation operates on **three orthogonal dimensions**:

1. **Epistemic state** — what the reader knows, suspects, holds open, cannot determine. The most powerful and least failure-prone dimension. Operationalizes the information-withholding mechanism that drives narrative tension. Vocabulary: "Reader knows X happened but not why," "Reader suspects Y but has no confirmation," "Reader holds open whether Z is reliable." The Spoiler Alert (2026) finding — that explicit per-beat reader-uncertainty constraints raised tension scores from 0.606 to 0.747 — is the closest empirical anchor for this dimension.

2. **Affective quality (structural, not declarative)** — the *kind of* unsettlement / implication / resonance the prose engineers, NOT the named emotion. This dimension requires the most care: prompts that ask "what should the reader feel?" prime emotional labeling rather than dramatic conditions. Vocabulary: "Reader is unsettled in a way they cannot articulate," "Reader is implicated in the character's choice without consent," "Reader feels the wrongness before they can name it." Explicitly NOT: "Reader feels sad / hopeful / moved." Maass: *"Fiction writers assume readers will feel what their characters do, but they don't. Readers instead react by weighing, judging, comparing and creating their own emotional journey."* The agent's job is to engineer the conditions for that weighing, not to instruct the journey.

3. **Forward pull** — what unresolved question or tension carries the reader into the next scene. What page-turner work does this scene leave undone?

#### Voice calibration of the reader-state model

The voice spec from Stage 3 determines whether and how the reader-state model applies. Modernist-resistant voice modes (Beckett-inflected, Sebald-inflected, fragment-led, deliberately-difficult prose) may suppress the affective dimension or the entire reader-state element — these voices' aesthetic depends on resisting reader-satisfaction-oriented design. The PreThink prompt reads the voice spec and adapts.

#### Granularity

Scene-end is the right granularity. Beat-end is too fine for PreThink (micromanagement that overwhelms generation or collapses to labeling). Whole-story-end is too coarse (individual scene exits need their own specifications). This matches Brewer & Lichtenstein's discourse-structure work and Kahneman's peak-end rule, both of which argue scene/segment boundaries are where reader-state is most consequential for memory and engagement.

#### Empirical status

The reader-state-modeling element is **theoretically grounded but empirically untested for LLM prose generation**. Theoretical anchors: Rabinowitz's "authorial audience"; Booth's implied reader; Iser; Fish; Flower-Hayes writing-process research showing audience modeling is a marker of expert writing. Adjacent empirical anchor: Spoiler Alert's information-withholding pipeline. No published study compares reader-state-modeling prompts directly against mechanical-objectives prompts for LLM prose; pilot validation is mandated before full v1 commit.

Phase 1 ends with `finalize_pre_think()`. The orchestrator validates that all DAG nodes have plans before allowing transition to Phase 2.

### Phase 2 — DownDraft (per-scene prose with read-as-reader)

The agent writes prose to `story.md` scene-by-scene in topological order. Per scene, the loop is:

1. Read pre_think for the scene + neighboring context (prior scenes from the file, DAG context, voice spec from cached prefix).
2. Write the scene (`write_file` or `edit_file` to the manuscript).
3. Use the generic `think` tool to **re-read the just-written draft as a reader, not as its author** — per `voice_revise.txt`'s pattern: "Read each one more — not as its author, as a reader. Notice where the revision improved the prose and where it merely shuffled the problem." The reader-mode is prompt-side; the tool stays generic.
4. Optional `edit_file` if step 3 surfaced issues.
5. Move to next scene.

The read-as-reader micro-loop addresses voice drift at write-time, before it accumulates across scenes. It is metacognition (same agent, different mode), not a critic call. Complementary to Phase 3's external critic loop — different scale, different mechanism.

#### Goal: get the whole thing on the page

Lamott's shitty-first-draft principle, translated: the goal of DownDraft is to produce concrete prose for Phase 3's critics to grapple with. Quality is not the goal at this phase. The agent should resist polishing during DownDraft and trust Phase 3 to do the surgical work.

#### Reasoning settings

Phase 2 runs with thinking OFF, temperature 1.0 default. Anthropic notes thinking mode is "more detached" — not what we want for prose generation. (`project_voice_api_techniques.md`.) On DeepSeek, full sampler flexibility; on Claude, temperature is settable when thinking is off.

Phase 2 ends with `finalize_down_draft()`. The orchestrator validates story.md has prose for all scenes (or that the agent restructured intentionally) before allowing transition to Phase 3.

### Phase 3 — Revise (cycled sub-phases with critic ensemble)

Phase 3 is the load-bearing revision phase. The agent reads the finished down draft and works through cycles of critique-gather-then-revise until plateau, cycle cap, or call ceiling.

#### Sub-phase A — Critique gather

The agent has read access to the manuscript and tool access to call critics. It does NOT have edit tools in this sub-phase — the structural separation forces signal integration before commits, matching CritiCS's gather-then-revise validated pattern.

- **Tier A critics (fidelity) fire mandatory** — the orchestrator enforces that voice_fidelity, payload_enactment, continuity, and motif_fidelity all fire before allowing `finalize_critique_plan`. Per Huang ICLR 2024, LLMs cannot self-correct without external feedback; pure agent-judgment risks under-invocation of fidelity checks.
- **Tier B critics (resonance) fire at agent's judgment.** The 11 Tier B critics map to the 10 Resonance Dimensions from `docs/judging/overview.md` plus three cross-cutting cognitive-science operations (specificity grounding, defamiliarization, reader implication). Default budget 1 each per cycle; agent picks which to call when.
- **Conditional domain_expert critics fire when the work demands them.** Built at setup time by the pre-stage filter (see §"The Critic Suite" below).

The agent integrates the gathered critiques via `think` and `note_to_self`. Sub-phase A ends with `finalize_critique_plan(plan_summary)` — the agent's structured commitment to what it intends to revise, packaged for logging and the plateau check.

#### Sub-phase B — Revise

The agent applies the revision plan. It has full edit tools (read, write, edit_file) and can dispatch `surgical_edit` subagents for surgical work. It does NOT have critic tools — structural enforcement that critique gathering is done before revision begins.

Sub-phase B ends with `finalize_cycle()`. The orchestrator runs the plateau check; either the next cycle's sub-phase A begins, or Phase 3 exits.

#### Plateau detection (issue counts + severity tags)

The orchestrator's plateau detector uses **issue counts and severity tags**, not scalar scores. Each Tier A critic returns a structured list of flagged issues with `severity ∈ {severe, moderate, minor}` tags. The plateau detector watches:

- Total flagged-issue count not decreasing across 2 consecutive cycles → plateau
- No `severe` issues resolved across 2 consecutive cycles → plateau
- Agent declares no further improvement available (via `finalize_stage_4`) → plateau

This preserves the project's judges-don't-score posture (`feedback_judge_query_scope.md`) — critics observe and flag, they don't produce scalar scores that invite Goodhart-style optimization. Issue counts + severity carry the convergence signal without the Goodhart risk.

#### Backstops

Two backstops sit underneath the plateau detector:

- **Cycle cap: ~6 cycles maximum.** Empirical convergence: AutoNovel "after cycle 4, fixing one score drops another"; CritiCS validates 3 rounds; Self-Refine sees diminishing returns by round 3-5. Six is the ceiling of the empirical sweet spot.
- **Hard LLM call ceiling: 30-50 calls.** Independent of cycle count. Catches runaway loops the cycle counter misses.

Phase 3 ends when any of: plateau detected, cycle cap hit, call ceiling hit, or the agent calls `finalize_stage_4()` (allowed only after ≥1 complete cycle).

---

## The Critic Suite

Critics are subagents. Each has a system-prompt persona, a focus, and a constrained tool surface. The agent calls them via `call_critic(critic_id, focus?)` in Phase 3 sub-phase A.

**Critics return structured observations and suggestions, not rewrites.** The same rule as Stage 3's `ask_judge` (`feedback_judge_query_scope.md`): critics evaluate; the agent edits. Critics may flag specific passages, point at problems, suggest *kinds* of moves the prose could make — but they don't produce replacement text.

### Tier A — Upstream-genome fidelity (mandatory after every draft commit)

| Critic | What it checks | Persona | Default budget |
|---|---|---|---|
| `voice_fidelity` | Prose matches voice spec + renderings. **Has tool access to `stylometry`, `slop_score`, `writing_style`** to ground qualitative read in metric ensemble. The Stage 3 winning voice agent persona, full context. | Yes (Stage 3 winner) | 3 |
| `payload_enactment` | For each DAG edge with a payload, did the relevant scene **dramatize** it or only **state** it? | Yes (developmental editor archetype) | 3 |
| `continuity` | Entity log; established-fact reconciliation; story_constraints honored where they hold; character_arcs consistent. Reads the file end-to-end. | No (criteria are direct) | 2 |
| `motif_fidelity` | Per-node motifs surface in their declared mode (introduced / embodied / performed / agent / echoed / inverted). | No | 1 |

Tier A critics are the floor. Their failures invalidate everything Phases 1-2 produced; they must fire on every revision cycle so structural failures get caught before the agent moves on.

### Tier B — Resonance critics (cognitive-science-grounded)

The Tier B set maps to the **10 Resonance Dimensions** in `docs/judging/overview.md` plus three cross-cutting lit-theory operations. Each critic checks a research-grounded mechanism for what makes stories grip readers.

| Critic | Resonance dim(s) | What it checks | Persona | Default budget |
|---|---|---|---|---|
| `transportation` | Transportation/Immersion (Green & Brock 2000) | Dream-breaking elements; absence of vivid imagery; affective inertness; meta-awareness slips. Three-component check (cognitive + affective + imagery). | Yes (immersive reader) | 1 |
| `tension_curiosity` | Suspense + Curiosity (Brewer & Lichtenstein 1982; Loewenstein) | Withheld-information mechanics; resolvable vs permanent gaps; build-then-release patterns. | Yes | 1 |
| `emotional_peaks` | Emotional Depth + Arc (Radford; Reagan et al. 2016; Leong et al. 2025 brain-network integration) | Where are the emotional intensities? Are they set up and earned? Performed emotion vs. genuine. | Yes (emotionally literate reader) | 1 |
| `causal_inevitability` | Causal Coherence + Surprise/Post-dictability (Chen & Bornstein TiCS 2024; Bissell et al. 2025) | Tight causal chain → "inevitable" feel. Surprises that feel inevitable in retrospect (Bissell ideal: low predict + high post-dict). | Yes | 1 |
| `ending_quality` | Ending (Kahneman peak-end) | Does the ending land? Emotional resolution + thematic resonance + the "click." Endings disproportionately determine memory of the whole. | Yes | 1 |
| `flow_check` | Flow/Pacing (Thissen et al. — strongest predictor of reading pleasure) | Effortless processing; loss of time awareness; momentum. Where does processing stutter? | No (criteria direct + corroborable via burstiness) | 1 |
| `signature_image` | Memorability/Distinctiveness (Mar story superiority effect; Leong 2025) | The indelible image — would a reader recall this story days later? Where is the prose's silhouette in cultural memory? | Yes | 1 |
| `specificity_grounding` | Cross-cutting (Saunders, Paivio dual-coding) | Concrete sensory anchors. Load-bearing detail check — remove any detail; would the scene be different? | Yes | 1 |
| `defamiliarization` | Cross-cutting (Shklovsky) | Does the prose make the familiar strange? Where does it slow attention? Or is it skimming over what could be seen anew? | Yes (poet/imagist sensibility) | 1 |
| `reader_implication` | Cross-cutting (Sebald/Saunders complicity pattern) | Where does the prose implicate the reader? Working with the reader's prior assumptions/wishes against them. | Yes | 1 |
| `irreducible_remainder` | Emotional Depth + Memorability (overlap) | What in this prose can't be paraphrased away? Where would summary lose something load-bearing? | Yes | 1 |

#### Why all Tier B critics fire by default

Per the long-form research (`lab/deep-research/runs/2026-04-29-stage-4-iterative-critique-architectures/`), each Tier B sweep costs ~$0.05/critic at sonnet-4-6 (~$0.55 for 11 critics). Cheaper than building a concept-tagged-applicability filter that introduces misclassification risk. The misdirection risk on stories where a dimension doesn't apply is handled by **explicitly allowing "this dimension isn't load-bearing here; the prose is pursuing X instead" as a valid critic output**. A voice/compression piece (Hills Like White Elephants) won't have suspense in the Brewer & Lichtenstein sense; `tension_curiosity` should observe that and confirm what the prose IS pursuing rather than demanding suspense be present.

#### Conditional critics — the `domain_expert` factory

Some critics earn their place only on specific kinds of stories: hard-SF needs physics expertise, period pieces need historical accuracy, medical fiction needs procedural correctness, cultural fiction set outside the author's lived experience needs cultural verification. Rather than hardcoding a critic per domain (combinatorial sprawl), Stage 4 uses a **generic `domain_expert` critic class parameterized at setup time** by a pre-stage classification call.

The pre-stage filter (cheap haiku-class call at session start) reads the concept + DAG + voice spec and returns 0-N `ExpertNeed` specs:

```python
class ExpertNeed(BaseModel):
    needed: bool
    domain: str                  # specific (e.g., "quantum optics", "Roman
                                 # military history of the late Republic",
                                 # "competitive bridge")
    expertise_focus: list[str]   # what the prose will likely need to get right
    persona_hint: str            # one-paragraph description of who has this
                                 # expertise, in ordinary-specific register
                                 # ("a postdoc in atomic physics who reviews
                                 # fiction manuscripts as a side gig" — NOT
                                 # "a Nobel-winning physicist")
    web_search_recommended: bool # true when factual verification matters
                                 # more than tacit understanding
```

When `experts` is non-empty (capped at 2 in v1), each spec instantiates a `domain_expert` critic at setup with persona, focus, and `web_search` allowlist filled in from the spec. The critic's system prompt template is fixed (workshop frame, observations-not-rewrites contract, classification taxonomy: `verisimilitude_break / dramatic_license_acceptable / uncertain_recommend_review`); only the variable parts are runtime-rendered.

The pattern this implements is a *factory* for critics, not a *registry* of pre-built ones. Right for the open-ended-expertise space (we can't anticipate every domain — competitive bridge play, deaf culture, historical Yugoslav punk — runtime instantiation handles all of them).

### Persona / non-persona split

Per CritiCS Fig 2 (persona-critics 59/49/56 vs non-persona 37/42/38 on Interesting/Coherence/Creative), persona-driven critics catch coherence-relevant issues that non-persona critics miss. Most Stage 4 critics get personas. Two exceptions: `continuity` and `motif_fidelity` — both have direct, mechanical criteria where persona adds noise rather than insight. `flow_check` is also non-persona because its signal is largely corroborable via burstiness metrics.

### Persona discipline — ordinary-specific, not famous-author

Same rule as Stage 3 voice agents (`feedback_few_shot_exemplars_for_voice.md`): ordinary-specific personas with material work backgrounds (the developmental-editor; the reader-who-marks-poetry; the entomologist-who-reads-fiction-for-pattern). Famous-author archetypes collapse to caricature (CoMPosT EMNLP 2023; Mikros 2025).

Each critic persona's system prompt should:
- Open with the cognitive/literary mechanism the critic is checking, in research register (not mystical theater)
- Set workshop frame, not assistant register (`feedback_assistant_register.md`)
- Specify what observations the critic returns, with examples in its own voice
- Forbid score predictions and rewrite requests (`feedback_judge_query_scope.md`)
- Allow "this dimension isn't load-bearing here; the prose is pursuing X instead" as a valid output
- Include 2-3 prose exemplars of the dimension *operating well* in canonical prose, where the dimension benefits from few-shot prose anchoring

---

## Architectural Invariants

Three architectural commitments hold across Stage 4 and shape every other decision.

### 1. Cross-family critics (in-context reward hacking safeguard)

The Stage 4 generating agent runs on Claude (Opus prod / Sonnet dev) or DeepSeek-v4-pro (dev). **Critics MUST run on a different model family from the generator.** This is mechanistic: in-context reward hacking (arxiv 2407.04549, validated on creative writing) emerges at iteration 1-5 when the same model generates and judges. Judge scores inflate while human-rated quality stagnates or decreases. Cross-family is not a cost preference; it's a safeguard against convergence gaming.

The voice_fidelity critic is the principled exception (continuity of authorship from Stage 3's winning voice agent). Even there, partial diversification is preferred (e.g., generator on Opus, voice_fidelity on Sonnet) for additional ICRH protection.

### 2. No writer-persona for the agent

The voice spec from Stage 3 IS the aesthetic commitment. Adding a writer-persona to the agent would compete with the voice spec — which one wins when they conflict? The Stage 4 agent runs with workshop framing + task description + voice spec + DAG + concept + audience framing in its system prompt. No archetype, no aesthetic identity beyond what the voice spec carries. The voice_fidelity critic provides aesthetic continuity (the same persona that produced the voice spec evaluates the prose's adherence to it). The agent's job is execution; the voice spec carries the stance.

### 3. The manuscript file is the single source of truth for prose state

State held in conversation context drifts and decays across long phases. The manuscript file is the canonical prose. The agent reads from it and writes to it; critics read from it; surgical-edit subagents read from it and write to constrained scopes within it. There is no "prose held only in context, written to file at finalize." Every write is committed.

---

## Audience Framing

The pre-stage filter call also returns the work's **implied audience** (Rabinowitz's "authorial audience" — who the work is for, not who's writing it). This is populated into the agent's system prompt and serves as a target the writing aims at.

The audience framing is concept+voice-derived: cheap setup-phase classification reads concept and voice spec, returns a paragraph-shape description of the implied reader (their tastes, prior literary exposure, what they bring to the text, what they recognize). This complements the voice spec without competing with it:
- Voice spec: HOW to write (aesthetic stance, characteristic moves, prose texture)
- Audience framing: WHO it's for (implied reader, what they bring)

The reader-state model in PreThink layers on top: WHO the reader is (audience framing) → what they feel/know/hold open at scene boundaries (reader-state model). The two are consistent and additive; they operate at different scales.

---

## Tool Surface

The Stage 4 agent has 12 tools, with phase-specific allowlists determining which fire when.

| Tool | Available to | Description |
|---|---|---|
| `read_file(path, offset?, limit?)` | agent (all phases), critics, surgical-edit subagents | Read manuscript / pre_think / any run file |
| `write_file(path, content)` | agent (Phases 1, 2, 3-B), surgical-edit subagents (within scope) | Full overwrite |
| `edit_file(path, find, replace, replace_all?)` | agent (Phases 1, 2, 3-B), surgical-edit subagents (within scope) | Find/replace edit |
| `think(thought)` | agent (all phases) | Generic in-the-moment reasoning. Returns "Considered." Thought stays in tool-use history within phase, doesn't persist across phases. Reused from `owtn/stage_3/tools.py` |
| `note_to_self(text)` | agent (all phases) | Cross-phase memo. Visible to agent across calls; NOT visible to critics. Reused from `owtn/stage_3/tools.py` |
| `call_critic(critic_id, focus?)` | agent (Phase 3 sub-phase A only) | Dispatch a critic subagent. Returns CriticReport (issue list + severity tags) |
| `dispatch_surgical_edit(scope_description, instruction)` | agent (Phase 3 sub-phase B only) | Spawn surgical-edit subagent constrained to a natural-language-described scope |
| `web_search(query, n?)` | domain_expert critics only (per-critic allowlist) | Provider-agnostic Python wrapper around Exa (or equivalent external search API). Does NOT use Anthropic native web_search (dev runs on DeepSeek which has no native equivalent) |
| `finalize_pre_think()` | agent (Phase 1 end) | Explicit commit ending Phase 1. Orchestrator validates pre_think.md coverage |
| `finalize_down_draft()` | agent (Phase 2 end) | Explicit commit ending Phase 2. Orchestrator validates story.md coverage |
| `finalize_critique_plan(plan_summary)` | agent (Phase 3 sub-phase A end) | Ends sub-phase A. plan_summary packages gathered signals + intended responses for logging and plateau check |
| `finalize_cycle()` | agent (Phase 3 sub-phase B end) | Ends sub-phase B. Orchestrator runs plateau check |
| `finalize_stage_4()` | agent (Phase 3 only, after ≥1 cycle) | Explicit "story complete" signal. Ends Stage 4. Orchestrator can also force on cycle-cap / call-ceiling |

### File format

`story.md` is markdown with `## scene_id` headings matching DAG node ids. Optional frontmatter for run metadata. The convention is established by the setup scaffolding; the agent may follow or restructure as it judges. There is no scene-aware tooling enforcing the structure — the agent uses string operations on the file the way Claude Code finds code.

---

## Surgical-Edit Dispatch — The Surgical-Edit Mechanism

`dispatch_surgical_edit` is the load-bearing surgical-edit mechanism. It exists for the case where the agent wants to polish or fix a specific passage without risk of unintended changes elsewhere.

When the agent calls `dispatch_surgical_edit(scope_description, instruction)`:

1. **Cheap haiku-class subagent translates scope to bounds.** Reads the file + scope_description, returns structured bounds: `{anchor_before: str, anchor_after: str, line_range: [start, end] | null, scene_heading: str | null}`. ~$0.01.
2. **Surgical-edit subagent (sonnet-class with reasoning) does the edit.** Gets bounds in its system prompt + the parent agent's `instruction` + read access to the full file + `edit_file` access wrapped to validate edits are within bounds.
3. **Post-hoc validation.** A cheap diff check confirms edits stayed in-bounds. Out-of-scope edits are rolled back; subagent retries with the scope reinforced.

The flavor is "translate then constrain" rather than "trust" or "reflect post-hoc." Out-of-scope edits are an architectural failure mode the design eliminates rather than detects.

The surgical-edit subagent reads the full manuscript (so it has surrounding context) but its writes are scoped. It can't break what it can't write to.

---

## Parent-Agent Configuration

### Model

- **Production:** claude-opus-4-7 (1M context). Long stories + voice renderings + DAG + critic reports + scratchpad fit comfortably; creative writing strength + reasoning capability matter at this stage.
- **Development:** deepseek-v4-pro (cost-driven, per `feedback_dev_default_llm`). Architecture validation happens at deepseek prices before paying for Opus quality.

### Persona

None — Option X from the architecture scoping. Workshop frame + task description + voice spec + DAG + concept + audience framing carry the work without naming a persona. Mirrors Claude Code's shape (no persona, just tool surface and task).

### Sampler / reasoning settings

| Phase | Reasoning | Temperature | Notes |
|---|---|---|---|
| PreThink | ON (medium) | 1.0 | Forced on Claude with thinking; settable on DeepSeek. Reasoning helps with narrative/structural planning |
| DownDraft | OFF | 1.0 default | Thinking mode is "more detached" per `project_voice_api_techniques.md` — off for prose generation. Temperature settable on Claude when thinking is off |
| Revise | ON (medium) | 1.0 | Critic-signal integration is reasoning-heavy work |

### Memory across phases

`note_to_self` persists across phases; visible to the agent in subsequent phases but NOT visible to critics. Critics evaluate prose, not the agent's intentions; if critics could see the agent's plans, they might pattern-match to them rather than to the actual prose (anchoring bias).

---

## How Stage 4 Differs from Earlier Stages

Stage 4 runs on a different shape than Stages 1-3. What carries over is the LLM client, the judge-panel philosophy applied to critic personas, the pairwise comparison machinery (used by Stage 6 if cross-candidate work happens there), and the compost heap.

| Aspect | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| Engine | ShinkaEvolve (async population, islands) | MCTS (per-concept tree per preset) | Multi-agent session (5 phases) | Single agent + critic ensemble (3 phases) |
| Genome | Concept JSON | DAG JSON | Voice spec + 3 renderings | Manuscript file (5-10k words markdown) |
| State medium | Database | DAG renders | YAML payload | **File on disk** (new) |
| Diversity engine | Islands + operator bandit | Pacing presets | Voice persona casting | (None within Stage 4 — produces one prose; cross-candidate work moves to Stage 6) |
| Selection | Pairwise champion succession + Swiss tournament | UCB1 + within-concept tournament | Pairwise judge tournament + agent Borda | (None — see selection note above) |
| Critic shape | 3 judge panel, pairwise per-criterion | Same panel; 8 dims | Same panel; 5 dims; voice exemplars | 15-critic ensemble (Tier A + Tier B + dynamic experts); observations-not-rewrites |
| Cost (light) | $5-10 / run | $30-48 / run | $5-20 / run | $8-17 / candidate (provisional; pilot to validate) |

### What carries over

- **LLM client** (`owtn/llm/`) — caching, multi-provider, cost tracking, bandit ensemble
- **Tool infrastructure** — `_think_handler`, `_note_to_self_handler` reused directly from `owtn/stage_3/tools.py`
- **Persona-loading patterns** — copy structure from `owtn/stage_3/personas.py`, adapt to critic-persona shape
- **Filter patterns** — copy structure from `owtn/stage_3/casting.py` for the pre-stage classification call
- **Stylometry/slop/writing-style tools** — `voice_fidelity` critic calls these directly
- **Judge personas adapted as critics** — the 4 contest judges' aesthetics inform some critic-persona designs
- **`docs/judging/overview.md` Resonance Dimensions** — the Tier B critic taxonomy is operationalized from these

### What does not carry over

- **ShinkaEvolve.** No population evolution; one prose per tuple in v1.
- **MCTS / pacing presets.** Not relevant at the prose layer.
- **Pairwise within Stage 4.** Stage 4's critic ensemble is descriptive (issue counts + severity tags), not pairwise. Cross-candidate pairwise work is Stage 6.
- **MAP-Elites archive.** Prose space doesn't have obvious orthogonal axes for grid archiving in v1.

---

## Handoff From Stage 3

Stage 4 receives for each (concept, struct, voice) tuple advancing from Stage 3:

1. **Complete voice genome** — pov / tense / lit-theory fields / description / diction / positive_constraints / prohibitions / **3 renderings** (the load-bearing field)
2. **Adjacent scene drafts** — both the neutral baselines and the winner's voice-transformed renderings
3. **Stylometric signals from Stage 3 pre-screening** — drift detection baseline
4. **Identity of the Stage 3 winning voice agent** — carried forward to instantiate `voice_fidelity` critic with full Stage 3 context (persona prompt + reasoning during Stage 3 + the renderings they produced)
5. **Complete DAG** (Stage 2 forward) — nodes / edges / motif_threads / character_arcs / story_constraints / concept_demands
6. **Concept genome** (Stage 1 forward) — premise / target_effect / anchor_scene / constraints / style_hint / character_seeds / setting_seeds

### How Stage 4 uses the handoff

The voice genome populates the cached system-prefix block (renderings as few-shot exemplars; voice spec as conditioning). The DAG becomes scene scaffolding for `story.md` (one heading per node) and the source for PreThink's per-scene context. The concept genome populates the agent's system prompt (with goal-fields wrapped per the prompting guide). The voice agent identity instantiates `voice_fidelity` at setup.

`target_effect`, `thematic_engine`, and edge payloads are wrapped per `docs/prompting-guide.md` §"Goals Cannot Ride in Plaintext" to prevent literal narration.

---

## Handoff to Downstream

Stage 4 produces:

```
results/run_<ts>/stage_4/
├── by_tuple/
│   ├── c_a8f12e_struct_0_voice_R/
│   │   ├── story.md                 # The manuscript (the output)
│   │   ├── pre_think.md             # Phase 1 plans
│   │   ├── critiques/
│   │   │   ├── cycle_1/
│   │   │   │   ├── voice_fidelity.json
│   │   │   │   ├── payload_enactment.json
│   │   │   │   └── ...
│   │   │   └── cycle_n/
│   │   ├── parent_log.yaml          # Decisions, edits made, cycle structure
│   │   ├── plateau_log.yaml         # Issue counts + severity by cycle
│   │   ├── stylometric_drift.json   # Voice fidelity tracking across run
│   │   └── run.log
│   └── ...
├── handoff_manifest.json            # What advances downstream
└── stage_4_run.log
```

The manuscript is the load-bearing artifact. Everything else is logging and signal preserved for cross-stage analysis.

### Stage 4 → Stage 5 / Stage 6 (open question)

The original 6-stage spec defines Stage 5 as "Refinement — editorial critique-revise cycles (2-3 rounds max)" and Stage 6 as "Selection & Archive — quality-diversity archiving, feedback to Stage 1." With Stage 4's Phase 3 already running a critic-revise loop, Stage 5's role becomes ambiguous. Two possibilities:

- **Stage 5 absorbed.** Stage 4 does the revise work; Stage 5 is gone or repurposed.
- **Stage 5 retained for macro revision.** Stage 4 produces a near-final draft; Stage 5 handles harder, more aggressive revision (kill darlings, restructure, possibly Stage-2 re-entry) — Lamott's "broader category" revision that goes beyond Stage 4's mid-level work.

In v1, **Stage 4 produces one prose per tuple, deferring both Stage 5's role and any cross-candidate aggregation to Stage 6**. Stage 5 and Stage 6 remain spec-only until pilot data informs whether Stage 4's revise loop catches macro-level issues. If pilot shows Stage 4's parent agent doesn't escape the down-draft's gravity for structural issues, Stage 5 retains its macro-revision role; otherwise it folds into Stage 4.

This is the genuinely flimsy part of Stage 4's commitments — the cross-stage boundary is deferred to two future stages neither of which has a settled role.

---

## Validation Protocol

Stage 4 has lighter formal validation than upstream stages — most "validation" happens via the critic ensemble in Phase 3, which IS the evaluation surface during the run. End-of-run validation:

### Gate 1: Schema validation

- `story.md` exists and contains prose for all DAG nodes (or the agent restructured intentionally and the orchestrator accepted at `finalize_down_draft`)
- All `finalize_*` calls completed in valid order
- No phase exited via exception

### Gate 2: Tier A critic floor

- Final-cycle Tier A critics return no `severe` findings, OR
- The agent explicitly justified in `note_to_self` why a `severe` finding was accepted (rare; usually a voice/aesthetic decision the agent stands behind despite the critic's flag)

### Gate 3: (deferred) Tier-A pre-screening on the final manuscript

End-of-run stylometric checks (slop_score, burstiness, MATTR, function-word distance vs voice centroid) on `story.md`. **Not implemented in v1** — the long-form research recommended this as optional and we deferred. Add only if pilot shows we need a final hard-pass gate.

---

## Cost Estimate

Per-tuple cost at light config (provisional; pilot to validate):

| Step | Cost |
|---|---|
| Pre-stage filter (haiku-class) | $0.05 |
| Audience framing call (haiku-class) | $0.05 |
| Phase 1 PreThink (8-15 scenes) | $0.50-1.50 |
| Phase 2 DownDraft (8-15 scenes with read-as-reader) | $1.50-3.50 |
| Phase 3 critic sweeps (~3-5 cycles × 5-8 critics each) | $4-10 |
| Phase 3 revisions + surgical-edit dispatches | $2-4 |
| **Total per candidate** | **~$8-19** |

Heavy config (more critic budget, more cycles): $30-60.

Production with Opus 4.7 instead of dev DeepSeek: 2-4× the dev cost depending on call distribution.

The voice_fidelity critic with metric tools (stylometry, slop_score, writing_style) costs slightly more per call than other critics — these are local Python tools but they involve real computation.

---

## Open Questions

These are flagged for resolution during pilot or in subsequent docs:

1. **Stage 4 / Stage 5 / Stage 6 boundaries.** As above — Stage 5's role and Stage 6's role both depend on what Stage 4's revise loop actually achieves. Pilot will tell.

2. **Reader-state-model effectiveness.** The PreThink reader-state element is theoretically grounded (Rabinowitz, Booth, Iser, Flower-Hayes; adjacent Spoiler Alert empirical evidence) but empirically untested for LLM prose generation. A/B pilot mandated before full v1 commit (see `lab/issues/2026-04-29-stage-4-architecture-scoping.md` Steps).

3. **Voice fidelity decay curve at 5-10k.** The single highest-uncertainty load-bearing claim. Per-scene voice_fidelity tracking across a full run will quantify decay. If decay is steep, hosted fine-tuning (v1.5) becomes more attractive.

4. **15-critic ensemble saturation.** Extrapolated above CritiCS's 5 critics. Justified by task-diversity but unvalidated. Pilot signal: does the agent reach for many critics or settle into a small subset?

5. **Mid-story focus for continuity.** ConStory-Bench's 40-60% finding measured at 8-10k stories with 19 models including older ones. Severity on Opus 4.7 / Sonnet 4.6 likely less than benchmark headline. Pilot signal: do continuity issues actually cluster mid-story for our generator?

6. **Critic-call autonomy (Tier B).** The agent decides when to invoke each Tier B critic. Risk of under-invocation (declares done before catching all issues). Mitigation: Tier A mandatory floor catches the highest-leverage failures regardless of agent judgment. Pilot signal: does the agent reach for the right Tier B critics for each story?

7. **Plateau detection thresholds.** Issue counts + severity tags carry the convergence signal, but the specific thresholds (e.g., "no severe resolved across 2 cycles" — is 2 the right number?) are pilot-tunable.

8. **Cycle cap calibration.** ~6 cycles based on AutoNovel + CritiCS empirical convergence. May be too generous or too tight for our specific architecture. Pilot signal: where does plateau actually fire vs. cycle cap?

9. **Hosted fine-tuning v1.5.** Memory entry `project_voice_api_techniques.md` previously cited Mosca 2025 as 97-98% style compliance with 50-100 exemplars at ~$0.03 — corrected 2026-04-29 (claim is unverified; no published paper matches). Realistic FT path is LoRA-per-axis adapters (StyleRemix-style) or DDPO/GRPO with synthetic data, NOT hosted fine-tuning at small-exemplar regime. Defer to v1.5; voice fidelity decay measurement (Q3) is the decision driver.

10. **Final Tier-A pre-screening (Gate 3).** Optional in v1. Add only if pilot shows critic-loop quality occasionally drops below the slop/burstiness floor we'd accept.

11. **Adaptation re-entry to Stage 2.** Out of scope for v1. v1.5 mechanics if pilot shows structural problems Stage 4 can't fix and would benefit from DAG-level mutation rather than compost-heap rejection.

---

## Where to Look for More Detail

- **`lab/issues/2026-04-29-stage-4-architecture-scoping.md`** — full architecture scoping with all decisions, pushback, and provisional commitments
- **`lab/deep-research/runs/2026-04-29-reader-state-modeling-prompts/`** — empirical/theoretical grounding for the reader-state element
- **`lab/deep-research/runs/2026-04-29-stage-4-iterative-critique-architectures/`** — long-form architecture audit + literature review
- **`docs/judging/overview.md`** — the 10 Resonance Dimensions that ground the Tier B critic suite
- **`docs/stage-3/overview.md`** — Stage 3 voice genome handoff; the source of the voice spec, renderings, and `voice_fidelity` critic
- **`docs/prompting-guide.md`** — goal-field wrapping, decision-chain prompts, additive context
- **`owtn/stage_3/tools.py`** — `think` and `note_to_self` handlers Stage 4 reuses directly
- **`owtn/orchestration/`** — agent / phase / tool primitives Stage 4 builds on
- **`configs/stage_4/critics/`** (TBD) — critic persona YAMLs as they're written
- **`docs/stage-4/critics.md`** (TBD) — per-critic persona detail and operational rationale
- **`docs/stage-4/surgical_edit.md`** (TBD) — surgical-edit dispatch implementation detail
- **`docs/stage-4/implementation.md`** (TBD) — module structure, build order, test strategy
