# One Within the Novelist — Ideas

An evolutionary AI short story writing system that combines ShinkaEvolve-style
population-based optimization with a diverse simulated reader panel, staged
narrative graph exploration, and craft-informed fitness functions.

---

## Core Thesis

Current AI story generation fails in two fundamental ways: **convergence** (all
stories sound the same) and **shallowness** (technically competent prose that
lacks genuine surprise, emotional depth, and character interiority). These
failures are not just prompt engineering problems — they are architectural
consequences of how LLMs are trained (RLHF collapses diversity, instruction
tuning reduces informational surprise 2-4x below human writing).

The fix requires treating story writing as an **evolutionary search problem**
over a quality-diversity landscape, not as a single-shot generation task.
Maintain a *population* of stories evolving under selection pressure from a
*diverse panel of judges*, with explicit mechanisms to prevent mode collapse
and reward genuine novelty.

---

## Architecture Overview

### Staged Evolutionary Pipeline

Each stage has its own evolve-evaluate loop before feeding winners into the next
stage. The "genome" at each stage is different.

**Stage 1 — Concept/Premise Evolution**
- Genome: structured premise (theme + conflict + character seeds + "what if")
- Mutations: King's collision method (smash two premises together), Bradbury's
  noun-list recombination, Le Guin's thought experiment reframing, Gaiman's
  compost recombination (combine fragments from an accumulation archive)
- Population: 30-60 diverse premises
- Evaluation: judge panel scores originality, emotional potential, narrative
  tension, thematic richness
- Exit: top N premises (diverse, via MAP-Elites) advance to Stage 2

**Stage 2 — Structure/Narrative Graph Evolution**
- Genome: causal event DAG (nodes = scene beats, edges = causal/temporal links)
- Mutations: add/remove/reorder nodes, branch at decision points, merge
  subplots, apply structural frameworks (Kishotenketsu, Story Circle, Freytag),
  BiT-MCTS climax-first expansion (generate climax, expand bidirectionally)
- The population is a *forest* of graph variants per premise
- Evaluation: judge panel scores causal coherence, tension arc quality,
  surprise + post-dictability balance, structural completeness
- L4D Director-style pacing: track narrative tension as continuous variable,
  gate which beats can fire at each point (Build-up → Peak → Fade → Relax)
- Selection: MCTS for structure exploration; salience-based selection (Emily
  Short's QBN) for which storylets fire; Narrative Information Theory for
  measuring pivotal moments
- Exit: top N narrative graphs advance to Stage 3

**Stage 3 — Prose/Scene Evolution**
- Genome: the prose itself, or the prompt/instructions that generate prose for
  each scene in the narrative graph
- Mutations: style shifts, voice changes, detail level, dialogue rewrites,
  Hemingway iceberg passes (build full context then omit), Carver cuts (radical
  compression)
- Characters as Nemesis-style mutable state objects: traits, memories, wounds,
  goals, voice patterns — updated as scenes are generated, feeding back into
  subsequent scene generation
- GOAP for character behavior: characters with explicit goals + action
  repertoires; plot emerges from goal collisions
- Evaluation: prose quality dimensions (transportation, flow, lexical diversity,
  anti-slop), character depth, emotional authenticity, show-vs-tell ratio
- Affective reranking (AffGen): at sentence level, prioritize high-emotion
  candidates as proxy for engagement
- Exit: top N full stories advance to Stage 4

**Stage 4 — Holistic Refinement**
- Genome: full story + editing instructions
- CritiCS-style multi-critic refinement: 2-3 rounds of diverse-criteria
  critique-then-revise (diminishing returns after 3 rounds)
- Poe's Unity of Effect audit: does every element serve the target emotional
  response?
- Chekhov's Gun audit: is everything on the page paid off?
- Saunders' escalation check: does every beat change the story's state?
- Anti-slop pass: tiered word banning + structural anti-pattern detection
- Peak-end rule optimization: ensure ending quality is high (endings
  disproportionately determine overall evaluation)

### Cross-Stage Mechanisms

- **5-layer co-evolution** (from Nous autonovel): Voice, World, Characters,
  Outline, Chapters + Canon database — changes propagate bidirectionally across
  stages
- **Post-hoc rationalization** (from Caves of Qud): generate events first, then
  construct causal connective tissue afterward — inverts the hard forward
  planning problem into the easier retrospective rationalization problem
- **Dynamic outlining** (from DOME): outlines adapt during generation rather
  than being rigid upfront plans
- **Reflexion-style episodic memory**: maintain a running memory of what didn't
  work in previous generations ("too predictable," "inconsistent character,"
  "weak climax") as critique context for future mutations

---

## The Judge Panel

### Architecture

A panel of 5-10 parameterized reader personas, each evaluating stories
independently on specific dimensions.

**Composition rule** (from Synthetic Reader Panels, arXiv:2602.14433):
- 40% target audience anchors (age, genre matching)
- 30% adjacent/stretch audience
- 20% random diversity (age, reading sophistication, genre)
- 10% domain experts (literary critics, professional editors)

**Implementation principles** (from PoLL, LitBench, bias research):
- Use models from **different families** for generation vs. evaluation (prevents
  self-preference bias)
- **Single-turn independent evaluations** per persona (multi-turn debate causes
  conformity/confabulation)
- **0-5 grading scale** (ICC 0.853 vs 0.805 for 0-10)
- **Chain-of-thought reasoning** required before scoring
- Rubric with **criterion + per-score descriptions** (Prometheus 2 achieves
  0.897 Pearson with humans using this approach)
- **Score clustering detection**: if all judges give similar scores, the
  evaluation is degenerate — regenerate

### 10 Evaluation Dimensions

Grounded in cognitive science of narrative engagement:

1. **Transportation / Immersion** — imagery + affect + attentional absorption
   (Green & Brock transportation theory)
2. **Suspense Architecture** — protagonist threat, outcome uncertainty, tension
   build/release (Brewer & Lichtenstein, Zillmann excitation transfer)
3. **Curiosity / Information Gaps** — seeded mysteries, unanswered questions
   that pull forward (Loewenstein)
4. **Character Depth** — mental state language, internal conflict, goal clarity,
   wound-fear-lie framework (Mar & Oatley ToM simulation)
5. **Emotional Arc Coherence** — satisfying trajectory; does each shift feel
   earned? (Reagan et al. 6 arcs)
6. **Causal Coherence** — tight cause-effect chains; events connected, not
   arbitrary (Chen & Bornstein — causally central events better remembered)
7. **Surprise + Post-dictability** — unexpected yet retrospectively inevitable
   (Bissell et al. 6-criterion framework)
8. **Ending Quality** — peak-end rule; endings disproportionately determine
   overall satisfaction (Kahneman)
9. **Flow / Pacing** — narrative momentum, page-turner quality, absence of
   cognitive interruption (Thissen RFSS — flow is strongest predictor of reading
   pleasure)
10. **Memorability / Distinctiveness** — at least one indelible image; what
    would be recalled days later (story superiority effect, emotional arousal →
    memory)

### Anti-Slop Fitness (Negative Signals)

4-tier detection framework, from cheapest to most expensive:

**Tier 1 — Regex/stats (run on every candidate):**
- Slop word frequency (Tier 1 banned: delve, tapestry, nuanced, paradigm...)
- Sentence burstiness (stdev/mean sentence lengths; <0.4 = penalize)
- MATTR-500 lexical diversity (<0.7 = penalize)
- Personal pronoun density (below human baseline = penalize)
- Em-dash overuse (>2/page = penalize)
- "Not-X-But-Y" pattern density
- Nominalization rate, participial clause rate

**Tier 2 — NLP (run on promising candidates):**
- Per-paragraph entropy variance (uniform = penalize)
- Sentiment arc flatness
- Character voice differentiation (stylometric distance between characters)
- Dialogue-to-prose ratio

**Tier 3 — LLM judge (run on top candidates):**
- Over-explanation detection (showing then telling)
- Show-vs-tell ratio
- Dialogue naturalness (false starts, interruptions)
- All 12 Nous Research structural anti-patterns

**Tier 4 — Long-form only:**
- Character trait drift across scenes
- Foreshadowing resolution tracking
- Cross-scene echo detection

---

## Diversity Maintenance

The central risk: evolutionary optimization against judge scores will converge
on "high-scoring average" — technically competent but generic stories. This is
confirmed by the NeurIPS 2025 Best Paper ("Artificial Hivemind"): reward models
and LLM judges actively punish diversity.

### Mechanisms

- **MAP-Elites / QDAIF**: maintain a grid archive indexed by diversity
  dimensions (genre, emotional arc shape, structure type, POV, setting, theme,
  prose register). New stories must be non-dominated in their cell to survive.
  5-8 dimensions is practical.
- **Fitness sharing / niching**: penalize stories too similar to others in the
  population, forcing exploration across story space
- **ShinkaEvolve's novelty rejection-sampling**: if a new candidate is >0.95
  cosine similar to existing population members, reject before expensive
  evaluation
- **ShinkaEvolve's bandit-based LLM ensemble**: use UCB1 to dynamically select
  which LLM to use as writer, tracking which models produce bold improvements
  vs. incremental refinements
- **Inter-judge disagreement as positive signal**: a story that gets 10/10 from
  some judges and 3/10 from others is doing something bold. Explicitly preserve
  polarizing stories rather than selecting for consensus.
- **Go-Explore restarts**: periodically restart evolution from archived but
  undeveloped story seeds, not just from current best
- **Coevolution of evaluators**: as stories improve at gaming certain metrics,
  periodically update/retrain judge criteria to prevent Goodhart's Law

---

## Character System

Characters as mutable state vectors (Nemesis system-inspired):

```
Character {
  name, age, appearance
  wound: formative trauma
  fear: what they dread
  lie: false belief from wound
  want: external conscious goal
  need: internal truth they must face
  voice: speech patterns, vocabulary, verbal tics
  memories: []  // accumulate across scenes
  relationships: {}  // to other characters, evolving
  current_state: emotional/physical
}
```

- GOAP planning: given goals + action repertoire + world state, compute what
  each character would logically do next
- Character interview technique: surface backstory through character's own logic
- Voice differentiation test: remove dialogue tags; if you can't tell who's
  speaking, revise
- O'Connor's principle: character drives plot, not the reverse

---

## Craft Principles Encoded as Fitness Criteria

From the authors' techniques research:

- **Poe's Unity of Effect**: define target emotional response before generating;
  evaluate every element against it
- **Hemingway's Iceberg**: generate full backstory/world-building, then omit
  7/8; authority comes from knowing more than you write
- **Chekhov's Gun**: if it's on the page, it must pay off
- **Saunders' "Always Be Escalating"**: every beat must change the story's
  state; no repeated beats
- **Saunders' P/N Meter**: track reader's positive/negative emotional state
  moment to moment — story works when the needle keeps moving
- **Gardner's Vivid and Continuous Dream**: anything that breaks the reading
  dream (inconsistencies, wrong tone, meta-awareness) damages the story
- **Salesses' cultural awareness**: craft conventions encode cultural
  assumptions; evaluation should be relative to the story's chosen aesthetic,
  not a universal standard
- **Kishotenketsu option**: not all stories need conflict; revelation and
  reframing can drive a story

---

## Key Open Questions

1. **What models to use for generation vs. evaluation?** The research is clear
   that they should be from different families. Candidate: generate with Claude,
   evaluate with a mix of GPT-4, Gemini, and open-weight models (Prometheus 2).

2. **How many generations per stage?** AlphaWrite used 5 generations with 60
   initial stories. ShinkaEvolve achieves SOTA in ~150 samples. Need to
   empirically test convergence speed vs. compute cost.

3. **Short story target length?** 1,000-5,000 words. Most evaluation research
   covers this range. Longer introduces the consistency problems that ConStory-
   Bench documents (errors peak mid-narrative beyond ~8k words).

4. **Human-in-the-loop vs. fully autonomous?** Nous autonovel is fully
   autonomous. But a human curator selecting from the population at key stages
   (premise selection, structure selection) could dramatically improve quality
   with minimal effort. Consider a "guided evolution" mode.

5. **Training vs. orchestration?** Nous autonovel uses pure API orchestration
   (no fine-tuning). Sudowrite Muse trained their own model. The RL research
   (Alibaba RLCS, Edinburgh COLM 2025) suggests that RL post-training on
   narrative-theory rewards improves quality. Could the evolutionary loop itself
   serve as the RL signal — evolve prompts/pipelines rather than model weights?

6. **How to handle the uncertainty gap?** Human writing is 2-4x more
   informationally surprising. Can evolutionary pressure toward surprise close
   this gap, or does it require architectural changes (e.g., decoding strategies
   that increase entropy, or base models without RLHF)?

---

## Existing Systems to Learn From

| System | Approach | Key Lesson |
|---|---|---|
| **Nous autonovel** | 4-phase pipeline, 5-layer co-evolution, anti-slop | Iterative refinement works; anti-slop is essential |
| **AlphaWrite** | Evolutionary story generation, Elo judging | Validates the approach at small scale; warns of homogenization |
| **QDAIF** (ICLR 2024) | MAP-Elites + LLM feedback for creative text | Quality-diversity is the formal framework |
| **ShinkaEvolve** | LLM + evolutionary algorithms | Adaptive parent sampling, novelty rejection, bandit ensemble |
| **Agents' Room** (DeepMind) | Multi-agent decomposition | Specialized agents outperform single-agent |
| **CritiCS** (EMNLP 2024) | Multi-persona critics | Diverse criteria > single critic |
| **Narrative Studio** | MCTS + entity graph | Tree search for narrative exploration |
| **BiT-MCTS** | Climax-first bidirectional expansion | Generate the powerful moment first, build around it |
| **Rimworld** | Three storyteller personalities | Different drama curves as population members |
| **Caves of Qud** | Post-hoc rationalization | Generate events first, rationalize causation after |
| **L4D Director** | 4-state tension management | Pacing as continuous variable |
| **Nemesis system** | Characters as mutable state objects | History accrues automatically |

---

## Calibration Data

Immediately downloadable datasets for judge calibration:

| Dataset | What | Size | License |
|---|---|---|---|
| HANNA | 6-dim human ratings | 1,056 stories, 19k annotations | MIT |
| CR4-NarrEmote | Passage-level emotions | 200k+ annotations | CC0 |
| Fiction-1B | Quality fiction prose | 1B+ words | MIT |
| Gutenberg3 DPO | Genre-labeled chosen/rejected | 5,652 pairs | Public domain |
| CoSER | Character dialogues + thoughts | 17,966 characters | HuggingFace |
| LitBank | Entity/coreference/quotes | 100 novels | CC BY 4.0 |
| Reagan arcs | 6 emotional arc shapes | 1,327 stories | CC BY 4.0 |
| Lechmazur v4 | 18-rubric multi-judge scores | 15,500+ stories | GitHub |

Requires access but critical: LitBench (46k pairs, email Stanford), StoryER
(100k pairs), StoryAlign (ICLR 2026, pending release).

---

## Research Base

Full reports in `.claude/deep-research/runs/`:

1. `224321_ai-fiction-writing` — SOTA survey
2. `224329_nous-auto-novelist` — Nous autonovel deep dive
3. `224342_evolutionary-creative-writing` — Evolutionary approaches
4. `224512_llm-simulated-panels` — Judge panel design
5. `224722_story-gen-pipelines` — Staged pipelines & narrative graphs
6. `224847_fiction-writing-craft` — Authors' techniques
7. `225001_procedural-narrative-games` — Game narrative techniques
8. `225659_creative-fiction-eval` — Story quality metrics
9. `225936_narrative-engagement-cog-sci` — Cognitive science of engagement
10. `225959_creative-evo-diversity` — Preventing mode collapse
11. `232126_fiction-datasets-judge-calibration` — Datasets inventory
12. `232150_ai-fiction-failure-modes` — Why AI fiction fails
