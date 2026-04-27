# Stage 2: Rubric Anchors

Stage 2 judges evaluate structures on 8 dimensions via per-criterion pairwise comparison. This document specifies each dimension's definition, sub-criteria, and endpoint anchors (score 1 and score 5) plus a score-3 exemplar, following `lab/references/reference-answer-bias/` (ρ improvement 0.62 → 0.76 with score-3 exemplar) and `lab/references/judge-design-choices/` (endpoint-only anchors outperform full 1–5 scales).

The rubric was reduced from 9 to 8 dimensions pre-implementation (2026-04-19): Edge-Type Appropriateness was merged into Causal Soundness as "Edge Logic" because the dimensions collapse on the same failure mode (causal edges that should be a different type, or edges with generic payloads). Post-dictability was kept separate from Tension & Information Architecture despite overlap — surprise-that-feels-inevitable is a distinct craft property worth its own judgment; revisit if empirical co-vote rate > 0.85.

Anchors use endpoint-only format: only scores 1 and 5 are fully described; 2, 3, and 4 are interpolated, with a named score-3 exemplar for calibration. This matches Stage 1's format after the 2026-04-04 rubric redesign.

Each dimension is tied to cognitive-science research (cited in `docs/ideas.md` 10-dimension list or in the scoping issue §10) and to a specific downstream bias — what this dimension, by being evaluated at Stage 2, forces Stages 3–5 to respect.

Literary examples are drawn from published canonical fiction, never from AI-generated text. Sources: Jackson, Chiang, Hemingway, O'Connor, Le Guin, Carver, Carson, Borges, Saunders, Kafka, Oates, Munro, Woolf, Ishiguro, Machado.

---

## 1. Edge Logic

**Definition**: does each edge have the right type for the relationship it names, and does its payload actually hold? This combines two concerns: causal edges' `realizes` fields must name plausible mechanisms (Causal Soundness), and the broader edge-type distribution must match the relationships actually present (Edge-Type Appropriateness). A merely-sequential beat labeled causal fails both: wrong type AND an unearned payload.

**Cog-sci anchor**: Trabasso & van den Broek (1985) — causal-chain membership is the strongest predictor of story recall. Chen & Bornstein (2024) — causally central events are better remembered. LLM causal-reasoning failure modes (arXiv:2410.23884): positional bias conflates temporal order with causation. IPOCL typed-edge theory (Riedl & Young 2010) — causal vs. motivational distinction is formally load-bearing; our 5-edge taxonomy extension defended in `lab/deep-research/runs/20260418_034956_stage2-typed-edges/final_report.md`.

**Downstream bias**: Stage 3 voice register aligned with structural mode (implication-heavy → clinical; disclosure-heavy → flat-surface-deep-tension; constraint-heavy → spare). Stage 4 prose preserves cause-effect flow and honors edge-type-specific moves. Stage 5 Chekhov-gun audit has a logical spine to check.

**Sub-criteria**:

- **Mechanism specificity**: the `realizes` field names a specific mechanism, not a generic one ("he was angry, so he acted" is weak; "his childhood fear of the dark, re-triggered by the blackout, pushed him past caution" is strong). Analogous specificity expected of `entails`, `prohibits`, `reframes`/`withheld`, and `goal`/`stakes` fields.
- **Temporal-causal distinction**: merely-sequential beats are not labeled causal. A beat that happens after another without being caused by it is not causally linked — it's an implication, a disclosure, or an untyped adjacency.
- **Mode match**: the dominant edge type matches the concept's mode. A thought-experiment concept leans implication-heavy; a reveal concept leans disclosure-heavy; a voice-constraint concept leans constraint-heavy; a character-collision concept leans causal + motivates. A purely single-type DAG is unusual and should be justified by the concept.
- **Edge quality per type**: causal edges have specific mechanisms; disclosure edges have meaningful withheld information (reader-facing); constraint edges name a concrete in-world prohibition; motivates edges have specific character goals, consistent agents, and *local* scope (source and target near-adjacent; whole-story arcs belong in `character_arcs`, not edges).
- **Implication is strict entailment, not thematic rhyme**: an `implication` edge names an in-world logical consequence — given A, a character could reason that B must hold. If A and B merely share thematic resonance (the story argues they parallel each other), that's motif-carried recurrence, not implication. Mis-use: "the heptapod script entails the physicist's variational principle" — that's authorial rhyme, not entailment. Correct use: "if the bomb went off in the room at noon, the room is destroyed at 12:01." Judges should demote `implication` edges whose `entails` field reads as a thematic claim rather than a logical consequence.
- **Disclosure vs. constraint**: disclosure hides/reveals from the reader (`withheld` is reader-facing); constraint prohibits in the world (`prohibits` is diegetic). A single event that does both (villagers not naming the lottery's lethal purpose) should be encoded as both edges — the constraint names the social rule, the disclosure names the reader-reframe.
- **Chain length integrity**: long causal chains (5+ edges) maintain specificity throughout, not degrading into "A happened, B happened, C happened" filler.
- **No decorative edges**: removing any edge should visibly degrade the DAG. Edges whose removal changes nothing are labels without semantics.

**Score 5**:
Every edge has the right type for its relationship, and every payload field is specific and load-bearing. The edge type distribution matches the concept's mode with intentional variation. A Chiang-mode concept might be 60% implication, 20% causal, 15% motivates, 5% disclosure. A Jackson-mode concept might be 40% causal, 40% disclosure, 10% motivates, 10% implication. Whatever the distribution, it is intentional and serves the concept. Example: O'Connor's "A Good Man Is Hard to Find" — the grandmother's vanity causes the detour; the detour causes the crash; the crash causes the encounter with the Misfit. Each causal link is specific; motivates edges anchor the grandmother's arc; the Misfit's dialogue carries implication edges.

**Score 3 (exemplar)**:
Most edges have the right type; 1–2 are miscoded (merely-sequential beats labeled causal, or a reveal that should be a disclosure edge labeled causal). Edge type distribution is roughly appropriate but slightly miscalibrated — a thought-experiment concept has too few implication edges, or a reveal concept has disclosure edges but they're concentrated at the end rather than dispersed. Most payloads are specific; 1–2 are generic.

**Score 1**:
Edges are labeled but their payloads are generic ("A leads to B") or contradicted by the beat sketches. Edge type distribution is default — mostly causal with a token disclosure edge at the climax, regardless of concept. The LLM has not engaged with the typed-edge distinction; edges are labeled but not chosen thoughtfully. Multiple edges are decorative — removing them wouldn't change the DAG's meaning.

---

## 2. Motivational Coherence

**Definition**: do the genome's motivational structures — `character_arcs` (whole-story trajectories) and `motivates` edges (local intention installations) — together ground character behavior in specific mental states? Do they cohere across beats without contradicting each other?

**Cog-sci anchor**: Graesser et al. (1991, 1994) — story comprehension requires intentionality attribution. Riedl & Young IPOCL (JAIR 2010) — causal and motivational links are formally distinct. CoG 2025 benchmark: LLMs fail at motivational planning 53% of the time when implicit.

**Downstream bias**: Stage 3 voice has interiority to shape; Stage 4 character actions grounded in specific intentions, not generic emotion.

**Sub-criteria**:

- **Agent consistency (across arcs and edges)**: the same character is named the same way in `character_arcs[].agent`, `motivates[].agent`, and beat sketches. Personality and goals cohere.
- **Scope discipline**: spanning trajectories live in `character_arcs` (≥3 non-adjacent touches); local installations live in `motivates` edges (near-adjacent src/dst). A long-span intention encoded as an edge with arbitrary endpoints — or a local installation that happens to be an arc — is a miscoded motivation.
- **Arc-edge consistency**: when a `motivates` edge installs a local goal on an agent who also has a `character_arc`, the local goal should be compatible with the arc's lifetime goal (e.g., the grandmother's local goal "persuade Bailey to take the dirt road" is consistent with her arc goal "to be seen as a lady"). Contradictions flag miscoding.
- **Goal specificity (arcs and edges)**: goals name specific intentions, not generic drives ("she wanted to escape" is weak; "she wanted to be seen as someone who had never compromised" is strong).
- **Stakes groundedness**: when present, `stakes` names what is specifically risked by this character's pursuit of this goal.
- **Arc presence fidelity**: the character named in an arc must actually surface in the beat sketches — not necessarily every beat, but at enough non-adjacent beats that the arc is visibly operative. A character arc whose agent never appears in any sketch is abstract-only and should be demoted. (Earlier drafts carried an explicit `touches` list on arcs; dropped because it duplicated signal derivable from sketches — see `overview.md` §Character arcs "No `touches` list".)
- **Intentionality causation (edges)**: a `motivates` edge links a specific earlier event to a specific adjacent action sequence; the link is not "the character was generally motivated by X" but "event A installed goal G in character C, anchoring action B."

**Score 5**:
Every `character_arc` names a specific agent, a specific lifetime goal, and specific stakes; the agent surfaces in beat sketches at enough non-adjacent beats to be visibly operative across the story. Every `motivates` edge installs a local goal that is compatible with its agent's arc (if any), with a specific antecedent event. Different characters have different trajectories; when arcs conflict (grandmother vs. Misfit in O'Connor), the structural consequence is visible. Example: O'Connor's "A Good Man Is Hard to Find" — the grandmother's lifetime arc ("to be seen as a lady") anchors local motivations at the detour and the pleading scene; her arc is exposed and reframed at the climax.

**Score 3 (exemplar)**:
Most arcs and edges have specific agents and goals. One arc's agent is named but only weakly present in sketches, or one motivates edge is vague. Arc-edge compatibility is mostly maintained with one minor slippage.

**Score 1**:
Arcs are absent or vague (generic agent, generic goal) where the concept calls for them. Motivates edges exist but with generic agents, vague goals, or endpoints that span the whole story (uncoded long-span arcs). Arc and local edge goals for the same agent contradict each other. The LLM filled in motivational fields without grounding them.

---

## 3. Tension & Information Architecture

**Definition**: does the DAG manage suspense and information gaps to produce narrative momentum? Do disclosure edges withhold and reveal information in a sequence that creates and maintains curiosity?

**Cog-sci anchor**: Brewer & Lichtenstein (1982) — suspense requires uncertainty about resolution. Zillmann excitation transfer — residual arousal amplifies later beats. Loewenstein (1994) — curiosity is perception of an information gap. Narrative Information Theory (Schulz 2024) — formal framework for suspense and curiosity as information-theoretic quantities.

**Downstream bias**: Stage 4 paces reveals and manages information density correctly; ending arrives at the right entropy state.

**Sub-criteria**:

- **Information-gap placement**: disclosure edges create readable gaps — the reader knows something is hidden without knowing what it is.
- **Suspense curve**: tension builds across the opening-to-climax span; the climax is the peak entropy moment; resolution reduces entropy in a specific way.
- **Curiosity management**: the DAG opens information gaps and closes them at rates appropriate to the concept's target effect.
- **Resolvable vs. permanent gaps**: the DAG distinguishes gaps that will be resolved from gaps that will remain mysterious. Both can be valid; the choice is intentional.

**Score 5**:
Disclosure edges establish a clear information-gap structure. The reader's curiosity is managed across the DAG — new gaps open as others close. The climax's peak entropy is earned by setup. No information gap feels arbitrary; each serves the concept's specific suspense goals. Example: Jackson's "The Lottery" — the cheerful opening opens a latent gap ("what IS this lottery?") that isn't even recognized as a gap until late, and the climax retroactively makes the entire opening a delayed disclosure.

**Score 3 (exemplar)**:
The DAG has a clear tension arc. Some information gaps are managed well; others are resolved too quickly or never fully opened. The climax is a peak but feels slightly unearned or slightly anticlimactic.

**Score 1**:
Information is delivered linearly, in the order it would be known. No disclosure edges or they serve no purpose. The DAG has no tension arc — beats happen in sequence without a rising-stakes structure.

---

## 4. Post-dictability

**Definition**: do the DAG's surprises feel inevitable in retrospect? When a disclosure edge reveals information that reframes earlier beats, does the reframe feel earned — like the story was always going there?

**Cog-sci anchor**: Bissell et al. (WNU 2025) 6-criteria surprise framework — low predictability × high post-dictability is the combined ideal. Narrative Information Theory on pivot JSD. Split from Tension Architecture because post-dictability is a distinct structural property worth its own judgment.

**Downstream bias**: Stage 4 surprises land as inevitable, not arbitrary; Stage 5 audit for unearned reveals has a target.

**Sub-criteria**:

- **Retrospective coherence**: after a disclosure edge's reveal, the reader should be able to trace the withheld information backwards through the beats that hinted at it.
- **Earned-not-forced**: surprising turns are grounded in earlier structural material, not imposed by LLM default ("suddenly, everything changed").
- **Initiatoriness**: Bissell's term — how well the ending explains prior events. A strong ending doesn't contradict the setup; it reframes it.
- **Immutability preservation**: the ending doesn't violate established facts of the story world (the DAG doesn't have the climax happen in a world that the opening shows is impossible).

**Score 5**:
Surprising disclosures feel inevitable on re-reading. Each `withheld` field's content is traceable through earlier beats — the setup was always pointing here, even if the reader couldn't see it. Example: the final revelation in Ishiguro's "Never Let Me Go" — the reader re-reads the opening and sees it was always about organs, always about their fate, hidden in every scene.

**Score 3 (exemplar)**:
Most surprising moments are retrospectively coherent. One or two reveals feel slightly imposed — the setup hints are present but thin. The ending is consistent with established facts but doesn't strongly reframe earlier beats.

**Score 1**:
Surprises feel arbitrary — the `withheld` fields reference information the beats never hinted at. The ending contradicts or ignores established setup. Disclosure edges are labeled but don't produce retrospective coherence.

---

## 5. Arc Integrity & Ending Strength

**Definition**: does the tension trace across the DAG form a satisfying arc? Is the ending — specifically the climax beat and the resolution that follows — strong enough to carry the story's weight?

**Cog-sci anchor**: Reagan et al. — 6 fundamental emotional arc shapes in human-authored fiction. **Kahneman peak-end rule** — endings disproportionately determine overall evaluation of an experience. This is the single most load-bearing downstream bias: prose-level polish cannot rescue a structurally weak ending.

**Downstream bias**: Stage 4's resolution is the peak of the story; Stage 5 peak-end audit has a target to preserve.

**Sub-criteria**:

- **Climax definition**: the `role: climax` node is a clear peak in the DAG's tension/emotional trajectory. It is not merely "the last event"; it is the emotional center.
- **Setup earning the climax**: the beats leading to the climax establish the stakes, characters, and tensions that make the climax feel earned.
- **Resolution coherence**: the post-climax beats resolve the arc without dissipating the emotional effect. They leave the reader with a clear emotional impression (whatever the target effect is — dread, insight, ache, understanding).
- **Peak-end discipline**: the ending carries emotional or thematic weight that the reader will remember; it is not a shrug.

**Score 5**:
The tension trace rises, peaks at the climax, and resolves into a definite emotional state. The climax feels inevitable given the setup and transformative given what follows. The ending is the strongest beat of the DAG. Example: O'Connor's "A Good Man Is Hard to Find" — the grandmother's moment of grace with the Misfit is the peak; the final "she would of been a good woman, if it had been somebody there to shoot her every minute of her life" is the ending that makes the story unforgettable.

**Score 3 (exemplar)**:
The DAG has a clear climax and resolution. The climax is earned but slightly conventional. The resolution is adequate but doesn't transform what came before — the ending is fine rather than strong.

**Score 1**:
No clear climax, or a climax that feels arbitrary given the setup. Resolution is absent, rushed, or dissipates the tension rather than resolving it. The ending is weak, generic, or anticlimactic. A reader would not remember this story's end.

---

## 6. Structural Coherence

**Definition**: does every beat earn its place? Is the DAG's scope appropriate for the target prose length? Does the structure remove what doesn't serve and retain what does?

**Cog-sci anchor**: Chekhov's gun. Saunders' "Always Be Escalating" — every beat must change the story's state. Stage 1 inherited this concept from its own rubric. For Stage 2, scope calibration against the `target_prose_length` is also here — the node count and structural density should fit the target.

**Downstream bias**: Stage 4 has no filler scenes; Stage 5 Unity-of-Effect audit has a clean spine.

**Sub-criteria**:

- **Every-beat-earns**: removing any node should visibly degrade the DAG's meaning.
- **Compression fit**: the node count matches the target prose length (3–5 for 1K, 5–8 for 3K, 7–12 for 5K, 10–18 for 10K — from scoping §7).
- **No redundancy**: no two beats accomplish the same structural job.
- **Saunders escalation**: every beat changes some structural state (reader knowledge, character intention, world state) relative to its predecessors.
- **Legitimate sparsity**: low-edge-count DAGs are valid for constraint-heavy / minimal-prose concepts (Hemingway-mode). When the story's force comes from what's not-on-the-page, the `story_constraints` list carries structural weight that would otherwise live on edges. A sparse-edge / rich-constraints Hemingway genome is *not* a failure of structural coherence; it's the correct encoding for that concept. Judges should weigh `story_constraints` + `character_arcs` alongside edge density when scoring this dimension.

**Score 5**:
Every beat is load-bearing. Removing any node would break a causal chain, lose a disclosure setup, or eliminate a state change. The DAG is the exact structural size for its target length. Example: Hemingway's "Hills Like White Elephants" — maybe four structural beats (they arrive; he raises it; she deflects; the train comes). Nothing extraneous; each beat is maximum compression.

**Score 3 (exemplar)**:
Most beats are load-bearing; 1–2 are slightly redundant or could be compressed. Scope is appropriate for target length but with 1–2 beats of slack.

**Score 1**:
Multiple beats are redundant (same structural function as neighbors). Scope exceeds or falls short of the target length. Beats exist that could be removed without visible damage. Filler is present.

---

## 7. Beat Quality

**Definition**: are beat sketches specific and evocative? Do they plant material that Stage 4 can build on? Are candidate indelible moments positioned in the DAG for Stage 4 to deliver?

**Cog-sci anchor**: Mar et al. (2021) story-superiority effect — stories are better recalled than expository text, because of their specificity. Sensorimotor simulation research — specific details activate reader embodied processing. Stage 1's Indelibility dimension returns here at beat granularity.

**Downstream bias**: primary content-quality downstream bias. Specific evocative beats seed specific evocative prose; generic beats seed generic prose. Beats marked as `role: climax` or `role: reveal` force Stage 4 to land them specifically.

**Sub-criteria**:

- **Specificity**: beat sketches contain concrete details, not abstract placeholders. "She asked whether the elephants looked like clouds" > "they made small talk."
- **Evocative language**: sketches activate sensory or emotional imagination in the reader without becoming prose themselves.
- **Indelibility planting**: at least one beat contains material that could become an indelible moment in prose — a specific image, an unusual action, a load-bearing object.
- **Variety**: beats don't read like variations of each other; each has its own character.

**Score 5**:
Every beat is specific and evocative. Multiple beats contain planted candidate indelible moments — specific images, unusual actions, concrete objects that Stage 4 can elevate into memorable prose. Example: the beats of Oates' "Where Are You Going, Where Have You Been?" — Connie washing her hair; Arnold Friend's gold car; the screen door; each beat plants specific images that become indelible in the prose.

**Score 3 (exemplar)**:
Most beats are specific. 1–2 beats are generic or default LLM-style ("she had an important conversation with him"). One clear indelible-moment candidate is planted; others are ordinary.

**Score 1**:
Beats are generic placeholders ("something significant happens"). No concrete imagery, no specific actions, no load-bearing objects. The sketches read like outline slots rather than beat content. Stage 4 would have nothing to build on.

---

## 8. Concept Fidelity & Thematic Resonance

**Definition**: does the structure + beats together deliver the concept's target emotional effect? Do the beats embody the concept's thematic tension rather than merely state it?

**Cog-sci anchor**: Poe's Unity of Effect (1846) — every element serves the predetermined emotional response. Green & Brock transportation theory — thematic coherence aids reader immersion. Stage 1 inherited this concept; Stage 2 tests it at the structural level.

**Downstream bias**: Stage 4 every scene serves the target effect; Stage 5 Unity-of-Effect audit has a specific target to verify.

**Sub-criteria**:

- **Target effect delivery**: the climax beat and the resolution together land the concept's `target_effect` field. If the concept says "the ache of knowing something beautiful is temporary," the climax should be structured to produce that ache.
- **Thematic embodiment**: thematic tensions appear in specific beats, not just as general aboutness. A freedom-vs-security thematic tension should manifest in specific choice points, specific objects, specific character actions.
- **Tonal coherence**: beats don't undercut the concept's intended tone (a concept with a dread target effect shouldn't have whimsical beats unless the whimsy is structurally meaningful).
- **Constraint honoring**: if the concept has constraint fields ("the word 'abortion' never appears"; "single scene, real-time"), the structure honors them.

**Score 5**:
The structure actively serves the target effect at every scale — the climax delivers it; the setup earns it; the resolution sustains it. Thematic tensions are embodied in specific beats, not stated. Constraints from the concept are honored in the structure. Example: Hemingway's "Hills" — the concept's thematic tension (the unnamed thing) is embodied structurally by the constraint edges (the word 'abortion' never appears), not just stated as a theme.

**Score 3 (exemplar)**:
Structure mostly serves the target effect. 1–2 beats feel off-tone or don't contribute. Themes are present but more told than shown. Constraints are honored with minor slippage.

**Score 1**:
Structure doesn't deliver the target effect — climax is flat, setup doesn't earn it, or resolution undermines the effect. Themes are stated rather than embodied. Constraints are violated (forbidden words appear; unity of scene broken without reason).

---

## Judging the Rubric Against Itself

Rubric anchors should be periodically sanity-checked:

- **Inter-judge agreement**: if judges systematically disagree on a dimension despite identical rubric, the rubric is ambiguous. Refine.
- **Low-discrimination dimensions**: if a dimension consistently produces ties in pairwise comparison, it is not discriminating between good and bad structures. Review its sub-criteria.
- **Runaway positive correlation**: if two dimensions always win or lose together, they are effectively one dimension. Consider merging.
- **Literary example re-check**: literary examples should age well. Periodic review to ensure examples still serve as anchors.

These checks are deferred to post-v1 empirical review. The initial 8 dimensions are a considered design; monitoring will tell us if any need to be revised, merged, or split.

---

## Open Questions Surfaced in Rubric Drafting

1. **Distinguishing Beat Quality from Prose Quality**. Beat Quality evaluates the *sketch* — 1–2 sentences describing a beat. Judges may unconsciously evaluate it as if it were prose. The rubric explicitly distinguishes these (sketches are specificity-at-plan-level, not craft-level prose), but pilot data may show judges still slipping. If so, add an explicit "do not evaluate this as prose; evaluate it as a plan for prose" instruction.

2. **Post-dictability co-vote with Tension & Information Architecture**. Kept separate pre-implementation on the bet that surprise-that-feels-inevitable is a distinct craft property from gap management. If pilot co-vote rate > 0.85, fold Post-dictability's sub-criteria into Tension & Info Arch as a "Retrospective coherence" sub-criterion.

3. **Structural Coherence overlaps Scope Calibration (which it absorbed)**. Stage 1 had separate Concept Coherence and Scope Calibration dimensions. I merged them here because for structure they felt like one concern. Pilot may show they should be split back out.

4. **Concept Fidelity is the hardest to judge without prose.** Judges must evaluate whether the structure will deliver the target effect without seeing the prose. This requires forward imagination — hard for LLMs. **Pre-scale gate (upgraded from soft touch-back per `2026-04-19-stage-2-third-review-followups.md` Item 7):** after the first pilot run, read every Concept Fidelity judge-reasoning chain. Count the fraction citing prose-forecasting language ("this would read as X in prose," "I imagine the prose would land," "once written this would feel..."). **If that fraction exceeds 30%, apply the structural-focus rephrase *before* scaling to medium/heavy** — this is a hard block, not a revisit. Rephrased framing: "Does this structure *carry the structural weight* of the target effect — is the anchor beat positioned as the structural peak, do setup beats plant the material the effect requires, do any beats dilute or contradict it, does the resolution sustain or dissipate?" Keeps judges in the structural domain. Below 30%, ship as-is and monitor. Rationale for the 30% floor: that's the fraction at which the dimension's signal is meaningfully contaminated by a task (prose forecasting) LLMs are documented to be bad at — below it, judges are mostly reasoning structurally; above it, the rubric itself is the problem.

5. **Sub-criteria count varies per dimension.** Most have 4 sub-criteria; some have 3. Stage 1 had consistent counts. Check with user whether consistency matters.

6. **Beat Quality absolute-band collapse risk.** Stage 2 has only weak levers driving sketch specificity (the system-message paragraph on generic-vs-specific sketches, plus motif threads) compared to Stage 1's seed bank + 11 operators enforcing concept specificity. LLM defaults toward generic placeholder text may compress Beat Quality ratings into a narrow band across most DAGs, reducing the dimension's discriminatory power in pairwise comparison. **Monitor** via the per-dimension absolute vote distribution metric (`implementation.md` §Metrics exported). If Beat Quality ties exceed 60% of pairwise comparisons, the dimension isn't discriminating — strengthen the expansion-prompt language about specificity, or add a pre-expansion "rewrite sketch" step that elevates placeholder beats before they enter the tree.
