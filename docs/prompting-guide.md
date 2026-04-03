# Prompt Engineering Principles

These principles were learned through iterative development — first with GLM-4 for evaluation tasks, then expanded through research and experimentation on creative generation. The two domains (analytical and creative) require different — sometimes opposite — approaches. Know which mode you're in.

---

## Tail-Wagging-The-Dog

Prompt structure determines reasoning flow. If you ask for X before Y, the model will derive Y from X—even when causality should run the other way.

**Example (wrong):**
1. Define nations
2. Assign regions to nations
3. Name regions

Result: Nations come first, then regions are shoehorned to fit. Geography serves politics.

**Example (right):**
1. Study the geography (from region analyses)
2. Name regions based on terrain
3. Form nations from natural boundaries

Result: Geography determines politics. Nations emerge from the land.

**The principle:** The order you ask for things IS the causal order the model assumes. If you want X to determine Y, ask for X first.

---

## Decision Chains > Pattern + Exemption

Rules like "X is a violation UNLESS Y" fail because the model fires on X before checking Y.

**Wrong:**
Flag APPOSITIVE: ", her [noun]" — UNLESS the possessive refers to an object.

Model sees ", her silver hair" → fires APPOSITIVE → never checks exemption.

**Right:**
1. Does it start with possessive (her/his/its)?
   → NO: Not this pattern
   → YES: Continue

2. What does possessive refer to?
   → INANIMATE OBJECT: Fine, not a violation
   → CONSCIOUS CHARACTER: APPOSITIVE, flag it

Decision chains force condition checking in order before reaching a verdict.

---

## Irrelevant Context = Noise, Prior Reasoning = Signal

The model doesn't need the premise to spot grammatical patterns. But it absolutely needs:
- Its own prior analysis when evaluating
- Its own fix plan when rewriting
- The accumulated conversation, not a fresh start

**Wrong:** Fresh context each step
```
Skeleton: System + User → Assistant
Evaluation: System + User (pretend skeleton happened) → Assistant
```

**Right:** Additive context
```
Skeleton: System + User → skeletonMessages
Evaluation: skeletonMessages + User → evalMessages
Rewrite: evalMessages + User → final
```

Each step continues the conversation. The model sees what it already reasoned about.

---

## Understanding > Reinforcement

More examples won't help if the model doesn't understand WHY something is wrong.

Example: Model claimed `${2Gender}[male]` was "fixed as male" when it's actually a placeholder with a default value.

**Wrong approach:** Add more examples of gender violations.

**Right approach:** Explain the semantics:
```
PARAMETER BLOCK SEMANTICS:
Format: Number + Label[default]:description

CRITICAL: The [default] is a STARTING VALUE the user can change. It is NOT fixed.
- 2Gender[male] → User might change to "female" → prose cannot assume male
```

Once the model understood what the format MEANS, discrimination followed.

---

## FIX PLAN Forces Commitment

Initial rewrite attempts achieved ~40% fix rate. Model was swapping synonyms without fixing underlying patterns.

Solution: Force explicit transformation plan before prose:

```
FIX PLAN:
1. "her arms wrapped" - APPOSITIVE → Split: "Liora curled inward. Her arms wrapped..."
2. "not with judgment, but with" - NOTXBUTY → Direct: "with the focused curiosity of..."

---

[rewritten prose]
```

The `---` separator is a commitment boundary. Each plan entry shows exact new text, not vague intentions.

---

## Explicit > Implicit

Rules that "should be obvious" will fail. Make everything algorithmic.

If you find yourself thinking "the model should know that..." — you're about to have a bad time. Write it out explicitly. Decision chains. Worked examples. No assumptions.

**Creative generation caveat:** This principle applies to structural requirements (JSON schema, output format) but is counterproductive for creative content. Over-specification reduces diversity — the model produces exactly what you asked for, and it's boring. Leave interpretive space for the creative work; be explicit only about the container it needs to fit into.

---

# Creative Generation Principles

The principles above were developed for evaluation and coding tasks. Creative generation is a different discipline with different — sometimes opposite — rules. RLHF alignment training makes LLMs default toward safe, typical, convergent outputs. The principles below are designed to counteract that.

These are grounded in research: a cognitive science literature review (Amabile, Higgins, Martindale, Haught-Tromp, Damadzic), empirical LLM studies (CreativeDC, Verbalized Sampling, Price of Format, Artificial Hivemind), and practitioner knowledge (Gwern, Kaj Sotala, SillyTavern community, NousResearch). Full citations in `lab/deep-research/runs/`.

---

## The Medium Is the Message

The form of a prompt shapes the output as much as its content. A prompt that reads like a technical specification produces outputs that read like technical completions — regardless of whether it says "be creative." A prompt that reads like exploratory prose produces outputs in that register.

This operates at every level:
- **Template structure.** Chat templates with role markers and system headers act as behavioral anchors that collapse output diversity, independent of content. "Simple steer" (bare instruction) outperforms "full template" for diversity. (Price of Format, EMNLP 2025)
- **Instruction register.** "Step 1: Identify the core energy. Step 2: Find the connection point." activates compliance mode. "Start with what snags you. Follow where it leads." activates exploration mode. Same creative logic, different outputs.
- **Schema proximity.** The moment the model reads `"premise": "A concrete story premise (string, min 20 chars)"`, it is reading a spec. It will produce spec-shaped output — syntactically correct, semantically generic. Keep the spec as far from the creative work as possible.
- **First message tone.** LLMs pick up style and register from the opening of the prompt more reliably than from instructions buried later. The first thing the model reads sets the distributional neighborhood it generates from.

The implication: write creative prompts *in the register you want the output to have*. If you want exploratory, surprising, specific output, the prompt itself must be exploratory, surprising, and specific — not a bulleted list of requirements for those qualities.

---

## Describe What IS, Not What ISN'T

Telling a model about failure modes anchors toward them. LLMs have fundamental representational difficulty with negation — negation tokens have limited effect on learned representations. Describing what bad output looks like activates those patterns rather than suppressing them.

**Wrong:** "LLMs fail at fiction because RLHF training rewards prosocial, resolved, comfortable stories. This produces reconciliation arcs, grief meditations, sanitized conflicts..."

The model has just read a detailed description of exactly the patterns you don't want. It is now primed toward them.

**Right:** "The concepts that survive are the ones rooted in concrete, particular human experience — specific where they could be general, committed where they could hedge, alive with tension that doesn't resolve cleanly."

Same intent, opposite mechanism. Describe the positive target. Let the model move toward something rather than trying to run away from something.

**Evidence:** Anchoring bias is strong in LLMs and CoT doesn't reduce it (arXiv 2412.06593). In diffusion models, negative prompts in early generation stages literally produce the specified content via reverse activation (arXiv 2406.02965). The negation finding: "negation tokens have limited effect on learned representations" (arXiv 2503.22395).

**The one exception:** Syntactic bans on specific constructions work empirically when restated as what to do instead. "Describe what is present, not what is absent — no 'It was not X, but Y' constructions" is effective because the ban is paired with a positive directive. This was developed independently by Gwern (gwern.net/system-prompts-2025) and NousResearch (autonovel ANTI-SLOP.md).

---

## Focusing Constraints > Exclusionary Constraints

A constraint that specifies a generative starting point ("your concept must involve X") outperforms a constraint that specifies a prohibition ("your concept must not use Y"). Focusing constraints direct attention toward a productive region of the creative space. Exclusionary constraints narrow the space but give no direction.

**Exclusionary:** "Write a story that does NOT use dialogue."
**Focusing:** "Write a story told entirely through action and internal thought."

Same constraint, different framing. The focusing version gives the model somewhere to go. The exclusionary version only takes something away.

**Evidence:** Haught-Tromp 2017 ("Green Eggs and Ham Hypothesis") — focusing constraints increased judged creativity, and the effect persisted even after constraints were removed. Damadzic 2022 meta-analysis (111 studies) — the type and timing of constraints determine whether they help or hurt.

---

## Early Constraints Help, Late Constraints Hurt

The same constraint has opposite effects depending on when it's introduced. Constraints presented at the start of the creative process shape the generative space productively. Constraints presented after the model has already committed to a direction force retroactive satisfaction — narrowing rather than guiding.

This means: put creative constraints (genre, tonal register, structural requirements) early in the prompt, before the creative work begins. Put formatting constraints (JSON schema, output delimiters) late — after the creative work is done.

**Wrong ordering:** Creative instructions → convergence pattern checklist → JSON schema → generate.
The model's last input before generating is format compliance. It generates from a compliance frame.

**Right ordering:** Creative instructions → generate creative content → then structure into required format.
The model's creative work happens before the spec kicks in.

**Evidence:** Damadzic et al. 2022 meta-analysis of 111 studies: early-stage constraints significantly and positively affected creativity; the same pattern reversed for late-stage constraints. This is one of the most robust findings in the creativity constraints literature.

---

## Brainstorm Before Commit

Asking the model to explore multiple directions before committing to one increases diversity 1.6-2.1x with no quality loss. A single-pass prompt collapses to the model's modal (most typical) output. Requesting a distribution over options forces the model to approximate a broader range of its generative space.

**Single-pass:** "Generate a story concept about..."
**Brainstorm-first:** "Explore 3 different directions this concept could go — note the core surprise and feeling of each. Then develop the most unexpected one."

The brainstorming IS the creative work. The final output is selection, not generation.

**Evidence:** Verbalized Sampling (arXiv 2510.01171) — 1.6-2.1x diversity improvement in creative writing tasks, training-free, model-agnostic. Gwern's "Ming" test prompt: "brainstorm different ideas, genres, time periods... rate each one, select the best one." Luminate (CHI 2024): three-phase decomposition (dimensions → values → content) prevents premature convergence.

---

## Prompt Diversity > Sampling Diversity

Issuing genuinely different prompts produces far more collective diversity than running the same prompt at different temperatures. Temperature adds noise within a semantic region; different prompts access different regions entirely.

For the evolutionary pipeline, this means operator diversity (different creative methods per generation) matters more than within-operator stochasticity. The 11 operators (collision, discovery, thought experiment, etc.) are the primary diversity mechanism. Temperature and sampling are secondary.

**Evidence:** Doshi & Hauser 2024 (Science Advances): "Issuing the same prompt hundreds of times results in many different ideas, but not nearly as diverse ideas as would result from hundreds of different prompts." Temperature research (ICCC 2024): high temperature does not access broader semantic regions — it increases noise within the same region. The "Price of Format" paper confirms: prompt structure dominates temperature for diversity.

---

## Register as Generative Cue

The register (tone, formality, style) of a prompt primes the distributional neighborhood the model generates from. An instructional register activates the vast corpus of technical documentation, procedure manuals, and specification-following text. A literary register activates fiction, essays, and creative prose.

A single-sentence change in framing measurably changes creative output in humans (Zabelina & Robinson 2010 — "you are 7 years old" increased creative scores). For LLMs, the mechanism is associative rather than psychological, but the behavioral effect is the same: the opening register selects which "mode" the model operates in.

Regulatory Focus Theory (Higgins 1998) provides the framework: promotion focus ("explore these possibilities," "what is the most surprising version?") activates eager, exploratory processing. Prevention focus ("make sure you don't...," "avoid these patterns...") activates vigilant, convergent processing. The register of the prompt determines which focus is active.

---

## Analytical vs. Creative — When Principles Conflict

| Principle | Analytical (eval/coding) | Creative (generation) |
|-----------|-------------------------|----------------------|
| Explicit > Implicit | Yes — make everything algorithmic | Only for structure, not for content |
| Decision Chains | Yes — force condition checking | No — compliance mode kills creativity |
| Fix Plans | Yes — commit before executing | Yes, as brainstorm-before-commit |
| Order = Causality | Yes | Yes — put creative seeds before constraints |
| Prior Reasoning = Signal | Yes | Yes — the model's own exploration is signal |
| Understanding > Reinforcement | Yes | Yes |

The general rule: be explicit and algorithmic about the *container* (output format, schema, structural requirements). Be invitational and spacious about the *content* (what the model actually generates creatively).
