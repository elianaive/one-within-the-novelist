# Prompt Engineering Principles

These principles were learned through iterative development with GLM-4. They apply broadly to prompt design.

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
