# Stage 1: MAP-Elites Classification

After a concept passes Gate 3 (judge panel evaluation), it must be classified
before archive insertion. The classifier assigns all 5 behavioral dimensions defined in population.md.
Three are grid axes (determining the MAP-Elites cell): concept_type, arc_shape,
and constraint_density. The other 2 (tonal_register, thematic_domain) are stored
as tracked metadata for analysis and potential future promotion.

Grid: concept_type (6) × arc_shape (6) × constraint_density (3) = 108 cells.
Constraint density is rule-based (no LLM call). See population.md for promotion
criteria.

---

## 1. Constraint Density (Rule-Based)

No LLM call needed. Count items in the genome's `constraints` array.

| Count | Classification |
|-------|---------------|
| 0, missing, or null | `unconstrained` |
| 1–2 | `moderate` |
| 3+ | `heavy` |

**Edge cases:**

- `constraints` field absent or `null` → `unconstrained`
- `constraints` is an empty array `[]` → `unconstrained`
- One constraint that is extremely severe (e.g., "entire story must be a single
  sentence") → still `moderate`. Severity is captured by the evaluation
  dimensions (scope calibration, generative fertility), not by classification.
  Count-based rules are simple, deterministic, and non-gameable.
- Constraint strings that are empty or whitespace-only → skip when counting

```python
def classify_constraint_density(genome: dict) -> str:
    constraints = genome.get("constraints") or []
    count = sum(1 for c in constraints if isinstance(c, str) and c.strip())
    if count == 0:
        return "unconstrained"
    elif count <= 2:
        return "moderate"
    else:
        return "heavy"
```

---

## 2. Subjective Dimensions (Single LLM Call)

The remaining 4 dimensions — concept type, arc shape, tonal register, thematic
domain — require interpretation. They are classified in a single LLM call to
minimize cost and ensure cross-dimension consistency (the same reading of the
concept informs all 4 assignments).

### Input

The classifier receives:
- The full concept genome (JSON)
- The judge panel's per-dimension scores from `public_metrics` (provides signal
  about what the judges saw in the concept)

### 2a. Concept Type

| Value | Detection Criteria |
|-------|-------------------|
| `thought_experiment` | Premise is a speculative or philosophical "what if" pushed to consequences. Thematic tension is prominent. Characters are secondary to the idea. |
| `situation_with_reveal` | Premise contains hidden information. The story's energy comes from what the reader doesn't yet know. A normal surface concealing something underneath. |
| `voice_constraint` | The `constraints` field is the most important element. The story IS the formal restriction. Premise may be simple — the constraint generates complexity. |
| `character_collision` | Two or more characters with incompatible worldviews, desires, or needs forced into proximity. Rich character seeds. The encounter IS the story. |
| `atmospheric_associative` | Meaning emerges from mood, imagery, and juxtaposition rather than argument or plot. Multiple images/fragments rather than a single premise. Style hint is prominent. |
| `constraint_driven` | A formal generative constraint comes first and the content follows from it. Distinct from `voice_constraint`: here the constraint *generates* the narrative (e.g., "every paragraph contains exactly one lie"), not just restricts voice. |

**Ambiguity handling — priority order:**

When a concept fits multiple types, the classifier assigns whichever type best
describes *where the story's primary energy comes from*:

1. If the concept has a dominant formal constraint that generates story content
   → `constraint_driven`
2. If the constraints field is the most important element but restricts rather
   than generates → `voice_constraint`
3. If the concept hinges on hidden information / a reveal → `situation_with_reveal`
4. If two characters' incompatibility is the central engine → `character_collision`
5. If the concept is a speculative premise explored for consequences → `thought_experiment`
6. If none of the above, and the concept is primarily imagistic/atmospheric → `atmospheric_associative`

The classifier may disagree with this priority order for a specific concept —
that's fine. The priority is a tiebreaker for genuinely ambiguous cases, not
a rigid hierarchy. The confidence field captures uncertainty.

### 2b. Emotional Arc Shape

| Value | Detection Criteria |
|-------|-------------------|
| `rise` | Protagonist/situation moves from negative to positive. Rags to riches. The target effect is hopeful, triumphant, or warming. |
| `fall` | Protagonist/situation moves from positive/neutral to negative. Tragedy. The target effect is devastating, cautionary, or bleak. |
| `fall_rise` | Things get worse before they get better. Man in a hole. The target effect involves earned recovery or hard-won understanding. |
| `rise_fall` | Things improve before collapsing. Icarus. The target effect involves loss of something gained, hubris, or irony. |
| `rise_fall_rise` | Complex arc with recovery after a fall. Cinderella. The target effect is bittersweet triumph or transformation through suffering. |
| `fall_rise_fall` | Brief hope crushed. Oedipus. The target effect is tragic irony — the attempted escape makes things worse. |

**Classification basis:** The arc shape is inferred from the concept's *implied
trajectory*, not from explicit plot points (concepts don't have plots yet).
Primary signals:
- The `target_effect` field (a devastating target effect implies a fall somewhere)
- The `premise` (does it set up a positive or negative starting state?)
- The `thematic_tension` (some tensions imply specific arc shapes — "freedom vs.
  security" often produces rise_fall or fall_rise)

**Ambiguity handling:** When the implied trajectory is unclear, prefer the
simpler arc. If `rise` vs `rise_fall_rise` is ambiguous, assign `rise`. The
MAP-Elites grid doesn't need perfect classification — it needs consistent-enough
classification that similar concepts land in similar cells.

### 2c. Tonal Register

| Value | Detection Criteria |
|-------|-------------------|
| `comedic` | The concept invites humor, absurdity, or wit. The target effect includes amusement, even if dark. |
| `tragic` | The concept centers on irreversible loss, suffering, or the impossibility of getting what one needs. Weight and gravity. |
| `ironic` | The concept's energy comes from gap between appearance and reality, expectation and outcome, or stated and actual meaning. Distance and observation. |
| `earnest` | The concept is sincere and direct about its emotional stakes. No protective distance. Vulnerability is the mode. |
| `surreal` | The concept operates by dream logic, impossible juxtaposition, or reality distortion. The strangeness IS the point. |
| `matter_of_fact` | Extraordinary content delivered with flat, uninflected tone. The restraint IS the effect. Hemingway, Carver, early Cormac McCarthy. |

**Classification basis:** Primary signals are the `style_hint`, the
`target_effect`'s emotional texture, and the premise's implied attitude toward
its own content.

**Ambiguity handling:** Tone is the most subjective dimension. When ambiguous:
- If the concept has ironic distance AND another tone, prefer `ironic` (irony is
  a mode, not just a flavor)
- If earnest AND tragic, prefer whichever the `target_effect` more directly
  signals
- If the concept genuinely blends two tones with no clear primary, the
  classifier picks whichever is more *distinctive* for the concept (the one that
  would better differentiate it in the archive)

### 2d. Thematic Domain

| Value | Detection Criteria |
|-------|-------------------|
| `interpersonal` | Central concern is relationships — family, love, betrayal, friendship, the space between people. |
| `societal` | Central concern is institutions, communities, power structures, class, justice at the group level. |
| `philosophical` | Central concern is epistemology, ethics, metaphysics, identity, or the nature of reality/knowledge. |
| `existential` | Central concern is mortality, meaning, isolation, transcendence, the individual confronting the void. |
| `mundane_elevated` | Everyday experience rendered extraordinary. The concept takes something ordinary and reveals the depth in it. |

**Classification basis:** Primary signal is the `thematic_tension` field (if
present). Secondary: what the premise is *about* at its deepest level.

**Ambiguity handling:**
- `interpersonal` vs `existential`: if the relationship is the vehicle for
  exploring existential themes (mortality, meaning), assign `existential`. If
  the existential backdrop serves the relationship story, assign `interpersonal`.
- `philosophical` vs `existential`: philosophical concepts ask questions that
  could be answered; existential concepts sit with questions that can't be.
- `mundane_elevated` is the catch-all for concepts that don't fit the other four
  — but only if the mundane-to-extraordinary transformation is genuinely the
  concept's core move. Not every "doesn't fit" concept is mundane_elevated.

---

## 3. The Classification Prompt

```
You are classifying a story concept into behavioral dimensions for a MAP-Elites quality-diversity archive.

CONCEPT GENOME:
{genome_json}

JUDGE SCORES (0-5 scale):
{scores_json}

Classify this concept on exactly 4 dimensions. For each, pick ONE value from the allowed set and rate your confidence.

CONCEPT_TYPE — where does this story's primary energy come from?
- thought_experiment: speculative/philosophical "what if" explored for consequences
- situation_with_reveal: normal surface hiding something; energy from hidden information
- voice_constraint: a restriction on expression IS the story; constraints field dominates
- character_collision: incompatible people forced together; the encounter is the engine
- atmospheric_associative: meaning from mood, imagery, juxtaposition; not plot or argument
- constraint_driven: a formal rule generates the narrative content (not just restricts voice)

ARC_SHAPE — what emotional trajectory does the concept imply?
- rise: negative → positive
- fall: positive/neutral → negative
- fall_rise: worse before better
- rise_fall: better before collapse
- rise_fall_rise: recovery after fall
- fall_rise_fall: brief hope crushed

TONAL_REGISTER — what is the concept's dominant mode?
- comedic: humor, absurdity, wit (even if dark)
- tragic: irreversible loss, weight, gravity
- ironic: gap between appearance and reality
- earnest: sincere, direct, vulnerable
- surreal: dream logic, impossible juxtaposition
- matter_of_fact: extraordinary content, flat delivery

THEMATIC_DOMAIN — what is the concept fundamentally about?
- interpersonal: relationships, the space between people
- societal: institutions, communities, power structures
- philosophical: epistemology, ethics, identity, nature of reality
- existential: mortality, meaning, isolation, transcendence
- mundane_elevated: everyday experience revealed as extraordinary

When a concept fits multiple values, pick the one that best describes its PRIMARY energy source. When genuinely ambiguous, prefer simpler/more distinctive classifications.

Respond with JSON only:
{
  "concept_type": "<value>",
  "concept_type_confidence": "<low|medium|high>",
  "arc_shape": "<value>",
  "arc_shape_confidence": "<low|medium|high>",
  "tonal_register": "<value>",
  "tonal_register_confidence": "<low|medium|high>",
  "thematic_domain": "<value>",
  "thematic_domain_confidence": "<low|medium|high>"
}
```

---

## 4. Expected Output Format

The classification result is merged into `public_metrics` before archive
insertion. Full schema of the classification fields within `public_metrics`:

```json
{
  "map_elites_cell": {
    "concept_type": "thought_experiment",
    "arc_shape": "fall_rise",
    "tonal_register": "ironic",
    "thematic_domain": "philosophical",
    "constraint_density": "moderate"
  },
  "classification_confidence": {
    "concept_type": "high",
    "arc_shape": "medium",
    "tonal_register": "high",
    "thematic_domain": "medium"
  }
}
```

`map_elites_cell` is the cell key used by ShinkaEvolve's MAP-Elites archive
strategy. `classification_confidence` is metadata for monitoring — low
confidence flags concepts that might be misclassified.

---

## 5. Example Classifications

### Example A: Clear Classification

```json
{
  "premise": "A linguistics professor discovers that a deaf student's sign language poetry operates on spatial dimensions that hearing poetry cannot access — and that translating it necessarily destroys it.",
  "target_effect": "The vertigo of realizing that every translation is a loss, and that entire art forms exist in spaces you can never enter.",
  "thematic_tension": "access vs. authenticity",
  "constraints": ["No character is a villain or hero."],
  "style_hint": "Academic precision dissolving into wonder."
}
```

**Classification:**
- `concept_type`: `thought_experiment` (high) — speculative premise explored for philosophical consequences
- `arc_shape`: `fall` (medium) — the realization is a loss; the target effect is vertigo, not resolution
- `tonal_register`: `earnest` (high) — sincere wonder, no ironic distance
- `thematic_domain`: `philosophical` (high) — epistemology of translation, limits of knowledge
- `constraint_density`: `moderate` — 1 constraint

### Example B: Ambiguous, Tiebreaker Needed

```json
{
  "premise": "Two retired Cold War spies — one American, one Soviet — meet weekly at a Denny's in Tucson. Each believes the other doesn't know. Both are right and wrong.",
  "target_effect": "The comedy of mutually maintained fictions collapsing into something unexpectedly tender.",
  "character_seeds": [
    {"label": "Frank", "sketch": "Ex-CIA, now widowed. Comes for the coffee and the routine. Spots the Russian on day one but says nothing.", "want": "To not be alone.", "need": "To stop performing."},
    {"label": "Volodya", "sketch": "Ex-KGB, defected decades ago. Lonely. Recognized Frank immediately. Maintains cover out of habit.", "want": "To preserve the game.", "need": "To let someone see him."}
  ],
  "thematic_tension": "performance vs. authenticity",
  "constraints": [],
  "style_hint": "Deadpan. Let the absurdity speak for itself."
}
```

**Classification:**
- `concept_type`: `character_collision` (medium) — this is close to
  `situation_with_reveal` (both maintain hidden knowledge) but the story's
  energy comes from the two characters' incompatible performances grinding
  against each other, not from a single reveal. The reveal is mutual and
  ongoing, which makes it a collision. Tiebreaker: the rich character seeds
  with want/need confirm character collision as primary.
- `arc_shape`: `fall_rise` (medium) — the maintained fictions are a "fall"
  (isolation disguised as connection), collapsing into tenderness is the "rise."
  Could be `rise` if you read the whole situation as positive from the start.
  Tiebreaker: prefer simpler arc when ambiguous → but `fall_rise` better
  matches the target effect's "collapsing into something unexpectedly tender"
  (the unexpected part implies a turn).
- `tonal_register`: `comedic` (medium) — the target effect says "comedy" and
  the style hint says "deadpan absurdity." But the emotional landing is tender,
  not funny. This is a case where `comedic` wins because it's the dominant mode
  even though the destination is earnest. If the target effect said "tenderness"
  without "comedy," this would be `earnest`.
- `thematic_domain`: `interpersonal` (high) — despite the Cold War backdrop,
  this is about two lonely people. The societal/political context is setting,
  not subject.
- `constraint_density`: `unconstrained` — empty array

### Example C: Constraint-Driven Edge Case

```json
{
  "premise": "A story told entirely in the language of a home inspection report.",
  "target_effect": "The horror of realizing what happened in this house, revealed only through what the inspector notes and — more importantly — what they professionally decline to note.",
  "constraints": [
    "Every sentence must be plausible as a line in a home inspection report.",
    "No dialogue.",
    "No named characters.",
    "The inspector's emotional state can only be inferred from increasingly terse or detailed observations."
  ],
  "style_hint": "Clinical. Bureaucratic. The form is the content."
}
```

**Classification:**
- `concept_type`: `constraint_driven` (high) — the formal constraint (home
  inspection report format) generates the narrative. Without the constraint,
  there's no story. This is NOT `voice_constraint` — the constraint doesn't
  just restrict how things are said, it determines what can be said at all.
- `arc_shape`: `rise` (low) — this is the hardest dimension here. The concept
  implies escalating revelation, which reads as a "rise" in tension/horror. But
  it's not a positive rise. Arc shape is about emotional trajectory, not
  tension trajectory. Could be `fall` (things are revealed to be worse than
  they seemed). Low confidence is appropriate — the arc will become clear in
  Stage 2, not here.
- `tonal_register`: `matter_of_fact` (high) — clinical delivery of horrifying
  content is textbook matter-of-fact register
- `thematic_domain`: `interpersonal` (medium) — whatever happened in the house
  happened between people. Could be `existential` if the horror is more
  abstract. The premise suggests domestic violence or crime, which is
  interpersonal.
- `constraint_density`: `heavy` — 4 constraints

---

## 6. Cost and Integration

### Token Cost

- **Input:** ~200 tokens (genome) + ~100 tokens (scores) + ~350 tokens (prompt template) = ~650 tokens
- **Output:** ~80 tokens (JSON response)
- **Total:** ~730 tokens per classification call
- At Haiku-class pricing, this is negligible relative to the judge panel evaluation that precedes it

### Model Selection

Use a Haiku-class model (cheapest available). This is a structured
classification task with clear definitions — it doesn't require the reasoning
depth of a Sonnet/Opus-class model. The confidence field lets us monitor
whether the cheap model is adequate; if low-confidence classifications are
common, upgrade.

The classifier model CAN be the same family as the generator or evaluator.
The self-preference bias concern from judging doesn't apply here —
classification has objectively better/worse answers, unlike quality evaluation.

### Pipeline Position

```
Gate 1 (validation) → Gate 2 (anti-cliche) → Gate 3 (judge evaluation)
    → CLASSIFICATION → archive insertion
```

Classification runs after Gate 3 because it uses judge scores as input signal.
It runs before archive insertion because the MAP-Elites cell key determines
where the concept is placed and whether it replaces an existing occupant.

### Integration with ShinkaEvolve

The classification output is written to `public_metrics.map_elites_cell` and
`public_metrics.classification_confidence` by `evaluate.py`. ShinkaEvolve's
MAP-Elites archive strategy (the targeted edit in population.md) reads
`public_metrics.map_elites_cell.concept_type`,
`public_metrics.map_elites_cell.arc_shape`, and
`public_metrics.map_elites_cell.constraint_density` to determine cell placement.
The remaining dimensions (tonal_register, thematic_domain) are stored for
analysis but do not affect placement.

The `constraint_density` field is computed in Python before the LLM call. The
4 subjective dimensions come from the LLM response. Both are merged into the
same `map_elites_cell` object.

---

## 7. Classifier Validation Plan

Before the first evolutionary run, validate that the classifier produces stable
assignments on the two grid dimensions. Instability in cell assignment undermines
MAP-Elites' core mechanism — if the same concept gets different cells on
different runs, the archive becomes noisy and within-cell competition is
meaningless.

### Protocol

1. Generate 50 concept genomes using a mix of operators (cold-start allocation)
2. Run judge evaluation on each to produce scores (needed as classifier input)
3. Classify each concept 3 times using the production model (Haiku-class)
4. For each concept, check whether all 3 classifications agree on: (a) concept_type, (b) arc_shape. Constraint density is rule-based and deterministic — no validation needed for that axis.
5. Check the combined cell key (concept_type, arc_shape, constraint_density) stability

### Acceptance Criteria

- **Cell-key stability ≥ 80%:** At least 40 of 50 concepts land in the same (concept_type, arc_shape, constraint_density) cell across all 3 classifications. Since constraint_density is deterministic, instability comes only from the 2 LLM-classified axes.
- **Per-dimension stability:** concept_type ≥ 85%, arc_shape ≥ 80% (arc shape is inherently more ambiguous for concepts that don't yet have plots)
- **Confidence correlation:** low-confidence classifications should account for the majority of instability (validates the confidence signal)

### Fallbacks

If stability is below threshold:
- **Upgrade model:** Try Sonnet-class for classification. Cost increase is ~5x per classification call but classification is cheap (~730 tokens) relative to judge evaluation
- **Reduce grid:** Fall back to concept_type × constraint_density only (18 cells, 2D grid). Arc shape becomes metadata. Re-attempt 3D after Stage 2 produces structure graphs where arc shape is manifest rather than inferred
- **Add classification consensus:** Classify twice, use the cell both agree on. If they disagree, use the higher-confidence classification. Cost doubles but may be worthwhile for grid integrity
