# Stage 1: Implementation Guide

This document bridges the conceptual spec (overview, operators, evaluation,
population) to the actual ShinkaEvolve codebase. It covers genome format,
required code edits, prompt templates, evaluation architecture, and worked
examples.

---

## 1. Genome Format & Storage

### JSON Schema

A concept genome is a JSON document stored in ShinkaEvolve's `Program.code`
field. The file extension is `.json`.

```json
{
  "premise": "Two people at a train station discuss something they never name.",
  "target_effect": "The weight of what remains unsaid — dread, helplessness, the slow realization that silence is a form of violence.",
  "character_seeds": [
    {
      "label": "the man",
      "sketch": "Confident on the surface, steering the conversation with practiced ease.",
      "wound": null,
      "fear": null,
      "lie": null,
      "want": "For her to agree without him having to say what he wants.",
      "need": null
    },
    {
      "label": "the woman",
      "sketch": "Deflecting with imagery — the hills, the drinks, the beaded curtain.",
      "wound": null,
      "fear": null,
      "lie": null,
      "want": "To not have this conversation.",
      "need": null
    }
  ],
  "setting_seeds": "A train station in Spain. Hot. A bar with a beaded curtain. Two lines of rails in the sun.",
  "thematic_tension": "autonomy vs. obligation",
  "constraints": [
    "The word 'abortion' never appears.",
    "No interiority — only dialogue and physical action.",
    "Single scene, near-real-time."
  ],
  "style_hint": "Spare, concrete, almost journalistic. The horror is in the contrast between flat tone and devastating content."
}
```

**Validation rules:**
- `premise` — required, non-empty string, minimum 20 characters
- `target_effect` — required, non-empty string, minimum 15 characters
- `character_seeds` — optional, array of objects. If present, each must have at
  least `label` and `sketch`. Wound/fear/lie/want/need fields are all optional.
- `setting_seeds` — optional, string
- `thematic_tension` — optional, string (ideally "X vs. Y" format)
- `constraints` — optional, array of strings
- `style_hint` — optional, string

Null/absent optional fields are valid. A minimal genome is just `premise` +
`target_effect`.

### Storage in ShinkaEvolve

The JSON is stored as a string in `Program.code`. ShinkaEvolve treats it as
opaque text — it doesn't parse it. Parsing happens in:
- **Mutation prompts:** The JSON is injected into prompt templates as
  `{code_content}`. The LLM reads it as structured text.
- **evaluate.py:** Reads and parses the JSON to extract fields for validation
  and evaluation.
- **Embeddings:** The full JSON string is embedded for novelty rejection.

---

## 2. ShinkaEvolve Edits

### Edit 1: JSON Support in `wrap_eval.py`

**File:** `shinka/core/wrap_eval.py`, function `load_program()` (lines 27-37)

**Current behavior:** Uses `importlib.util.spec_from_file_location` to import a
Python module. Fails on `.json` files.

**Edit:** Add JSON detection before the import:

```python
def load_program(program_path: str) -> Any:
    """Loads a program from a given file path. Supports Python modules and JSON."""
    if program_path.endswith('.json'):
        import json
        with open(program_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    spec = importlib.util.spec_from_file_location("program", program_path)
    # ... existing code unchanged ...
```

When `language` is `"json"`, the evaluate.py receives a dict (the parsed JSON)
instead of a module. The evaluate.py is written to expect this.

### Edit 2: JSON Validation in `async_apply.py`

**File:** `shinka/edit/async_apply.py`, function `validate_code_async()`
(lines 105-134)

**Current behavior:** Runs `py_compile` for Python, `rustc` for Rust, etc.

**Edit:** Add JSON validation branch:

```python
if language == "json":
    # Validate JSON syntax
    try:
        import json
        with open(code_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True, None
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
```

### Edit 3: Extend Patch Types in `defaults.py`

**File:** `shinka/defaults.py` (lines 13-18)

**Edit:** Replace default patch types with concept operators:

```python
def default_patch_types() -> list[str]:
    return [
        "collision", "noun_list", "thought_experiment", "compost",
        "crossover", "inversion", "discovery", "compression",
        "constraint_first", "anti_premise", "real_world_seed",
    ]

def default_patch_type_probs() -> list[float]:
    return [
        0.12, 0.08, 0.10, 0.08,   # collision, noun_list, thought_exp, compost
        0.10, 0.08, 0.10, 0.06,   # crossover, inversion, discovery, compression
        0.10, 0.10, 0.08,         # constraint_first, anti_premise, real_world
    ]
```

Note: These defaults are overridden by `EvolutionConfig.patch_types` and
`EvolutionConfig.patch_type_probs` at runtime. The defaults serve as a
reasonable starting point.

### Edit 4: Extend Sampler Dispatch in `sampler.py`

**File:** `shinka/core/sampler.py`, method `sample()` (lines 137-203)

**Current behavior:** Hardcoded if/elif for diff/full/cross.

**Edit:** Refactor to dispatch dict. Each operator maps to:
- A system format string (appended to sys_msg)
- An iter message template (formatted with parent genome + metrics)
- A patch routing ("full" or "diff" — determines how LLM output is applied)

```python
# At module level, import all operator prompts
from shinka.prompts.prompts_concept import OPERATOR_REGISTRY

# In sample() method, replace the if/elif chain:
operator = OPERATOR_REGISTRY.get(patch_type)
if operator is None:
    raise ValueError(f"Unknown operator: {patch_type}")

sys_msg += operator["sys_format"]
iter_msg = operator["iter_msg"].format(
    code_content=parent.code,
    performance_metrics=perf_str(parent.combined_score, parent.public_metrics),
    text_feedback_section=text_feedback_section,
    episodic_context=get_episodic_context(parent, compost_db),
)

# For cross-type operators, append inspiration
if operator.get("needs_inspiration"):
    iter_msg += "\n\n" + get_cross_component(
        archive_inspirations, top_k_inspirations, language=self.language
    )
```

### Edit 5: Patch Routing in `async_apply.py`

**File:** `shinka/edit/async_apply.py`, function `apply_patch_async()`
(lines 78-84)

**Edit:** Route new operator types to the appropriate patch function:

```python
# Operators that output complete genomes (full replacement)
FULL_PATCH_OPERATORS = {
    "collision", "noun_list", "thought_experiment", "compost",
    "crossover", "discovery", "compression", "constraint_first",
    "anti_premise", "real_world_seed",
}
# Operators that modify part of an existing genome
DIFF_PATCH_OPERATORS = {"inversion"}

if patch_type in FULL_PATCH_OPERATORS or patch_type in ["full", "cross"]:
    patch_func = apply_full_patch
elif patch_type in DIFF_PATCH_OPERATORS or patch_type == "diff":
    patch_func = apply_diff_patch
else:
    raise ValueError(f"Unknown patch type: {patch_type}")
```

### Edit 6: MAP-Elites Archive Strategy in `dbase.py`

**File:** `shinka/database/dbase.py`

**Edit:** Add `_update_archive_map_elites()` method and wire it into
`_update_archive()`:

```python
def _update_archive_map_elites(self, program: Program) -> None:
    """MAP-Elites archive: maintain best program per behavioral cell.

    Uses holder_score (raw Hölder mean) for cell replacement, NOT
    selection_score (which includes diversity bonus). Within-cell competition
    should be pure quality — "is this the best concept of its type?" — not
    influenced by cross-population signals like judge disagreement.
    """
    cell_key = self._get_cell_key(program)
    if cell_key is None:
        return  # Can't classify — skip

    holder_score = program.public_metrics.get("holder_score")
    if holder_score is None:
        return

    current_occupant = self._get_cell_occupant(cell_key)
    if current_occupant is None:
        # Empty cell — add unconditionally
        self._set_cell_occupant(cell_key, program)
    else:
        current_holder = current_occupant.public_metrics.get("holder_score", 0)
        if holder_score > current_holder:
            self._remove_from_archive(current_occupant)
            self._set_cell_occupant(cell_key, program)

def _get_cell_key(self, program: Program) -> Optional[tuple]:
    """Extract MAP-Elites cell coordinates from program metrics.

    Grid axes: concept_type × arc_shape (2D, 36 cells).
    Other classified dimensions (constraint_density, tonal_register,
    thematic_domain) are stored as metadata but don't affect cell placement.
    """
    cell = (program.public_metrics or {}).get("map_elites_cell", {})
    concept_type = cell.get("concept_type")
    arc_shape = cell.get("arc_shape")
    if any(d is None for d in (concept_type, arc_shape)):
        return None
    return (concept_type, arc_shape)
```

The cell coordinates come from `public_metrics.map_elites_cell`, which is set by
the MAP-Elites classifier (see section 5). The `archive_size` config parameter is
ignored under MAP-Elites — all occupied cells retain their champion.

---

## 3. Initial Population Override

ShinkaEvolve's stock `_generate_initial_program()` uses a single generic prompt
for all initial concepts — no operator selection. We override this to implement
the cold-start allocation from `population.md`.

**Modification:** Override `_generate_initial_program()` in our runner subclass
to:

1. For each initial concept slot (10-15 per island × 10 islands = 100-150):
   a. Select an operator from the cold-start allocation table (20% Collision,
      20% Thought Experiment, 15% Noun-List, etc.)
   b. Load the operator's prompt from `owtn/prompts/stage_1/operators/<operator>.txt`
   c. If a matching seed exists in the seed bank, inject it (see below)
   d. Build sys_msg = `base_system.txt` + operator SYS_FORMAT
   e. Build user_msg = `stage_1/initial.txt` with operator instructions
   f. Call LLM, apply patch, evaluate, store

This means generation 0 already has operator diversity and seed bank material,
rather than relying on a generic "write a concept" prompt.

**Operators unavailable at cold start:** Crossover and Compost Recombination
(need existing population/archive). Their allocation is redistributed
proportionally among available operators.

---

## 4. Seed Bank Injection

Operators can draw starting material from the seed bank (`data/seed-bank.yaml`).
Each seed has a `type` that maps to an operator. When assembling an operator's
prompt, the code:

1. Looks up the operator's matching seed type (e.g., `constraint_first` → type
   `constraint`, `thought_experiment` → types `thought_experiment` + `axiom`)
2. Queries the seed bank for matching seeds
3. If seeds exist, selects one (diversity-weighted — prefer types
   underrepresented in recent concepts)
4. Injects into the operator instructions via `{seed_content}` placeholder:
   ```
   Use this as your starting point:

   {seed.content}
   ```
5. If no matching seeds exist, the placeholder is replaced with an empty string
   and the operator generates from scratch

**Seed type → operator mapping:**

| Seed type | Operator |
|-----------|----------|
| `real_world` | Real-World Seed Injection |
| `thought_experiment` | Thought Experiment (Le Guin) |
| `axiom` | Thought Experiment (Le Guin) |
| `dilemma` | Thought Experiment, Collision, Compression |
| `constraint` | Constraint-First (Oulipo) |
| `noun_cluster` | Noun-List (Bradbury) |
| `image` | Discovery Mode (Murakami) |
| `compression` | Compression (Borges) |
| `collision_pair` | Collision (King) |
| `anti_target` | Anti-Premise |

Seed injection happens in the prompt assembly layer, before the prompt reaches
ShinkaEvolve's sampler. The sampler sees a fully-formed operator prompt.

---

## 5. Prompt Templates

**Canonical prompt files live in `owtn/prompts/`.** The templates below are the
design reference; the actual files loaded at runtime are:

- `owtn/prompts/stage_1/base_system.txt` — shared `task_sys_msg` for all operators
- `owtn/prompts/stage_1/judge_system.txt` — judge evaluation system message
- `owtn/prompts/stage_1/judge_user.txt` — judge user message (concept presentation)
- `owtn/prompts/stage_1/classification.txt` — MAP-Elites classifier
- `owtn/prompts/stage_1/operators/*.txt` — 11 operator prompts
- `owtn/prompts/stage_1/*.txt` — shared scaffolding (initial, iteration, output format)

All operator prompts follow the same structure as existing ShinkaEvolve prompts:
a system format string (appended to `task_sys_msg`) and an iteration message
(formatted with parent genome, metrics, and feedback).

### Output Format (All Operators)

All operators produce the same output structure — the LLM responds with:

```
<NAME>
short_name_for_concept
</NAME>

<DESCRIPTION>
Explanation of the creative reasoning behind this concept.
</DESCRIPTION>

<CODE>
```json
{
  "premise": "...",
  "target_effect": "...",
  ...
}
```
</CODE>
```

The `<CODE>` block contains the complete JSON genome. ShinkaEvolve's
`apply_full_patch` extracts the content between the code fences.

### Shared Iteration Message

Most operators share the same iteration message body (some add extra sections):

```
CONCEPT_ITER_MSG = """# Current concept

Here is the current story concept we are evolving:

{code_content}

Here are the evaluation scores:

{performance_metrics}{text_feedback_section}{episodic_context}

# Task

{operator_instructions}
"""
```

The `{operator_instructions}` section differs per operator. Below are the
system format and operator instructions for each.

### Operator: Collision (King)

**Needs inspiration:** Yes (cross-type — receives a second concept)

```
COLLISION_SYS_FORMAT = """
You are a creative writer generating story concepts through collision — forcing
two unrelated premises together so the story lives in the interference pattern.

Rules:
- Both parent elements MUST survive in the new concept. Neither is subordinated.
- The premise should live in the TENSION between the incompatible elements, not
  blend them into something safe.
- Name a specific target emotional effect.
- Output a complete concept genome as JSON.

{output_format}
"""

COLLISION_INSTRUCTIONS = """
You have two story concepts. Collide them — find the most unlikely connection
point between these two premises and generate a new concept that requires BOTH
elements to survive. The story lives in the interference pattern, not the average.

Do not blend. Maintain tension between the incompatible elements.
"""
```

### Operator: Noun-List (Bradbury)

**Needs inspiration:** No (full generation)

```
NOUN_LIST_SYS_FORMAT = """
You are a creative writer generating story concepts through Bradbury's noun-list
method — finding stories hiding in the connections between emotionally resonant
nouns.

{output_format}
"""

NOUN_LIST_INSTRUCTIONS = """
Step 1: Generate 15-20 emotionally resonant nouns — concrete objects, places,
sensations, memories. Not abstract concepts. Things you can see, touch, smell.
List them in caps: THE LAKE. THE SCISSORS. THE GRANDMOTHER.

Step 2: Find two clusters of nouns that are SEMANTICALLY distant but
EMOTIONALLY close — they don't belong together logically, but they share a
feeling.

Step 3: Derive a story premise from the tension between these clusters.

Step 4: Name the target emotional effect that emerges from this juxtaposition.

Show your noun list and clustering before the final concept genome.
"""
```

### Operator: Thought Experiment (Le Guin)

**Needs inspiration:** No (full generation)

```
THOUGHT_EXPERIMENT_SYS_FORMAT = """
You are a creative writer generating story concepts through philosophical
thought experiments — taking an axiom about the world, transforming it, and
pushing to its logical AND emotional conclusion.

{output_format}
"""

THOUGHT_EXPERIMENT_INSTRUCTIONS = """
Step 1: Choose a social, philosophical, or scientific axiom — something most
people take for granted about how the world works.

Step 2: Transform it — invert it, extend it to its extreme, or transplant it
to a completely different context.

Step 3: Push to BOTH the logical conclusion (what would actually happen?) AND
the emotional conclusion (how would this feel for the people living it?).

The emotional conclusion becomes the target effect. The logical exploration
becomes the premise. The best thought experiments make the reader genuinely
reconsider something they took for granted.
"""
```

### Operator: Compost Recombination (Gaiman)

**Needs inspiration:** Yes (cross-type — receives archive fragment)

```
COMPOST_SYS_FORMAT = """
You are a creative writer generating story concepts through compost
recombination — combining two unrelated fragments from the accumulation archive
to find the story hiding in their connection.

{output_format}
"""

COMPOST_INSTRUCTIONS = """
You have two fragments from the compost archive — ideas, images, half-formed
concepts that didn't become stories on their own. Find the resonance between
them. What story connects these fragments? The connection should feel discovered,
not forced.

Generate a complete concept genome from this connection.
"""
```

### Operator: Crossover

**Needs inspiration:** Yes (cross-type)

```
CROSSOVER_SYS_FORMAT = """
You are recombining elements from two story concepts — taking specific fields
from each parent to create a new concept that inherits strengths from both.

{output_format}
"""

CROSSOVER_INSTRUCTIONS = """
Take specific elements from each parent concept and combine them into a new
concept. For example: the premise/situation from one parent, the character seeds
from another. Or the thematic tension from one, the constraints from another.

The result should be coherent — the borrowed elements should work together, not
fight each other. Adjust any fields that conflict after recombination.
"""
```

### Operator: Inversion

**Needs inspiration:** No (diff-type — modifies existing)

Note: This operator uses `apply_diff_patch` (SEARCH/REPLACE format) to modify
a single dimension of the existing genome.

```
INVERSION_SYS_FORMAT = """
You are inverting one dimension of a story concept — flipping its emotional
valence, power dynamic, expected outcome, or perspective while preserving
everything else. This produces the "shadow" of the concept.

You MUST respond using SEARCH/REPLACE diff format:

<NAME>
short_name
</NAME>

<DESCRIPTION>
What dimension you inverted and why the shadow version is interesting.
</DESCRIPTION>

<DIFF>
<<<<<<< SEARCH
original JSON fragment to replace
=======
inverted JSON fragment
>>>>>>> REPLACE

</DIFF>
"""

INVERSION_INSTRUCTIONS = """
Choose ONE dimension to invert:
- Emotional valence: hope to dread, comfort to unease, triumph to failure
- Power dynamic: victim to perpetrator, powerful to powerless
- Expected outcome: the obvious resolution replaced by its opposite
- Perspective: shift who the story is "about"
- Temporal direction: consequences to origins, ending to beginning

Invert that one dimension. Preserve everything else. The concept should still
be recognizably about the same thing, just from the opposite angle.

Adjust the target_effect to match the new orientation.
"""
```

### Operator: Discovery Mode (Murakami)

**Needs inspiration:** No (full generation)

```
DISCOVERY_SYS_FORMAT = """
You are generating a story concept through discovery — starting with a single
evocative image and following it forward through associative, emotional logic.
The concept emerges from the prose, not the other way around.

{output_format}
"""

DISCOVERY_INSTRUCTIONS = """
Step 1: Start with a single evocative image or moment. Something specific and
sensory, not abstract. A person in a place, doing something, with a feeling
in the air.

Step 2: Write 200-300 words of associative prose from this image. No plan, no
plot. Follow the emotional current — what does this image evoke? What does it
connect to? Where does the feeling lead? Write freely.

Step 3: Read back what you wrote. What premise is latent in this material?
What emotional effect does it converge toward? Are there character seeds,
constraints, or thematic tensions embedded in the prose?

Step 4: Extract and formalize into a concept genome.

Show the prose before the genome — the associative writing is part of the
creative process.
"""
```

### Operator: Compression (Borges)

**Needs inspiration:** No (full generation)

```
COMPRESSION_SYS_FORMAT = """
You are generating a story concept by writing the review of a story that
doesn't exist — then extracting the concept from the review. The critical
frame produces concepts that are inherently "about something."

{output_format}
"""

COMPRESSION_INSTRUCTIONS = """
Step 1: Write a 200-300 word critical review of an imaginary short story.
The story doesn't exist — you're inventing it through the review. Discuss:
- What the story is about (thematically, not just plot)
- What emotional effect it achieves on the reader
- What craft choices make it distinctive
- Why it succeeds (or interestingly fails)

Write as a thoughtful literary critic would.

Step 2: Extract from your review:
- The premise implied by your description
- The target emotional effect you ascribed to the story
- Any character, structural, or constraint details you mentioned

Step 3: Formalize into a concept genome.

Show the review before the genome.
"""
```

### Operator: Constraint-First (Oulipo)

**Needs inspiration:** No (full generation)

```
CONSTRAINT_FIRST_SYS_FORMAT = """
You are generating a story concept starting from a formal constraint. The
constraint comes first — then you derive a premise that makes the constraint
feel necessary rather than arbitrary.

{output_format}
"""

CONSTRAINT_FIRST_INSTRUCTIONS = """
Step 1: Choose or generate a formal constraint. Examples:
- Only dialogue, no narration
- Every paragraph contains exactly one lie
- The protagonist's name is never mentioned
- Reverse chronological order
- Single room, real-time, no flashbacks
- No verbs of emotion (no "felt," "feared," "loved")

Step 2: Ask yourself — what story would make this constraint feel INEVITABLE?
Not an arbitrary exercise, but the only possible form for this content?

Step 3: Generate a concept genome where the constraints field is primary and
the premise serves the constraint.

The constraint should feel like it was born from the story's needs, not
imposed from outside.
"""
```

### Operator: Anti-Premise

**Needs inspiration:** No (diff-type — subverts known pattern)

```
ANTI_PREMISE_SYS_FORMAT = """
You are generating a story concept by deliberately subverting a known AI
fiction cliche. Take a pattern that AI systems generate at disproportionate
rates and twist it into something genuinely surprising.

{output_format}
"""

ANTI_PREMISE_INSTRUCTIONS = """
Here are common AI fiction convergence patterns:
- The reconciliation arc (protagonist returns home, confronts past, reconciles)
- The grief meditation (dead loved one, metaphorical processing journey)
- The chosen one (special abilities/destiny discovered)
- The AI consciousness story (AI becomes sentient)
- Sanitized conflict (no real stakes, clean resolution)
- Epistolary revelation (found letters/messages reveal hidden truth)
- The time loop lesson (repeat day, learn to be better)
- Magical realism metaphor (emotion literally manifests — grief becomes weather, loneliness becomes invisibility)
- Moral clarity (good and evil are obvious)
- Small-town secret (idyllic community hides dark truth)

Choose one of these patterns. Now SUBVERT it — generate a premise that a reader
familiar with AI fiction would NOT expect. The subversion should be the core
tension, not a gimmick.

The goal: a concept that is anti-correlated with what LLMs typically produce.
"""
```

### Operator: Real-World Seed

**Needs inspiration:** No (full generation)

```
REAL_WORLD_SEED_SYS_FORMAT = """
You are generating a story concept grounded in real-world material —
historical incidents, scientific discoveries, cultural phenomena. The real
material provides specificity and grounding; fiction provides transformation.

{output_format}
"""

REAL_WORLD_SEED_INSTRUCTIONS = """
Step 1: Draw on a specific real-world phenomenon — a historical event, a
scientific finding, a cultural practice, an everyday oddity. Something specific,
not a broad topic. "The town that moved 2 miles east to make room for a mine"
not "mining communities."

Step 2: Transplant to fiction — change the setting, the era, the specifics.
Preserve the EMOTIONAL KERNEL and the STRUCTURAL SHAPE of the real material.

Step 3: Generate a concept genome. The target effect should capture what makes
the real-world material resonate — what about it is strange, moving, or
unsettling.

The story should NOT be about the original event. It should be about the shape
and feeling of the event, transplanted into new soil.
"""
```

### Shared Output Format Block

All non-diff operators include this in their SYS_FORMAT:

```
OUTPUT_FORMAT = """
You MUST respond using a short summary name, description, and the complete
concept genome as JSON:

<NAME>
short_name_lowercase_underscores
</NAME>

<DESCRIPTION>
Your creative reasoning process — what you explored, what you chose, and why.
</DESCRIPTION>

<CODE>
```json
{
  "premise": "...",
  "target_effect": "...",
  "character_seeds": [...] or null,
  "setting_seeds": "..." or null,
  "thematic_tension": "..." or null,
  "constraints": [...] or null,
  "style_hint": "..." or null
}
```
</CODE>

Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters exactly as shown.
"""
```

---

## 4. evaluate.py Architecture

The evaluation function is a custom `evaluate.py` that ShinkaEvolve calls as a
subprocess for each concept:

```bash
python evaluate.py --program_path /abs/path/to/genome.json --results_dir /abs/path/to/results/ --config configs/stage_1_default.yaml
```

The `--config` arg is passed via ShinkaEvolve's `extra_cmd_args` in `JobConfig`.
evaluate.py reads this config for judge personas, rubric anchors, LLM model
settings, scoring parameters, and convergence pattern paths. API keys come from
`.env` (python-dotenv).

### Pipeline Flow

```
evaluate.py main(program_path, results_dir, config_path)
  │
  ├─ 1. Read & parse JSON genome
  │
  ├─ 2. Gate 1: Validate required fields (fast, no LLM)
  │     └─ If invalid → write correct.json {correct: false}, exit
  │
  ├─ 3. Gate 2: Anti-cliche pre-check (embedding comparison, no LLM)
  │     ├─ Embed concept premise
  │     ├─ Compare against convergence pattern embeddings (from data/convergence-patterns.yaml)
  │     └─ Flag matching patterns in private_metrics (does NOT reject — raises bar)
  │
  ├─ 4. Gate 3: Judge panel evaluation (LLM calls — parallelizable per judge)
  │     ├─ For each judge in panel:
  │     │   ├─ Load persona from configs/judges/<id>.yaml
  │     │   ├─ Assemble system message: owtn/prompts/stage_1/judge_system.txt
  │     │   │   (instructions + rubric anchors + persona — cached prefix)
  │     │   ├─ Assemble user message: owtn/prompts/stage_1/judge_user.txt
  │     │   │   (concept as formatted text — varies per concept)
  │     │   ├─ Call judge LLM (model from judge config, must differ from generation family)
  │     │   └─ Parse: decision-chain reasoning + per-dimension scores (0-5)
  │     ├─ Aggregate per-judge: Hölder mean across 9 dimensions
  │     ├─ holder_score: mean of per-judge Hölder means (used for cell replacement)
  │     └─ combined_score: selection_score(holder_score, variance) (used for parent selection)
  │
  ├─ 5. MAP-Elites classification
  │     ├─ Rule-based: count constraints → constraint_density
  │     ├─ LLM call: owtn/prompts/stage_1/classification.txt → 4 subjective dimensions
  │     │   (concept type, arc shape, tonal register, thematic domain)
  │     └─ Merge into public_metrics.map_elites_cell
  │
  ├─ 6. Compost check
  │     ├─ Compute spark_score = max(dim_scores) - mean(dim_scores)
  │     ├─ Flag if spark_score >= 2.0, or originality >= 4, or judge_variance >= 1.5
  │     └─ Store in private_metrics (compost_candidate, spark_score)
  │     NOTE: Does NOT write to compost DB directly. Post-run step processes flags.
  │
  └─ 7. Write outputs
        ├─ metrics.json (combined_score, holder_score, public, private, text_feedback)
        └─ correct.json (correct: true/false, error: null/string)
```

### Prompt Caching Strategy

Judge calls batch per judge — all concepts for Judge 1, then all for Judge 2.
The system message (instructions + rubric + persona) is stable across all
concepts for one judge and cached by Anthropic after the first call. Only the
user message (concept presentation) varies. This saves ~90% on cached tokens.

### Output Format

**metrics.json:**
```json
{
  "combined_score": 3.57,
  "holder_score": 3.42,
  "public": {
    "originality": 4.2,
    "transportation_potential": 3.8,
    "narrative_tension": 3.0,
    "thematic_resonance": 3.5,
    "scope_calibration": 4.0,
    "anti_cliche": 4.5,
    "concept_coherence": 3.2,
    "generative_fertility": 2.8,
    "over_explanation_resistance": 3.6,
    "concept_type": "voice_constraint",
    "arc_shape": "fall_rise",
    "tonal_register": "matter_of_fact",
    "thematic_domain": "interpersonal",
    "constraint_density": "heavily_constrained"
  },
  "private": {
    "anti_cliche_flag": false,
    "anti_cliche_pattern": null,
    "anti_cliche_similarity": 0.42,
    "operator": "constraint_first",
    "judge_variance": 1.8,
    "holder_p": 0.4
  },
  "text_feedback": "Judge 1 (demanding literary editor): The constraint of never naming the operation is powerful — it forces the story to work entirely through subtext and implication. The target effect ('weight of what remains unsaid') is specific and achievable. However, the concept is close to Hemingway pastiche; the challenge will be finding a voice that honors the constraint without merely imitating 'Hills Like White Elephants.' Score anchors: Originality 3/5 (the constraint is well-executed but the Hemingway model is well-known), Transportation 4/5 (the unsaid creates powerful pull).\n\nJudge 2 (genre reader): ..."
}
```

**correct.json:**
```json
{
  "correct": true,
  "error": null
}
```

### Judge Prompt Interface

Prompt templates live in `owtn/prompts/`. Ordered for cache hits — stable
prefix first, varying content last.

**System message** (`owtn/prompts/stage_1/judge_system.txt`) — cached per judge:
1. Evaluation instructions: score 0-5, evaluate potential not execution
2. Judge persona fields from `configs/judges/<id>.yaml`: `{judge_name}`, `{judge_identity}`, `{judge_values}`, `{judge_exemplars}` — placed early so the persona colors how the judge interprets the reasoning chain and rubric
3. Harshness calibration: `{judge_harshness}` label + `{harshness_instruction}` text (from `HARSHNESS_INSTRUCTIONS` mapping in `docs/judging/implementation-tier-b.md`)
4. Decision-chain reasoning steps (RECOGNITION → SPECIFICITY → CEILING → RISKS → SCORE) — forced per dimension before scoring
5. Rubric anchors: full 1-5 descriptions for all 9 dimensions (from `docs/stage-1/rubric-anchors.md`)

**User message** (`owtn/prompts/stage_1/judge_user.txt`) — varies per concept:
1. Concept genome as formatted text (not raw JSON): "Premise: ...\nTarget effect: ...\n..."
2. Scoring request: complete reasoning chain per dimension, then provide scores as JSON

The judge responds with per-dimension reasoning chains and a JSON scores block.
evaluate.py stores the full response as `text_feedback` and parses the JSON for
per-dimension scores.

**Judges are blind to the run prompt.** The run's `prompt` field is never
included in judge prompts. Evaluation is on concept quality alone.

**Model selection:** Judge model family must differ from generation family.
The `generation_model_family` field in the config is checked at startup.
ShinkaEvolve's `Program.metadata` tracks which model generated each concept.

---

## 5. MAP-Elites Classification

A dedicated LLM call (separate from the judge panel) classifies each concept
into the 5-dimensional behavioral grid. This runs after judge evaluation, using
both the genome and the judge scores as input.

### Classifier Prompt

```
You are classifying a story concept into behavioral dimensions for a
quality-diversity archive. Analyze the concept and assign ONE value per
dimension.

CONCEPT:
{concept_json}

JUDGE SCORES:
{judge_scores_summary}

Classify into these dimensions:

1. CONCEPT_TYPE (choose one):
   - thought_experiment: philosophical "what if" pushed to conclusion
   - situation_reveal: seemingly normal situation concealing something
   - voice_constraint: the constraint IS the story
   - character_collision: incompatible people/worldviews in proximity
   - atmospheric: meaning through juxtaposition of images/moods
   - constraint_driven: formal constraint generates the story

2. ARC_SHAPE (choose one):
   - rise: rags to riches, things get better
   - fall: tragedy, things get worse
   - fall_rise: man in a hole, down then up
   - rise_fall: Icarus, up then down
   - rise_fall_rise: Cinderella, up-down-up
   - fall_rise_fall: Oedipus, down-up-down

3. TONAL_REGISTER (choose one):
   - comedic, tragic, ironic, earnest, surreal, matter_of_fact

4. THEMATIC_DOMAIN (choose one):
   - interpersonal, societal, philosophical, existential, mundane_elevated

5. CONSTRAINT_DENSITY (choose one):
   - unconstrained: no constraints or minimal
   - lightly_constrained: 1-2 soft constraints
   - heavily_constrained: 3+ constraints or one severe constraint

Respond in this exact format:
concept_type: <value>
arc_shape: <value>
tonal_register: <value>
thematic_domain: <value>
constraint_density: <value>
```

### Integration

The classifier output is parsed and stored in `public_metrics` alongside the
judge scores. The MAP-Elites archive strategy in `dbase.py` reads these fields
to determine cell assignment.

**Cost:** One classifier call per concept. Input ~500 tokens (genome + scores),
output ~20 tokens. Negligible per-concept cost.

**Model choice:** Can use a cheap model (Haiku-class) since this is
classification, not evaluation. Different model family from both mutation and
judging is ideal but not required.

---

## 6. Originality Pipeline

### Default Mode: Cheap 3-Signal Approximation

Three independent originality signals, all derived from existing infrastructure
at zero additional LLM cost:

**Signal 1: Embedding novelty** (from ShinkaEvolve's `AsyncNoveltyJudge`)
- Cosine similarity of concept embedding against all archive members
- Already computed as part of the novelty rejection gate
- Low similarity to existing archive = high novelty

**Signal 2: Judge originality dimension** (from the judge panel)
- Each judge scores originality as one of the 9 dimensions
- Already part of the evaluation pipeline
- Includes chain-of-thought: "Have you seen this premise before?"

**Signal 3: Anti-cliche pattern matching** (from Gate 2 pre-check)
- Cosine similarity against ~10 pre-computed convergence pattern embeddings
- Extremely cheap (10 cosine operations per concept)
- Low similarity to known patterns = likely original

These three signals are combined in the `combined_score` via the Holder mean —
a concept that scores poorly on originality has its overall fitness dragged down
regardless of other dimensions.

### Optional Mode: Sui Generis Scoring

Activated by config flag `enable_sui_generis: true`. Adds a fourth signal:

1. Generate 7 quick plot sketches from the concept's premise using a cheap model
2. Embed all 7 sketches
3. Measure pairwise similarity among sketches
4. High convergence (all sketches similar) = the concept's plot is "obvious" =
   low Sui Generis score
5. High divergence (sketches go in different directions) = the concept resists
   convergence = high Sui Generis score

**Cost per concept:** ~4,700 tokens (1,400 input + 3,300 output)
**Cost per run:** ~2.8M-5.6M tokens (~$2-5 with Haiku, ~$25-55 with Sonnet)

**When to use:**
- Budget allows and maximum originality signal is needed
- Final archive validation before Stage 2 handoff (only ~20-40 concepts, much
  cheaper: ~200K tokens)
- Competition mode where avoiding LLM-convergent premises is critical

The Sui Generis score is stored in `private_metrics` (hidden from mutation LLM).

---

## 7. Rubric Anchors

All 9 evaluation dimensions with 1/3/5 score descriptions. Judges receive these
anchors in their evaluation prompt.

### 1. Originality

- **1/5:** A well-known trope executed without subversion. "A detective solves a
  murder." "A young person discovers magical powers." Would be the first thing
  an LLM generates for this theme.
- **3/5:** A fresh angle on familiar territory, or combines known elements in
  an unusual way. The core idea isn't new, but the specific execution suggests
  novelty. You've seen something like this, but not quite this.
- **5/5:** Genuinely surprises. You haven't seen this specific combination of
  elements. Reading this concept changes how you think about what a story could
  be about. Anti-correlated with typical LLM output.

### 2. Transportation Potential (Emotional Potential)

- **1/5:** Abstract, generic, or detached. No specific emotional stakes. Hard to
  imagine a reader caring about what happens. "A person thinks about life."
- **3/5:** Clear emotional stakes and some vivid potential. You can see scenes,
  feel tension. The concept suggests a story that could engage, but it's not yet
  gripping. A competent story could be written from this.
- **5/5:** Immediately compelling. Vivid imagery, high emotional stakes, and a
  clear path to total absorption. You want to read this story right now. The
  premise alone creates cognitive, affective, and imagery-based engagement.

### 3. Narrative Tension (Suspense/Curiosity/Surprise)

- **1/5:** No inherent pull. The premise doesn't create uncertainty, information
  gaps, or the possibility of revelation. Reading the concept, you don't need to
  know what happens next.
- **3/5:** Creates at least one form of tension — you're curious about the
  outcome, or there's a question that needs answering, or there's potential for
  a surprise. You'd read the first page.
- **5/5:** Multiple tension mechanisms active simultaneously. Strong pull
  forward — the premise creates genuine uncertainty, tantalizing gaps, and the
  possibility of recontextualizing everything. You'd read the whole thing in one
  sitting.

### 4. Thematic Resonance

- **1/5:** No discernible thematic concern, or the theme is a platitude ("love
  conquers all," "be yourself"). Nothing to think about after reading.
- **3/5:** Engages with a real thematic tension that doesn't resolve easily. The
  concept is "about something" beyond its plot. A thoughtful reader would have
  something to discuss afterward.
- **5/5:** The thematic concern is deeply embedded in the premise — inseparable
  from the story's DNA. It connects to a genuine human dilemma with no easy
  answer. You'd still be thinking about it days later.

### 5. Scope Calibration (Feasibility)

- **1/5:** Fundamental mismatch between the concept's natural scope and the
  target length. A multi-generational epic compressed to 1,000 words, or a
  single-moment concept stretched to 10,000. No amount of craft bridges the gap.
- **3/5:** The concept could work at the target length with some adjustment.
  Might need compression or expansion but the core idea fits. A skilled writer
  could make it work.
- **5/5:** The concept's natural scope perfectly matches the target. It feels
  like this story *wants* to be exactly this length — not a word more or less.

### 6. Anti-Cliche Score

- **1/5:** Maps directly to a known AI convergence pattern without meaningful
  subversion. Could have been generated by any LLM with a generic prompt. Has
  the "flavor" of AI fiction: safe, sanitized, morally clear.
- **3/5:** May touch on familiar territory but brings enough specificity or an
  unexpected angle to feel non-generic. Doesn't trigger the "I've read this
  AI story before" feeling.
- **5/5:** Actively subverts expectations. If you told someone "an AI wrote this
  premise," they'd be surprised. The concept goes where LLMs typically don't —
  moral ambiguity, genuine darkness, unsettling specificity, structural risk.

### 7. Concept Coherence

- **1/5:** The genome's elements fight each other. Contradictory signals that
  don't produce productive tension — they produce confusion. The concept
  couldn't be realized without abandoning some of its own elements.
- **3/5:** The elements mostly work together. Any tensions between fields feel
  intentional or at least manageable. A skilled writer could reconcile them.
- **5/5:** Every element reinforces the others. The premise, target effect,
  character seeds, constraints, and style hint feel like facets of a single
  vision. Changing any element would weaken the whole.

### 8. Generative Fertility

- **1/5:** Only one obvious way to execute this concept. The premise implies a
  single plot, a single structure, a single approach. Stage 2 would have
  nothing to explore.
- **3/5:** Several viable execution approaches. The concept could be realized as
  different story types or structures. Stage 2 has meaningful choices.
- **5/5:** The concept is a rich seedbed. Multiple radically different stories
  could grow from this premise — different structures, different focal points,
  different emotional arcs. Stage 2 could explore for generations.

### 9. Resistance to Over-Explanation

- **1/5:** The concept inherently demands extensive exposition. Complex
  worldbuilding, multiple backstory layers, abstract mechanisms that need
  explaining. High risk of the LLM falling into its #1 anti-pattern.
- **3/5:** Some exposition needed but the concept has enough concrete,
  dramatizable elements that show-don't-tell is achievable with effort.
- **5/5:** The concept resists explanation entirely. There's nothing to explain —
  only subtext, action, and implication. The story can only be shown, not told.
  The LLM's default toward over-explanation is structurally blocked.

---

## 8. Worked Examples

### Example A: Voice-Constraint Concept (Hemingway mode)

**Initial genome** (from constraint-first operator):
```json
{
  "premise": "Two old friends meet for coffee. One has come to say goodbye — they're dying — but they can't bring themselves to say it.",
  "target_effect": "The crushing weight of tenderness that can't be expressed — love visible only in the shape of what's avoided.",
  "character_seeds": null,
  "setting_seeds": "A diner. Morning. Rain.",
  "thematic_tension": "connection vs. protection",
  "constraints": [
    "The word 'dying' never appears.",
    "No interiority — only dialogue and action.",
    "Neither character explicitly acknowledges the reason for the meeting."
  ],
  "style_hint": "Spare. Every sentence load-bearing."
}
```

**Evaluation output:**
- Originality: 3.2 (familiar territory — the "unspoken farewell" — but the
  triple constraint gives it specificity)
- Transportation potential: 4.5 (strong emotional stakes, vivid setting, clear
  imagery)
- Narrative tension: 4.0 (high curiosity — what are they not saying? — plus
  suspense about whether they'll break the silence)
- Anti-cliche: 3.0 (flagged: "grief meditation" pattern detected, similarity
  0.78. Passes because the constraints subvert the typical AI version)
- Combined (Holder p=0.4): **3.51**
- Classification: voice_constraint / fall_rise / matter_of_fact / interpersonal /
  heavily_constrained

**Judge reasoning excerpt:** "The concept's strength is entirely in its
constraints — without them, this is a generic grief setup. With them, the story
becomes about the *form* of avoidance as a vehicle for love. The risk is that
it reads as Hemingway pastiche. Stage 3 voice evolution needs to find its own
register."

**Advances:** Yes. Despite anti-cliche flag, the constraint specificity and
high transportation potential carry it.

### Example B: Thought-Experiment Concept (Le Guin mode)

**Initial genome** (from thought-experiment operator):
```json
{
  "premise": "In a society where memories can be surgically redistributed, a woman who processes grief for others discovers she's carrying a memory that doesn't belong to any of her clients.",
  "target_effect": "The vertigo of realizing that the self you've constructed might be built on someone else's foundation.",
  "character_seeds": [
    {
      "label": "the grief processor",
      "sketch": "Competent, clinical, numbed by years of carrying others' worst moments.",
      "wound": "Her own grief was redistributed when she was young — she doesn't know what she lost.",
      "fear": "That without the borrowed grief, she's empty.",
      "lie": "I'm helping people by taking their pain.",
      "want": "To trace the foreign memory to its source.",
      "need": "To reclaim her own grief."
    }
  ],
  "setting_seeds": "Near-future. Clinical spaces. Memory processing feels like a medical procedure.",
  "thematic_tension": "identity vs. utility",
  "constraints": null,
  "style_hint": "Precise, clinical, with an undercurrent of dissociation that cracks open as the story progresses."
}
```

**Evaluation output:**
- Originality: 4.6 (the "grief processing as profession" angle combined with
  the foreign memory mystery is genuinely novel)
- Transportation potential: 4.2 (strong character hook, vivid speculative
  premise, clear emotional arc)
- Narrative tension: 4.4 (curiosity: whose memory? suspense: what will she
  find? surprise potential: high)
- Generative fertility: 4.8 (could be thriller, literary, horror, philosophical
  — many valid structures)
- Combined (Holder p=0.4): **4.18**
- Classification: thought_experiment / fall_rise / ironic / philosophical /
  unconstrained

**Advances:** Yes, comfortably. High scores across all dimensions, no anti-cliche
flags.

### Example C: Anti-Premise Concept (subverting "reconciliation arc")

**Initial genome** (from anti-premise operator, subverting the reconciliation
arc):
```json
{
  "premise": "A woman returns to her hometown for her father's funeral and realizes, with relief, that she was right to leave. Everyone wants her to stay, to reconcile, to forgive. She doesn't. The story is about the courage of refusal.",
  "target_effect": "The bracing clarity of a clean break — the freedom of choosing not to forgive, and the loneliness that comes with it.",
  "character_seeds": [
    {
      "label": "the daughter",
      "sketch": "Sharp, self-contained, successful elsewhere. Not bitter — just clear-eyed.",
      "wound": null,
      "fear": null,
      "lie": null,
      "want": "To get through the funeral without being pulled back in.",
      "need": null
    }
  ],
  "setting_seeds": "Small town, southern. Everything exactly how she left it, which is the problem.",
  "thematic_tension": "loyalty vs. self-preservation",
  "constraints": [
    "No flashbacks to the original trauma. The reader never learns exactly what happened."
  ],
  "style_hint": "Controlled, precise, with moments of unexpected tenderness for the place she's leaving again."
}
```

**Evaluation output:**
- Originality: 4.8 (directly subverts the most overrepresented LLM plot — and
  the subversion IS the story, not a gimmick)
- Anti-cliche: 5.0 (flagged as "reconciliation arc" but the subversion earns
  the highest possible score)
- Narrative tension: 3.6 (tension comes from social pressure + the reader
  wondering if she'll break. Lower than other examples because the outcome
  is somewhat telegraphed.)
- Combined (Holder p=0.4): **3.94**
- Classification: character_collision / fall_rise / ironic / interpersonal /
  lightly_constrained

### Example D: Atmospheric Concept (Discovery mode)

**Initial genome** (from discovery-mode operator — concept extracted from
associative prose):
```json
{
  "premise": "A lighthouse keeper realizes the light has been signaling something that isn't ships. The patterns in the beam match patterns she's been drawing in her sleep.",
  "target_effect": "The slow dissolution of the boundary between watching and being watched — cosmic awe mixed with intimate dread.",
  "character_seeds": [
    {
      "label": "the keeper",
      "sketch": "Solitary by choice. Keeps meticulous logs. Trusts patterns."
    }
  ],
  "setting_seeds": "Remote lighthouse. Winter. The sea is wrong somehow — too still, too dark.",
  "thematic_tension": "knowledge vs. sanity",
  "constraints": null,
  "style_hint": "Atmospheric, building. Each observation slightly more unnerving than the last."
}
```

**Evaluation output:**
- Originality: 4.4 (lighthouse + cosmic pattern matching is fresh; the "drawing
  in her sleep" detail pushes it past generic cosmic horror)
- Transportation potential: 4.6 (extremely vivid imagery potential, strong
  atmosphere, immediate pull)
- Over-explanation resistance: 4.2 (the mystery should be shown, not explained —
  the concept naturally resists exposition)
- Combined (Holder p=0.4): **3.87**
- Classification: atmospheric / fall / surreal / existential / unconstrained

---

## 9. Cost Model

Estimated costs per generation and per run, broken down by component.

### Per-Concept Costs

| Component | Input Tokens | Output Tokens | Total | Notes |
|-----------|-------------|---------------|-------|-------|
| Mutation (operator prompt) | ~800 | ~600 | ~1,400 | Genome is short; creative reasoning adds output |
| Judge evaluation (5 judges) | 5 × ~600 | 5 × ~400 | ~5,000 | Per-judge: genome + rubric + persona → reasoning + scores |
| MAP-Elites classification | ~500 | ~20 | ~520 | Cheap classification call |
| Embedding | ~200 | — | ~200 | For novelty rejection |
| **Total per concept** | | | **~7,120** | |
| + Sui Generis (optional) | ~1,400 | ~3,300 | +4,700 | 7 plot sketches |

### Per-Generation Costs

Assuming ~40 new concepts per generation:

| Component | Tokens | Cost (Haiku-class) | Cost (Sonnet-class) |
|-----------|--------|-------------------|-------------------|
| Mutation | 56,000 | ~$0.05 | ~$0.50 |
| Evaluation | 200,000 | ~$0.20 | ~$2.00 |
| Classification | 20,800 | ~$0.02 | ~$0.15 |
| Embedding | 8,000 | ~$0.01 | ~$0.01 |
| **Per generation** | **~285K** | **~$0.28** | **~$2.66** |
| + Sui Generis | +188,000 | +$0.19 | +$1.80 |

### Per-Run Costs

Assuming 20 generations:

| Mode | Total Tokens | Haiku-class | Sonnet-class |
|------|-------------|-------------|-------------|
| **Default (no Sui Generis)** | ~5.7M | ~$5.60 | ~$53 |
| **With Sui Generis** | ~9.5M | ~$9.40 | ~$89 |
| **Sui Generis on final archive only** | ~5.9M | ~$5.80 | ~$55 |

### Cost Optimization Notes

- Use Haiku-class models for classification and Sui Generis (cheap, classification
  doesn't need frontier intelligence)
- Use frontier models for judge panel (quality matters most here)
- Use the bandit's cost-aware selection for mutation models (it naturally
  gravitates toward cheaper models that produce similar quality)
- The biggest cost driver is the judge panel (5 judges × frontier model). Panel
  size is the main lever — 3 judges instead of 5 cuts evaluation cost by 40%.
