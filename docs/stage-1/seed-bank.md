# Stage 1: Seed Bank

The seed bank is the primary external input to Stage 1. It provides concrete
starting material so operators don't rely purely on LLM latent knowledge — which
tends toward convergent, overexposed ideas. Each seed type maps to one or more
mutation operators, giving different creative methods different fuel.

The actual seeds live in `data/seed-bank.yaml`. This document specifies the
schema, seed types, selection criteria, and maintenance strategy.

---

## Schema

```yaml
- id: string           # human-readable kebab-case
  type: string         # real_world | thought_experiment | axiom | dilemma | constraint | noun_cluster | image | compression | collision_pair | anti_target
  content: string|list # varies by type (string for most, list for noun_cluster and collision_pair)
  source: string       # optional — for real_world and axiom where provenance matters
  tags: [string]       # freeform, for diversity-weighted selection and theme filtering
```

Notes:

- **`content` is polymorphic by type.** Most types use a string. `noun_cluster`
  uses a list of 6-8 nouns. `collision_pair` uses a list of exactly two items.
- **`source` is optional.** Required for `real_world` (must be verifiable).
  Recommended for `axiom` (attribution matters). Unnecessary for other types.
- **No runtime tracking fields.** Usage counts, fitness associations, and
  staleness tracking belong in the run database, not in the seed file. The seed
  bank is a static curated resource.

---

## Seed Types

### `real_world`

**What it is:** A true, specific, surprising fact — a historical event,
scientific finding, or cultural phenomenon.

**Feeds:** Real-World Seed Injection (operator #11)

**What makes a good one:** Must be TRUE and citable. Specific enough to anchor
a story (not "climate change" but a particular village relocating). Surprising
enough that the reader pauses. Suggests human implications beyond the fact
itself. Not overexposed in LLM training data.

**Content format:** String. 1-2 sentences stating the fact.

---

### `thought_experiment`

**What it is:** A scenario that dramatizes a philosophical or ethical question —
not just the claim, but the concrete situation that makes the claim feel alive.
Mary's Room isn't just "qualia exist" — it's the woman, the black-and-white room,
the moment she sees red.

**Feeds:** Thought Experiment / Le Guin (operator #3)

**What makes a good one:** Must be a specific scenario with characters, stakes,
and a question that has no clean answer. The scenario should be richer than the
thesis it illustrates — it should suggest stories, not just arguments. Already
halfway to fiction.

**Content format:** String. The scenario described concretely, ending with the
question it raises.

---

### `axiom`

**What it is:** A bare philosophical, scientific, or ethical position — not a
scenario, but an arguable claim about reality, consciousness, identity, or
how people should live.

**Feeds:** Thought Experiment / Le Guin (operator #3)

**What makes a good one:** Must be a real position someone has defended, not a
strawman. Should be arguable from multiple sides. Should have emotional stakes —
not just an intellectual puzzle but something that changes how it feels to be
alive if true.

**Content format:** String. The position stated clearly enough to be inverted,
extended, or transplanted.

---

### `dilemma`

**What it is:** A specific situation where two legitimate values conflict with no
clean resolution. Not an abstract ethical puzzle — a concrete scenario with
characters who face an impossible choice.

**Feeds:** Thought Experiment / Le Guin (operator #3), Collision / King
(operator #1), Compression / Borges (operator #8)

**What makes a good one:** Both sides must be genuinely defensible. The situation
must be specific enough to imply characters and stakes. Should resist the "moral
clarity" convergence pattern — the whole point is that there's no right answer.
The best dilemmas make you change your mind twice while reading them.

**Content format:** String. The situation described with enough specificity that
the bind is felt, not just understood.

---

### `constraint`

**What it is:** A formal structural, linguistic, content, or POV constraint that
shapes how a story can be told.

**Feeds:** Constraint-First Generation / Oulipo (operator #9)

**What makes a good one:** Must be formally precise — implementable, not vague
("make it feel dreamlike" is not a constraint; "no sentence may exceed seven
words" is). Must be generative: it should create stories, not merely restrict
them. The best constraints make you immediately think "what story would need to
be told this way?"

**Content format:** String. The constraint stated precisely.

---

### `noun_cluster`

**What it is:** A list of 6-8 concrete, emotionally charged nouns grouped by
emotional resonance rather than semantic category.

**Feeds:** Noun-List Recombination / Bradbury (operator #2)

**What makes a good one:** Nouns must be concrete — objects, places, sensations,
not abstractions. Each noun is preceded by THE (following Bradbury's practice),
which converts a category into a specific memory — THE CREEK is not creeks in
general but the one that mattered. The cluster's coherence should be felt, not
explained — dredged, not curated. A good cluster makes you sense a story without
being able to articulate why those words belong together.

**Content format:** List of 6-8 strings, each prefixed with "the".

---

### `image`

**What it is:** A specific evocative scene — a starting image for associative
generation.

**Feeds:** Discovery Mode / Murakami (operator #7)

**What makes a good one:** Must be a specific scene, not a vague mood. Should
contain at least one surprising or incongruous element. Should resist easy
interpretation — the image should be richer than any single explanation of it.

**Content format:** String. A scene described in 1-2 sentences.

---

### `compression`

**What it is:** A dense thematic kernel that implies an entire story world —
what a story is ABOUT, not what happens in it.

**Feeds:** Compression / Borges (operator #8)

**What makes a good one:** Dense enough to imply characters, setting, and
emotional stakes without naming them. Should suggest theme, not plot. Should
feel like the last line of a review of a story you desperately want to read.

**Content format:** String. 1-3 sentences.

---

### `collision_pair`

**What it is:** Two semantically distant items that could produce a story in the
interference pattern between them.

**Feeds:** Collision / King (operator #1)

**What makes a good one:** The two items must be semantically distant but
emotionally or thematically connectable. The connection should not be obvious —
if you can immediately see the story, the pair is too close. If you can't
imagine any connection at all, they're too far.

**Content format:** List of exactly two strings.

---

### `anti_target`

**What it is:** A named convergence pattern (one of the 10 known LLM cliches)
paired with a specific subversion angle.

**Feeds:** Anti-Premise Generation (operator #10)

**What makes a good one:** Must name a specific convergence pattern, not a
vague complaint about AI writing. The subversion angle must be specific and
generative — not just "do the opposite" but a particular way into the same
thematic territory that the cliche forecloses.

**Content format:** String. Names the pattern and states the subversion angle.

---

## Seed Type Summary

| Type | Feeds | Content format |
|------|-------|----------------|
| `real_world` | Real-World Seed Injection | string: 1-2 sentence fact |
| `thought_experiment` | Thought Experiment (Le Guin) | string: scenario + question |
| `axiom` | Thought Experiment (Le Guin) | string: bare philosophical position |
| `dilemma` | Thought Experiment, Collision, Compression | string: impossible choice |
| `constraint` | Constraint-First (Oulipo) | string: formal constraint |
| `noun_cluster` | Noun-List (Bradbury) | list: 6-8 emotionally charged nouns |
| `image` | Discovery Mode (Murakami) | string: specific evocative scene |
| `compression` | Compression (Borges) | string: dense thematic kernel |
| `collision_pair` | Collision (King) | list: two semantically distant items |
| `anti_target` | Anti-Premise | string: convergence pattern + subversion angle |

---

## Selection Criteria

### Universal (all types)

A good seed must satisfy ALL of:

1. **Specific enough to anchor a story.** Not a broad topic ("grief") but a
   concrete starting point that implies characters, stakes, and setting.
2. **Surprising enough to resist LLM default convergence.** If the seed would
   naturally produce a story you've read a hundred times, it's too familiar.
3. **Suggests human implications.** Not just a cool fact or clever constraint —
   there must be something at stake for people. How does this change what it
   means to be a person, to love someone, to make a choice?

### Per-type criteria

- **`real_world`:** Must be TRUE. Cite source. Not overexposed in LLM training
  data (if it shows up in fun-facts listicles, dig deeper).
- **`axiom`:** Must be a genuine philosophical position (not a strawman). Should
  be arguable from multiple sides. Must have emotional stakes, not just
  intellectual interest.
- **`constraint`:** Must be formally precise (implementable, not vague). Should
  be generative (creates stories, not just restricts them). The best constraints
  make you immediately wonder what story would need this form.
- **`noun_cluster`:** Nouns must be concrete objects/places/sensations, not
  abstractions. Should have emotional resonance that isn't obvious from
  semantics. The cluster should feel right without being explainable.
- **`image`:** Must be a specific scene, not a vague mood. Should contain at
  least one surprising element. Should resist reduction to a single
  interpretation.
- **`compression`:** Dense enough to imply an entire story world. Should suggest
  theme, not just plot. Should make you want to read the story it describes.
- **`collision_pair`:** The two items must be semantically distant but
  emotionally or thematically connectable. The connection shouldn't be obvious.
- **`anti_target`:** Must name a specific convergence pattern AND provide a
  subversion angle that isn't just "the opposite."

---

## Content Filtering

Exclude seeds that:

- Are themselves cliched or would naturally produce one of the 10 known
  convergence patterns without subversion (reconciliation arc, grief
  meditation, chosen one, AI consciousness, sanitized conflict, epistolary
  revelation, time loop lesson, magical realism metaphor, moral clarity,
  small-town secret)

---

## Operator Integration

- Each operator queries the bank by matching `type`. The Collision operator
  pulls `collision_pair` seeds. The Thought Experiment operator pulls `axiom`
  seeds. And so on.
- **Graceful fallback.** If no seeds of the needed type exist, the operator
  falls back to pure LLM generation. Seeds are enrichment, not hard dependency.
- **Type-diversity weighting.** Prefer seed types that are underrepresented in
  the current run's population. If recent concepts are heavy on `real_world`
  seeds, weight toward `axiom` or `constraint` or `image`.
- **Steering-based tag filter.** When the run config includes a `steering`
  prompt, pre-filter seeds by `tags` relevance to the steering direction. Fall
  back to unfiltered random if no tags match. See `docs/stage-1/overview.md`
  for the full steering mechanism.

---

## Refresh Strategy

- **Manual curation** for now. Add seeds when interesting material surfaces.
- **File location:** `data/seed-bank.yaml`
- **Growth target:** ~100 seeds within the first months of operation.
- **Balance:** No single type should exceed 25% of the bank.
- **Pruning:** Periodically remove seeds that consistently produce low-scoring
  concepts (tracked in the run database, not in the seed file). Seeds that
  spawned high-scoring concepts can be retired too — they've done their work.

---

## Curated Starter Set

61 seeds across all 10 types, in `data/seed-bank.yaml`. Every seed should make
someone pause and think "oh, that's interesting." No filler.

| Type | Count | Notes |
|------|-------|-------|
| `real_world` | 21 | Verified facts with sources cited |
| `thought_experiment` | 5 | Scenarios that dramatize philosophical questions (Jackson, Nozick, Parfit, Davidson) |
| `axiom` | 5 | Bare philosophical positions (Clark & Chalmers, Ubuntu, Williams, Weil, Parfit) |
| `dilemma` | 5 | Impossible choices with no clean resolution |
| `constraint` | 5 | Formally precise, generative constraints |
| `noun_cluster` | 4 | 6-8 emotionally resonant nouns per cluster |
| `image` | 4 | Specific evocative scenes with incongruous elements |
| `compression` | 4 | Dense thematic kernels |
| `collision_pair` | 4 | Semantically distant, emotionally connectable pairs |
| `anti_target` | 4 | Named convergence patterns with specific subversion angles |