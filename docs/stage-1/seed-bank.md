# Stage 1: Seed Bank

The seed bank is the primary external input to Stage 1. It provides concrete starting material so operators don't rely purely on LLM latent knowledge — which tends toward convergent, overexposed ideas. Each seed type maps to one or more mutation operators, giving different creative methods different fuel.

Seeds live in `data/seed-bank.yaml`. This document specifies the schema, what makes a good seed, per-type criteria, and maintenance strategy.

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

- **`content` is polymorphic by type.** Most types use a string. `noun_cluster` uses a list of 6-8 nouns. `collision_pair` uses a list of exactly two items.
- **`source` is optional.** Required for `real_world` (must be verifiable). Recommended for `thought_experiment` and `axiom` (attribution matters). Unnecessary for other types.
- **No runtime tracking fields.** Usage counts, fitness associations, and staleness tracking belong in the run database, not in the seed file. The seed bank is a static curated resource.

---

## What Makes a Good Seed

### The literary critic pause test

The single best heuristic: when a knowledgeable reader encounters the seed, do they mentally pause? Not a dramatic pause — a *thinking* pause. "Huh, that's interesting. I haven't encountered that framing before." If instead the reader mentally reaches for a comparison ("that's just like [famous story]"), the seed is too overexposed.

### Universal criteria (all types)

Every seed must satisfy all three:

1. **Specific enough to anchor a story.** Not a broad topic ("grief") but a concrete starting point that implies characters, stakes, and setting. A good seed makes you immediately visualize a specific situation, not a category of situations.

2. **Surprising enough to resist LLM default convergence.** LLMs have strong patterns for story generation baked in from training data. A seed must be orthogonal to those patterns. If the seed would naturally produce a story you've read a hundred times, it's too familiar. The 10 known convergence patterns (reconciliation arc, grief meditation, chosen one, AI consciousness, sanitized conflict, epistolary revelation, time loop lesson, magical realism metaphor, moral clarity, small-town secret) are the primary traps.

3. **Human implications.** Not just a cool fact or clever constraint — there must be something at stake for people. How does this change what it means to be a person, to love someone, to make a choice? The seed must touch something that matters.

### What the evaluation system rewards

Seeds are scored indirectly — the concepts they generate are evaluated on 9 dimensions. A good seed naturally leads to concepts that score well on:

- **Originality** — the seed avoids convergence patterns
- **Emotional potential** — the seed implies emotional stakes
- **Narrative tension** — the seed supports suspense, curiosity, or surprise
- **Anti-cliché score** — the seed is orthogonal to LLM training defaults
- **Generative fertility** — the seed suggests multiple stories, not just one

Seeds that are too generic will produce concepts that score poorly. Seeds that are too narrow will produce one good concept and then exhaust themselves.

### The overexposure problem

LLMs flatten what they've seen thousands of times. A thought experiment that appears in every intro philosophy textbook, a fact that shows up in fun-facts listicles, a constraint that's been done in every MFA program — these are pre-flattened. The LLM already has strong defaults for them, which means the seed can't do its job of pushing generation into unfamiliar territory.

When evaluating a candidate seed, ask: how much of this has the LLM already seen? If the answer is "a lot," the seed needs to be either more specific (a less-known angle on a known idea) or replaced with something genuinely underexposed.

---

## Seed Types

### `thought_experiment`

**What it is:** A scenario that dramatizes a philosophical or scientific question. Not a claim — a concrete situation with characters, stakes, and a question that has no clean answer.

**Feeds:** Thought Experiment / Le Guin (operator #3)

**Content format:** String. The scenario described concretely, ending with or implying the question it raises.

**What makes a good one:**

The key insight from studying thought-experiment fiction (Borges, Le Guin, Chiang, Kafka, Vasubandhu, Zhuangzi) is the distinction between *illustration* and *genuine dramatization*. In illustration, the narrative can be replaced by the explicit argument without loss — the story is just a vehicle. In genuine dramatization, the narrative does irreducible work. The story couldn't exist as an argument.

The structural features that make a thought experiment work as a fiction seed:

1. **The scenario must be halfway to fiction already.** It should imply characters, a setting, and a tension. "Is perception reliable?" is a philosophical question. "A ghost, a human, and a fish approach the same river and each experiences a completely different world" is a scenario.

2. **The question must be genuinely unresolvable.** Not just hard but structurally ambiguous. The Grand Inquisitor's speech has no rebuttal — Christ just kisses him. That silence is the fiction's answer. A thought experiment with a clean solution is a puzzle, not a seed.

3. **It must carry an irreplaceable emotional payload.** The scenario should make you *feel* the philosophical stakes before you can articulate them. Mencius's child-at-the-well forces an intuitive moral response before the reader can theorize it. Mary's Room makes you feel the gap between knowing everything about red and seeing red.

4. **Low exposure in training data matters more for this type than any other.** LLMs have strong defaults for famous thought experiments (trolley problem, Chinese room, brain in a vat, Ship of Theseus). These are pre-flattened. The bank should draw from underexposed traditions: non-Western philosophy (Vasubandhu, Mozi, Gongsun Long, Al-Ghazali), underexposed Western philosophy (Parfit's combined spectrum, not his teleporter; Jackson's Fred, not Mary), and science-derived scenarios (Boltzmann brains, split-brain research) that are known in their home disciplines but rare as fiction seeds.

5. **It should suggest multiple possible stories, not just one.** A thought experiment that implies a single narrative is a plot, not a seed. A thought experiment that implies a dozen possible stories — each taking the scenario in a different direction — is generative.

---

### `constraint`

**What it is:** A formal structural, linguistic, content, or POV constraint that shapes how a story can be told.

**Feeds:** Constraint-First Generation / Oulipo (operator #9)

**Content format:** String. The constraint stated precisely.

**What makes a good one:**

There are two categories of constraint, and the bank should contain both:

**Structural constraints (Oulipo-style):** Formally precise rules about what the prose can and cannot do. "No sentence may exceed seven words." "Each section is half the word count of the previous." "The narrator can only describe what the protagonist is NOT doing." These must be implementable (not vague), and generative (they create stories, not just restrict them). The best structural constraints make you immediately wonder what story would *need* this form.

**Format constraints (format-as-voice):** The story is presented in a non-literary format that carries cultural, social, or epistemological context. Greentext, incident report, ritual instructions, trial transcript, hagiography, almanac, herbalist's compendium. These are distinct from structural constraints because the container implies a worldview — a relationship to knowledge, authority, truth, or audience. The format forces the story into an alien shape, and the tension between the story's emotional content and the format's conventions is where the fiction lives.

The strongest format constraints carry an entire epistemology. Ritual instructions imply a cosmology where the ritual is necessary. A trial transcript from an ecclesiastical court implies a world where "did you consort with the devil" is a legally meaningful question. An almanac implies a cyclical relationship to time that industrial modernity has lost. The more alien the epistemology, the harder it is for the LLM to fall back on trained defaults — which is exactly the point.

Research finding (Damadzic meta-analysis of 111 studies): constraint timing matters. Constraints injected at generation-start are more generative than constraints added mid-story. Seeds are injected early, so this works in the pipeline's favor.

Research finding (Amabile et al.): controlling framing ("follow these steps precisely") produces less creative output than informational framing ("explore this space"). Constraint seeds should frame the constraint as a generative puzzle, not a restriction.

---

### `real_world`

**What it is:** A true, specific, surprising fact — a historical event, scientific finding, or cultural phenomenon.

**Feeds:** Real-World Seed Injection (operator #11)

**Content format:** String. 1-2 sentences stating the fact. `source` field required.

**What makes a good one:** Must be TRUE and citable. Specific enough to anchor a story (not "climate change" but a particular village relocating). Surprising enough that the reader pauses. Suggests human implications beyond the fact itself. Not overexposed in LLM training data — if it shows up in fun-facts listicles, dig deeper.

---

### `axiom`

**What it is:** A bare philosophical, scientific, or ethical position — not a scenario, but an arguable claim about reality, consciousness, identity, or how people should live.

**Feeds:** Thought Experiment / Le Guin (operator #3)

**Content format:** String. The position stated clearly enough to be inverted, extended, or transplanted. `source` field recommended.

**What makes a good one:** Must be a real position someone has defended, not a strawman. Should be arguable from multiple sides. Should have emotional stakes — not just an intellectual puzzle but something that changes how it feels to be alive if true.

---

### `dilemma`

**What it is:** A specific situation where two legitimate values conflict with no clean resolution. Not an abstract ethical puzzle — a concrete scenario with characters who face an impossible choice.

**Feeds:** Thought Experiment / Le Guin (operator #3), Collision / King (operator #1), Compression / Borges (operator #8)

**Content format:** String. The situation described with enough specificity that the bind is felt, not just understood.

**What makes a good one:** Both sides must be genuinely defensible. The situation must be specific enough to imply characters and stakes. Should resist the "moral clarity" convergence pattern — the whole point is that there's no right answer. The best dilemmas make you change your mind twice while reading them.

---

### `noun_cluster`

**What it is:** A list of 6-8 concrete, emotionally charged nouns grouped by emotional resonance rather than semantic category.

**Feeds:** Noun-List Recombination / Bradbury (operator #2)

**Content format:** List of 6-8 strings, each prefixed with "the".

**What makes a good one:** Nouns must be concrete — objects, places, sensations, not abstractions. Each noun is preceded by THE (following Bradbury's practice), which converts a category into a specific memory — THE CREEK is not creeks in general but the one that mattered. The cluster's coherence should be felt, not explained — dredged, not curated. A good cluster makes you sense a story without being able to articulate why those words belong together.

The worst clusters are thesis statements — a set of nouns assembled to make a point about a topic (e.g., all digital-age artifacts to make a point about modern alienation). The best clusters feel found, not argued.

---

### `image`

**What it is:** A specific evocative scene — a starting image for associative generation.

**Feeds:** Discovery Mode / Murakami (operator #7)

**Content format:** String. A scene described in 1-2 sentences.

**What makes a good one:** Must be a specific scene, not a vague mood. Should contain at least one surprising or incongruous element that creates productive tension. Should resist reduction to a single interpretation — the image should be richer than any single explanation of it. The best images make you want to know what happened before and after, without giving you enough information to be sure.

---

### `compression`

**What it is:** A dense thematic kernel that implies an entire story world — what a story is ABOUT, not what happens in it.

**Feeds:** Compression / Borges (operator #8)

**Content format:** String. 1-3 sentences.

**What makes a good one:** Dense enough to imply characters, setting, and emotional stakes without naming them. Should suggest theme, not plot. Should feel like the last line of a review of a story you desperately want to read.

---

### `collision_pair`

**What it is:** Two semantically distant items that could produce a story in the interference pattern between them.

**Feeds:** Collision / King (operator #1)

**Content format:** List of exactly two strings.

**What makes a good one:** The two items must be semantically distant but emotionally or thematically connectable. The connection should not be obvious — if you can immediately see the story, the pair is too close. If you can't imagine any connection at all, they're too far. The sweet spot is where you think "these shouldn't go together, but I can feel why they might." The creative work is in *finding* the connection, not executing a known pattern.

---

### `anti_target`

**What it is:** A named convergence pattern (one of the 10 known LLM clichés) paired with a specific subversion angle.

**Feeds:** Anti-Premise Generation (operator #10)

**Content format:** String. Names the pattern and states the subversion angle.

**What makes a good one:** Must name a specific convergence pattern, not a vague complaint about AI writing. The subversion angle must be specific and generative — not just "do the opposite" but a particular way into the same thematic territory that the cliché forecloses. The subversion should suggest stories, not just negate a pattern.

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

## Content Filtering

Exclude seeds that are themselves clichéd or would naturally produce one of the 10 known convergence patterns without subversion (reconciliation arc, grief meditation, chosen one, AI consciousness, sanitized conflict, epistolary revelation, time loop lesson, magical realism metaphor, moral clarity, small-town secret).

---

## Operator Integration

- Each operator queries the bank by matching `type`. The Collision operator pulls `collision_pair` seeds. The Thought Experiment operator pulls `axiom` seeds. And so on.
- **Graceful fallback.** If no seeds of the needed type exist, the operator falls back to pure LLM generation. Seeds are enrichment, not hard dependency.
- **Type-diversity weighting.** Prefer seed types that are underrepresented in the current run's population. If recent concepts are heavy on `real_world` seeds, weight toward `axiom` or `constraint` or `image`.
- **Steering-based tag filter.** When the run config includes a `steering` prompt, pre-filter seeds by `tags` relevance to the steering direction. Fall back to unfiltered random if no tags match. See `docs/stage-1/overview.md` for the full steering mechanism.

---

## Refresh Strategy

- **Manual curation** for now. Add seeds when interesting material surfaces.
- **File location:** `data/seed-bank.yaml`
- **Balance:** No single type should exceed 25% of the bank.
- **Pruning:** Periodically remove seeds that consistently produce low-scoring concepts (tracked in the run database, not in the seed file). Seeds that spawned high-scoring concepts can be retired too — they've done their work.
