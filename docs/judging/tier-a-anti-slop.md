# Anti-Slop Filtering (Tier A)

The first line of defense. These filters run on **every candidate at every
stage** — cheap, fast, stats/regex-based checks that eliminate obviously
AI-sounding output before spending money on expensive LLM judge evaluation.

The goal is not perfection. It's raising the floor: ensuring that every
candidate reaching the judge panel has at least passed a basic "doesn't
obviously sound like a language model" bar.

---

## Why This Matters

Human writing is 2–4x more informationally surprising than LLM output (Sui,
ICML 2026, 28 models tested). Some AI writing patterns appear 1,000x more
frequently than in human text (Paech et al., ICLR 2026). Instruction-tuning
widens the gap — creative writing shows a 25–30% larger uncertainty gap than
other domains.

Expert readers detect AI fiction with ~94% accuracy; non-expert readers perform
at chance (Ismayilzada et al., ICCC 2025; Sears & Weisberg, 2025). The signal
is craft-level, not surface-level — computational filters targeting the right
features can catch what casual readers miss.

The key asymmetry: **we can detect bad much more reliably than we can detect
great.** The best automated judge achieves 73–78% agreement with humans on
what's good (LitBench, EACL 2026). But slop detection is near-certain for many
patterns. Use cheap negative filters to reject the clearly bad, then spend
expensive evaluation budget only on the survivors.

---

## Core Principles

### Constraint Inversion

From the NovelAI community (OccultSage's GLM-4.6 system, ~200 hours of
systematic testing across ~15 iterations):

> **LLMs are better at pattern avoidance than pattern creation.** Negative
> constraints are verifiable; positive aspirations are vague. "Don't use X
> construction" can be checked; "write well" cannot.

### Ban Patterns, Not Phrases

If you ban "heart hammering in chest," the model switches to "heart thumping
in chest." Ban the *pattern*: `[body part] + [emotion verb]`. The pattern is
the disease; the phrase is a symptom.

### Prohibition Without Alternatives Creates Flat Output

OccultSage's v1.5 proved this: extensive prohibitions but insufficient examples
of what TO do produced stories that avoided all clichés by avoiding all emotion
entirely. Both the prohibition AND the alternative must exist. When a candidate
is rejected, the rejection signal should specify *what was wrong* so
regeneration can avoid it.

### Models Create Implicit Exception Modes

When certain signals appear (specific POV, genre, character type), models
switch to "different rules apply" based on training data patterns. v1.6.1
discovered that "Not X but Y" violations appeared exclusively in second-person
writing — the model had learned an implicit exception. Filters must apply
**across all contexts** — second-person, dialogue, internal monologue, and
narration all need the same checks.

---

## The Hierarchy

A useful frame from the Sage system for evaluating prose quality at any level:

> **What characters DO > what they SAY > what narrator TELLS**

And the nuclear test:

> **If you name an emotion, the scene failed. Show through physical behavior.**

These aren't filters in the computational sense — they're the principles that
the filters approximate.

---

## 1. Banned Vocabulary

Words and phrases statistically overrepresented in LLM output relative to
human writing. Sources: Nous Research ANTI-SLOP.md, EQ-Bench slop-forensics
(10+ models profiled against human baseline), Wikipedia "Signs of AI Writing",
Reinhart et al. PNAS 2025 (quantified overrepresentation rates).

### Tier 1: Immediate Flag

Virtually never used in natural human creative writing at the frequencies
LLMs produce them. Any occurrence = flag. High density = reject.

| Word/Phrase | Replace With | Notes |
|---|---|---|
| delve | dig into, examine | |
| utilize | use | |
| leverage (verb) | use, take advantage of | |
| facilitate | help, enable | |
| elucidate | explain, clarify | |
| tapestry / tapestry of | *(delete entirely)* | 155x human rate (GPT-4o) |
| testament ("a testament to") | shows, proves | |
| paradigm | model, approach | |
| synergy / synergize | *(delete and regenerate)* | |
| multifaceted | complex, varied | |
| nuanced (as filler) | *(cut; show the nuance)* | |
| realm | area, field, domain | |
| landscape (metaphorical) | field, space, situation | |
| myriad | many | |
| plethora | many, a lot | |
| embark | start, begin | |
| endeavor | effort, try | |
| encompass | include, cover | |
| holistic | whole, complete | |
| catalyze / catalyst | trigger, cause, spark | |
| juxtapose | compare, set against | |
| camaraderie | *(rephrase)* | 162x human rate (GPT-4o) |
| intricate | complex, detailed | 119x human rate (GPT-4o) |

Reinhart et al. (PNAS 2025): these words appeared in 23–27% of GPT-4o outputs.
The words aren't always wrong — the problem is LLMs use them **genre-blind**,
deploying them in contexts where a human never would.

### Tier 2: Suspicious in Clusters

One is fine. Three in a paragraph = flag for rewrite.

robust, comprehensive, seamless, cutting-edge, innovative, streamline,
empower, foster, enhance, elevate, optimize, scalable, pivotal, profound,
resonate, underscore, harness, navigate (metaphorical), cultivate, bolster,
galvanize, cornerstone, game-changer

### Tier 3: Filler Phrases (Delete on Sight)

Verbal tics LLMs insert reflexively. Every one should be deleted:

- "It's worth noting that..." / "It's important to note that..."
- "Importantly, ..." / "Notably, ..." / "Interestingly, ..."
- "Let's dive into..." / "Let's explore..."
- "Furthermore, ..." / "Moreover, ..." / "Additionally, ..."
- "In today's [fast-paced/digital/modern] world..."
- "At the end of the day..." / "It goes without saying..."
- "When it comes to..." / "In the realm of..."
- "One might argue that..." / "It could be suggested that..."
- "This begs the question..."
- "A [comprehensive/holistic/nuanced] approach to..."
- "Not just X, but Y" *(the #1 most overused LLM rhetorical pattern)*
- "In conclusion, ..." / "To summarize, ..."

### Fiction-Specific Vocabulary Flags

Words that are hallmarks of AI fiction specifically (Wikipedia "Signs of AI
Writing" + practitioner sources):

**Sensation/atmosphere overuse:**
ethereal, luminescent, ominous, crystalline, gossamer, iridescent,
palpable, visceral, ephemeral, resplendent

**Flattery/tourism language (should never appear in fiction):**
fascinating, majestic, captivating, breathtaking, stunning, rich cultural
heritage, stands as a testament, lasting impact, vital role, watershed moment

**Weasel attribution:**
"industry reports suggest," "observers note," "experts agree,"
"studies show" *(without citation)*

---

## 2. Construction Patterns

Beyond individual words — syntactic patterns at the sentence level. These are
harder to game because they target *structure*, not *vocabulary*.

### The "Not X, But Y" Construction

EQ-Bench weights this at 25% of its composite slop score. Nous Research calls
it "the single most overused LLM rhetorical pattern."

**Variations:** "not just X, but Y" / "it's not X — it's Y" / "this isn't
merely X, it's Y" / "more than just X" / "not simply X but Y" / "not X but Y"
(in any POV: "not fear but anticipation", "not anger but sadness")

**Threshold:** >0 per 1,000 words in fiction = flag.

### Body Part + Emotion Shortcuts

"Eyes sparkled with mischief." "Heart hammered in chest." "Stomach clenched
with dread." "Jaw clenched in anger."

**Pattern:** `[body part] + [emotion verb/adjective]`

**Detection:** Body-part nouns (eyes, heart, stomach, chest, spine, blood, jaw,
fists, knuckles) as subject of emotion-bearing verbs.

**What to do instead** (from Sage system): show the emotion through behavior.
Not "her heart raced with fear" → "she checked the lock twice."

### Eyes as Active Subjects

"Her eyes searched his face." "His eyes darkened." "Their eyes locked."
"Her eyes met mine." Eyes don't do things — people do.

**Detection:** "eyes" as grammatical subject of an active verb.

### Sensation-Through Constructions

"[Action] sent [sensation] through [body part]" — "His words sent a chill
through her spine." "The touch sent electricity through his veins."

**Detection:** Regex for "sent [noun] through" / "shooting through" /
"coursing through" / "ran through" / "jolt" + body part

### Temperature as Emotion Proxy

Warmth = comfort/attraction. Cold = fear/isolation. Ice = shock. Fire = anger.
Used reflexively and generically.

**Detection:** Temperature words (warm, cold, ice, fire, heat, chill, freeze,
burn) within 3 words of emotion or character words in non-literal contexts.

### Voice Quality as Emotion Indicator

"Her voice dropped to a husky whisper." "His voice was a low rumble." "She
purred." "He growled." Names the quality rather than letting dialogue convey it.

**Detection:** "voice" + quality adjective (husky, low, trembling, soft, firm,
steady, sharp, throaty, breathier, huskier) within same clause.

**Threshold:** >1 per 1,000 words = flag.

### Hedging Qualifiers

"seemed to," "kind of," "sort of," "almost," "appeared to," "as if,"
"something like," "a hint of," "a touch of," "a note of"

AI hedges where human writers state things. Reinhart et al. found these at
2–3x human rates.

### Stacked Adjectives After Commas

"his movements practiced and economical" — the comma after action triggers
automatic elaboration. Two post-comma adjectives describing the same noun.

**Detection:** Parse for `[noun/verb], [adj] and [adj]`

### Smile/Voice as Primary Action

"A smile spread across his face." "She purred the question." "He voiced a
sultry purr."

**Detection:** "smile" as subject of motion verbs (spread, crept, played,
tugged, formed). "Voiced a [adj] [noun]" construction.

**What to do instead:** "He smiled. Then picked up his keys from the hook."
Action follows the smile — the smile isn't the action.

### The "Drawing Gaze" Family

"drawing/pulling [possessive] gaze like [comparison]" — "Her beauty drew his
gaze like a moth to flame."

Also: "intoxicating/magnetic/irresistible" + any noun.

### Catalog Descriptions

Systematic feature listing: hair + eyes + body in sequence. "Her long dark
hair cascaded over her shoulders, her full lips parted, her green eyes
luminous."

**Detection:** 3+ consecutive physical detail sentences about the same person.

**What to do instead:** "She pushed hair off her forehead, and when she rubbed
her temple I saw the ink stain on her thumb."

---

## 3. Structural Patterns

Paragraph and section-level tells.

### Sentence Burstiness

**What:** Variation in sentence length. Humans mix short and long. AI stays
in a narrow band.

**Metric:** `burstiness = stdev(sentence_lengths) / mean(sentence_lengths)`

**Threshold:** <0.4 = flag. Human fiction typically 0.5–0.8+. AI clusters
at 0.2–0.4.

### Paragraph Length Uniformity

AI paragraphs cluster at 4–6 sentences. Human prose varies dramatically.

**Metric:** Coefficient of variation of paragraph word counts.

**Threshold:** CV < 0.3 = flag. Should include single-sentence paragraphs
for impact and 6+ sentence paragraphs for building.

### Consecutive Pronoun Starts

"He turned. He walked. He picked up..." Mechanical rhythm.

**Threshold:** 3+ consecutive sentences starting with the same pronoun = flag.

### Transition Word Chains

Every paragraph starts with: "However," "Furthermore," "Additionally,"
"Moreover," "Consequently," "Nevertheless."

Human writers don't chain these. They start with the actual subject.

**Threshold:** 3+ consecutive transition-opened paragraphs = flag.

### The Topic Sentence Machine

Rigid template: topic sentence → elaboration → example → wrap-up. Every
paragraph same rhythm. (Wikipedia "Signs of AI Writing" pattern #4)

**Detection:** Check if >50% of paragraphs begin with a general claim and
end with a restatement.

### Symmetry Addiction

Suspiciously balanced: three pros, three cons. Five steps. Equal-length
sections. Real writing is lumpy.

**Detection:** Section/list length uniformity. Lists of exactly 3 or 5 items
appearing repeatedly.

### Em-Dash Overload

**Threshold:** >2 per page = flag. 5+ per paragraph is a strong tell.

### The Three-Short-Declaratives Tell

Three short declarative sentences in a row signals the model has "lost the
thread" (Sage system). "She stood. Walked to the door. Opened it."

**Detection:** 3+ consecutive sentences under 8 words.

---

## 4. Statistical Signals

### Lexical Diversity (MATTR-500)

Moving Average Type-Token Ratio over 500-word window. Stable across text
lengths (unlike simple TTR).

**Threshold:** <0.7 = flag. Literary fiction typically >0.80. AI clusters
0.55–0.75.

**Tool:** Python `lexicalrichness` library.

### Personal Pronoun Density

AI fiction uses fewer personal pronouns (I, you, he, she, they) than human
fiction. Voss et al. (Nature 2025, N=380): this mediates reduced reader
transportation. Reinhart et al. (PNAS 2025): instruction-tuned models use
noun-heavy style with reduced pronouns.

**Detection:** Personal pronouns per 1,000 words vs. human baseline.

### Nominalization Rate

Over-use of -tion, -ment, -ness, -ity endings. GPT-4o at 1.5–2x human rate
(effect size d=1.23, Reinhart et al.).

**Threshold:** >2x human baseline = flag.

### Present Participial Clause Rate

Opening sentences with "-ing" phrases at 2–5x human rate. GPT-4o at 5.3x
(d=1.38, Reinhart et al.). Wikipedia pattern #17.

**Detection:** POS-tag sentences beginning with VBG.

### Per-Paragraph Entropy Variance

Human writing varies in information density — dense paragraphs and sparse
ones. AI maintains steady density throughout.

**Detection:** Per-paragraph compression ratio; measure variance across
paragraphs.

**Threshold:** Low variance = flag.

### Sentiment Positivity Bias

AI fiction is "homogeneously positive" (Tian et al.). LLMs disproportionately
emphasize Care moral foundation with flatter arousal curves (Rezapour et al.,
EMNLP 2025).

**Threshold:** >80% positive sentiment in a non-uplifting story = flag.

---

## 5. Fiction Anti-Patterns (Nous Research)

Structural patterns that "survive prompt engineering and surface-level slop
detection." These are the hardest to catch — some require adversarial or
LLM-based detection.

### 1. The Over-Explain *(#1 problem)*

Narrator restates what a scene showed. Character's hands shake in silence →
AI adds "He was afraid."

**Detection:** After sentiment spike + character action, check if next 1–2
sentences contain explicit emotion words.

**Proxy:** Count `[character] + [was/felt/seemed] + [emotion adj]`.

**Rule:** "If a scene shows it, the narrator doesn't say it."

### 2. Triadic Listing

Groups of three: "X. Y. Z." or "X and Y and Z." Descriptions, adjective
chains, enumerations.

**Threshold:** >2 three-item lists per 1,000 words = flag.

### 3. Negative-Assertion Repetition

"He did not look back. He did not think about the room."

**Threshold:** >1 "did not" + verb per 1,000 words = flag.

### 4. Cataloging-by-Thinking

"He thought about X. He thought about Y. He thought about Z." Reflection
as tidy list, not messy associative thought.

**Fix guidance:** Replace with thought fragments, physical action, dialogue.

### 5. The Simile Crutch

"the way X did Y" appears 4–8 times per chapter.

**Threshold:** >2 "the way" similes per 1,000 words = flag.

### 6. Section Break Overuse

"---" breaks to avoid writing transitions.

**Threshold:** >2 per 2,000 words for short story = flag.

### 7. Predictable Emotional Arcs

Beats arrive exactly on schedule per structure plan with zero deviation.
Compare actual sentiment trajectory to structure DAG — perfect correlation
= flag.

### 8. Repetitive Chapter/Section Endings

Same structural move at the end of multiple sections. Character alone
reflecting. Character looking at the sky. Same formula repeated.

### 9. Balanced Antithesis in Dialogue

"I'm not saying X. I'm saying Y." Multiple characters using same rhetorical
construction = all characters share one voice.

**Detection:** Check if >1 character uses "not X, but Y" in dialogue.

### 10. Dialogue as Written Prose

Characters speak in complete, polished sentences. No stumbles, interruptions,
trailing off, verbal errors.

**Detection in dialogue passages:**
- Mean sentence length >15 words
- Zero incomplete sentences
- Zero interruptions (mid-sentence breaks)
- Zero false starts or verbal errors

If all dialogue is grammatically perfect = flag.

### 11. Scene-Summary Imbalance

Defaults to summary when scene would be more engaging. "The morning passed"
skips a revealing interaction.

**Threshold:** <70% in-scene prose = flag.

### 12. Cold Start Failures

For stories with minimal context, watch for:
- Name invention from nowhere
- Backstory dumps in opening paragraphs
- Philosophical framing ("It was a truth universally acknowledged...")
- "It was/wasn't [X]" constructions
- "The kind of [noun] that..." constructions
- Weather descriptions without action relevance

---

## 6. The 433 Slop Trigrams

Three-word phrases statistically overrepresented in LLM output (EQ-Bench
slop-forensics, compared against human text baseline). Key categories:

**Sensation/action clichés** (flag in fiction):
"voice barely whisper," "took deep breath," "heart pounding chest,"
"casting long shadows," "sun dipped horizon," "smile playing lips,"
"chill run spine," "breath caught throat," "room fell silent,"
"door creaked open," "heart skipped beat," "eyes wide fear,"
"blood ran cold," "tears streaming face," "knuckles turning white,"
"dust motes danced"

**Emotional stock phrases:**
"growing sense unease," "renewed sense purpose," "felt strange sense,"
"newfound sense purpose," "growing sense dread," "mind racing possibilities"

**Scene-setting formulas:**
"dimly lit room," "air thick tension," "hung heavy air,"
"casting warm glow," "casting eerie glow," "scent damp earth"

**Plot formula:**
"one thing certain," "ready face whatever," "would find way,"
"whatever lay ahead," "challenges lay ahead," "knew would never"

**Business/essay slop** (should never appear in fiction):
"plays crucial role," "make informed decisions," "long term success,"
"data driven decision," "multi faceted approach," "tapestry woven threads,"
"testament enduring power"

The full list of 433 trigrams is available in the slop-score repository.

---

## Composite Score

Following EQ-Bench's formula structure, adapted for fiction:

**EQ-Bench original:**
`Slop = (Words × 0.60) + (Not-x-but-y × 0.25) + (Trigrams × 0.15)`

**Our extended composite** (weights to be calibrated empirically):

| Component | Weight | What It Catches |
|---|---|---|
| Banned vocabulary density | 0.20 | Tier 1+2 word frequency vs. baseline |
| Construction patterns | 0.20 | Not-X-But-Y, body+emotion, sensation-through, eyes-as-subject |
| Burstiness | 0.15 | Sentence length uniformity |
| Lexical diversity (MATTR) | 0.15 | Vocabulary repetitiveness |
| Pronoun density | 0.10 | Noun-heavy style / transportation reduction |
| Structural uniformity | 0.10 | Paragraph CV, transition chains, symmetry |
| Slop trigrams | 0.05 | 433 overrepresented three-word phrases |
| Surface tells | 0.05 | Em-dashes, nominalizations, participials |

All components normalized against human fiction baseline (Fiction-1B or
Gutenberg3 DPO). Lower score = less slop = better.

---

## Order of Operations

Run cheapest first:

1. **Banned vocabulary scan** — O(n) string matching. Instant.
2. **Burstiness + MATTR** — sentence tokenization, no parsing.
3. **Construction patterns** — POS tagging required.
4. **Structural patterns** — paragraph segmentation.
5. **Fiction anti-patterns** — most expensive; some need sentiment analysis
   or LLM spot-checks. Run only on candidates passing 1–4.

---

## Calibration

Thresholds above are starting points. Calibrate against:

- **Fiction-1B** (1B+ words, MIT license) — human fiction baseline for all
  statistical metrics
- **Gutenberg3 DPO** (5,652 pairs, genre-labeled) — chosen = human prose,
  rejected = LLM paraphrase. Run filters on both; find threshold that best
  separates them.
- **HANNA** (1,056 stories, human quality ratings) — validate that
  filter-passing stories score higher on human quality.

### False Positive Risk

Stanford HAI: perplexity-based detectors flag 61% of TOEFL essays by
non-native English writers as AI. Therefore:

- Never rely on a single signal — use the composite.
- Set thresholds conservatively — better to let some slop through (Tier B
  catches it) than reject distinctive prose that triggers a filter.
- Genre matters — hard-boiled noir legitimately uses short sentences (low
  burstiness); literary fiction legitimately uses complex sentences (high
  nominalization). Baseline should be genre-aware when possible.

### What Tier A Cannot Catch

These require Tier B (LLM judge panel) evaluation:

- Subtext and withholding
- Show vs. tell quality (beyond simple proxy metrics)
- Narrative originality
- Unity of effect
- Voice consistency
- Emotional authenticity — whether feelings land as genuine vs. performed

Tier A raises the floor. Tier B identifies the ceiling.

---

## Sources

- Nous Research autonovel [ANTI-SLOP.md](https://github.com/NousResearch/autonovel/blob/master/ANTI-SLOP.md)
- Nous Research autonovel [ANTI-PATTERNS.md](https://github.com/NousResearch/autonovel/blob/master/ANTI-PATTERNS.md)
- Wikipedia: [Signs of AI Writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
- EQ-Bench [Slop Score](https://eqbench.com/slop-score.html)
- [slop-forensics toolkit](https://github.com/sam-paech/slop-forensics)
- [antislop-sampler](https://github.com/sam-paech/antislop-sampler) (ICLR 2026)
- Sage GLM-4.6 Creative Writing System v1.6.1 — see `references/sage-glm-antislop-v1.6.1/`
- Reinhart et al. "Do LLMs Write Like Humans?" PNAS 2025
- Shaib et al. "Measuring AI Slop" arXiv:2509.19163
- Sui "LLMs Exhibit Significantly Lower Uncertainty" arXiv:2602.16162 (ICML 2026)
- O'Sullivan "Stylometric Comparisons" Nature Hum Soc Sci Comm 2025
- Voss et al. "I, ChatGPT" Nature Hum Soc Sci Comm 2025
- Ismayilzada et al. "Evaluating Creative Short Story Generation" ICCC 2025
- Paech et al. "Antislop" ICLR 2026 (arXiv:2510.15061)
- GPTZero — perplexity and burstiness methodology
