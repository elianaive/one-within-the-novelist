# Rubric Anchors: 9 Stage 1 Concept Evaluation Dimensions

Full score-level descriptions for all 9 concept evaluation dimensions. Judges
receive these anchors in their evaluation prompts. Each dimension has explicit
descriptions for scores 1 through 5.

These anchors evaluate **potential**, not execution. The question is always
"could a great story be written from this concept?" — never "is this
well-written?" The concepts being evaluated are short JSON structures (premise,
target_effect, character_seeds, thematic_tension, constraints), not prose.

Anchors are grounded in cognitive science research (cited per dimension) and
follow Prometheus 2's rubric format, which achieves 0.897 Pearson correlation
with human evaluators when using criterion-specific per-score descriptions.

---

## 1. Originality

*Has this premise been done before? Would an LLM naturally converge on it?*

Grounded in Echoes-in-AI (Xu et al., PNAS 2025): LLM plots converge at 6-7x
the rate of human plots. AI uncertainty gap (Sui, ICML 2026): human writing is
2-4x more informationally surprising.

| Score | Description |
|-------|-------------|
| **1** | The premise is a well-known trope executed without subversion. "A detective solves a murder." "A young person discovers they have magical powers." "An AI becomes sentient and questions its existence." If you gave this premise to five different LLMs, they would all produce essentially the same story. Sui Generis score would be in the 6-7 range — maximum convergence. |
| **2** | Familiar territory with minor variation. The core shape is recognizable — estranged parent reconciles with child, protagonist learns to accept loss — but there's one specific element that isn't default. The variation feels like a coat of paint on a known structure rather than a genuine departure. An LLM could easily generate this with light prompting. |
| **3** | A fresh angle on familiar territory, or a combination of known elements in an unusual way. The core idea isn't new, but the specific execution suggests novelty. "What if the grief meditation were told from the perspective of the dead person's coworker who barely knew them?" — recognizable emotional territory, but the entry point is unexpected. Multiple LLMs would produce noticeably different stories from this premise. |
| **4** | The premise surprises. The evaluator pauses — this combination of elements is unexpected. It might subvert a known pattern in a way that creates genuine tension (not just inverting the trope, but finding an angle the trope didn't know it had). Or it might occupy territory that doesn't map cleanly to existing categories. The evaluator has to think about what story this would produce, rather than immediately knowing. |
| **5** | The premise changes how the evaluator thinks about what's possible. This specific combination of elements, angle, or framing hasn't been encountered before — not in published fiction, not in AI-generated fiction. Reading this concept produces the feeling of discovery. Like encountering Borges's "The Library of Babel" or Chiang's "Story of Your Life" as premises for the first time — the idea itself is an invention. Sui Generis score would be 13+ because every execution would diverge. |

---

## 2. Transportation Potential

*Could this premise pull a reader in — absorb their attention, engage their emotions, and generate vivid mental imagery?*

Grounded in Green & Brock's transportation theory (2000): full transportation
requires cognitive absorption, affective involvement, and vivid imagery
operating simultaneously. Peak-end rule (Kahneman et al., 1993): stories are
remembered by their emotional peaks and endings.

| Score | Description |
|-------|-------------|
| **1** | The concept is abstract, generic, or inert. No emotional stakes are apparent. No concrete imagery is suggested — the premise lives entirely in the realm of ideas with no sensory grounding. The target effect (if present) is vague ("sadness," "wonder") or missing. There is nothing here to care about, nothing to see, and nothing to hold attention. A reader encountering this story would have no reason to keep reading past the first paragraph. |
| **2** | One of Green & Brock's three components is present but the others are absent. Perhaps the premise suggests vivid imagery (a striking setting) but no emotional stakes. Or there are clear emotional stakes but the concept is so abstract that the reader can't picture anything. The target effect exists but is generic — "a sense of loss" without specificity about what kind of loss or why it matters. Peak-end potential is unclear. |
| **3** | Two of three transportation components are present. The concept suggests both emotional stakes and concrete imagery, or both intellectual interest and emotional engagement. The target effect is specific enough to guide downstream decisions — "the ache of watching someone you love choose something that will harm them" rather than just "sadness." There's a natural climax moment visible in the premise. A competent writer could pull a reader in, but the concept doesn't make it easy. |
| **4** | All three transportation components are strongly implied. The premise suggests cognitive depth (something to think about), emotional stakes (something to feel), and sensory specificity (something to see). The target effect is precise and achievable — reading it, you can already feel the ghost of the intended emotion. There's a natural peak moment and a natural ending beat. The concept does half the work of transportation before a single sentence of prose is written. |
| **5** | The concept is a transportation engine. Even as a bare premise, it produces an emotional response — the evaluator feels a flicker of the target effect just from reading the concept description. All three Green & Brock components are not just present but synergistic: the intellectual interest deepens the emotional stakes, the emotional stakes make the imagery more vivid, the imagery grounds the intellect. Like reading the premise of Shirley Jackson's "The Lottery" or Kazuo Ishiguro's *Never Let Me Go* — the concept itself is haunting before execution begins. Peak-end structure is built into the premise's DNA. |

---

## 3. Narrative Tension Potential

*Does the concept create inherent tension — suspense, curiosity, or surprise potential — that will pull readers forward?*

Grounded in Schulz et al. (2024): suspense, curiosity, and surprise are
computationally distinguishable information-theoretic quantities. A concept
should strongly support at least one.

| Score | Description |
|-------|-------------|
| **1** | The concept is static. There is no uncertainty about outcomes (no suspense), no information withheld (no curiosity), and no room for recontextualization (no surprise). The premise describes a situation but not a tension. "A woman reflects on her life" — reflects on what? Why? What's at stake? The concept suggests a mood piece with no engine to drive it forward. |
| **2** | Tension exists but is weak or generic. The suspense is of the "will they or won't they?" variety without specificity. The curiosity gap is shallow — the reader might mildly wonder what happens but wouldn't lose sleep over it. Or the concept relies entirely on a single tension type that's thin: a mystery with an obvious answer, stakes that feel manufactured, a reveal that's predictable from the premise. |
| **3** | The concept supports at least one tension type at a meaningful level. There's genuine suspense (uncertain outcome for someone the reader might care about), genuine curiosity (an information gap the reader would want closed), or genuine surprise potential (room for a revelation that reframes events). The tension is specific enough to sustain a short story. Like a premise that clearly sets up "Hills Like White Elephants"-style curiosity — you know the subtext engine is there, even if it's not yet running. |
| **4** | The concept supports multiple tension types or one type at exceptional strength. Suspense AND curiosity operate simultaneously — the reader both wants to know what happens and wants to understand something hidden. Or the concept has built-in surprise architecture: a premise where revelation is structurally inevitable but the content of the revelation is genuinely uncertain. The tension is inherent in the situation, not imposed by plot machinery. |
| **5** | The concept IS tension. The premise itself is an unresolved force — reading it creates the need for resolution. Like the premise of "The Ones Who Walk Away from Omelas" (curiosity: what is the terrible secret? suspense: what will the protagonist do? surprise: the reveal reframes the utopia). Or Chiang's "Story of Your Life" (curiosity: why is she addressing her daughter? suspense: what happens in the alien encounter? surprise: the temporal structure reframes everything). Multiple tension types are not just present but feed each other — resolving one raises another. |

---

## 4. Thematic Resonance

*Does this concept connect to something universal? Does it resist easy answers?*

Grounded in narrative transportation research (Green & Appel, 2024): thematic
depth is a primary driver of story impact and memorability. Themes that resist
simple resolution produce richer reader engagement than themes that deliver
messages.

| Score | Description |
|-------|-------------|
| **1** | No thematic content, or theme is a platitude. "Love conquers all." "Be yourself." "Technology is dangerous." The concept has a message rather than a theme — it knows its answer before asking its question. Or the premise is purely situational with no connection to anything beyond its own plot mechanics. A reader finishing this story would not think about anything larger than what happened. |
| **2** | Theme is present but shallow or bolted on. The concept has a situational premise and a thematic label that don't organically connect — the theme feels like a post-hoc justification rather than the engine of the story. "A story about a fisherman, exploring themes of isolation" — the theme is stated, not embedded. Or the thematic territory is real but the concept approaches it without tension: the answer is already implicit in the setup. |
| **3** | Theme is embedded in the premise and connects to genuine human experience. The concept raises a question that real people face — duty vs. desire, the cost of honesty, what we owe strangers. The theme emerges naturally from the situation rather than being imposed on it. But the thematic tension may be familiar (well-trodden philosophical ground) or the concept may lean toward resolving it too neatly. A good theme, competently embedded. |
| **4** | The thematic tension is the premise. The concept doesn't just contain a theme — the theme IS the reason the story exists. The situation is constructed such that the thematic question is inescapable, and the question genuinely resists resolution. Like the premise of "The Ones Who Walk Away from Omelas" — the utilitarian dilemma isn't added to the story, it IS the story. The evaluator finds themselves thinking about the thematic question independently of the story mechanics. |
| **5** | The concept reframes a universal tension in a way that makes it feel new. The thematic territory is deep (mortality, identity, obligation, freedom, meaning) but the angle of approach reveals an aspect of it that feels undiscovered. Not a new theme — there are no new themes — but a new way of experiencing an eternal one. Like how Ishiguro's *Never Let Me Go* makes mortality feel different by removing the randomness, or how "Story of Your Life" reframes free will by making foreknowledge a gift rather than a curse. The concept makes the evaluator see something familiar as if for the first time. |

---

## 5. Scope Calibration

*Can this concept produce a satisfying story in a short story word count (~1,000-8,000 words)?*

Grounded in narrative structure research and the practical constraints of short
fiction. Scope mismatch is one of the most common failure modes in AI-generated
stories — either too much premise for the word count or too little.

| Score | Description |
|-------|-------------|
| **1** | Severe scope mismatch. The concept either requires novel-length treatment (multiple characters with separate arcs, extensive worldbuilding, complex political dynamics, multi-generational timelines) or is so thin it couldn't sustain 500 words (a single observation, a mood without situation, a premise that resolves in one beat). "The rise and fall of an interstellar empire" at 3,000 words. "A cat sits in a window" at 8,000 words. |
| **2** | Noticeable mismatch that would require significant compression or expansion. The concept has 2-3 elements that each need room to breathe but collectively exceed the target range. Or the concept's natural scope is very narrow (~500 words) being stretched to fill a longer target. A skilled writer could make it work but would be fighting the concept's natural size the whole time. |
| **3** | Workable scope with some adjustment needed. The concept's natural size is in the right neighborhood but not a perfect fit. Perhaps it naturally wants 10,000 words and needs compression to fit 8,000, or it's a tight 800-word concept being asked to fill 3,000. The adjustment is feasible — cut a subplot, expand a scene — without fundamentally changing what the story is. Most competent concepts land here. |
| **4** | Good scope fit. The concept's natural size aligns well with the target range. The number of characters, the complexity of the situation, the amount of worldbuilding needed, and the emotional arc all fit comfortably within the word count. The concept has enough material to sustain the length without padding, and little enough that nothing essential gets cut. Like Carver's premises at flash length, or Munro's premises at 6,000-8,000 words — the concept and the form feel matched. |
| **5** | The concept and the short story form are made for each other. The premise has a natural compression that rewards economy — every word counts, nothing is wasted, and the constraint of the word count actually improves the story. Hemingway's iceberg principle is built into the concept's DNA: what's left unsaid is as powerful as what's said. The concept would be *worse* as a novel — expansion would dilute it. Like "Hills Like White Elephants" or "The Lottery" — premises that achieve their maximum power at exactly short story scale. |

---

## 6. Anti-Cliche Score

*Does this concept avoid known AI convergence patterns — or, if it maps to one, does it subvert the pattern with enough specificity to justify the familiar territory?*

Grounded in Echoes-in-AI (Xu et al., PNAS 2025): LLM-generated fiction
converges on a narrow set of plot shapes and thematic patterns. Nous Research
anti-patterns: 12 structural failure modes common in AI fiction.

| Score | Description |
|-------|-------------|
| **1** | The concept maps directly to a known convergence pattern without subversion. It IS the cliche, unmodified. A reconciliation arc (protagonist returns home, confronts past, reconciles). A grief meditation (dead spouse, metaphorical journey through loss). AI consciousness (machine becomes sentient, questions existence). Sanitized conflict (tension with no real stakes, clean resolution). The concept reads like an LLM's default output when asked to "write a literary short story." |
| **2** | The concept maps to a known pattern with superficial variation. It's a time loop lesson set on Mars instead of a suburb. It's an epistolary revelation using text messages instead of letters. The small-town secret is set in a space station. The pattern is immediately recognizable beneath the cosmetic changes. Or the concept combines two known patterns (grief meditation + magical realism metaphor) without the combination producing anything new. |
| **3** | The concept has echoes of known patterns but brings enough specificity to create distance. It might occupy the territory of a grief meditation but the specific loss is unusual enough (grief over a living person who has changed beyond recognition) that the execution could diverge from the template. Or it maps to a convergence pattern but the genre, setting, or structural approach is unexpected enough to disrupt the default execution. The concept is in the neighborhood of cliche but isn't standing on the doorstep. |
| **4** | The concept either avoids all 10 known convergence patterns, or it maps to one but subverts it in a way that makes the pattern work against itself. A small-town secret where the "secret" is that there is no secret and the paranoia is the point. A reconciliation arc where reconciliation is revealed to be impossible and the story is actually about accepting that. The concept uses the reader's pattern-recognition against them — the familiar shape creates expectations that the story can then violate productively. Alternatively, the concept occupies territory that AI fiction simply doesn't reach — specificity of experience, cultural particularity, professional knowledge, or emotional granularity that LLMs don't default to. |
| **5** | The concept is unclicheable. It occupies territory so specific, so particular, so rooted in concrete human experience that no convergence pattern applies. Or it takes a known pattern and transforms it so completely that the pattern becomes raw material for something unrecognizable — the way Jackson's "The Lottery" uses "small-town tradition" as a delivery mechanism for horror, or Chiang's "Story of Your Life" uses "grief meditation" as the surface of a story that's actually about determinism and love. The concept doesn't avoid AI convergence patterns by being weird — it avoids them by being true. The 10 known patterns (reconciliation arc, grief meditation, chosen one, AI consciousness, sanitized conflict, epistolary revelation, time loop lesson, magical realism metaphor, moral clarity, small-town secret) are not just absent but irrelevant. |

---

## 7. Concept Coherence

*Do the genome's elements work together — do they create productive tension or contradictory signals?*

Grounded in Gestalt principles of narrative coherence and the practical
constraint that downstream stages (structure, voice, prose) need a concept
whose elements point in a compatible direction, even if that direction involves
internal tension.

| Score | Description |
|-------|-------------|
| **1** | The elements actively contradict each other with no productive purpose. A comedic premise paired with "devastating grief" as target effect, where the comedy and grief don't inform each other — they just coexist incoherently. Character seeds that have no relationship to the premise. Constraints that make the core premise impossible to execute. Thematic tension that has nothing to do with the situation described. The genome reads like it was assembled from separate concepts. |
| **2** | Elements are loosely connected but don't reinforce each other. The premise and target effect are compatible but the character seeds feel generic — they could belong to any story. Or the constraints are present but arbitrary — they don't create interesting pressure on the premise. The genome hangs together at the surface level but the elements aren't doing work for each other. A writer given this concept would likely ignore half the genome elements. |
| **3** | Elements are compatible and functional. The premise leads naturally to the target effect. Character seeds make sense in the situation. Constraints are relevant. Thematic tension connects to the premise. Everything points in roughly the same direction. But the elements don't surprise each other — the character seeds are exactly who you'd expect in this premise, the constraints are the obvious ones, the thematic tension is the first one you'd think of. Coherent but predictable. |
| **4** | Elements create productive tension. The genome's components aren't just compatible — they push against each other in ways that generate energy. A comedic premise paired with a devastating target effect where the comedy IS the delivery mechanism for devastation (dark comedy that earns its darkness). Character seeds that are unexpected for the premise but whose presence creates new possibilities. Constraints that force the story away from the obvious execution. The elements surprise each other and the surprise creates something richer than any element alone. |
| **5** | The genome is a system. Every element is load-bearing and every element transforms every other element. Remove the constraints and the premise becomes generic. Remove the character seeds and the thematic tension loses its grounding. Change the target effect and the whole concept becomes a different story. The elements create a resonance — like a chord where each note changes the quality of every other note. The concept has the feeling of inevitability: these specific elements belong together even though you'd never have predicted the combination. Like a concept where premise, character, theme, and constraint all independently point to the same deep truth from different angles. |

---

## 8. Generative Fertility

*Does this concept suggest multiple possible stories, or only one obvious execution?*

Grounded in the evolutionary premise of the pipeline: concepts are starting
points for Stage 2 (structure evolution). High-fertility concepts give
downstream stages more room to explore, producing a richer population of
structural variants.

| Score | Description |
|-------|-------------|
| **1** | The concept implies a single execution path with no alternatives. The premise dictates the plot: "A man realizes at his retirement party that he wasted his life" — there's one story here, and it's the obvious one. The concept is a plot summary disguised as a premise. Any writer given this concept would produce essentially the same story. No branching potential, no structural ambiguity, no room for Stage 2 to explore. |
| **2** | The concept suggests 1-2 possible executions, but they're minor variations on the same basic shape. You could tell this story chronologically or in reverse. You could focus on character A or character B. But the core arc, the key scenes, and the emotional trajectory are predetermined by the premise. Stage 2 could adjust the structure but couldn't genuinely surprise anyone with it. |
| **3** | The concept supports several meaningfully different executions. The premise could be told as a thriller, a character study, or an absurdist comedy and each version would be a legitimate realization. The core situation is open enough that different structural choices produce different stories, not just different arrangements of the same story. A writer given this concept would make real choices about what kind of story to write. |
| **4** | The concept is a generative engine. It suggests many possible stories across different genres, structures, and emotional registers. The premise has enough depth and ambiguity that each execution reveals a different facet. Like Chiang's "Story of Your Life" as a premise — it could be a linguistics puzzle, a love story, a philosophical meditation, a first-contact thriller, or all of these simultaneously. Stage 2 would produce a diverse population of structural variants, each genuinely distinct. |
| **5** | The concept is inexhaustible. It's the kind of premise that different writers across different decades would execute completely differently — and each version would be valid. It has the quality of a great prompt: specific enough to constrain (not "anything could happen") but open enough to generate (the constraint creates rather than limits). Like the premise of Borges's "The Garden of Forking Paths" or Calvino's *If on a winter's night a traveler* — the concept contains a generative principle, not just a situation. Even after reading one excellent execution, you'd want to see another. |

---

## 9. Over-Explanation Resistance

*Does this concept inherently invite or resist exposition — the #1 AI fiction failure mode?*

Grounded in Nous Research's identification of over-explanation as the most
common AI fiction anti-pattern: the narrator explains what a scene already
showed. Some concepts structurally invite this failure; others structurally
resist it.

| Score | Description |
|-------|-------------|
| **1** | The concept demands heavy exposition to function. The premise requires extensive worldbuilding (a novel political system, an invented technology, a complex alternate history) or backstory (multiple characters with intertwined pasts) that must be explained before the story can begin. An LLM writing this story would spend the first third on setup. The concept is a trap — it looks interesting but it front-loads explanation, which is exactly where AI prose is weakest. |
| **2** | The concept needs moderate exposition. There's a speculative element or situational complexity that requires some explaining, but not fatally so. A skilled writer could weave it in; an LLM would likely dump it in the first few paragraphs. The concept doesn't demand over-explanation but it doesn't resist it either — it's neutral territory where the writer's (or model's) habits will determine the outcome. |
| **3** | The concept is balanced. Some elements need grounding (a character's profession, a cultural context, a key relationship) but the exposition required is modest and can be delivered through action and dialogue rather than narration. The concept doesn't pull toward explanation but also doesn't actively push away from it. Most realistic-fiction premises land here — they need some context but not a lecture. |
| **4** | The concept actively resists explanation. The premise is grounded in observable action, sensory experience, or interpersonal dynamics that show rather than tell. Exposition would feel intrusive — the story wants to dramatize, not explain. Like a concept built around a single charged conversation, or a physical task performed under pressure, or a sensory experience that resists abstraction. The LLM would have to fight its explaining instinct to write this concept, which means the concept is doing protective work. |
| **5** | The concept makes over-explanation structurally impossible. The premise, constraints, or point-of-view inherently forbid it. A child narrator who doesn't understand what they're witnessing. A constraint that prohibits internal monologue. A premise where the central truth is precisely what cannot be said — like "Hills Like White Elephants," where the power is in what's never named. Or a concept so rooted in physical action and concrete imagery that abstraction would be absurd. The concept doesn't just resist over-explanation; it turns the absence of explanation into a source of power. Everything must be shown because telling would destroy the effect. |

---

## Usage Notes

### These Are Concept Anchors, Not Prose Anchors

Every description above evaluates **potential**, not achievement. A concept
scores 5 on transportation potential if the premise strongly suggests a
story that would fully immerse readers — not because immersion has already
been achieved. Judges must project forward: given this concept, what is the
ceiling for a skilled execution?

### Not All Dimensions Apply Equally

The Holder mean (p ~ 0.3-0.5) aggregation means a single catastrophic score
drags down the whole concept. But not every dimension is equally load-bearing
for every concept. A pure voice experiment might score low on narrative tension
potential and that's fine — if the concept's strengths are elsewhere and the
low score is a 2 (modest) rather than a 1 (absent). The dynamic rubric system
adjusts weights per concept type.

### Relationship to Prose Rubric Anchors

These 9 dimensions are **Stage 1-specific** and evaluate concept potential.
The 10 shared resonance dimensions in `docs/judging/rubric-anchors.md` evaluate
executed prose. There is deliberate overlap (transportation, tension, surprise)
because the same qualities matter at both levels — but the anchors here ask
"could this concept support it?" while the prose anchors ask "did the story
achieve it?"

### Calibration Target

These anchors should be validated by evaluating known story premises:
- Concepts from HANNA dataset stories (known human quality scores)
- Premises extracted from LitBench preference pairs
- Premises of canonical short stories (Jackson, Carver, Munro, Chiang, Borges,
  O'Connor) — these should consistently score 4-5 on most dimensions
- LLM-generated premises from Echoes-in-AI convergence studies — these should
  score 1-2 on originality and anti-cliche

If the judge panel's scores on these known quantities diverge from expectations,
the anchors need revision.
