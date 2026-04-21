# Stage 1: Seed Bank Curation Playbook

Companion to `seed-bank.md` (which specifies *what* seeds look like). This doc is the *how*: the methodology for expanding the bank without quality drift.

Produced after the 2026-04-21 expansion session that added 132 seeds across 5 types (axiom, image, thought_experiment, constraint, real_world) and then audited pre-session material against the filters developed during the run. The method is replicable. This doc makes it so.

---

## The cycle

Per type, the rhythm is:

1. **Direction.** A single load-bearing axis for the type, not a laundry list. Written in the parent issue.
2. **Dispatch** 3-4 parallel deep-research streams with explicitly non-overlapping territories (detailed briefs; see template below). Run in the background.
3. **Wait.** Each stream produces a `final_report.md` with 10-15 drafts and a sources registry.
4. **Distill per stream.** Read each `final_report.md`, pick ~6-7 strongest.
5. **Cross-stream curation.** Dedup across streams, check author/mechanism redundancy with existing bank, balance for judge-taste coverage.
6. **Staging scratch** (`lab/issues/<date>-<type>-staging-scratch.md`) with each pick's full YAML and *explicit cut rationale* for each drop.
7. **Commit** to `data/seed-bank.yaml` at the right section boundary.
8. **Test** (`uv run pytest tests/test_models/test_stage_1/test_seed_bank.py tests/test_prompts/test_registry.py -q`).
9. **Log** in the parent issue with bank counts before/after, judge-alignment summary, and link to staging scratch.

The commit gate is where the user approves. Everything upstream is autonomous.

---

## Writing a deep-research brief

This is where the depth is produced. A vague brief ("find more axioms") produces slop. A concrete brief produces research-grade material you can actually commit.

### Template

```
You are researching <type> seeds for "One Within the Novelist" — <one-line project context>.

**Project context**: full CLAUDE.md at <path>. Judge profiles at
<path to judges/>. Read these before drafting.

**Your stream**: `<type>-<sub-slug>` — one of <N> parallel research streams
expanding the `<type>` seed type.

**Stream-specific direction**: <one paragraph — the single load-bearing axis
for this stream. What territory it owns, what aesthetic it produces, what
the reader/judge gets from it.>

**Target territories** (produce at least 2 from each cluster):

**Cluster A — <name>**:
- <specific target 1: name + brief description>
- <specific target 2>
- <6-10 per cluster>

**Cluster B — <name>**:
- <specific targets>

<3-4 clusters total. Specific targets, not categories. "Stasi IM reports
(Ministerium für Staatssicherheit)" not "surveillance documents."
Name the thing.>

**Aesthetic requirements**:
- <register, tense, length>
- <must-include structural elements — named mechanism, cited source, disturbing detail>
- <explicit rejection criteria — what NOT to produce>

**Format requirements**:
- <sentence count / prose style>
- <what metadata must be present>
- <source-citation requirement>

**Existing <type> bank** (<N> seeds) — DO NOT duplicate. Existing IDs:
<enumerate ALL existing IDs, or the canonical ones plus a note about
recently-committed ones from adjacent types>.

<Specific overlap-watches: "we have X covering Y angle — your picks must
differ mechanically, not just choose different examples of the same
mechanism">

**Judge-taste targets** (from project judge profiles):
- <Judge 1> (primary for this stream): <specific alignment criterion>
- <Judge 2> (secondary): <criterion>
- <all 4 judges with criteria — primary + secondary designated>

**Output format per draft**:
```
### Draft N: [short title]
**Phenomenon / Source / Thinker**: [named, specific]
**Source**: [citation with year + DOI/URL where possible]
**Content (3-5 sentences)**: [the seed text in final-register prose]
**What makes it story-generative**: [1 sentence]
**Judge-alignment**: [primary + secondary]
**Suggested tags**: [4-6 tags]
**Proposed id**: `<type>-<kebab-case>`
```

**Sources to mine**: <specific named archives, journals, databases, scholars>

**Stop conditions**: 10-15 distinct drafts spanning at least <M> target
territories with ≥2 each. Each must pass: (1) <specificity gate>,
(2) <non-duplication>, (3) <judge-alignment>, (4) <source quality>.

Final report to `final_report.md`. Standard structure: worklog, sources,
important_findings, final_report.
```

### Why each element matters

- **Non-overlapping territories across streams**: forces each agent to drill into *their* niche instead of competing on the same ground. Territories should be different enough that you can't accidentally pick the same item from two streams.
- **Target territories listed as specific named things**: "Aztec Matrícula de Tributos" forces research; "documents from obsolete empires" produces vibes.
- **Existing-IDs enumeration**: forces the agent into the gaps. Without this, streams converge on the material adjacent to what's already in the bank.
- **Overlap-watches**: call out specific existing seeds that the new draft must differ from *mechanically* (not just lexically). `axiom-ibn-arabi-barzakh` exists, so a new Ibn ʿArabī seed must be a structurally distinct claim, not "Ibn ʿArabī on cosmology."
- **Judge-taste per draft required**: anchors "what makes this good" in concrete perspectives instead of general craft principles.
- **Source citation requirement**: filters out speculation. The agent must verify and link.
- **Stop conditions**: tell the agent when to quit. Without these, drafts drift toward completion-theater (rewriting old drafts to meet quota).

---

## Quality filters (hardened 2026-04-21)

These are the checks to apply when distilling per-stream and during cross-stream curation. They are also the checks to apply when auditing *existing* bank content for cuts.

### 1. LLM-canonical test

Is this concept in every intro philosophy/psychology/science syllabus? Will it appear thousands of times in training corpora? **Reject.**

The failure mode: mutation operators cannot produce novel work on a deeply-worn groove. Any mutation on Mary's Room / Pascal's Mugging / the rubber-hand illusion-as-canonical-demo reverts to the standard pattern.

**Examples cut 2026-04-21:** `te-marys-room`, `te-teleporter-duplicate`, `te-utility-monster`, `te-swampman`, `te-combined-spectrum` — all direct-to-canon analytic philosophy.

**Exception:** a specific underexposed angle on a canonical case can survive. `rw-rubber-hand` was rewritten (not cut) because the *clinical-correlate* angle (schizophrenia faster susceptibility, Parkinson's reduced before motor symptoms) is not canonical; the illusion itself is.

### 2. Pop-science fun-fact test

Has the item escaped to Atlas Obscura / BuzzFeed / daytime NPR? Is it a cocktail-party fact without a mechanism? **Reject or rewrite.**

The failure mode: fun facts don't disturb naive physicalism; they just decorate it. They produce "interesting" concepts, which is the mid register the contest explicitly rejects.

**Rewrite test:** is there a specific *mechanism* or *contested element* buried under the pop-surface? If yes, rewrite to foreground it. If no, cut.

- `dancing-plague` (rewrite): the Strasbourg council building *wooden stages* to accelerate the cure is the buried mechanism.
- `pepys-cheese-burial` (cut): no buried mechanism. Just a cute fact.

### 3. Author-redundancy

If the bank has claim X from a thinker, does a new claim Y from the same thinker add mechanical distinct material, or just another sentence from the same author? **Cut the redundant; keep the distinct.**

Author-match ≠ redundant. Mechanism-match = redundant.

- `axiom-weil-attention-as-prayer` + `axiom-weil-malheur`: kept both. Distinct mechanisms (attention as ethical practice vs. affliction-stamp as soul-mark).
- `axiom-zapffe-sublimation-trap` + `te-zapffe-cosmic-panic`: kept both. Sublimation-trap is the specific mechanism; cosmic-panic is the four-mechanism framework plus the "what happens when the mechanism fails" scenario.

### 4. Mechanism-overlap across types

If `axiom-X` is the claim-form and `te-X` is the scenario-form of the *same* claim, pick one. Prefer the tighter framing.

- `te-mainlander-god-corpse` cut because `axiom-mainlander-god-corpse` is tighter.
- `te-cotard-il-y-a` cut because `rw-cotard-denial-existence` has stronger sourcing.

### 5. Mechanism-overlap within types

Two seeds that do different literary work but rely on the same underlying mechanism compete for the same mutation slot. Pick the one whose specifics are more narratively generative.

- Whitehead "events-not-substances" dropped; `axiom-sadra-motion` kept. Both claim "change is what being is." Sadra's graded-existence framing is underexposed; Whitehead is LLM-canonical.

### 6. Media-anchored treatment

Different rules for different types:

- **Images tagged with a source work** (`manga`, `VN`, `JRPG`, `Blame`, `gem`) are fine. Images work as *inspiration for a scene*, not as the scene itself — the pipeline treats them as starting material, not target. A manga frame generates a story about that image; it doesn't become the story.
- **Axioms and thought_experiments anchored to specific works** need the claim to stand alone. If the content depends on knowing *Umineko* or *Houseki no Kuni* to make sense, cut or rewrite. Today's audit kept the VN/JRPG/manga-tagged axioms because the *claims* are standalone philosophical content (the tag just notes the origin medium).

### 7. Contested-over-resolved

A seed with an unsettled mechanism generates more mutation space than one with a settled mechanism. "Scientists dispute X" > "the mechanism of X is Y." The pipeline's generative operators thrive on residue the facts haven't absorbed.

- `rw-saiga-mass-death`: 100% simultaneous mortality, mechanism unknown. Strong.
- `rw-muon-g2-contradiction`: two Standard Model calculations disagree with each other. Strong.
- `rw-greenland-shark-maturity`: 156-year pre-reproductive phase, molecular mechanism unknown. Strong.

### 8. Judge-alignment required per pick

Every pick should hit at least 2 of 4 judge profiles explicitly (primary + secondary). This forces concrete "what makes this good" instead of general craft gestures.

- **Gwern**: discipline-internal specificity, foreign archive, anti-listicle, technical precision (actual citations, specific mechanisms)
- **Roon**: AI as ontology not metaphor (no human-in-a-box for AI content); civilizational stakes compressed; shape-rotator structure over wordcel performance
- **Wales**: mechanical load-bearing elements; rules have second-order effects; cold reading rewards
- **Wahls**: untold before / untold after; consent paradoxes; medium-enacts-meaning; transformation arc

Pick without judge-alignment rationale = pick by vibe. Don't.

---

## Rituals (the state-producing discipline)

### Cut rationale is mandatory

When distilling per-stream and cross-stream, the *written cut rationale* for each drop is the thing that creates the filter. Not a reporting convenience — the discipline of articulating *why* this one and not that one is what hardens judgment across the session.

Without written cut rationale, the first dozen cuts are principled and the next hundred are vibes.

The staging scratch format does this structurally:

```
**<Stream> (N kept, M cut)**:
Kept: `<id-1>` (<rationale>), `<id-2>` (<rationale>), ...
Cut: `<id-3>` (<rationale>), `<id-4>` (<rationale>), ...
```

Every drop gets a sentence. The sentence names the filter it failed.

### Staging scratch before commit

Every bank expansion creates a staging scratch at `lab/issues/<date>-<type>-staging-scratch.md`. Structure:
- Parent-issue reference
- Streams run (with run directory + draft counts)
- Curation rationale (per-stream kept/cut with reasons)
- Full YAML for each final pick
- Summary (judge-alignment, commit plan)

This is the review artifact. User approves by approving this, not the raw YAML commit.

### Parent-issue log append

Each type's completion adds a block to the parent issue with:
- Direction recap
- Streams table (slug | run dir | draft count)
- N-total-drafts → N-final-picks
- Picks by stream (one-line each, with the hook that got it in)
- Judge-alignment summary
- Key cuts with rationale
- Committed confirmation (bank total, type total, tests)

This creates a durable record of *why* the bank has the shape it has. Future audits (see `2026-04-21-seed-bank-quality-audit.md`) depend on this.

---

## Research-first stance

Every seed comes with a source. No exceptions.

This is the cognitive mode that produces depth: before you write the seed text, you have cited material in hand. You are verifying, triangulating, checking primary sources. The output inherits that discipline.

"Brainstorm creative ideas" is a different mode — it produces creative-writing-register content, which reads to judges as vibes. Research-first produces discipline-internal-register content, which reads as specificity.

Rules:
- **Primary source preferred.** Cite the paper, not the Wikipedia summary. When primary isn't accessible, cite the review article that engages with primary, not the pop-science article.
- **Verify before citing.** Agents occasionally fabricate DOIs. Cross-check any DOI you haven't seen before.
- **Named cases beat generic claims.** "Patient LM (Zihl et al. Brain 1983)" is stronger than "motion-blindness patients."
- **Specific numbers beat qualitative claims.** "156 ± 22 years at sexual maturity" is stronger than "extreme longevity."

If the seed can't be sourced, it can't go in. The 3 manga-tagged images are the edge case — they're fictional scenes, but they're explicitly labeled as such and serve as inspiration material not factual claims.

---

## How to collaborate with Claude on this (user-facing)

The curation state is partly produced by the user's collaboration rhythm. Three patterns from the 2026-04-21 session that mattered:

### Approve at commits, not at dispatch

Dispatching briefs, running streams, distilling per-stream, and writing staging scratches are autonomous work. The commit is the approval gate. If you approve each brief, Claude becomes conservative on brief quality because each approval step introduces friction. If you approve each stream's picks, the curation loop slows to a crawl.

**The gate**: the staging scratch. Review it, redirect with specifics, approve the commit. Then Claude moves to the next category.

### Redirect with specifics, not wholesale

"Manga images are fine as inspiration, not the story — take another pass on the others" is a precise redirect. It kept 3 picks, cut 0 from the original pass, and re-audited 15 others. Wholesale reversals ("start over") destroy the context that produced the quality.

The pattern: name what you're keeping, name what you're changing, name the criterion.

### Give token permission early if you're committing

Early in a long session, state the cost posture explicitly. "Don't be afraid to deep research" (actual quote from 2026-04-21) removed friction on stream-dispatch decisions. Without it, Claude budgets toward minimum-viable briefs.

If you're planning a multi-category session, token-permission up front produces better briefs, which produces better research, which produces better picks. Post-hoc "go deeper" is more expensive than ex-ante permission.

---

## Technical gotchas

### YAML tags with bare integers

Pydantic's `List[str]` validation rejects bare-integer tags. `tags: [engineering, 1981, walkway]` fails; `tags: [engineering, "1981", walkway]` passes.

This caused a test failure on 2026-04-21; two seeds had year-tags that needed quoting.

Year tags are often desirable (historical anchoring), so this comes up. Always quote bare-number tags.

### Folded vs literal scalar

Use `content: >` (folded) for prose content. The folded scalar joins lines with spaces, which is the right register for paragraph-level prose.

Use `content: |` (literal) only when whitespace is load-bearing (e.g., code snippets, tabular data). Seeds almost never need literal.

### ID conventions

- Kebab-case
- Type prefix: `rw-`, `te-`, `axiom-`, `image-`, `constraint-`, `compression-`, `collision-`, `dilemma-`, `anti-`, `nouns-`
- Unique globally, not just within type (the validation checks cross-type uniqueness)

### Section boundaries

`data/seed-bank.yaml` is sectioned by type with comment banners:

```yaml
# ============================================================
# <type>
# ============================================================
```

New seeds go at the end of their type section, immediately before the next section banner. Don't interleave types; the pipeline doesn't care about order within a type, but the sectioning is human-facing and should be preserved.

---

## Related documents

- `docs/stage-1/seed-bank.md` — the *what*: schema, quality criteria per type, maintenance strategy
- `docs/stage-1/operators.md` — how seeds become operator prompts
- `lab/references/unslop-contest/judges/` — judge profiles referenced throughout
- `lab/issues/2026-04-21-seed-bank-expansion-unslop-informed.md` — the parent issue for the session this doc codifies
- `lab/issues/2026-04-21-seed-bank-quality-audit.md` — the retrospective audit pass
