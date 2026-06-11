# Changelog

## 2026-04-30: Stage 4 v1 implemented and live-validated

The scoped architecture (entry below) is now code: `owtn/stage_4/` (session loop, PreThink/DownDraft/Revise phases, plateau detection), `owtn/models/stage_4/`, `owtn/prompts/stage_4/`, configs under `configs/stage_4/`, full offline test suite in `tests/test_stage_4/`.

- **Critic ensemble shipped (15 personas + domain_expert factory).** 4 Tier A fidelity critics mandatory, 11 Tier B resonance critics agent-invoked (`configs/stage_4/critics/`). Personas are ordinary-specific practitioners (working librarians, nurses, poets), not author archetypes — named-author personas collapsed empirically. `voice_fidelity` is a tool-using critic seeded from the Stage 3 winning persona, with stylometry/slop_score/writing_style access.
- **domain_expert factory.** A pre-stage filter classifies the concept's expertise demands; matching domain critics are instantiated at setup with `web_search` + `fetch_page` for factual verification. Scales to open-ended expertise spaces without code-per-domain.
- **Surgical-edit dispatch ("dental" renamed).** Haiku scope-translator converts natural-language edit scope to anchor-text bounds; the subagent edits only within them; post-edit re-validation. The dental metaphor didn't carry across model families — empirical probe, hence the rename. A 3-way propose→pick→execute wrapper was tested and rejected: the status quo already suffices under production voice context.
- **Per-agent sandbox.** File ops isolated at `run_dir/sandbox/{agent_id}/`; agents can't touch orchestrator logs or metadata.
- **Prompt audit.** All agent-facing surfaces stripped of pipeline/phase/stage framing; workshop register only. Fixed the drafter's "mediocrity license" by reframing scope discipline.
- **Nudge fallbacks restricted to finalize-only tools.** Writers and tool-using critics that exhaust their loop budget previously kept editing instead of committing; the nudge turn now exposes only the `finalize_*` tool.

**Live validation** (`lab/issues/2026-04-30-stage-3-stage-4-live-validation.md`): full end-to-end run on a dense multi-domain concept — 3 revise cycles, 2 domain experts instantiated, 13 surgical edits, 4,911 words / 9 scenes. ~$1.48 Stage 4 + ~$3.21 Stage 3 per (concept, structure, voice) tuple.

Issues: `lab/issues/2026-04-29-stage-4-*.md`, `lab/issues/2026-04-30-stage-4-*.md`

---

## 2026-04-30: Cross-stage pipeline (Stages 1 → 4 under one config)

New `owtn/pipeline.py` + `configs/pipeline/`: a pipeline config points at one per-stage YAML each and adds orchestration-only knobs (eval-job concurrency, tier, max-concept caps) — per-stage schemas stay authoritative. Output lands under a single `results/run_<ts>/` root with per-stage subdirectories. `owtn/__main__.py` rewritten around it. `submission.yaml` / `dry_run.yaml` configs added for every stage. Cross-stage transcript helpers (`owtn/orchestration/transcript.py`) render an agent's full LLM chain as a Markdown chat transcript — tool-use loops no longer collapse to one opaque composite entry in the call log.

---

## 2026-04-30: Stage 2 Tier 3 — concept-demand fidelity

Every preset's final terminal DAG is now checked against its `concept_demands` list (`owtn/evaluation/stage_2_tier3.py`) — predicates the structure must realize that escape the 8-dimension pairwise rubric (reader-address, form-as-device, structural rhyme, deliberate irresolution). One classifier-model call per terminal verdicts each demand as `satisfied | partial | failed`; entries with `failed` verdicts rank below all-satisfied entries in the tournament. Config cleanup alongside: `scalar_brief_re_summarize_every` exposed as a real knob, scalar+simulate warning added, redundant `scoring_handoff_top_k` deleted. Scalar champion-brief prompts split out (`champion_brief_scalar.txt`, `champion_brief_scalar_lineage.txt`).

Issue: `lab/issues/2026-04-30-stage-2-tier3-and-config-cleanup.md`

---

## 2026-04-30: LLM layer — adaptive thinking, validation telemetry, cache fixes

- **Opus 4.7 adaptive thinking.** The model rejects the legacy `thinking.type=enabled` shape and deprecates `temperature`. Provider now dispatches per model: callers keep passing `reasoning_effort`, the Anthropic provider decides legacy budget vs. `thinking.type=adaptive` + `output_config.effort`, and `resolve_temperature` omits temperature where deprecated.
- **No silent LLM payment loss.** When structured output fails Pydantic validation on a paid 200, providers now raise `LLMValidationError` (`owtn/llm/errors.py`) carrying the raw payload, token counts, and cost; `api.py` logs a diagnostic yaml and re-raises so retry loops behave as before.
- **Anthropic prompt caching actually fires in tool-use loops.** Two-breakpoint pattern (system + most-recent-user, tools array as third) caches the full prefix every turn — ~40-60% cost reduction on Stage 3 runs.
- **DeepSeek reasoning fixes.** `build_call_kwargs` was silently skipped on three call sites, defaulting reasoning models to thinking-disabled (refusal rate ~75% → 0% after fix); terminal tool-use turns now retain `reasoning_content` so follow-on calls that reuse history don't break. Stage 2 expansion factories pass `reasoning_effort="low"` explicitly — a disabled-thinking default had produced 100%-hallucinated DAG node IDs.

Issues: `lab/issues/2026-04-30-opus-4-7-adaptive-thinking.md`, `2026-04-30-llm-raw-payload-logging-on-validation-failure.md`, `2026-04-30-anthropic-prompt-cache-miss.md`, `2026-04-30-deepseek-thinking-reasoning-content-dropped.md`, `2026-04-30-stage-2-expansion-reasoning-disabled.md`

---

## 2026-04-30: Cross-stage tools — NL reference lookup, web tools, metric fixes

- **`lookup_reference` goes natural-language.** Slug/tag interface replaced by a haiku resolver translating agent intent to corpus keys, with a prompt-cached catalog (~30-40K token prefix) making per-lookup cost trivial. Compound queries intersect by default; entry-id fast-path bypasses the LLM. 13/13 pilot queries resolved.
- **`web_search` (Exa HTTP) + `fetch_page` (trafilatura)** in `owtn/tools/` — provider-agnostic by design since the dev-default DeepSeek has no native search. Stage 4 domain experts are the current consumer; the surface is cross-stage.
- **Voice metric bug fixes.** `vocab_level` no longer scales with sentence length (Flesch-Kincaid → mean syllables per word, thresholds recalibrated); burstiness returns `None` on single-sentence passages instead of recommending variety to deliberate incantations; `stylometry.target_styles` → `style_queries` routed through the same NL resolver.
- New stylometric voice-reference data: `data/voice-references-stylometric.yaml`.

Issues: `lab/issues/2026-04-30-voice-metrics-tool-bugs.md`, `2026-04-30-domain-expert-fetch-page-tool.md`

---

## 2026-04-29: Stage 4 architecture scoped

Designed and committed to `docs/stage-4/overview.md`. Stage 4 produces a finished short story (5-10k words across ~8-15 scenes) as a markdown manuscript file, one prose per (concept, structure, voice) tuple. Architecture: single agent + critic ensemble, three orchestration phases (PreThink → DownDraft → Revise), file-based interaction (Claude-Code-shape), no writer-persona.

**Key design commitments:**

- **File-based state.** Manuscript on disk, generic file tools (read_file/write_file/edit_file). Mirrors Claude Code; scales beyond context limits.
- **Critic ensemble (15 critics + dynamic domain_expert).** Tier A fidelity (4 critics, mandatory) maps to upstream-genome enforcement; Tier B resonance (11 critics, agent-invoked) maps to the 10 Resonance Dimensions in `docs/judging/overview.md` plus three cross-cutting operations. Conditional `domain_expert` factory class instantiated at setup by a pre-stage filter for technical / historical / cultural / canon-specific work, with provider-agnostic web_search.
- **Cycled Phase 3 sub-phases.** Sub-phase A (gather) has critic tools but no edit tools; sub-phase B (revise) has edit tools but no critic tools. Forces signal integration before commits, matching CritiCS gather-then-revise validated pattern.
- **Plateau detection via issue counts + severity tags.** Preserves judges-don't-score posture. Cap ~6 cycles + 30-50 LLM call ceiling as backstops.
- **Cross-family critics as architectural invariant.** Generator on Claude family → critics on DeepSeek/Gemini. Mechanistic safeguard against in-context reward hacking (arxiv 2407.04549, validated on creative writing).
- **Surgical-edit dispatch with bound-validated subagents.** Surgical-edit mechanism: cheap haiku translates natural-language scope to concrete bounds; surgical-edit subagent works within bounds; post-hoc diff check rolls back out-of-scope edits.
- **Reader-state model in PreThink (3 dimensions).** Epistemic / affective-structural / forward-pull, voice-mode-gated. Empirically untested for LLM prose; theoretically grounded (Rabinowitz authorial audience, Spoiler Alert 2026 epistemic constraint). A/B pilot mandated.
- **Read-as-reader micro-loop in DownDraft.** Per-scene write → use generic `think` tool with prompt instruction to read as reader → optional edit. Reuses Stage 3's existing `think` tool.
- **Audience framing at setup.** Cheap classification call returns the work's implied audience (Rabinowitz "authorial audience"); populated into agent system prompt. Complement to voice spec without competing.
- **Five `finalize_*` tools.** Explicit phase-boundary commits matching Stage 3's `finalize_voice_genome` pattern.

**Two deep-research runs informed the design:**

- `lab/deep-research/runs/2026-04-29-reader-state-modeling-prompts/` — empirical / theoretical grounding for the reader-state element. Finding: empirically untested intervention; theoretical case strong; primary failure mode is emotional labeling, mitigable via epistemic vocabulary; Spoiler Alert 2026 is closest empirical anchor.
- `lab/deep-research/runs/2026-04-29-stage-4-iterative-critique-architectures/` — long-form architecture audit. Finding: design is consistent with Agents' Room (ICLR 2025, Google DeepMind); CritiCS empirical sweet spot is 3 rounds with 2 trade-off cycles; ICRH validated on creative writing; "Mosca 2025" 50-100 exemplar FT claim is unverifiable (corrected memory).

**Stage 5 / Stage 6 boundary** is the genuinely flimsy part. Stage 5's original "editorial critique-revise" role has effectively been absorbed by Stage 4's Phase 3. Stage 5 may be (a) gone, (b) retained for harder macro revision, (c) repurposed for cross-candidate work. Resolution deferred to pilot signal from Stage 4.

Issue: `lab/issues/2026-04-29-stage-4-architecture-scoping.md`

---

## 2026-04-19: Run-prompt redesign (`steering` → `prompt`)

Renamed the run-config `steering` field to `prompt` and fixed a silent bug where the field only fired in gen 0. The mutation path in `lib/shinka-evolve/shinka/core/sampler.py` had been calling `build_operator_prompt` without forwarding the steering string, so every generation past the first ran unprompted. The user prompt now propagates from `StageConfig.prompt` → `PromptSampler(prompt=...)` → `build_operator_prompt(prompt=...)` for both genesis and mutation calls.

**Injection shape**: replaced the inline `Creative direction for this run: {steering}` field-style label with a separate prose block (the *Magnes* template at `owtn/prompts/stage_1/run_prompt.txt`) that wraps the user's prompt in a literary frame in the same register as the operator personae. The block sits between the tonal-steering paragraph and the base task description, so directional pressure is in place before the structural contract appears. When `prompt` is empty, the block is omitted entirely — `base_system.txt` is now a complete invocation on its own.

**Tonal steering untouched.** `tonal_steering` and `sample_tonal_steering` are a separate, sampler-internal mechanism (random affective register × literary mode per concept). Those still work exactly as before.

Issue: `lab/issues/2026-04-19-rename-steering-to-prompt.md` — includes the literary-editor critique that produced the *Magnes* draft and four backup drafts (Present-tense ongoing, Three sentences, Absent agent, Inside-the-weights). To swap drafts, edit `owtn/prompts/stage_1/run_prompt.txt` directly.

---

## 2026-04-05: Pairwise Selection

Replaced pointwise scoring with pairwise comparison. Absolute LLM scoring compressed all concepts into a 0.3-point band regardless of rubric design — the leniency bias is structural (RLHF), not fixable by prompt engineering. Pairwise comparison discriminates where scoring cannot.

**Selection**: Per-criteria pairwise voting (9 dims x 3 judges x 2 orderings) replaces Holder mean + diversity bonus. Each judge compares two concepts on all 9 dimensions independently, picks a winner per dimension. Position bias mitigated via dual orderings — if a judge picks different winners in different orderings, that vote is discarded as a tie.

**Evaluation**: Inline via `eval_function` on `JobConfig` (no subprocess). Validation + pairwise comparison in one blocking step, so ShinkaEvolve sees real scores before selecting next parent.

**Islands**: Champion-based succession. Each island maintains one champion. New concepts challenge the champion; winner takes over. Equal island scheduling (`EqualIslandSampler`) ensures balanced allocation.

**Archive**: Fitness-based (was MAP-Elites). MAP-Elites barely fired at current scale (7-9 cells occupied out of 36 with 12 concepts per run).

**Tournament**: Swiss-system tournament ranks island champions after evolution completes. Buchholz tiebreaking.

**Score**: Win percentage `(dim_wins + 0.5 * ties) / 9`, not Holder mean. Champions start at 0.5. Tournament applies final ranking.

**Removed**: Pointwise scoring functions (`holder_mean`, `aggregate_judge_scores`), subprocess evaluation entry point (`evaluation/__main__.py`), old pointwise prompts (`judge_system.txt`, `judge_user.txt`), harshness instructions, classification LLM call, dead config fields (`tier_a_enabled`, `pairwise_enabled`, `dynamic_rubrics_enabled`).

Issue: `lab/issues/2026-04-05-pairwise-selection.md`

---

## 2026-04-04: Rubric Anchor Redesign

Restructured evaluation dimensions from 9 impression-based to 9 observable sub-criteria-based, grounded in deeper cognitive science analysis. Research finding: criteria decomposition is the highest-leverage structural fix for score discrimination (+17.7 accuracy points in the literature).

**Dimensions changed**:
- Merged Originality + Anti-Cliche → **Novelty** (3 sub-criteria: domain crossing, convergence distance, generative surprise)
- Renamed Transportation Potential → **Grip** (3 sub-criteria: the thing you can't look away from, emotional stakes, sensory seed)
- Renamed Narrative Tension → **Tension Architecture** (3 sub-criteria: suspense, information architecture with resolvable vs permanent gaps, reframing potential)
- Added **Emotional Depth** split from Transportation (4 sub-criteria: recognition, complexity, source, reader implication)
- Added **Indelibility** from prose-stage Memorability (3 sub-criteria: indelible image, irreducible remainder, silhouette)
- Removed Over-Explanation Resistance (absorbed into other dimensions)

**Format**: Endpoint-only anchors (scores 1 and 5 described; 2-4 are interpolations). "Name it" prompting — each sub-criterion requires the judge to cite specific evidence from the concept text.

**Six deep patterns** identified from analysis of stories that survive across decades: the invisible made visible, permanent withholding, reader complicity, structural irony, recognition, inevitability. These are distributed across the sub-criteria.

**All examples from published fiction**, not AI-generated concepts: Jackson, Kafka, Chiang, Borges, Hemingway, Nabokov, O'Connor, Le Guin, Carver, Joyce, Ishiguro, Keyes, Melville, Gilman, Oates, Machado, Swift.

Issue: `lab/issues/2026-04-04-rubric-anchor-redesign.md`

---

## 2026-04-03: Evolution Score Debugging

Diagnosed why evolutionary search wasn't improving scores across generations. The best score (4.530) was set in Gen 0 and never surpassed across 19 generations.

**Root causes found**:
- Mutation prompts drowning the model in 4,000-6,000 tokens of raw judge feedback
- Genesis calls contaminated by eval_history prepended unconditionally
- Score compression: all 29 concepts between 4.08-4.53 (0.45-point spread on 0-5 scale)
- Feedback not actionable through most operator instructions (only 5 of 11 reference it)
- Seed reuse across generations

**Fixes**: Compressed judge feedback for mutations via `build_mutation_feedback()` (26% token reduction). Added judge scoring calibration instruction (insufficient alone — led to rubric redesign). Discovered Gemini 2.5 Flash truncates reasoning in structured output mode.

Issue: `lab/issues/2026-04-03-evolution-not-improving-scores.md`

---

## 2026-04-03: Prompt Register Rewrite

Rewrote all prompts from instructional to invitational register based on research showing instructional framing suppresses creative diversity.

**Research basis**: "Price of Format" (EMNLP 2025) — structured templates collapse diversity independent of content. Regulatory Focus Theory — "avoid mistakes" framing activates convergent processing; "approach ideals" activates divergent. CreativeDC — 63.5% novelty improvement from separating divergent/convergent phases.

**Results** (13 concepts vs 28 baseline): Self-BLEU 48.88→33.38 (↓32%), distinct-1 0.198→0.310 (↑57%), earnest tone 100%→76.9% (first non-earnest concepts), cosine similarity 0.656→0.520 (↓21%).

Issue: `lab/issues/2026-04-03-prompt-register-rewrite.md`

---

## 2026-04-02: Convergence Monoculture Fix

Identified 6 root causes of concept convergence — every concept converging to literary psychological realism about morally ambiguous intimacy.

**Causes**: (1) Anti-convergence checklist acting as convergence funnel, (2) single example anchoring register, (3) judge monoculture (3 judges share same aesthetic), (4) evolution amplifying convergence (MAP-Elites cell key missing tonal register), (5) temperature too low (2/3 calls ≤0.5), (6) strong seeds dominating.

**Fixes**: Tonal steering injection (16 affective registers x 18 literary modes = 288 combinations), temperature annealing schedule (high early for exploration, low late for refinement), genesis ratio annealing, softened anti-convergence checklist.

Issue: `lab/issues/2026-04-02-convergence-monoculture.md`
