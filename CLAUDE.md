# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**One Within the Novelist** — an evolutionary AI short story writing pipeline.

**Core thesis:** AI writing fails because (1) LLMs generate too linearly while humans explore branching pathways, and (2) evaluating story quality is a fundamental bottleneck. We address both via population-based evolutionary search with rigorous multi-judge evaluation at every stage.

**Current state:** Stage 1 is functional end-to-end — concept generation, pairwise evaluation, island-based evolution with champion succession, Swiss tournament for final ranking. ShinkaEvolve integration complete. Stages 2-6 have high-level specs.

## How We Work

**Think first, implement second.** Before writing code:
1. Discuss the approach and get explicit approval
2. Document the plan in `lab/issues/<YYYY-MM-DD-descriptive-name>.md` (see `lab/issues/FORMAT.md`)
3. Then implement

**All work is grounded in issues.** Issues are the single source of truth for what's happening, what's blocked, and what's done. Update them continuously as work progresses.

**Push back.** Existing plans are subject to change. If something doesn't make sense, say so. If you have a better idea, raise it. Don't just execute blindly.

**Keep docs current.** When implementation changes a design, update the relevant docs.

## Lab Structure

`lab/` is the working directory for non-committed Claude artifacts.

| Path | Purpose |
|------|---------|
| `lab/scripts/` | Debugging scripts, one-off utilities, diagnostic tools |
| `lab/issues/` | All work must be grounded in issues here |
| `lab/references/` | External reference materials (PDFs, cloned repos) |
| `lab/deep-research/` | Deep research agent outputs (runs with reports, sources, findings) |

### Issue Discipline

**Naming:** `YYYY-MM-DD-descriptive-name.md`

**Every issue must include:** goal, steps, progress, blockers, resolution, related issues. Update continuously — the issue is the central place for information during active work.

**Parent/child linking is a 2-step atomic operation:**
1. Create child issue with parent reference in "Related Issues"
2. *Immediately* update parent issue to add child — verify both links before proceeding

See `lab/issues/FORMAT.md` for the full template and conventions.

## Debugging Rules

**Slow down. Isolate the root cause.**
1. Create an issue in `lab/issues/` — this is the central log for the entire debugging session
2. Add diagnostic logs or write debug tests in `lab/scripts/` — do NOT make code changes yet
3. Identify the root cause with evidence, not guesses
4. Then implement the fix
5. Update the issue with the resolution

**Do not:** make code changes before understanding root cause, try multiple fixes hoping one works, add complexity without evidence, assume anything without checking.

## Coding Style

**I want clean, maintainable code that serves the core purpose. My aversion to premature complexity is not a desire for sloppy code.**

- Concise, direct code. No fluff.
- Clear, readable code with descriptive variable names.
- Comments only for business logic — don't explain implementation details.
- Keep functions focused on single responsibilities.
- Basic error handling where it matters. Don't defensively code against impossible states.
- Structure code for easy modification, not premature scalability.

**Avoid:**
- Premature abstractions and over-engineered architecture
- Features not essential to the core value proposition
- Complex dependencies when simpler alternatives exist
- Optimization before identifying actual bottlenecks

**Function stubbing** is encouraged for mapping architecture: implement skeletal versions with log statements to track execution paths before committing to full implementations.

## Development

- **Always use `uv`** for running Python, pytest, and scripts (e.g. `uv run pytest`, not `pytest`).
- **API keys** live in `.env` at the project root. `uv run` loads them automatically.
- **Tests:** `uv run pytest tests/ -m "not live_api"` for fast offline tests (~2s). Tests marked `live_api` make real API calls and cost money — run the full suite with `uv run pytest tests/`.

### Running the Pipeline

```bash
uv run python -m owtn --config configs/stage_1/<config>.yaml [--max-eval-jobs N] [--max-proposal-jobs N]
```

Configs: `dry_run.yaml`, `light.yaml`, `medium.yaml`. See each file's header for cost estimates and concurrency recommendations.

## Test Organization

Tests live in `tests/`, organized by module:

| Directory | What it tests |
|-----------|--------------|
| `test_evaluation/` | Gate validation, pairwise prompt assembly, mocked pipeline E2E, ShinkaEvolve contract |
| `test_judging/` | Tier A filters (anti-patterns, vocabulary, structural, statistical, ngrams) |
| `test_llm/` | Query routing, cache keys, model resolution, pricing, prompt caching, QueryResult serialization |
| `test_models/` | Pydantic models: ConceptGenome, classification, config, seed bank, judge personas |
| `test_prompts/` | Operator registry structure, routing, seed types, prompt building, mutation feedback |
| `test_runner/` | Cold-start operator distribution, config building |
| `test_shinka/` | ShinkaEvolve integration: sampler dispatch, MAP-Elites archive, JSON patch application, program loading |

**Shared fixtures** in `tests/conftest.py`: canonical test genomes (`HILLS_GENOME`, `MINIMAL_GENOME`), `genome_file`/`results_dir` fixtures. Use these — don't redefine test data locally.

**Test principles:**
- Don't assert on LLM output content — it's non-deterministic. Assert structural validity only (correct fields, score ranges, files written).
- Mark tests requiring API keys with `@pytest.mark.live_api`.
- One concern per test file. Name files for what they test, not implementation history.

## Architecture

### 6-Stage Evolutionary Pipeline

Each stage evolves a different "genome" under selection pressure from a diverse judge panel:

1. **Concept** — premise/seed ideas (functional → `docs/stage-1/`)
2. **Structure** — typed-edge DAG of story beats
3. **Voice** — style specification (rhythm, diction, POV, constraints)
4. **Prose** — actual written text, scene by scene
5. **Refinement** — editorial critique-revise cycles (2-3 rounds max)
6. **Selection & Archive** — quality-diversity archiving, feedback to Stage 1

Pipeline overview, cross-stage mechanisms, and per-stage detail: `docs/stages.md`

### Selection System

**Pairwise comparison, not absolute scoring.** Absolute LLM scoring compresses all AI concepts into a narrow band (model-level leniency bias). Pairwise comparison ("which is better?") discriminates where scoring cannot.

**Per-criteria voting.** Each judge compares two concepts on all 9 dimensions independently, picks a winner per dimension, with position bias mitigation (dual orderings). Overall winner = most dimension-wins across all judges.

**Island champions.** Each island maintains a champion. New concepts challenge the champion via pairwise comparison. Winners become the new champion; losers are archived with their comparison feedback.

**Swiss tournament.** After evolution completes, island champions compete in a Swiss-system tournament for final ranking.

### Evaluation Dimensions (9)

Concepts are compared on these dimensions (sub-criteria in `owtn/prompts/stage_1/rubric_anchors.txt`):

1. **Novelty** — domain crossing, convergence distance, generative surprise
2. **Grip** — the thing you can't look away from, emotional stakes, sensory seed
3. **Tension Architecture** — suspense, information architecture (resolvable vs permanent gaps), reframing potential
4. **Emotional Depth** — recognition, complexity, source, reader implication
5. **Thematic Resonance** — question vs message, embeddedness
6. **Concept Coherence** — load-bearing elements, surface/depth architecture
7. **Generative Fertility** — execution diversity, generative principle vs situation
8. **Scope Calibration** — natural size, constraint as compression
9. **Indelibility** — indelible image, irreducible remainder, silhouette

### Judging System (`docs/judging/`)

Two-tier architecture:
- **Tier A** (`tier-a-anti-slop.md`): Fast regex/stats filters. Banned vocabulary, burstiness, MATTR, construction patterns, 433 slop trigrams, 12 Nous anti-patterns. Used for prose stages, not concepts.
- **Tier B**: Pairwise comparison with 3-judge panel across 9 dimensions. Per-criteria voting with position bias mitigation. Judge personas in `configs/judges/*.yaml`.

Key rules: different model families for generation vs. evaluation. Single-turn independent evaluations. Each judge sees both concepts in both orderings.

### ShinkaEvolve Integration

Stage 1 maps onto ShinkaEvolve's async evolution engine. Concept genomes are JSON in `Program.code`. Evaluation is inline (no subprocess) via `eval_function` on `JobConfig`. The eval function validates the concept, reads the island champion from disk, and runs pairwise comparison — all blocking, so ShinkaEvolve sees the real score before selecting the next parent.

Fork: `lib/shinka-evolve/`

Key additions to ShinkaEvolve:
- `eval_function` on `JobConfig` for inline evaluation (no subprocess)
- `parent_id` and `island_idx` passed through scheduler to eval function
- `get_island_champion()` and `update_program_score()` on `ProgramDatabase`
- `EqualIslandSampler` for balanced island allocation

## Key Documents

| Path | What |
|------|------|
| `docs/CHANGELOG.md` | What changed and why — pairwise selection, rubric redesign, convergence fixes |
| `docs/ideas.md` | Master design doc — thesis, architecture, judge panel, dimensions, open questions |
| `docs/stages.md` | All 6 stages with genome/operator/eval specs and worked examples |
| `docs/judging/overview.md` | Judging philosophy, panel architecture, resonance dimensions |
| `docs/stage-1/` | Overview, operators (11), evaluation (9 dims), population, implementation |
| `docs/judging/` | Tier A filters, Tier B pairwise tournament, rubric anchors |
| `docs/prompting-guide.md` | Prompt engineering principles (ordering matters, decision chains, additive context) |
| `lab/INDEX.md` | Claude working directory — issues, scripts, references |
| `lab/deep-research/runs/` | 18+ research reports backing the design |

## LLM-Fed Files — Token Hygiene

Files in `owtn/prompts/` and prose fields in `configs/judges/*.yaml` are fed directly to LLMs at runtime. Avoid wasting tokens on formatting that only helps human readers:
- **No hard-wrapping.** Don't break lines at 70-80 chars for readability. One logical sentence/paragraph = one line. Use YAML `>` (folded scalar), not `|` (literal scalar), for multi-line prose fields.
- **No decorative blank lines.** Blank lines between template fields or list items cost tokens. Use them only where they create meaningful structure for the LLM (e.g., separating instructions from content).

## Prompt Engineering Principles (from `docs/prompting-guide.md`)

These apply when writing any LLM prompts for the pipeline:
- **Order = causality.** The order you ask for things IS the causal order the model assumes.
- **Decision chains > pattern + exemption.** Force step-by-step condition checking.
- **Additive context.** Each step continues the conversation; don't restart fresh.
- **Understanding > reinforcement.** Explain semantics, not more examples.
- **Fix plans force commitment.** Require explicit transformation plan before prose output.
