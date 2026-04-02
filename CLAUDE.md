# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**One Within the Novelist** — an evolutionary AI short story writing pipeline.

**Core thesis:** AI writing fails because (1) LLMs generate too linearly while humans explore branching pathways, and (2) evaluating story quality is a fundamental bottleneck. We address both via population-based evolutionary search with rigorous multi-judge evaluation at every stage.

**Current state:** Stage 1 building blocks implemented (LLM client, data models, evaluation pipeline, Tier A filters, scoring, operator prompt registry, configs). ShinkaEvolve integration in progress. Stages 2-6 have high-level specs.

## How We Work

**Think first, implement second.** Before writing code:
1. Discuss the approach and get explicit approval
2. Document the plan in `docs/issues/<YYYY-MM-DD-descriptive-name>.md`
3. Then implement

**Push back.** Existing plans are subject to change. If something doesn't make sense, say so. If you have a better idea, raise it. Don't just execute blindly.

**Keep docs current.** When implementation changes a design, update the relevant docs.

## Development

- **Always use `uv`** for running Python, pytest, and scripts (e.g. `uv run pytest`, not `pytest`).
- **API keys** live in `.env` at the project root. `uv run` loads them automatically.
- **Tests take ~5 minutes** — some are LLM integration tests with real API calls. Use `-k "not integration"` for fast iteration.

## Coding Style

- Concise, direct code. No fluff.
- Comments only for business logic — don't explain obvious things.
- Don't defensively code. Let errors surface naturally; we'll catch them.
- No over-engineering. No abstractions for one-time operations.

## Architecture

### 6-Stage Evolutionary Pipeline

Each stage evolves a different "genome" under selection pressure from a diverse judge panel:

1. **Concept** — premise/seed ideas (fully specified → `docs/stage-1/`)
2. **Structure** — typed-edge DAG of story beats
3. **Voice** — style specification (rhythm, diction, POV, constraints)
4. **Prose** — actual written text, scene by scene
5. **Refinement** — editorial critique-revise cycles (2-3 rounds max)
6. **Selection & Archive** — MAP-Elites quality-diversity archiving, feedback to Stage 1

Pipeline overview, cross-stage mechanisms, and per-stage detail: `docs/stages.md`

### Judging System (fully specified → `docs/judging/`)

Two-tier architecture:
- **Tier A** (`tier-a-anti-slop.md`): Fast regex/stats filters on every candidate. Banned vocabulary, burstiness, MATTR, construction patterns, 433 slop trigrams, 12 Nous anti-patterns.
- **Tier B** (`tier-b-resonance.md`): Pairwise tournament with LLM judge panel. Swiss-system brackets. 10 resonance dimensions scored with Hölder mean (p≈0.4). Dynamic per-story rubrics.

Key rules: different model families for generation vs. evaluation. Single-turn independent evaluations. 0-5 scale. CoT before scoring.

Scoring math, calibration, stage integration: `implementation-scoring.md`, `implementation-tier-a.md`, `implementation-tier-b.md`, `rubric-anchors.md`

### ShinkaEvolve Integration

Stage 1 maps onto ShinkaEvolve's async evolution engine. Concept genomes are JSON in `Program.code`. Required edits to ShinkaEvolve documented in `docs/stage-1/implementation.md` (JSON support, MAP-Elites archive strategy, operator-level bandit tracking, compost heap).

Fork: `lib/shinka-evolve/`

## Key Documents

| Path | What |
|------|------|
| `docs/ideas.md` | Master design doc — thesis, architecture, judge panel, dimensions, open questions |
| `docs/stages.md` | All 6 stages with genome/operator/eval specs and worked examples |
| `docs/judging/overview.md` | Judging philosophy, panel architecture, resonance dimensions, scoring |
| `docs/stage-1/` | Fully specified: overview, operators (11), evaluation (9 dims), population, implementation |
| `docs/judging/` | Fully specified: Tier A filters, Tier B pairwise tournament, scoring math, rubric anchors |
| `docs/prompting-guide.md` | Prompt engineering principles (ordering matters, decision chains, additive context) |
| `references/INDEX.md` | All external reference materials with summaries |
| `.claude/deep-research/runs/` | 12 research reports backing the design |

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
