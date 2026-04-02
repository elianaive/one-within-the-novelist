# Stage 1 Evaluation Pipeline

**Date:** 2026-04-01
**Scope:** `owtn/evaluation/` — Gates 1-3, classification, ShinkaEvolve output format
**Depends on:** LLM client, data models, scoring, judge configs, prompts (all implemented)
**Blocks:** ShinkaEvolve integration testing

---

## Problem

All Stage 1 building blocks exist (LLM client, data models, Tier A filters, scoring math, judge configs, prompt templates, seed bank) but nothing orchestrates them. ShinkaEvolve needs an `evaluate.py` subprocess that receives a concept genome and returns scored metrics. Without it, the evolution loop can't run.

## Solution

New `owtn/evaluation/` package implementing the 3-gate evaluation pipeline:

1. **Gate 1 — Validation** (no LLM): Parse JSON genome, validate with ConceptGenome model, reject trivial/placeholder content
2. **Gate 2 — Anti-Cliche** (stub): Embedding-based convergence pattern check. Stubbed until embedding client lands. Returns not-flagged.
3. **Gate 3 — Judge Panel** (LLM): 3 judges evaluate independently via async parallel calls. Structured output via instructor with `reasoning` field first in model. Hölder mean per judge, cross-judge aggregation with diversity bonus.
4. **Classification** (LLM): Single call to classify into MAP-Elites dimensions. Merged with rule-based constraint_density.
5. **Output**: `metrics.json` (scores, classification, cell_key) + `correct.json` for ShinkaEvolve.

Also: rubric anchors prompt file extracted from `docs/stage-1/rubric-anchors.md`, and judge prompt updates to request single JSON (reasoning + scores) instead of free-text CoT then separate JSON.

## Files

| File | Purpose |
|------|---------|
| `owtn/evaluation/__init__.py` | Exports `evaluate()` |
| `owtn/evaluation/stage_1.py` | Core orchestration |
| `owtn/evaluation/models.py` | Pydantic models for judge output |
| `owtn/evaluation/prompts.py` | Template loading + persona formatting |
| `owtn/evaluation/anti_cliche.py` | Gate 2 stub |
| `owtn/evaluation/__main__.py` | CLI entry point |
| `owtn/prompts/stage_1/rubric_anchors.txt` | Rubric anchors for judge prompts |
| `owtn/prompts/stage_1/judge_system.txt` | Updated for instructor format |
| `owtn/prompts/stage_1/judge_user.txt` | Updated for instructor format |
| `tests/test_evaluation/` | Unit + integration tests |

## Key Decisions

- **Instructor with reasoning field**: Single LLM call per judge. `reasoning` field first in Pydantic model forces CoT before scores. No regex extraction, no doubled calls.
- **Gate 2 stubbed**: Embedding client doesn't exist yet. Stub returns not-flagged. Fills in when embeddings land.
- **Async judge calls**: All 3 judges evaluated in parallel via `asyncio.gather`.
