# ShinkaEvolve Integration — Wire the Evolutionary Loop

**Date:** 2026-04-01
**Scope:** 5 ShinkaEvolve edits + runner subclass → first evolutionary dry run
**Depends on:** LLM client, data models, evaluation pipeline, operator registry, compatibility bridge (all complete)
**Blocks:** First evolutionary run, empirical validation of Stage 1

---

## Problem

All Stage 1 building blocks are implemented and tested (230/230 pass) but nothing connects them into a running evolutionary loop. ShinkaEvolve needs 5 remaining edits to handle concept operators and MAP-Elites archiving, plus a runner subclass that configures the loop for Stage 1.

## Changes

### Edit 1: JSON loading in `wrap_eval.py`

`load_program()` uses `importlib` to import Python modules. Add JSON early-return for `.json` files so concept genomes can be loaded as dicts.

### Edit 3: Concept operators in `defaults.py`

Replace `default_patch_types()` / `default_patch_type_probs()` with 11 concept operators and their cold-start probabilities.

### Edit 4: Sampler dispatch in `sampler.py`

Replace hardcoded if/elif for diff/full/cross with registry-based dispatch. Import `load_registry` from `owtn.prompts.stage_1.registry`. Thread seed bank as optional parameter. Preserve backward compatibility for legacy patch types.

### Edit 5: Patch routing in `async_apply.py`

Route operator names to `apply_full_patch` or `apply_diff_patch` based on `OPERATOR_DEFS` routing metadata. Keep backward compat for "full", "diff", "cross".

### Edit 6: MAP-Elites archive in `dbase.py`

Add `_update_archive_map_elites()` with SQL-backed cell storage. Cell replacement uses `holder_score` (raw Hölder mean, no diversity bonus). Wire into `_update_archive()` before the `archive_size` guard (MAP-Elites ignores size caps).

### Runner subclass: `owtn/runner.py`

`ConceptEvolutionRunner(ShinkaEvolveRunner)` overriding `_generate_initial_program()` for cold-start allocation. Selects operators from weighted distribution, builds prompts via registry, extracts JSON from code fences, skips EVOLVE-BLOCK wrapping (invalid in JSON).

## Files

| File | Change |
|------|--------|
| `lib/shinka-evolve/shinka/core/wrap_eval.py` | JSON early-return in `load_program()` |
| `lib/shinka-evolve/shinka/defaults.py` | 11 concept operators + probabilities |
| `lib/shinka-evolve/shinka/core/sampler.py` | Registry-based operator dispatch |
| `lib/shinka-evolve/shinka/edit/async_apply.py` | Operator → patch function routing |
| `lib/shinka-evolve/shinka/database/dbase.py` | MAP-Elites archive strategy + SQL cell table |
| `owtn/runner.py` | ConceptEvolutionRunner subclass |
| `tests/test_shinka_edits/` | Unit tests for all 5 edits |
| `tests/test_runner/` | Runner unit + integration + smoke tests |

## Verification

1. Unit tests for each edit (no LLM calls)
2. Mocked integration test: 2 generations, 2 islands, verify full loop
3. Smoke test (API-gated): 1 generation, real LLM, verify valid genomes scored
4. Manual dry run: 5 generations, 2 islands, 3 judges — inspect scores, diversity, cell coverage
