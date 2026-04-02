# Operator Prompt Assembly Layer + Bug Fixes

**Date:** 2026-04-01
**Scope:** `owtn/prompts/stage_1/registry.py` — operator registry, prompt building, seed injection
**Depends on:** Prompt templates (all 11 exist), seed bank model, `OPERATOR_SEED_TYPES`
**Blocks:** ShinkaEvolve Edit 4 (sampler dispatch), runner class cold-start generation

---

## Problem

All Stage 1 building blocks exist (LLM client, models, evaluation pipeline, Tier A filters, scoring, prompts, configs, seed bank) but nothing connects the 11 operator prompt templates to ShinkaEvolve's sampler. The sampler needs a registry that maps operator names to system messages, iteration templates, patch routing, and seed injection. Without this bridge, Edit 4 (sampler dispatch) can't be implemented.

Two minor bugs also needed fixing.

## Changes

### Bug Fix 1: Stale test assertion

`tests/test_evaluation/test_stage_1.py:145` — `"Be fair but honest"` replaced with `"use your best judgment"` to match current `HARSHNESS_INSTRUCTIONS["moderate"]`.

### Bug Fix 2: Cross-family rule violation

`configs/judges/mira-okonkwo.yaml` — model was `claude-haiku-4-5-20251001` (Anthropic), same family as generation model (Anthropic Sonnet). Violates the design rule that generation and evaluation use different model families to prevent self-preference bias. Changed to `deepseek-chat`.

### Operator Prompt Registry

New file `owtn/prompts/stage_1/registry.py`:

- `OPERATOR_DEFS` — static metadata for all 11 operators (routing, cross-type flag)
- `OperatorDef` — dataclass with resolved prompt templates per operator
- `load_registry()` — loads all operator `.txt` files, resolves `{output_format}`, splits sys_format from instructions
- `inject_seed()` — selects a seed from the bank by operator type, formats for injection
- `build_operator_prompt()` — assembles complete `(system_msg, user_msg)` tuple for ShinkaEvolve

The registry splits each operator prompt at `---INSTRUCTIONS---`:
- Above: identity/role text → appended to base system message
- Below: task instructions → fills `{operator_instructions}` in iteration/initial templates

Seed injection follows `implementation.md` §4: look up operator's seed types via `OPERATOR_SEED_TYPES`, select one, format as "Use this as your starting point:" block, inject via `{seed_content}` placeholder.

### Tests

New file `tests/test_prompts/test_registry.py` — 25 tests covering:
- All 11 operators present and loadable
- Patch routing (full vs diff) matches spec
- Cross-type operators correctly flagged (collision, compost, crossover)
- Seed types match `OPERATOR_SEED_TYPES` from `seed_bank.py`
- Seed injection returns content for matching types, empty for non-matching
- Seed exclusion works
- Initial and iteration prompt structure
- Feedback/steering/episodic context injection
- Output format resolved in all operators
- Diff operator retains SEARCH/REPLACE format

## Verification

1. `uv run pytest tests/test_prompts/ -v` — 25 passed
2. `uv run pytest tests/ -q -k "not integration"` — 198 passed
3. `uv run python -c "from owtn.prompts.stage_1.registry import load_registry; print(list(load_registry().keys()))"` — prints all 11 operators

## What This Unblocks

1. **ShinkaEvolve Edit 4** — sampler dispatch consumes `load_registry()` directly
2. **Runner class** — `build_operator_prompt(is_initial=True)` for cold-start generation
3. **ShinkaEvolve Edit 5** — patch routing uses `OperatorDef.routing`
