# ShinkaEvolve Compatibility Bridge

**Status:** Complete

## Problem

Three contract mismatches between our evaluation pipeline (`owtn/evaluation/`) and ShinkaEvolve's async runner would silently break integration:

### 1. metrics.json field names

`async_runner.py` read `metrics_val.get("public", {})` but the `Program` dataclass and all downstream code use `public_metrics`. Our `EvaluationResult` correctly serialized as `public_metrics`/`private_metrics`, but the async runner would discard them.

**Fix:** Changed `async_runner.py` to read `public_metrics`/`private_metrics` (two locations: initial program eval ~L1326 and main loop ~L3276).

### 2. __main__.py argument format

ShinkaEvolve's scheduler calls `python evaluate.py --program_path <path> --results_dir <path>`. Our `__main__.py` used positional arguments.

**Fix:** Changed to `--program_path` (required), `--results_dir` (required), `--config_path` (default: `configs/stage_1_default.yaml`). Config path passed via ShinkaEvolve's `extra_cmd_args`.

### 3. apply_full_patch EVOLVE-BLOCK markers

`apply_full_patch` requires EVOLVE-BLOCK markers to know where to apply mutations. JSON genomes don't have these — the entire file is the genome. Without this fix, every mutation is silently discarded.

**Fix:** Added JSON early-return path in `apply_full_patch`. When `language in ("json", "json5")`: validate JSON, do whole-file replacement, skip all EVOLVE-BLOCK logic. Also fixed `validate_code_async` to use `json.loads` instead of shelling out to `jsonschema`.

## Additional changes

- Relaxed ShinkaEvolve's `httpx==0.27` pin to `httpx>=0.27` to resolve dependency conflict with `google-genai>=1.13`.
- Added `shinka-evolve` as editable dependency via `[tool.uv.sources]`.

## Files modified

| File | Change |
|------|--------|
| `lib/shinka-evolve/shinka/core/async_runner.py` | `"public"` → `"public_metrics"`, `"private"` → `"private_metrics"` (2 locations) |
| `lib/shinka-evolve/shinka/edit/apply_full.py` | JSON early-return path after code extraction |
| `lib/shinka-evolve/shinka/edit/async_apply.py` | JSON validation via `json.loads` instead of `jsonschema` subprocess |
| `lib/shinka-evolve/pyproject.toml` | `httpx==0.27` → `httpx>=0.27` |
| `owtn/evaluation/__main__.py` | Positional args → `--flags` |
| `pyproject.toml` | Added `shinka-evolve` dependency with local source |

## Tests added

| File | Tests |
|------|-------|
| `tests/test_evaluation/test_shinka_contract.py` | Validates metrics.json and correct.json field names match async_runner expectations |
| `tests/test_evaluation/test_apply_json.py` | JSON whole-file replacement, invalid JSON rejection, output file writing, Python still requires EVOLVE-BLOCK |

## What this unblocks

ShinkaEvolve can now invoke our evaluation pipeline as a subprocess and correctly read the results. Next steps:
1. Operator dispatch — wire `registry.py` into `sampler.py`
2. MAP-Elites archive — replace fitness-ranked archive in `dbase.py`
3. First evolutionary dry run
