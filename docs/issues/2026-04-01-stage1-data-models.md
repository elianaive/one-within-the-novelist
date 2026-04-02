# Stage 1 Data Models & Config Loader

**Date:** 2026-04-01
**Scope:** Create `owtn/models/` — typed Pydantic models for the concept genome, MAP-Elites classification, judge personas, stage config, and seed bank
**References:** `docs/stage-1/implementation.md` (genome schema), `docs/stage-1/classification.md` (enums), `configs/stage_1_default.yaml`, `configs/judges/*.yaml`, `data/seed-bank.yaml`

---

## Motivation

The infrastructure layer is complete: LLM client, Tier A filters, scoring math, prompts, judge configs, seed bank data. The next phase is the orchestration layer — evaluate.py, judge panel, seed bank injection, ShinkaEvolve edits. Every one of these components needs to parse, validate, and pass around the same data structures: concept genomes, classification results, judge personas, config values, seeds.

Without shared typed models, each component will reinvent validation, hardcode field names as strings, and disagree on edge cases (is `constraints: null` the same as `constraints: []`?). The models are the shared vocabulary.

## Why Now

- **Zero upstream dependencies.** Pure Pydantic + PyYAML. No LLM calls, no ShinkaEvolve integration.
- **Every downstream component depends on this.** evaluate.py needs `ConceptGenome` for Gate 1 validation. Judge orchestration needs `JudgePersona` for prompt assembly. MAP-Elites archive needs classification enums for cell keys. Config loader feeds thresholds to every subsystem.
- **Highly testable.** Validate against example genomes already in the specs and the existing YAML/data files in the repo.

## Changes

### 1. ConceptGenome — `owtn/models/concept_genome.py`

Pydantic model mirroring `implementation.md` lines 17-61:

```
premise: str              # required, min_length=20
target_effect: str        # required, min_length=15
character_seeds: list[CharacterSeed] | None
setting_seeds: str | None
thematic_tension: str | None
constraints: list[str] | None
style_hint: str | None
```

`CharacterSeed` nested model: `label` + `sketch` required, `wound`/`fear`/`lie`/`want`/`need` optional.

Methods:
- `classify_constraint_density() -> ConstraintDensity` — rule-based, from `classification.md` lines 36-44
- `to_prompt_fields() -> dict[str, str]` — returns template vars matching `judge_user.txt` placeholders, rendering None as empty string
- `from_code_string(cls, code: str) -> ConceptGenome` — classmethod, parses JSON string (how ShinkaEvolve stores genomes in `Program.code`)

### 2. Classification Enums & Result — `owtn/models/classification.py`

From `classification.md`:

| Enum | Values |
|------|--------|
| `ConceptType` | thought_experiment, situation_with_reveal, voice_constraint, character_collision, atmospheric_associative, constraint_driven |
| `ArcShape` | rise, fall, fall_rise, rise_fall, rise_fall_rise, fall_rise_fall |
| `ConstraintDensity` | unconstrained, moderate, heavy |
| `TonalRegister` | comedic, tragic, ironic, earnest, surreal, matter_of_fact |
| `ThematicDomain` | interpersonal, societal, philosophical, existential, mundane_elevated |

`ClassificationResult` Pydantic model with typed fields, confidence levels, and a `cell_key() -> tuple` method returning `(concept_type, arc_shape, constraint_density)`.

### 3. Judge Persona — `owtn/models/judge.py`

Pydantic model for judge persona YAMLs:

```
id: str
name: str
identity: str
values: list[str]
exemplars: list[str]
harshness: Literal["lenient", "moderate", "demanding"]
priority: Literal["primary", "secondary", "contrarian"]
model: list[str]
```

`load_panel(judges_dir, panel_ids) -> list[JudgePersona]` function.

### 4. Stage Config — `owtn/models/config.py`

Hierarchical Pydantic models parsing `stage_1_default.yaml`:
- `EvolutionConfig` — num_generations, language, patch_types, patch_type_probs, etc.
- `DatabaseConfig` — num_islands, archive_size, archive_selection_strategy, etc.
- `OperatorBanditConfig` — enabled, warmup_generations, exploration_constant, min_probability_floor
- `LLMConfig` — generation_models, generation_model_family, judge_models, classifier_model, embedding_model
- `JudgesConfig` — panel (list of IDs), judges_dir, min_demanding_ratio
- `EvaluationConfig` — holder_p, diversity_weight, std_threshold, anti_cliche (nested), tier_a_enabled, pairwise_enabled, dynamic_rubrics_enabled
- `HandoffConfig` — strategy, max_concepts
- `PathsConfig` — seed_bank, convergence_patterns
- `StageConfig` — top-level, composes all above + `from_yaml(path)` classmethod

### 5. Seed Bank — `owtn/models/seed_bank.py`

- `Seed` model: `id`, `type`, `content`, `source`, `tags`
- `SeedBank` class: `load(path)`, `get_by_type(seed_type)`, `select(seed_type, exclude_ids)` (random selection with exclusion)
- `SEED_OPERATOR_MAP` constant dict from `implementation.md` lines 313-326

### 6. Tests — `tests/test_models/`

- `test_concept_genome.py` — parse Hills Like White Elephants example, validation failures (missing premise, too-short), constraint density classification, round-trip JSON serialization
- `test_classification.py` — enum membership, ClassificationResult parsing, cell_key()
- `test_judge.py` — load 3 actual judge YAMLs from configs/judges/
- `test_config.py` — load stage_1_default.yaml, verify all fields parse correctly
- `test_seed_bank.py` — load data/seed-bank.yaml, verify seed counts by type, operator mapping coverage

## What This Unblocks

1. **evaluate.py** — Gate 1 = `ConceptGenome.model_validate_json()`. Config thresholds from `StageConfig`.
2. **Judge orchestration** — `JudgePersona` fills `judge_system.txt`. `ConceptGenome.to_prompt_fields()` fills `judge_user.txt`.
3. **Seed bank injection** — `SeedBank.get_by_type()` in operator prompt assembly.
4. **ShinkaEvolve Edit 6** — `ClassificationResult` enums for MAP-Elites cell keys.

## Verification

1. `uv run pytest tests/test_models/ -v` — all model tests pass
2. `uv run pytest tests/ -k "not tier_a"` — no regressions in existing 67 passing tests
3. Manual: `uv run python -c "from owtn.models.config import StageConfig; c = StageConfig.from_yaml('configs/stage_1_default.yaml'); print(c.evaluation.holder_p)"` prints `0.4`
