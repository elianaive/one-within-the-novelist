# Engineering Decisions

Cross-cutting software engineering decisions for the pipeline. Not stage-specific
— these apply everywhere.

---

## Repository Structure

```
one-within-the-novelist/
├── CLAUDE.md                    # project instructions
├── pyproject.toml               # package config + dependencies
├── docs/                        # design docs, specs
│   ├── ideas.md                 # master design doc
│   ├── stages.md                # all 6 stages overview
│   ├── prompting-guide.md       # prompt engineering principles
│   ├── engineering.md           # this file
│   ├── issues/                  # decision logs (YYYY-MM-DD-name.md)
│   ├── judging/                 # judging system specs
│   ├── stage-1/                 # stage 1 specs
│   └── stage-{2-6}/             # future stage specs
├── owtn/                        # implementation package ("one within the novelist")
│   ├── __init__.py
│   ├── prompts/                 # prompt templates (.txt files with {placeholders})
│   │   └── stage_1/             # Stage 1: concept evolution
│   │       ├── base_system.txt  # generation system message
│   │       ├── judge_system.txt # judge evaluation system message
│   │       ├── judge_user.txt   # judge user message (concept presentation)
│   │       ├── classification.txt # MAP-Elites classifier
│   │       ├── initial.txt      # gen-0 scaffolding
│   │       ├── iteration.txt    # gen-1+ scaffolding (parent + scores)
│   │       ├── output_format.txt # shared output format for operators
│   │       └── operators/       # 11 mutation operator prompts
│   ├── llm/                     # canonical LLM client (extracted from ShinkaEvolve)
│   │   ├── __init__.py
│   │   ├── client.py            # OWTNClient: caching, judge interface
│   │   ├── providers/           # per-provider implementations (from ShinkaEvolve)
│   │   ├── query.py             # routing layer
│   │   ├── kwargs.py            # hyperparameter sampling
│   │   └── prioritization.py    # bandit model selection
│   ├── judging/                 # shared judging infrastructure
│   │   ├── __init__.py
│   │   ├── tier_a.py            # anti-slop filters
│   │   ├── tier_b.py            # LLM judge panel
│   │   ├── scoring.py           # Hölder mean, disagreement signal
│   │   └── personas.py          # judge persona loading
│   ├── stage_1/                 # concept evolution
│   │   ├── __init__.py
│   │   ├── evaluate.py          # ShinkaEvolve evaluate.py (pluggable)
│   │   ├── classify.py          # MAP-Elites classification
│   │   ├── operators.py         # operator registry + prompt templates
│   │   └── config.py            # stage 1 evolution config
│   └── common/                  # shared utilities
│       ├── __init__.py
│       └── embeddings.py        # embedding client (novelty, anti-cliche)
├── data/
│   └── seed-bank.yaml           # curated seed bank (all types)
├── configs/                     # run configurations
│   └── stage_1_default.yaml     # default Stage 1 config
├── shinka-evolve/               # forked ShinkaEvolve (git submodule)
│   └── ...
├── references/                  # research papers, external references (read-only)
│   └── INDEX.md
└── tests/
    ├── test_llm/
    ├── test_judging/
    └── test_stage_1/
```

### Key Principles

**`owtn/` is our code. `shinka-evolve/` is their code (with our edits).**

Our pipeline code lives in `owtn/`. ShinkaEvolve is a dependency we modify
minimally. Clean separation means we can pull upstream changes without merge
hell, and our code isn't scattered across two package trees.

**`owtn/llm/` is the canonical LLM client.** Extracted from ShinkaEvolve, owned
by us. ShinkaEvolve's fork is modified to import from `owtn.llm` instead of
`shinka.llm`. Dependency flows one direction: `shinka-evolve` → `owtn`. See
"LLM Client" section below.

**`owtn/judging/` is shared across all stages.** The two-tier judging system
(Tier A filters, Tier B LLM panel) is used by Stages 1-5. Stage-specific
evaluation dimensions and judge personas are configured per stage, but the
scoring math, panel orchestration, and anti-slop filters are shared code.

**`references/` stays read-only.** Research papers, datasets, external reference
materials. Never modified by the pipeline. The ShinkaEvolve submodule moves out
of `references/` to the repo root (see "ShinkaEvolve Fork" section) since it's
a dependency we modify, not a reference.

---

## ShinkaEvolve Fork Strategy

### Current State

ShinkaEvolve is a git submodule pointing at our fork:

```
[submodule "lib/shinka-evolve"]
    path = lib/shinka-evolve
    url = https://github.com/elianaive/ShinkaEvolve-Concept.git
```

### Upstream Tracking

Periodically merge from `SakanaAI/ShinkaEvolve:main` into our fork's `main`,
then rebase our branch.

### What We Edit in the Fork

From `docs/stage-1/implementation.md`:

1. `wrap_eval.py` — JSON genome support
2. `async_apply.py` — JSON validation + operator routing
3. `defaults.py` — 11 concept operators + probabilities
4. `sampler.py` — operator dispatch refactoring
5. `async_apply.py` — patch routing (full vs diff per operator)
6. `dbase.py` — MAP-Elites archive strategy
7. Operator-level bandit tracking (new, extends existing model bandit)
8. Compost heap tables (new, in existing SQLite DB)
9. **Replace `shinka.llm` imports with `owtn.llm`** — point at our LLM client

All changes are additive or clearly scoped to specific functions. We don't
restructure their codebase beyond rewiring the LLM import.

---

## LLM Client

### Why We Extract It

ShinkaEvolve's `shinka/llm/` package is solid multi-provider infrastructure:
8 providers (Anthropic, Bedrock, OpenAI, Azure, OpenRouter, DeepSeek, Gemini,
local OpenAI-compatible), async-native, cost tracking, native structured
output per provider (Anthropic forced tool use; OpenAI Responses API
`text_format=`; DeepSeek `response_format=json_object` with our own schema
prompt; Gemini `response_schema=`), exponential backoff retries.

We extract it into `owtn/llm/` and make it the single LLM client for the whole
project because:

1. **Clean dependency direction.** Our `evaluate.py` (in `owtn/`) needs LLM
   calls. ShinkaEvolve's evolution engine needs LLM calls. If the LLM client
   lives in ShinkaEvolve and we import it, but ShinkaEvolve also imports our
   evaluate.py, we have a circular dependency. Extracting it breaks the cycle:
   both `owtn` and `shinka-evolve` depend on `owtn.llm`.

2. **Prompt caching.** The biggest gap in their client. We need it for judge
   evaluations. Better to add it in our owned copy than maintain patches in a
   fork.

3. **Every stage needs it.** Stages 2-6 won't use ShinkaEvolve's evolution
   engine (only Stage 1 does), but they all need LLM calls. The client should
   live where all stages can reach it.

### What We Copy

The entire `shinka/llm/` directory:

```
shinka/llm/
├── client.py           # provider factory functions
├── llm.py              # LLMClient, AsyncLLMClient
├── query.py            # routing layer
├── kwargs.py           # hyperparameter sampling
├── prioritization.py   # bandit model selection (UCB, Thompson)
└── providers/
    ├── __init__.py
    ├── client.py       # per-provider query functions
    ├── anthropic.py    # Anthropic provider
    ├── openai.py       # OpenAI provider
    ├── deepseek.py     # DeepSeek provider
    ├── gemini.py       # Gemini provider
    ├── local_openai.py # local OpenAI-compatible
    ├── model_resolver.py
    ├── pricing.py      # cost calculation
    ├── result.py       # QueryResult dataclass
    └── pricing.csv     # pricing table
```

This is a one-time copy. After extraction, `owtn/llm/` is our code — we
maintain it, we extend it, we own it. ShinkaEvolve's fork gets a one-line
import change per file that uses it.

### What We Add

**Prompt caching.** Anthropic's prompt caching gives 90% cost reduction on
cached tokens. Our judge calls are perfectly suited:

```
System message (CACHED — same across all concepts for one judge):
├── Rubric anchors (~1,500 tokens) — identical for all Stage 1 evaluations
├── Judge persona (~300 tokens) — identical per judge
└── Evaluation instructions (~200 tokens) — identical per stage

User message (NOT CACHED — varies per concept):
└── Concept genome + scoring request (~500 tokens)
```

Implementation:
- Add `cache_control` support to the Anthropic provider
- Structure the client API so callers pass `system_prefix` (cacheable) and
  `system_suffix` (per-call) separately
- Batch concepts per judge — evaluate all concepts for Judge 1, then all for
  Judge 2 — to keep the system message prefix stable across the batch
- For non-Anthropic providers, rely on automatic prefix matching (OpenAI caches
  automatically when prefixes match)

**Judge-friendly query interface.** A method that takes persona + rubric +
concept and handles the prompt assembly, cache-friendly ordering, and structured
output parsing.

**Trade-off:** We lose automatic upstream LLM updates from ShinkaEvolve (new
providers, pricing updates). But these are easy to cherry-pick from their repo,
and we'd need to verify compatibility anyway.

### ShinkaEvolve Rewiring

In our fork, replace LLM imports across ShinkaEvolve's codebase:

```python
# Before (in shinka-evolve files):
from shinka.llm import AsyncLLMClient, QueryResult

# After:
from owtn.llm import AsyncLLMClient, QueryResult
```

The interface is identical — same class names, same methods, same return types.
The only change is where the import comes from. Files affected:

- `shinka/core/async_runner.py` (3 LLMClient instances)
- `shinka/core/novelty_judge.py`
- `shinka/core/summarizer.py`
- `shinka/core/async_summarizer.py`
- `shinka/core/prompt_evolver.py`

After rewiring, delete `shinka/llm/` from the fork entirely. One source of
truth for LLM calls.

---

## Package Installation

### Development Setup

```bash
# Clone with submodule
git clone --recurse-submodules <repo-url>
cd one-within-the-novelist

# Install with uv (handles venv + deps in one step)
uv sync

# API keys
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.
```

### pyproject.toml

```toml
[project]
name = "owtn"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    # LLM providers (from ShinkaEvolve's deps):
    "anthropic>=0.49",
    "openai>=1.75",
    "google-genai>=1.13",
    "backoff",
    # Data / ML:
    "numpy",
    "pandas",
    "scikit-learn",
    "pydantic>=2.0",
    # Utilities:
    "aiofiles",
    "python-dotenv",
    "pyyaml",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "ruff"]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[dependency-groups]
dev = ["pytest", "pytest-asyncio", "ruff"]
```

ShinkaEvolve is installed as a workspace member or path dependency:

```toml
[tool.uv.sources]
shinka-evolve = { path = "shinka-evolve", editable = true }
```

This way `uv sync` installs both `owtn` and `shinka` as editable packages in
one command. Code changes in either are immediately reflected.

---

## Run Lifecycle

A Stage 1 run is managed by a thin CLI wrapper around ShinkaEvolve's
`AsyncShinkaEvolveRunner`:

```
owtn run-stage-1 --config configs/stage_1_default.yaml

1. SETUP
   ├─ Load config
   ├─ Load seed bank from data/seed-bank.yaml
   ├─ Compute/cache convergence pattern embeddings from data/convergence-patterns.yaml
   ├─ Validate judge panel (model families differ from generation, demanding ratio met)
   └─ Initialize ShinkaEvolve runner with our config

2. INITIAL POPULATION (modified runner)
   ├─ For each slot: select operator from cold-start allocation
   ├─ Inject seed bank material if matching type exists
   ├─ Generate concept, evaluate, store
   └─ Unavailable operators (crossover, compost) redistributed

3. EVOLUTION (ShinkaEvolve takes over)
   ├─ Per generation: propose → evaluate → select → archive
   ├─ Operator bandit adapts weights (UCB1, after warmup)
   ├─ Meta-summarizer runs every N generations
   └─ text_feedback flows back into mutation prompts (use_text_feedback: true)

4. TEARDOWN
   ├─ Export winners to output/stage_1_run_<timestamp>/winners/
   ├─ Export archive snapshot
   ├─ Process compost candidates (read flags from DB → write to compost table)
   └─ Write run summary (cost, generations, convergence signals)
```

evaluate.py runs as a subprocess per concept. It receives `--program_path`,
`--results_dir`, and `--config` (via ShinkaEvolve's `extra_cmd_args`). It
handles Gates 1-3, judge evaluation, classification, and compost flagging.

---

## Configuration

YAML files in `configs/`. One per run configuration. Judge personas are
separate files in `configs/judges/` — the stage config references them by ID.

See `configs/stage_1_default.yaml` for the complete reference config with all
parameters documented inline.

---

## Testing Strategy

### Unit Tests (no API keys)

- `test_judging/` — scoring math (Hölder mean, disagreement signal), Tier A
  filters, persona loading, config validation
- `test_stage_1/` — genome validation, constraint density classification,
  operator prompt formatting, MAP-Elites cell key computation
- `test_llm/` — cache key generation, prompt assembly, cost calculation

### Integration Tests (require API keys)

- End-to-end: seed concept → mutate → evaluate → classify → archive
- Judge panel: run 3 judges on a known concept, verify score structure
- Cost tracking: verify token counts match expectations

### Smoke Tests

- Classification prompt produces valid JSON for 10 diverse test concepts
- All 11 operator prompts produce parseable genomes
- Gate 1 validation catches malformed genomes

---

## Error Handling

Don't over-engineer. The pipeline is exploratory — individual concept failures
are expected and cheap.

- **LLM call fails:** Retry (backoff handles this). After max retries, mark
  concept as `correct: false` and move on. One failed concept out of 180 doesn't
  matter.
- **JSON parse fails:** Mark `correct: false`. The evolutionary loop discards
  invalid genomes.
- **Judge disagrees wildly:** That's signal, not error. Track variance, apply
  diversity bonus.
- **Cost overrun:** ShinkaEvolve has budget tracking. Set a hard cap per run.
  Stop gracefully when hit.

Don't catch and silence errors. Let them surface. Fix patterns, not instances.

---

## Logging

Use Python's `logging` module (ShinkaEvolve already does). Key events to log:

- Per-generation: concepts generated, evaluated, archived, composted
- Per-concept: operator used, model used, scores, MAP-Elites cell
- Per-run: total cost, archive occupancy, convergence signals
- Warnings: high similarity rejection, cost approaching budget, stagnation

ShinkaEvolve's web UI provides real-time visualization. Our custom evaluate.py
metrics should be compatible with their dashboard format.
