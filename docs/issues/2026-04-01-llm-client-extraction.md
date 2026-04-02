# LLM Client Extraction from ShinkaEvolve

**Date:** 2026-04-01
**Scope:** Extract `shinka/llm/` into `owtn/llm/`, add Anthropic prompt caching, add response caching via cachier
**References:** `docs/engineering.md` lines 152-262 (original plan), ShinkaEvolve `shinka/llm/` (14 source files)

---

## Motivation

Both `owtn` and our ShinkaEvolve fork need LLM calls. If the client lives in ShinkaEvolve and we import it, but ShinkaEvolve also imports our `evaluate.py`, we get a circular dependency. Extracting into `owtn/llm/` breaks this: both packages depend on `owtn.llm`. Additionally, Stages 2-6 won't use ShinkaEvolve's evolution engine but still need LLM calls.

## Decisions

- **Copy all, trim later.** Wholesale copy of all 7 providers (Anthropic, OpenAI, Azure, Bedrock, DeepSeek, Gemini, OpenRouter, local). Remove unused ones after we know which we need.
- **Keep instructor** for structured output parsing. Already integrated, works across providers.
- **Add Anthropic prompt caching.** `cache_control` on system message content blocks for judge evaluations. ~90% cost savings on cached tokens.
- **Add response caching via cachier.** `@cachier()` decorator on query functions for dev/test memoization. Opt-in via `OWTN_CACHE_ENABLED` env var. Pickle storage. Custom hash function for deterministic keys.
- **Scope: client extraction only.** No judge interface, no evaluate.py, no ShinkaEvolve import rewiring.

## Changes

1. Copy `shinka/llm/` → `owtn/llm/` (16 files)
2. Replace `shinka.env.load_shinka_dotenv()` with `python-dotenv`'s `load_dotenv()` in `client.py`
3. Add `system_prefix` kwarg to Anthropic provider for cache_control content blocks
4. Add `cache_read_tokens` / `cache_creation_tokens` to QueryResult
5. Add cachier decorator integration in `query.py` with custom hash function
6. Merge `system_prefix` into plain string for non-Anthropic providers in `query.py`
7. Update `pyproject.toml` with new dependencies
8. Add `.cache/` to `.gitignore`
9. Tests in `tests/test_llm/`
