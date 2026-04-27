"""Public API of owtn.llm.

`owtn.llm` is a utility layer: provider plumbing, the query path, and
telemetry. Stage- or pipeline-specific behavior (self-critic, lineage briefs,
etc.) lives outside this package.

Importing this package loads `.env` at the project root so providers can
read API keys via `os.environ[...]` without callers having to set up
dotenv themselves. This is important: `uv run` does NOT auto-load `.env`
(despite some folklore — verified 2026-04-27).
"""

from dotenv import load_dotenv as _load_dotenv

_load_dotenv()

# Public API is exposed via submodules:
#   owtn.llm.api     → query, query_async
#   owtn.llm.result  → QueryResult
#   owtn.llm.call_logger → llm_context, llm_log_dir
