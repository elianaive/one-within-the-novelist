"""Public API of owtn.llm.

`owtn.llm` is a utility layer: provider plumbing, the query path, and
telemetry. Stage- or pipeline-specific behavior (self-critic, lineage briefs,
etc.) lives outside this package.
"""

# Public API is exposed via submodules:
#   owtn.llm.query   → query, query_async
#   owtn.llm.providers.result → QueryResult
#   owtn.llm.call_logger → llm_context, llm_log_dir
# This file is intentionally empty so importing `owtn.llm` is cheap.
