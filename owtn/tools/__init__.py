"""Cross-stage tools — agent-callable utilities used by voice agents
(Stage 3) and potentially by Stage 4/5 prose-generation and refinement
loops.

Each tool is its own file; this module re-exports the public surface.

Available tools:
- `stylometry()` — stylometric signals + centroid distances on a passage
- `lookup_exemplar()` — fetch reference passages (exemplars + baselines only;
  defaults are blocked)

Internal helpers:
- `_corpus` — voice-references corpus loader and signal cache (private to
  the tools module)

Unified CLI:
    uv run python -m owtn.tools <command> [options]
where <command> is one of: analyze | lookup | rebuild-cache
"""

from .lookup_exemplar import lookup_exemplar
from .stylometry import StylometricToolReport, rebuild_cache, stylometry

__all__ = [
    "StylometricToolReport",
    "lookup_exemplar",
    "rebuild_cache",
    "stylometry",
]
