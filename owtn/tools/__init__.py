"""Cross-stage tools — agent-callable utilities used by voice agents
(Stage 3) and potentially by Stage 4/5 prose-generation and refinement
loops.

Each tool is its own file; this module re-exports the public surface.

Available tools:
- `stylometry()` — stylometric signals + centroid distances on a passage
- `lookup_exemplar()` — fetch reference passages (exemplars + baselines only;
  defaults are blocked)
- `slop_score()` — EQ-Bench slop-score port: weighted composite of slop
  vocabulary, "not X but Y" contrast patterns, and slop n-grams
- `writing_style()` — surface-level register signals (vocab grade, sentence
  length, paragraph length, dialogue frequency) positioned against the
  project calibration corpus
- `thesaurus()` — Datamuse API wrapper for diction work (synonyms, phonetic
  neighbours, related/antonym/adjective/noun lookups)

Internal helpers:
- `_corpus` — voice-references corpus loader and signal cache (private to
  the tools module)

Unified CLI:
    uv run python -m owtn.tools <command> [options]
where <command> is one of:
    analyze | lookup | slop | style | thesaurus | rebuild-cache
"""

from .lookup_exemplar import lookup_exemplar
from .slop_score import SlopScoreReport, slop_score
from .stylometry import StylometricToolReport, rebuild_cache, stylometry
from .thesaurus import ThesaurusReport, thesaurus
from .writing_style import WritingStyleReport, writing_style

__all__ = [
    "SlopScoreReport",
    "StylometricToolReport",
    "ThesaurusReport",
    "WritingStyleReport",
    "lookup_exemplar",
    "rebuild_cache",
    "slop_score",
    "stylometry",
    "thesaurus",
    "writing_style",
]
