"""Stage-agnostic optimizer-state utilities.

Houses the summarizer machinery that distills per-lineage match history
(`LineageBrief`, today the per-parent brief) and — in Phase 2 — run-wide
cross-lineage signal (`PopulationBrief`). Per-stage adapters live in
`adapters.py`; the generic machinery does not know about Stage 1 concept
genome fields.

See `lab/issues/2026-04-22-global-optimizer-state.md` for the design and
`lab/issues/2026-04-24-refactor-feedback-to-optimizer-module.md` for the
refactor that created this module.
"""
