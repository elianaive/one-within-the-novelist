"""Surgical-edit dispatch — Pydantic shapes.

Three small models:

- `TranslatedBounds` — what the cheap haiku-class scope translator
  returns. Validated by Pydantic on the way in; the dispatcher then
  checks the anchors actually resolve uniquely against the current
  manuscript before letting the surgical-edit subagent run.
- `SurgicalBounds` — the validated runtime form. Same fields, just
  carries the verified-against-the-file invariant.
- `SurgicalEditCommit` — what the surgical-edit subagent calls
  `commit_surgical_edit` with. Just `new_content`, the replacement text
  for the bracketed region.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranslatedBounds(BaseModel):
    """Translator structured output. Anchors are verbatim slices of the
    current manuscript that bracket the editable region. Both must
    resolve to a unique location in the file; validator at dispatch
    time enforces that — the model can't reliably check uniqueness on
    its own, so we validate post-parse."""
    anchor_before: str = Field(min_length=10)
    anchor_after: str = Field(min_length=10)
    scene_heading: str | None = None
    rationale: str = Field(default="", description="One sentence on what region this brackets and why.")


class SurgicalBounds(BaseModel):
    """Runtime bounds — what the orchestrator passes to the surgical-edit
    subagent's prompt. Same fields as `TranslatedBounds`; this type
    carries the validated-against-the-file invariant."""
    anchor_before: str
    anchor_after: str
    scene_heading: str | None = None


class SurgicalEditCommit(BaseModel):
    """Body the surgical-edit subagent passes to `commit_surgical_edit`. Just
    the replacement text for the bracketed region — the surrounding
    prose is reattached verbatim by the handler."""
    new_content: str
