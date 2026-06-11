"""Surgical-edit dispatch — Flavor B (translate then constrain).

Two-step orchestration: a cheap haiku-class scope translator turns the
parent agent's natural-language scope into a verbatim anchor pair; a
sonnet-class surgical-edit subagent reads the full manuscript for context and
commits a replacement for the bracketed region only. The handler
splices the replacement back into the manuscript between the anchors,
re-validates, and returns a summary to the parent agent.

The architectural constraint: the surgical-edit subagent's only write surface
is `commit_surgical_edit(new_content)`, which by construction can only
modify text inside the anchor pair. Out-of-bounds edits are an
architectural impossibility, not a failure-to-detect.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from owtn.llm.kwargs import sample_model_kwargs
from owtn.llm.query import query_async
from owtn.llm.tool_use import query_async_with_tools
from owtn.models.stage_3 import VoiceGenome
from owtn.models.stage_4 import (
    AudienceFraming,
    SurgicalBounds,
    SurgicalEditCommit,
    TranslatedBounds,
)
from owtn.orchestration import ToolContext, ToolSpec, write_transcript
from owtn.prompts.stage_4 import _load, build_surgical_edit_subagent_prompt
from owtn.tools.file_ops import READ_FILE
from owtn.stage_4.manuscript import read_text, write_text


logger = logging.getLogger(__name__)


DEFAULT_TRANSLATOR_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_SURGICAL_EDIT_MODEL = "claude-sonnet-4-6"
DEFAULT_SURGICAL_EDIT_MAX_ITERS = 10


# ─── ContextVar slot ─────────────────────────────────────────────────────


_SURGICAL_EDIT_SLOT: ContextVar[dict | None] = ContextVar("_surgical_edit_slot", default=None)


# ─── Bounds validation ──────────────────────────────────────────────────


def _validate_bounds(text: str, bounds: TranslatedBounds) -> str | None:
    """Return None when bounds are valid against `text`, or an error
    string explaining why not. The dispatcher surfaces the error to the
    parent agent verbatim — it's the agent's signal that the scope
    description was ambiguous and needs refinement."""
    if text.count(bounds.anchor_before) != 1:
        return (
            f"anchor_before resolves to {text.count(bounds.anchor_before)} locations "
            "in the manuscript (must be exactly 1)"
        )
    if text.count(bounds.anchor_after) != 1:
        return (
            f"anchor_after resolves to {text.count(bounds.anchor_after)} locations "
            "in the manuscript (must be exactly 1)"
        )
    after_idx = text.find(bounds.anchor_after)
    before_end = text.find(bounds.anchor_before) + len(bounds.anchor_before)
    if after_idx < before_end:
        return "anchor_after appears before (or overlaps with) anchor_before"
    return None


def _between(text: str, bounds: SurgicalBounds) -> tuple[int, int, str]:
    """Return (start_idx, end_idx, region_text) — the locations and
    current text of the bracketed region, exclusive of the anchors
    themselves. Caller is responsible for having validated bounds."""
    start = text.find(bounds.anchor_before) + len(bounds.anchor_before)
    end = text.find(bounds.anchor_after, start)
    return start, end, text[start:end]


# ─── Translator ─────────────────────────────────────────────────────────


async def _translate_scope_to_bounds(
    *,
    scope_description: str,
    manuscript_text: str,
    model: str,
) -> TranslatedBounds | str:
    """Run the haiku scope translator. Returns a `TranslatedBounds` on
    success or an error string on validation failure. The error string
    is what the parent agent sees — actionable feedback for the agent
    to refine its scope description."""
    user_msg = (
        _load("surgical_edit_translator.txt")
        .replace("{MANUSCRIPT}", manuscript_text.strip())
        .replace("{SCOPE}", scope_description.strip())
    )
    try:
        result = await query_async(
            model_name=model,
            msg=user_msg,
            system_msg=(
                "You are translating a writer's natural-language description "
                "of a manuscript region into a verbatim anchor pair that "
                "brackets the editable area for a surgical edit."
            ),
            output_model=TranslatedBounds,
            **_translator_kwargs(model),
        )
    except Exception as e:
        logger.warning("surgical-edit translator failed (%s: %s)", type(e).__name__, e)
        return f"translator call failed ({type(e).__name__}: {e})"
    bounds = result.content
    if not isinstance(bounds, TranslatedBounds):
        return f"translator returned unexpected type {type(bounds).__name__}"
    err = _validate_bounds(manuscript_text, bounds)
    if err is not None:
        return f"could not resolve scope to a bounded region: {err}"
    return bounds


def _translator_kwargs(model: str) -> dict[str, Any]:
    out = sample_model_kwargs(
        model_names=[model],
        reasoning_efforts=["medium"],
        temperatures=[0.4],
        max_tokens=[8192],
    )
    out.pop("model_name", None)
    return out


# ─── Surgical-edit subagent ─────────────────────────────────────────────────────


async def _commit_surgical_edit_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    slot = _SURGICAL_EDIT_SLOT.get()
    if slot is None:
        return "ERROR: not in a surgical-edit dispatch context"
    if slot.get("commit") is not None:
        return "ERROR: surgical edit already committed; stop calling tools"
    try:
        body = SurgicalEditCommit.model_validate(params)
    except ValidationError as e:
        return (
            "ERROR: SurgicalEditCommit validation failed. Fix the issues and "
            f"call commit_surgical_edit again with corrected fields.\n\n{e}"
        )
    slot["commit"] = body
    return (
        "Surgical edit committed. Stop calling tools — the surrounding "
        "anchors will be reattached by the handler."
    )


COMMIT_SURGICAL_EDIT = ToolSpec(
    name="commit_surgical_edit",
    description=(
        "Submit your replacement text for the bracketed region. The "
        "anchors before and after are reattached verbatim by the "
        "handler — do not include them in `new_content`. After this "
        "call succeeds, stop calling tools."
    ),
    parameters=SurgicalEditCommit.model_json_schema(),
    handler=_commit_surgical_edit_handler,
)


def _build_subagent_prompt(
    *,
    bounds: SurgicalBounds,
    region_text: str,
    instruction: str,
    voice_genome: VoiceGenome,
    audience_framing,
) -> str:
    return build_surgical_edit_subagent_prompt(
        voice_genome=voice_genome,
        region_text=region_text,
        instruction=instruction,
        anchor_before=bounds.anchor_before,
        anchor_after=bounds.anchor_after,
        audience_framing=audience_framing,
    )


async def _run_surgical_edit_subagent(
    *,
    bounds: SurgicalBounds,
    region_text: str,
    instruction: str,
    voice_genome: VoiceGenome,
    state_payload: Mapping[str, Any],
    model: str,
    max_iters: int,
) -> tuple[SurgicalEditCommit | str, str, list[dict]]:
    """Run the surgical-edit subagent's tool-use loop. Returns the
    committed edit (or an error string), the system prompt, and the
    msg_history so the caller can write a transcript."""
    af = state_payload.get("audience_framing")
    system_msg = _build_subagent_prompt(
        bounds=bounds,
        region_text=region_text,
        instruction=instruction,
        voice_genome=voice_genome,
        audience_framing=af if isinstance(af, AudienceFraming) else None,
    )

    spec_by_name = {
        READ_FILE.name: READ_FILE,
        COMMIT_SURGICAL_EDIT.name: COMMIT_SURGICAL_EDIT,
    }
    tool_schemas = [
        {"name": s.name, "description": s.description, "parameters": dict(s.parameters)}
        for s in spec_by_name.values()
    ]

    async def dispatch(tool_name: str, params: dict) -> str:
        spec = spec_by_name.get(tool_name)
        if spec is None:
            return f"ERROR: tool {tool_name!r} not allowed for the surgical-edit subagent"
        ctx = ToolContext(
            session_id="surgical_edit_subagent",
            phase_id="surgical_edit",
            agent_id="surgical_edit",
            state_view=state_payload,
        )
        return await spec.handler(params, ctx)

    slot: dict[str, SurgicalEditCommit | None] = {"commit": None}
    token = _SURGICAL_EDIT_SLOT.set(slot)
    msg_history: list[dict] = []
    try:
        result = await query_async_with_tools(
            model_name=model,
            msg="Begin.",
            system_msg=system_msg,
            tools=tool_schemas,
            dispatch=dispatch,
            max_iters=max_iters,
            **_subagent_kwargs(model),
        )
        msg_history = list(result.new_msg_history)
    except Exception as e:
        logger.warning("surgical-edit subagent failed (%s: %s)", type(e).__name__, e)
        return f"surgical-edit subagent call failed ({type(e).__name__}: {e})", system_msg, msg_history
    finally:
        _SURGICAL_EDIT_SLOT.reset(token)

    commit = slot.get("commit")
    if commit is None:
        return (
            f"surgical-edit subagent did not call commit_surgical_edit within {max_iters} iters",
            system_msg,
            msg_history,
        )
    return commit, system_msg, msg_history


def _subagent_kwargs(model: str) -> dict[str, Any]:
    out = sample_model_kwargs(
        model_names=[model],
        reasoning_efforts=["medium"],
        temperatures=[0.7],
        max_tokens=[16384],
    )
    out.pop("model_name", None)
    return out


# ─── Apply + revalidate ─────────────────────────────────────────────────


def apply_bounded_edit(
    story_path: Path | str,
    bounds: SurgicalBounds,
    new_content: str,
) -> str | None:
    """Splice `new_content` into `story.md` between the anchors. The
    anchors themselves stay verbatim; only the region between is
    replaced.

    Returns None on success or an error string on failure (anchors
    couldn't be located uniquely; or the post-write re-read shows the
    anchors are no longer locatable, indicating new_content broke
    them). On failure the original file is left untouched.
    """
    p = Path(story_path)
    text = p.read_text(encoding="utf-8")
    if text.count(bounds.anchor_before) != 1:
        return f"anchor_before no longer resolves uniquely in {p}"
    if text.count(bounds.anchor_after) != 1:
        return f"anchor_after no longer resolves uniquely in {p}"

    start, end, _region = _between(text, bounds)
    new_text = text[:start] + new_content + text[end:]

    # Pre-flight re-validation: the new text must keep both anchors
    # uniquely resolvable, else future edits in the same region break.
    if new_text.count(bounds.anchor_before) != 1:
        return "new_content collides with anchor_before — refusing to apply"
    if new_text.count(bounds.anchor_after) != 1:
        return "new_content collides with anchor_after — refusing to apply"

    write_text(p, new_text)
    return None


# ─── Public dispatcher ──────────────────────────────────────────────────


async def dispatch_surgical_edit(
    *,
    scope_description: str,
    instruction: str,
    state_payload: Mapping[str, Any],
    translator_model: str = DEFAULT_TRANSLATOR_MODEL,
    surgical_edit_model: str = DEFAULT_SURGICAL_EDIT_MODEL,
    max_iters: int = DEFAULT_SURGICAL_EDIT_MAX_ITERS,
) -> str:
    """Run a surgical-edit dispatch end-to-end. Returns a JSON-shaped status
    string for the parent agent's tool-result channel."""
    story_path = state_payload.get("story_path")
    if story_path is None:
        run_dir = state_payload.get("run_dir")
        if run_dir is None:
            return _result(ok=False, error="run_dir not set in session state")
        story_path = str(Path(run_dir) / "story.md")
    p = Path(story_path)
    if not p.exists():
        return _result(ok=False, error=f"manuscript {p} does not exist")
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        return _result(ok=False, error="manuscript is empty")

    voice_genome = state_payload.get("voice_genome")
    if not isinstance(voice_genome, VoiceGenome):
        return _result(ok=False, error="voice_genome missing or wrong type in session state")

    translated = await _translate_scope_to_bounds(
        scope_description=scope_description,
        manuscript_text=text,
        model=translator_model,
    )
    if isinstance(translated, str):
        return _result(ok=False, error=translated)

    bounds = SurgicalBounds(
        anchor_before=translated.anchor_before,
        anchor_after=translated.anchor_after,
        scene_heading=translated.scene_heading,
    )
    _, _, region_text = _between(text, bounds)

    commit, system_msg, msg_history = await _run_surgical_edit_subagent(
        bounds=bounds,
        region_text=region_text,
        instruction=instruction,
        voice_genome=voice_genome,
        state_payload=state_payload,
        model=surgical_edit_model,
        max_iters=max_iters,
    )

    seq = _next_surgical_edit_seq(state_payload)
    cycle_idx = _current_cycle_idx(state_payload)
    label_prefix = f"cycle_{cycle_idx}_" if cycle_idx is not None else ""
    write_transcript(
        agent_id="surgical_edit_subagent",
        label=f"{label_prefix}dispatch_{seq}",
        title=(
            f"Surgical edit — dispatch {seq}"
            + (f" (cycle {cycle_idx})" if cycle_idx is not None else "")
            + f" — scope: {scope_description[:80]}"
        ),
        system_msg=system_msg,
        msg_history=msg_history,
        final_output=commit.model_dump_json(indent=2) if not isinstance(commit, str) else commit,
    )

    if isinstance(commit, str):
        return _result(ok=False, error=commit)

    err = apply_bounded_edit(p, bounds, commit.new_content)
    if err is not None:
        return _result(ok=False, error=err)

    return _result(
        ok=True,
        scene_heading=bounds.scene_heading,
        chars_replaced=len(region_text),
        chars_written=len(commit.new_content),
        rationale=translated.rationale,
    )


def _result(*, ok: bool, **fields: Any) -> str:
    return json.dumps({"ok": ok, **fields}, ensure_ascii=False, indent=2)


def _next_surgical_edit_seq(state_payload: Mapping[str, Any]) -> int:
    """1-indexed dispatch counter scoped to the session, used to give
    each surgical-edit transcript a distinct filename."""
    if not isinstance(state_payload, dict):
        return 1
    n = state_payload.get("_surgical_edit_dispatch_count", 0) + 1
    state_payload["_surgical_edit_dispatch_count"] = n
    return n


def _current_cycle_idx(state_payload: Mapping[str, Any]) -> int | None:
    """The cycle index this dispatch is happening inside, if any. None
    when surgical-edit is invoked outside a Phase 3 cycle (e.g. tests)."""
    p3 = state_payload.get("phase_3_revise") if isinstance(state_payload, dict) else None
    if not isinstance(p3, dict):
        return None
    cycles = p3.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        return None
    last = cycles[-1]
    if isinstance(last, dict):
        return last.get("cycle")
    return None
