"""Surgical-edit dispatch tests — bounds validation, bounded apply, full dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from owtn.models.stage_3 import (
    ConsciousnessRendering,
    Craft,
    DialogicMode,
    ImpliedAuthor,
    Rendering,
    VoiceGenome,
)
from owtn.models.stage_4 import (
    SurgicalBounds,
    SurgicalEditCommit,
    TranslatedBounds,
)
from owtn.stage_4.surgical_edit import (
    apply_bounded_edit,
    dispatch_surgical_edit,
    _commit_surgical_edit_handler,
    _SURGICAL_EDIT_SLOT,
    _validate_bounds,
)


# ─── Helpers ─────────────────────────────────────────────────────────────


SAMPLE_MANUSCRIPT = (
    "## opening\n"
    "\n"
    "The room was quiet. He set the cup down on the counter.\n"
    "She did not look up from the page she was reading.\n"
    "\n"
    "## middle\n"
    "\n"
    "Outside the window the snow continued to fall in the late afternoon.\n"
    "He waited for her to speak first.\n"
    "\n"
    "## closing\n"
    "\n"
    "The bell rang. Neither of them moved to answer it.\n"
)


class _FakeResult:
    def __init__(self, content, history=None, cost: float = 0.0):
        self.content = content
        self.new_msg_history = history or []
        self.cost = cost


def _voice() -> VoiceGenome:
    from tests.conftest import _signature_risk_for_test
    body = "He looked at her. She did not look back. " * 8
    return VoiceGenome(
        pov="third", tense="past",
        consciousness_rendering=ConsciousnessRendering(mode="external_focalization", fid_depth="none"),
        implied_author=ImpliedAuthor(stance_toward_characters="elegiac", moral_temperature="cool"),
        dialogic_mode=DialogicMode(type="monologic"),
        craft=Craft(sentence_rhythm="varied", crowding_leaping="leaping"),
        description="Spare third-person past, elegiac without emotional gloss.",
        diction="Plain Anglo-Saxon nouns; no abstractions.",
        positive_constraints=["Render scene exits with stripped declarative."],
        prohibitions=[],
        signature_risk=_signature_risk_for_test(),
        renderings=[
            Rendering(scene_id="ab", text=body),
            Rendering(scene_id="bc", text=body),
            Rendering(scene_id="cd", text=body),
        ],
        pair_id="c_test", agent_id="test",
    )


def _state_payload(tmp_path: Path) -> dict:
    story_path = tmp_path / "story.md"
    story_path.write_text(SAMPLE_MANUSCRIPT)
    return {
        "run_dir": str(tmp_path),
        "story_path": str(story_path),
        "voice_genome": _voice(),
    }


# ─── Bounds validation ──────────────────────────────────────────────────


def test_validate_bounds_accepts_valid():
    bounds = TranslatedBounds(
        anchor_before="The room was quiet. He set the cup down",
        anchor_after="Outside the window the snow continued",
    )
    assert _validate_bounds(SAMPLE_MANUSCRIPT, bounds) is None


def test_validate_bounds_rejects_non_unique_before():
    bounds = TranslatedBounds(
        anchor_before="He waited for her to speak first.\n",  # would repeat with similar text...
        anchor_after="zzzzzzzzzz",
    )
    # Make anchor_before non-unique by manufacturing a manuscript that contains it twice
    text = SAMPLE_MANUSCRIPT + "He waited for her to speak first.\n"
    err = _validate_bounds(text, bounds)
    assert err is not None
    assert "anchor_before" in err
    assert "2 locations" in err


def test_validate_bounds_rejects_missing_after():
    bounds = TranslatedBounds(
        anchor_before="The room was quiet. He set the cup down",
        anchor_after="this string is not in the manuscript at all",
    )
    err = _validate_bounds(SAMPLE_MANUSCRIPT, bounds)
    assert err is not None
    assert "anchor_after" in err
    assert "0 locations" in err


def test_validate_bounds_rejects_after_before_before():
    bounds = TranslatedBounds(
        anchor_before="Outside the window the snow continued",
        anchor_after="The room was quiet. He set the cup down",
    )
    err = _validate_bounds(SAMPLE_MANUSCRIPT, bounds)
    assert err is not None
    assert "anchor_after appears before" in err


# ─── apply_bounded_edit ─────────────────────────────────────────────────


def test_apply_replaces_bracketed_region(tmp_path: Path):
    p = tmp_path / "story.md"
    p.write_text(SAMPLE_MANUSCRIPT)
    bounds = SurgicalBounds(
        anchor_before="## middle\n\n",
        anchor_after="\n\n## closing",
    )
    new_content = "Outside the window the snow had stopped.\nHe spoke first this time."
    err = apply_bounded_edit(p, bounds, new_content)
    assert err is None
    text = p.read_text()
    # Anchors preserved verbatim
    assert "## middle\n\n" in text
    assert "\n\n## closing" in text
    # Old content gone, new content present
    assert "Outside the window the snow continued" not in text
    assert "snow had stopped" in text
    # Surrounding scenes untouched
    assert "The room was quiet. He set the cup down on the counter." in text
    assert "The bell rang. Neither of them moved to answer it." in text


def test_apply_rejects_when_new_content_breaks_anchor(tmp_path: Path):
    p = tmp_path / "story.md"
    p.write_text(SAMPLE_MANUSCRIPT)
    bounds = SurgicalBounds(
        anchor_before="## middle\n\n",
        anchor_after="\n\n## closing",
    )
    # The new_content contains the closing anchor — would make it
    # non-unique post-edit, so the apply must refuse.
    new_content = "Some prose.\n\n## closing\n\nMore prose."
    err = apply_bounded_edit(p, bounds, new_content)
    assert err is not None
    assert "anchor" in err.lower()
    # Original file untouched
    assert p.read_text() == SAMPLE_MANUSCRIPT


def test_apply_rejects_when_anchor_no_longer_unique(tmp_path: Path):
    """Defensive — the same bounds applied twice would fail on the
    second call because the anchor_before would resolve twice (the
    original location plus the one inside the previously-pasted region
    if it duplicated the anchor). We simulate by mutating the file
    between bounds capture and apply."""
    p = tmp_path / "story.md"
    p.write_text(SAMPLE_MANUSCRIPT + "## middle\n\nduplicate scene heading\n")
    bounds = SurgicalBounds(
        anchor_before="## middle\n\n",
        anchor_after="\n\n## closing",
    )
    err = apply_bounded_edit(p, bounds, "anything")
    assert err is not None
    assert "uniquely" in err.lower()


# ─── commit_surgical_edit handler ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_commit_handler_outside_dispatch_context():
    from owtn.orchestration import ToolContext
    ctx = ToolContext(session_id="s", phase_id="p", agent_id="a", state_view={})
    result = await _commit_surgical_edit_handler({"new_content": "x"}, ctx)
    assert result.startswith("ERROR")
    assert "surgical-edit dispatch context" in result


@pytest.mark.asyncio
async def test_commit_handler_stashes_body():
    from owtn.orchestration import ToolContext
    ctx = ToolContext(session_id="s", phase_id="p", agent_id="a", state_view={})
    slot: dict = {"commit": None}
    token = _SURGICAL_EDIT_SLOT.set(slot)
    try:
        result = await _commit_surgical_edit_handler({"new_content": "the new prose"}, ctx)
    finally:
        _SURGICAL_EDIT_SLOT.reset(token)
    assert "Surgical edit committed" in result
    assert isinstance(slot["commit"], SurgicalEditCommit)
    assert slot["commit"].new_content == "the new prose"


@pytest.mark.asyncio
async def test_commit_handler_double_commit_rejected():
    from owtn.orchestration import ToolContext
    ctx = ToolContext(session_id="s", phase_id="p", agent_id="a", state_view={})
    slot: dict = {"commit": None}
    token = _SURGICAL_EDIT_SLOT.set(slot)
    try:
        first = await _commit_surgical_edit_handler({"new_content": "one"}, ctx)
        assert "committed" in first
        second = await _commit_surgical_edit_handler({"new_content": "two"}, ctx)
        assert second.startswith("ERROR")
        assert "already committed" in second
    finally:
        _SURGICAL_EDIT_SLOT.reset(token)


# ─── End-to-end mocked dispatch ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_surgical_edit_end_to_end_success(tmp_path: Path):
    state = _state_payload(tmp_path)

    bounds_returned = TranslatedBounds(
        anchor_before="## middle\n\n",
        anchor_after="\n\n## closing",
        rationale="Brackets the middle scene's body — the writer wants to tighten it.",
    )

    async def fake_translator(**kwargs):
        return _FakeResult(bounds_returned)

    async def fake_subagent(**kwargs):
        # The subagent commits via the dispatch closure
        await kwargs["dispatch"]("commit_surgical_edit", {
            "new_content": "Outside the window the snow had stopped.\nHe spoke first.",
        })
        return _FakeResult("done", history=[], cost=0.05)

    with patch("owtn.stage_4.surgical_edit.query_async", new=fake_translator), \
         patch("owtn.stage_4.surgical_edit.query_async_with_tools", new=fake_subagent):
        result_str = await dispatch_surgical_edit(
            scope_description="the middle scene body",
            instruction="tighten this paragraph",
            state_payload=state,
        )

    payload = json.loads(result_str)
    assert payload["ok"] is True
    assert "snow had stopped" in Path(state["story_path"]).read_text()


@pytest.mark.asyncio
async def test_dispatch_surgical_edit_translator_returns_invalid_bounds(tmp_path: Path):
    """Translator returns anchors that don't resolve uniquely — dispatcher
    surfaces the validation error to the parent agent without touching
    the manuscript."""
    state = _state_payload(tmp_path)
    original = Path(state["story_path"]).read_text()

    bad_bounds = TranslatedBounds(
        anchor_before="this string is not in the manuscript at all",
        anchor_after="this one isn't either",
    )

    async def fake_translator(**kwargs):
        return _FakeResult(bad_bounds)

    with patch("owtn.stage_4.surgical_edit.query_async", new=fake_translator):
        result_str = await dispatch_surgical_edit(
            scope_description="something ambiguous",
            instruction="anything",
            state_payload=state,
        )

    payload = json.loads(result_str)
    assert payload["ok"] is False
    assert "could not resolve scope" in payload["error"]
    # Manuscript untouched
    assert Path(state["story_path"]).read_text() == original


@pytest.mark.asyncio
async def test_dispatch_surgical_edit_subagent_fails_to_commit(tmp_path: Path):
    """Subagent returns without calling commit_surgical_edit — dispatcher
    surfaces the failure; manuscript untouched."""
    state = _state_payload(tmp_path)
    original = Path(state["story_path"]).read_text()

    bounds_returned = TranslatedBounds(
        anchor_before="## middle\n\n",
        anchor_after="\n\n## closing",
    )

    async def fake_translator(**kwargs):
        return _FakeResult(bounds_returned)

    async def fake_subagent(**kwargs):
        # Don't dispatch commit; just return.
        return _FakeResult("done", history=[], cost=0.0)

    with patch("owtn.stage_4.surgical_edit.query_async", new=fake_translator), \
         patch("owtn.stage_4.surgical_edit.query_async_with_tools", new=fake_subagent):
        result_str = await dispatch_surgical_edit(
            scope_description="the middle scene body",
            instruction="tighten this",
            state_payload=state,
        )

    payload = json.loads(result_str)
    assert payload["ok"] is False
    assert "did not call commit_surgical_edit" in payload["error"]
    assert Path(state["story_path"]).read_text() == original


@pytest.mark.asyncio
async def test_dispatch_surgical_edit_subagent_commit_collides_with_anchor(tmp_path: Path):
    """If the subagent's new_content contains the closing anchor, the
    apply step rejects to keep future edits in the same region
    locatable. Manuscript stays untouched on that failure."""
    state = _state_payload(tmp_path)
    original = Path(state["story_path"]).read_text()

    bounds_returned = TranslatedBounds(
        anchor_before="## middle\n\n",
        anchor_after="\n\n## closing",
    )

    async def fake_translator(**kwargs):
        return _FakeResult(bounds_returned)

    async def fake_subagent(**kwargs):
        # new_content includes "## closing" — would make anchor_after non-unique
        await kwargs["dispatch"]("commit_surgical_edit", {
            "new_content": "Some prose.\n\n## closing\n\nMore prose.",
        })
        return _FakeResult("done", history=[], cost=0.0)

    with patch("owtn.stage_4.surgical_edit.query_async", new=fake_translator), \
         patch("owtn.stage_4.surgical_edit.query_async_with_tools", new=fake_subagent):
        result_str = await dispatch_surgical_edit(
            scope_description="the middle scene body",
            instruction="rewrite",
            state_payload=state,
        )

    payload = json.loads(result_str)
    assert payload["ok"] is False
    assert "anchor" in payload["error"].lower()
    assert Path(state["story_path"]).read_text() == original


@pytest.mark.asyncio
async def test_dispatch_surgical_edit_missing_voice_genome(tmp_path: Path):
    """Defensive — voice_genome must be present in state for the
    subagent prompt to render."""
    state = _state_payload(tmp_path)
    del state["voice_genome"]

    result_str = await dispatch_surgical_edit(
        scope_description="x", instruction="y", state_payload=state,
    )
    payload = json.loads(result_str)
    assert payload["ok"] is False
    assert "voice_genome" in payload["error"]


@pytest.mark.asyncio
async def test_dispatch_surgical_edit_missing_manuscript(tmp_path: Path):
    state = {"run_dir": str(tmp_path), "voice_genome": _voice()}
    result_str = await dispatch_surgical_edit(
        scope_description="x", instruction="y", state_payload=state,
    )
    payload = json.loads(result_str)
    assert payload["ok"] is False
    assert "does not exist" in payload["error"]
