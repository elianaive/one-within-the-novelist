"""Tests for owtn.stage_3.tools — handlers, schemas, gating contract."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from owtn.orchestration import ToolContext, ToolRegistry
from owtn.stage_3.tools import (
    ALL_VOICE_TOOLS,
    PHASE_1_TOOLS,
    PHASE_4_TOOLS,
    VOICE_PHASE_ALLOW,
    _note_to_self_handler,
    _render_adjacent_scene_handler,
)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _bench_payload() -> dict:
    """Minimal AdjacentSceneBench-shaped dict for state.payload."""
    return {
        "drafts": [
            {
                "scene_id": "morning-kitchen",
                "synopsis": "She makes coffee while he reads.",
                "demand": "Render slow domestic time without filler.",
                "why_distinct": "Tests stillness; others test motion.",
                "neutral_draft": (
                    "She poured the coffee. He turned a page. The kitchen "
                    "was cold. Outside, a car passed and was gone."
                ),
            },
            {
                "scene_id": "argument-at-door",
                "synopsis": "They disagree about whether to leave.",
                "demand": "Hold pressure inside ordinary speech.",
                "why_distinct": "Tests motion under speech.",
                "neutral_draft": (
                    "He said he would not go. She said it was not a choice. "
                    "Neither moved."
                ),
            },
            {
                "scene_id": "after-the-news",
                "synopsis": "They sit with what was said.",
                "demand": "Render aftermath without resolution.",
                "why_distinct": "Tests stillness after rupture.",
                "neutral_draft": (
                    "The news ended. The room held. Neither of them spoke "
                    "for a long time."
                ),
            },
        ],
        "bench_rationale": "rationale text long enough to pass validation",
        "picker_model": "claude-sonnet-4-6",
        "drafter_model": "claude-sonnet-4-6",
    }


def _ctx(state_view: dict, agent_id: str = "the-reductionist") -> ToolContext:
    return ToolContext(
        session_id="sess_test",
        phase_id="phase_1_private_brief",
        agent_id=agent_id,
        state_view=state_view,
    )


# ─── Spec / registry contracts ───────────────────────────────────────────


def test_all_tools_register_without_collision():
    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)
    names = registry.names()
    assert "render_adjacent_scene" in names
    assert "stylometry" in names
    assert "slop_score" in names
    assert "writing_style" in names


def test_phase_allow_references_only_known_tools():
    """ToolRegistry will raise on unknown allowlisted names — confirms our
    table doesn't drift from the actual tool set."""
    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)
    assert registry is not None  # constructor would have raised otherwise


def test_phase_1_includes_metric_ensemble():
    """Per the architecture commitment — stylometry + slop + writing_style
    in Phase 1 is the ensemble that prevents single-metric gaming."""
    assert "stylometry" in PHASE_1_TOOLS
    assert "slop_score" in PHASE_1_TOOLS
    assert "writing_style" in PHASE_1_TOOLS
    assert "lookup_reference" in PHASE_1_TOOLS
    assert "thesaurus" in PHASE_1_TOOLS


def test_phase_4_drops_lookup_reference():
    """Phase 4 is revision; reopening reference search would redo Phase 1."""
    assert "lookup_reference" not in PHASE_4_TOOLS
    # But still has the metric ensemble
    assert {"stylometry", "slop_score", "writing_style"}.issubset(PHASE_4_TOOLS)


def test_ask_judge_is_not_in_any_v0_1_phase_allow():
    """Phase 2 deferred to v0.2; ask_judge stays out of the v0.1 allowlist."""
    for tools in VOICE_PHASE_ALLOW.values():
        assert "ask_judge" not in tools


def test_phase_3_and_phase_5_are_absent_from_allowlist():
    """Pure structured-output phases — no tools."""
    assert "phase_3_reveal_critique" not in VOICE_PHASE_ALLOW
    assert "phase_5_borda" not in VOICE_PHASE_ALLOW


def test_schemas_for_phase_1_returns_expected_set():
    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)
    agent_tools = frozenset({
        "render_adjacent_scene", "think", "note_to_self", "lookup_reference",
        "stylometry", "slop_score", "writing_style", "thesaurus",
        "finalize_voice_genome", "ask_judge",
    })
    schemas = registry.schemas_for(agent_tools, "phase_1_private_brief")
    names = {s["name"] for s in schemas}
    assert names == PHASE_1_TOOLS
    assert "ask_judge" not in names
    assert "finalize_voice_genome" in names


# ─── render_adjacent_scene ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_render_adjacent_scene_returns_draft_for_known_scene():
    ctx = _ctx({"adjacent_scene_bench": _bench_payload()})
    result = await _render_adjacent_scene_handler(
        {"scene_id": "morning-kitchen"}, ctx,
    )
    parsed = json.loads(result)
    assert parsed["scene_id"] == "morning-kitchen"
    assert "She poured the coffee" in parsed["neutral_draft"]
    assert parsed["demand"].startswith("Render slow domestic")


@pytest.mark.asyncio
async def test_render_adjacent_scene_unknown_scene_returns_error():
    ctx = _ctx({"adjacent_scene_bench": _bench_payload()})
    result = await _render_adjacent_scene_handler(
        {"scene_id": "nonexistent"}, ctx,
    )
    assert result.startswith("ERROR")
    assert "morning-kitchen" in result  # lists what's available


@pytest.mark.asyncio
async def test_render_adjacent_scene_missing_bench_returns_error():
    ctx = _ctx({})
    result = await _render_adjacent_scene_handler(
        {"scene_id": "morning-kitchen"}, ctx,
    )
    assert result.startswith("ERROR")


# ─── note_to_self ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_note_to_self_acknowledges():
    ctx = _ctx({})
    result = await _note_to_self_handler(
        {"text": "Considering FID-deep but rejecting because the concept's "
                 "structural ignorance breaks any sustained interior."},
        ctx,
    )
    assert "Noted" in result


@pytest.mark.asyncio
async def test_note_to_self_empty_returns_error():
    ctx = _ctx({})
    result = await _note_to_self_handler({"text": "   "}, ctx)
    assert result.startswith("ERROR")


# ─── stylometry / slop_score / writing_style — wrappers ──────────────────


@pytest.mark.asyncio
async def test_stylometry_passes_caller_model_from_state_view():
    """The handler should pull the caller's model name from
    state_view["agent_models"][agent_id] and pass it to compute_stylometry."""
    from owtn.stage_3.tools import _stylometry_handler

    captured = {}

    def fake_compute_stylometry(passage, caller_model=None, neutral_baseline=None, **_):
        captured["caller_model"] = caller_model
        captured["neutral_baseline"] = neutral_baseline
        # Return a minimal stub with the right shape — just needs to be JSON-able
        return {"signals": {"burstiness": 0.42}, "caller_model": caller_model}

    state = {
        "adjacent_scene_bench": _bench_payload(),
        "agent_models": {"the-reductionist": "deepseek-v4-pro"},
    }
    ctx = _ctx(state)

    with patch("owtn.stage_3.tools.compute_stylometry", side_effect=fake_compute_stylometry):
        result = await _stylometry_handler(
            {"passage": "She poured the coffee.", "scene_id": "morning-kitchen"}, ctx,
        )

    parsed = json.loads(result)
    assert captured["caller_model"] == "deepseek-v4-pro"
    assert "She poured" in captured["neutral_baseline"]
    assert parsed["caller_model"] == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_stylometry_no_scene_id_omits_neutral_baseline():
    from owtn.stage_3.tools import _stylometry_handler

    captured = {}

    def fake_compute_stylometry(passage, caller_model=None, neutral_baseline=None, **_):
        captured["neutral_baseline"] = neutral_baseline
        return {"signals": {}}

    ctx = _ctx({"adjacent_scene_bench": _bench_payload(), "agent_models": {}})
    with patch("owtn.stage_3.tools.compute_stylometry", side_effect=fake_compute_stylometry):
        await _stylometry_handler({"passage": "x" * 40}, ctx)

    assert captured["neutral_baseline"] is None


@pytest.mark.asyncio
async def test_metric_handlers_reject_empty_passage():
    from owtn.stage_3.tools import (
        _slop_score_handler, _stylometry_handler, _writing_style_handler,
    )
    ctx = _ctx({"adjacent_scene_bench": _bench_payload()})

    for handler in (_stylometry_handler, _slop_score_handler, _writing_style_handler):
        result = await handler({"passage": "   "}, ctx)
        assert result.startswith("ERROR")


# ─── lookup_reference ────────────────────────────────────────────────────


def _ok_result(query: str) -> dict:
    return {
        "query": query, "match": "tags_only", "interpretation": "stub",
        "authors": [], "tags": ["x"],
        "n_returned": 2, "n_available": 5,
        "passages": [{"id": "x", "text": "..."}, {"id": "y", "text": "..."}],
        "note": "ok",
    }


def _empty_result(query: str, note: str = "no match") -> dict:
    return {
        "query": query, "match": "none", "interpretation": "stub",
        "authors": [], "tags": [],
        "n_returned": 0, "n_available": 0, "passages": [],
        "note": note,
    }


def _patch_lookup(side_effect):
    """Helper — async patch on the new lookup_exemplar_async path."""
    async def _async_side_effect(query, n=2, **_):
        return side_effect(query, n=n)
    return patch("owtn.stage_3.tools.lookup_exemplar_async", side_effect=_async_side_effect)


@pytest.mark.asyncio
async def test_lookup_reference_passes_query_through():
    from owtn.stage_3.tools import _lookup_reference_handler

    captured = {}

    def fake(query, n=2, **_):
        captured["query"] = query
        captured["n"] = n
        return _ok_result(query)

    ctx = _ctx({})
    with _patch_lookup(fake):
        result = await _lookup_reference_handler(
            {"query": "Morrison's incantatory mode", "n": 3}, ctx,
        )

    parsed = json.loads(result)
    assert captured["query"] == "Morrison's incantatory mode"
    assert captured["n"] == 3
    assert parsed["query"] == "Morrison's incantatory mode"


@pytest.mark.asyncio
async def test_lookup_reference_empty_query_returns_error():
    from owtn.stage_3.tools import _lookup_reference_handler
    ctx = _ctx({})
    result = await _lookup_reference_handler({"query": ""}, ctx)
    assert result.startswith("ERROR")


@pytest.mark.asyncio
async def test_lookup_reference_records_corpus_gap_on_no_results(tmp_path):
    """When the resolver returns no passages, the handler appends a JSONL
    entry capturing the NL query + interpretation + resolver note."""
    from owtn.stage_3.tools import _lookup_reference_handler
    from owtn.orchestration.session import session_log_dir

    def fake(query, n=2, **_):
        return {
            "query": query, "match": "none",
            "interpretation": "agent reaching for an absent author",
            "authors": [], "tags": [],
            "n_returned": 0, "n_available": 0, "passages": [],
            "note": "no Beckett entries; closest: joyce, davis (minimalist).",
        }

    ctx = _ctx({}, agent_id="the-reductionist")
    log_token = session_log_dir.set(str(tmp_path))
    try:
        with _patch_lookup(fake):
            await _lookup_reference_handler({"query": "Beckett's reductive monologue"}, ctx)
    finally:
        session_log_dir.reset(log_token)

    lines = (tmp_path / "corpus_gaps.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["query"] == "Beckett's reductive monologue"
    assert entry["match"] == "none"
    assert "Beckett" in entry["interpretation"] or "absent" in entry["interpretation"]
    assert "Beckett" in entry["note"] or "joyce" in entry["note"]
    assert entry["agent_id"] == "the-reductionist"
    assert entry["session_id"] == "sess_test"
    assert entry["phase_id"] == "phase_1_private_brief"


@pytest.mark.asyncio
async def test_lookup_reference_does_not_record_when_passages_returned(tmp_path):
    """A successful lookup must NOT pollute corpus_gaps.jsonl."""
    from owtn.stage_3.tools import _lookup_reference_handler
    from owtn.orchestration.session import session_log_dir

    ctx = _ctx({})
    log_token = session_log_dir.set(str(tmp_path))
    try:
        with _patch_lookup(lambda q, n=2: _ok_result(q)):
            await _lookup_reference_handler({"query": "free_indirect_discourse"}, ctx)
    finally:
        session_log_dir.reset(log_token)

    assert not (tmp_path / "corpus_gaps.jsonl").exists()


@pytest.mark.asyncio
async def test_lookup_reference_appends_multiple_gaps(tmp_path):
    """Multiple unmatched lookups append to the same JSONL file."""
    from owtn.stage_3.tools import _lookup_reference_handler
    from owtn.orchestration.session import session_log_dir

    ctx = _ctx({})
    log_token = session_log_dir.set(str(tmp_path))
    try:
        with _patch_lookup(lambda q, n=2: _empty_result(q)):
            await _lookup_reference_handler({"query": "unknown-style-1"}, ctx)
            await _lookup_reference_handler({"query": "unknown-style-2"}, ctx)
            await _lookup_reference_handler({"query": "unknown-style-3"}, ctx)
    finally:
        session_log_dir.reset(log_token)

    lines = (tmp_path / "corpus_gaps.jsonl").read_text().strip().split("\n")
    assert [json.loads(line)["query"] for line in lines] == [
        "unknown-style-1", "unknown-style-2", "unknown-style-3",
    ]


# ─── ask_judge stub ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thesaurus_passes_word_and_mode():
    from owtn.stage_3.tools import _thesaurus_handler

    captured = {}

    def fake_compute(word, mode="means_like", max_results=20, **_):
        captured["word"] = word
        captured["mode"] = mode
        captured["max_results"] = max_results
        return {"word": word, "mode": mode, "n_results": 0, "results": []}

    ctx = _ctx({})
    with patch("owtn.stage_3.tools.compute_thesaurus", side_effect=fake_compute):
        result = await _thesaurus_handler(
            {"word": "stillness", "mode": "means_like", "max_results": 5}, ctx,
        )

    parsed = json.loads(result)
    assert captured["word"] == "stillness"
    assert captured["mode"] == "means_like"
    assert captured["max_results"] == 5
    assert parsed["word"] == "stillness"


@pytest.mark.asyncio
async def test_thesaurus_unknown_mode_returns_error():
    from owtn.stage_3.tools import _thesaurus_handler
    ctx = _ctx({})
    result = await _thesaurus_handler({"word": "stillness", "mode": "thinks_like"}, ctx)
    assert result.startswith("ERROR")


@pytest.mark.asyncio
async def test_thesaurus_empty_word_returns_error():
    from owtn.stage_3.tools import _thesaurus_handler
    ctx = _ctx({})
    result = await _thesaurus_handler({"word": "  "}, ctx)
    assert result.startswith("ERROR")


@pytest.mark.asyncio
async def test_finalize_voice_genome_validates_and_stashes_body():
    """Tool call should validate args via VoiceGenomeBody and store the
    body in state.payload['_pending_commits'][agent_id] for the
    orchestrator to extract after the explore loop."""
    from owtn.stage_3.tools import _finalize_voice_genome_handler
    from owtn.models.stage_3 import VoiceGenomeBody

    valid_args = {
        "pov": "third",
        "tense": "past",
        "consciousness_rendering": {"mode": "external_focalization", "fid_depth": "none"},
        "implied_author": {"stance_toward_characters": "neutral", "moral_temperature": "cool"},
        "dialogic_mode": {"type": "monologic"},
        "craft": {"sentence_rhythm": "varied", "crowding_leaping": "balanced"},
        "description": (
            "A test voice for unit testing — flat, declarative, with the "
            "fluorescent honesty of a procedural register."
        ),
        "diction": "Plain. Direct. No ornament.",
        "positive_constraints": ["Render emotion through gesture, not adjective."],
        "renderings": [
            {"scene_id": f"scene-{i}", "text": "x" * 100} for i in range(3)
        ],
    }
    state_view: dict = {}
    ctx = _ctx(state_view, agent_id="the-reductionist")
    out = await _finalize_voice_genome_handler(valid_args, ctx)
    assert "Voice committed" in out
    assert "the-reductionist" in state_view["_pending_commits"]
    body = state_view["_pending_commits"]["the-reductionist"]
    assert isinstance(body, VoiceGenomeBody)
    assert body.pov == "third"
    assert body.consciousness_rendering.mode == "external_focalization"


@pytest.mark.asyncio
async def test_finalize_voice_genome_returns_validation_errors():
    """Invalid args should return an error string the model can act on,
    not raise. The agent gets feedback and can call again with corrections."""
    from owtn.stage_3.tools import _finalize_voice_genome_handler

    bad_args = {
        "pov": "omniscient",  # not a valid Literal value
        "tense": "past",
        # missing other required fields
    }
    ctx = _ctx({}, agent_id="the-reductionist")
    out = await _finalize_voice_genome_handler(bad_args, ctx)
    assert out.startswith("ERROR")
    assert "validation" in out.lower()


@pytest.mark.asyncio
async def test_finalize_voice_genome_rejects_double_commit():
    """If the agent calls finalize twice, the second should be rejected
    so the model is nudged to stop calling tools."""
    from owtn.stage_3.tools import _finalize_voice_genome_handler

    valid_args = {
        "pov": "third",
        "tense": "past",
        "consciousness_rendering": {"mode": "external_focalization", "fid_depth": "none"},
        "implied_author": {"stance_toward_characters": "neutral", "moral_temperature": "cool"},
        "dialogic_mode": {"type": "monologic"},
        "craft": {"sentence_rhythm": "varied", "crowding_leaping": "balanced"},
        "description": (
            "A test voice for unit testing — flat, declarative, with the "
            "fluorescent honesty of a procedural register."
        ),
        "diction": "Plain. Direct. No ornament.",
        "positive_constraints": ["Render emotion through gesture."],
        "renderings": [
            {"scene_id": f"scene-{i}", "text": "x" * 100} for i in range(3)
        ],
    }
    state_view: dict = {}
    ctx = _ctx(state_view, agent_id="the-reductionist")
    first = await _finalize_voice_genome_handler(valid_args, ctx)
    assert "Voice committed" in first
    second = await _finalize_voice_genome_handler(valid_args, ctx)
    assert second.startswith("ERROR")
    assert "already committed" in second.lower()


@pytest.mark.asyncio
async def test_ask_judge_returns_phase_2_deferral_message():
    from owtn.stage_3.tools import _ask_judge_handler
    ctx = _ctx({})
    result = await _ask_judge_handler(
        {"judge_id": "gwern", "question": "what makes a metaphor earn its place?"},
        ctx,
    )
    assert "deferred" in result.lower() or "v0.2" in result.lower()


# ─── Integration: registry dispatch through handlers ─────────────────────


@pytest.mark.asyncio
async def test_dispatch_render_adjacent_scene_through_registry():
    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)
    ctx = _ctx({"adjacent_scene_bench": _bench_payload()})

    result = await registry.dispatch(
        "render_adjacent_scene",
        {"scene_id": "argument-at-door"},
        ctx,
    )
    parsed = json.loads(result)
    assert parsed["scene_id"] == "argument-at-door"
