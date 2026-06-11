"""Tests for owtn.stage_4.tools — ToolSpec set, allowlist, finalize handlers."""

from __future__ import annotations

from pathlib import Path

import pytest

from owtn.orchestration import ToolContext, ToolRegistry
from owtn.stage_4.tools import (
    ALL_STAGE_4_TOOLS,
    PHASE_1_PRETHINK_TOOLS,
    PHASE_2_DOWNDRAFT_TOOLS,
    PHASE_3A_GATHER_TOOLS,
    PHASE_3B_REVISE_TOOLS,
    STAGE_4_PHASE_ALLOW,
    TIER_A_CRITICS,
    _finalize_critique_plan_handler,
    _finalize_cycle_handler,
    _finalize_down_draft_handler,
    _finalize_pre_think_handler,
    _finalize_stage_4_handler,
)
from owtn.tools.file_ops import (
    edit_file_handler as _edit_file_handler,
    read_file_handler as _read_file_handler,
    write_file_handler as _write_file_handler,
)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _ctx(state_view: dict, *, phase_id: str = "phase_1_prethink", agent_id: str = "stage_4_agent") -> ToolContext:
    return ToolContext(
        session_id="sess_test",
        phase_id=phase_id,
        agent_id=agent_id,
        state_view=state_view,
    )


def _state_with_sandbox(tmp_path: Path) -> dict:
    """Minimal state.payload shape: a sandbox_dir the file tools resolve
    against. The sandbox is `<tmp_path>/sandbox`; tests treat it as the
    writer agent's playground."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    return {"run_dir": str(tmp_path), "sandbox_dir": str(sandbox)}


# ─── Spec / registry contracts ───────────────────────────────────────────


def test_all_tools_register_without_collision():
    registry = ToolRegistry(ALL_STAGE_4_TOOLS, per_phase_allow=STAGE_4_PHASE_ALLOW)
    names = registry.names()
    assert "read_file" in names
    assert "write_file" in names
    assert "edit_file" in names
    assert "call_critic" in names
    assert "dispatch_surgical_edit" in names
    assert "web_search" in names
    for finalize in (
        "finalize_pre_think", "finalize_down_draft",
        "finalize_critique_plan", "finalize_cycle", "finalize_stage_4",
    ):
        assert finalize in names


def test_phase_allow_references_only_known_tools():
    """ToolRegistry raises on unknown allowlisted names — confirms our
    table doesn't drift from the actual tool set."""
    ToolRegistry(ALL_STAGE_4_TOOLS, per_phase_allow=STAGE_4_PHASE_ALLOW)


def test_phase_keys_are_the_committed_four():
    assert set(STAGE_4_PHASE_ALLOW.keys()) == {
        "phase_1_prethink",
        "phase_2_downdraft",
        "phase_3a_gather",
        "phase_3b_revise",
    }


def test_tier_a_critics_are_the_committed_four():
    assert TIER_A_CRITICS == frozenset({
        "voice_fidelity", "payload_enactment",
        "continuity", "motif_fidelity",
    })


def test_phase_3a_excludes_edit_tools():
    """Sub-phase A: critic tools, NO edits — structural enforcement."""
    assert "write_file" not in PHASE_3A_GATHER_TOOLS
    assert "edit_file" not in PHASE_3A_GATHER_TOOLS
    assert "dispatch_surgical_edit" not in PHASE_3A_GATHER_TOOLS
    assert "call_critic" in PHASE_3A_GATHER_TOOLS


def test_phase_3b_excludes_critic_tools():
    """Sub-phase B: edit tools, NO critics — gather-then-revise enforcement."""
    assert "call_critic" not in PHASE_3B_REVISE_TOOLS
    assert "write_file" in PHASE_3B_REVISE_TOOLS
    assert "edit_file" in PHASE_3B_REVISE_TOOLS
    assert "dispatch_surgical_edit" in PHASE_3B_REVISE_TOOLS


def test_web_search_is_critic_only_not_in_any_agent_phase():
    """web_search is domain_expert-only; no agent-phase exposes it."""
    for tools in STAGE_4_PHASE_ALLOW.values():
        assert "web_search" not in tools


def test_finalize_tools_are_phase_specific():
    assert "finalize_pre_think" in PHASE_1_PRETHINK_TOOLS
    assert "finalize_pre_think" not in PHASE_2_DOWNDRAFT_TOOLS
    assert "finalize_down_draft" in PHASE_2_DOWNDRAFT_TOOLS
    assert "finalize_down_draft" not in PHASE_3A_GATHER_TOOLS
    assert "finalize_critique_plan" in PHASE_3A_GATHER_TOOLS
    assert "finalize_critique_plan" not in PHASE_3B_REVISE_TOOLS
    assert "finalize_cycle" in PHASE_3B_REVISE_TOOLS
    assert "finalize_stage_4" in PHASE_3B_REVISE_TOOLS


def test_schemas_for_phase_1_returns_expected_set():
    registry = ToolRegistry(ALL_STAGE_4_TOOLS, per_phase_allow=STAGE_4_PHASE_ALLOW)
    agent_tools = frozenset(spec.name for spec in ALL_STAGE_4_TOOLS)
    schemas = registry.schemas_for(agent_tools, "phase_1_prethink")
    names = {s["name"] for s in schemas}
    assert names == PHASE_1_PRETHINK_TOOLS


# ─── File-op handlers ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_then_read_via_handlers(tmp_path: Path):
    state = _state_with_sandbox(tmp_path)
    ctx = _ctx(state)
    write_result = await _write_file_handler(
        {"path": "story.md", "content": "## a\n\nbody\n"}, ctx,
    )
    assert "Wrote" in write_result
    read_result = await _read_file_handler({"path": "story.md"}, ctx)
    assert read_result == "## a\n\nbody\n"


@pytest.mark.asyncio
async def test_read_file_missing_returns_actionable_error(tmp_path: Path):
    ctx = _ctx(_state_with_sandbox(tmp_path))
    result = await _read_file_handler({"path": "missing.md"}, ctx)
    assert result.startswith("ERROR")
    assert "does not exist" in result


@pytest.mark.asyncio
async def test_read_file_rejects_absolute_path(tmp_path: Path):
    ctx = _ctx(_state_with_sandbox(tmp_path))
    result = await _read_file_handler({"path": "/etc/passwd"}, ctx)
    assert result.startswith("ERROR")
    assert "absolute" in result


@pytest.mark.asyncio
async def test_read_file_rejects_path_escaping_sandbox(tmp_path: Path):
    """The sandbox is `<tmp_path>/sandbox`; `../secret.md` resolves to
    `<tmp_path>/secret.md`, outside the sandbox — rejected. Critic
    JSONs and parent_log.yaml live above the sandbox at run_dir level
    and must remain unreadable by agents."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (tmp_path / "secret.md").write_text("orchestrator-only data")
    ctx = _ctx({"run_dir": str(tmp_path), "sandbox_dir": str(sandbox)})
    result = await _read_file_handler({"path": "../secret.md"}, ctx)
    assert result.startswith("ERROR")
    assert "outside" in result


@pytest.mark.asyncio
async def test_read_file_without_sandbox_dir_returns_setup_error():
    ctx = _ctx({})
    result = await _read_file_handler({"path": "story.md"}, ctx)
    assert result.startswith("ERROR")
    assert "sandbox_dir" in result


@pytest.mark.asyncio
async def test_edit_file_via_handler(tmp_path: Path):
    ctx = _ctx(_state_with_sandbox(tmp_path))
    await _write_file_handler(
        {"path": "story.md", "content": "the brown fox"}, ctx,
    )
    result = await _edit_file_handler(
        {"path": "story.md", "find": "brown", "replace": "russet"}, ctx,
    )
    assert "1 replacement" in result
    final = await _read_file_handler({"path": "story.md"}, ctx)
    assert final == "the russet fox"


@pytest.mark.asyncio
async def test_edit_file_non_unique_match_returns_remediation(tmp_path: Path):
    ctx = _ctx(_state_with_sandbox(tmp_path))
    await _write_file_handler(
        {"path": "story.md", "content": "fox fox fox"}, ctx,
    )
    result = await _edit_file_handler(
        {"path": "story.md", "find": "fox", "replace": "cat"}, ctx,
    )
    assert result.startswith("ERROR")
    assert "matches 3 places" in result


# ─── finalize_pre_think / finalize_down_draft ────────────────────────────


@pytest.mark.asyncio
async def test_finalize_pre_think_marks_state_and_blocks_double_commit():
    state: dict = {"run_dir": "/tmp/x"}
    ctx = _ctx(state)
    first = await _finalize_pre_think_handler({}, ctx)
    assert "Pre-think committed" in first
    assert state["phase_1_prethink"]["committed_by"] == "stage_4_agent"
    second = await _finalize_pre_think_handler({}, ctx)
    assert second.startswith("ERROR")


@pytest.mark.asyncio
async def test_finalize_down_draft_marks_state():
    state: dict = {}
    ctx = _ctx(state)
    msg = await _finalize_down_draft_handler({}, ctx)
    assert "Down draft committed" in msg
    assert state["phase_2_downdraft"]["committed_by"] == "stage_4_agent"


# ─── finalize_critique_plan: Tier A enforcement ──────────────────────────


def _phase_3_initialized(cycle: int = 0, critic_calls: list[str] | None = None) -> dict:
    """Build a state.payload with Phase 3 set up + one active cycle."""
    return {
        "phase_3_revise": {
            "cycles": [
                {
                    "cycle": cycle,
                    "critic_calls": list(critic_calls or []),
                    "plan": None,
                    "completed": False,
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_finalize_critique_plan_rejects_when_phase_3_not_initialized():
    ctx = _ctx({})
    result = await _finalize_critique_plan_handler(
        {"plan_summary": "valid plan summary text", "intended_revisions": []}, ctx,
    )
    assert result.startswith("ERROR")
    assert "no active revision pass" in result


@pytest.mark.asyncio
async def test_finalize_critique_plan_rejects_when_tier_a_missing():
    state = _phase_3_initialized(critic_calls=["voice_fidelity", "continuity"])
    ctx = _ctx(state)
    result = await _finalize_critique_plan_handler(
        {"plan_summary": "valid plan summary text", "intended_revisions": []}, ctx,
    )
    assert result.startswith("ERROR")
    assert "Tier A critics missing" in result
    # The two missing names should appear in the message
    assert "payload_enactment" in result
    assert "motif_fidelity" in result


@pytest.mark.asyncio
async def test_finalize_critique_plan_accepts_when_all_tier_a_fired():
    state = _phase_3_initialized(critic_calls=list(TIER_A_CRITICS))
    ctx = _ctx(state)
    result = await _finalize_critique_plan_handler(
        {
            "plan_summary": "Tighten voice in scenes 2-3 and sharpen the disclosure beat.",
            "intended_revisions": ["Remove emotional gloss from weights-freeze opening.", "Restructure the testimony's Q-and-A rhythm."],
            "unresolved_observations": ["transportation flagged a subtle stutter mid-anchor; deferred."],
        },
        ctx,
    )
    assert "Critique plan committed" in result
    plan = state["phase_3_revise"]["cycles"][0]["plan"]
    assert plan["plan_summary"].startswith("Tighten voice")
    assert len(plan["intended_revisions"]) == 2


@pytest.mark.asyncio
async def test_finalize_critique_plan_rejects_invalid_plan_body():
    state = _phase_3_initialized(critic_calls=list(TIER_A_CRITICS))
    ctx = _ctx(state)
    result = await _finalize_critique_plan_handler(
        {"plan_summary": "short", "intended_revisions": []}, ctx,
    )
    assert result.startswith("ERROR")
    assert "validation failed" in result


# ─── finalize_cycle / finalize_stage_4 ───────────────────────────────────


@pytest.mark.asyncio
async def test_finalize_cycle_requires_committed_plan():
    state = _phase_3_initialized(critic_calls=list(TIER_A_CRITICS))
    ctx = _ctx(state)
    result = await _finalize_cycle_handler({}, ctx)
    assert result.startswith("ERROR")
    assert "has no committed plan" in result


@pytest.mark.asyncio
async def test_finalize_cycle_marks_completed_when_plan_present():
    state = _phase_3_initialized(critic_calls=list(TIER_A_CRITICS))
    state["phase_3_revise"]["cycles"][0]["plan"] = {"committed": True}
    ctx = _ctx(state)
    result = await _finalize_cycle_handler({}, ctx)
    assert "Revision pass complete" in result
    assert state["phase_3_revise"]["cycles"][0]["completed"] is True


@pytest.mark.asyncio
async def test_finalize_stage_4_rejects_before_any_cycle_completes():
    state = _phase_3_initialized(critic_calls=list(TIER_A_CRITICS))
    ctx = _ctx(state)
    result = await _finalize_stage_4_handler({}, ctx)
    assert result.startswith("ERROR")
    assert "at least one complete cycle" in result


@pytest.mark.asyncio
async def test_finalize_stage_4_accepts_after_one_cycle_completes():
    state = _phase_3_initialized()
    state["phase_3_revise"]["cycles"][0]["completed"] = True
    state["phase_3_revise"]["cycles"][0]["plan"] = {"committed": True}
    ctx = _ctx(state)
    result = await _finalize_stage_4_handler({}, ctx)
    assert "Stage 4 complete" in result
    assert state["phase_3_revise"]["committed_by"] == "stage_4_agent"


# ─── dispatch_surgical_edit input validation ────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_surgical_edit_rejects_empty_scope():
    """Real handler — empty scope_description rejected before any LLM call."""
    from owtn.stage_4.tools import _dispatch_surgical_edit_handler
    result = await _dispatch_surgical_edit_handler(
        {"scope_description": "", "instruction": "tighten this paragraph"}, _ctx({}),
    )
    assert result.startswith("ERROR")
    assert "scope_description" in result


@pytest.mark.asyncio
async def test_dispatch_surgical_edit_rejects_empty_instruction():
    from owtn.stage_4.tools import _dispatch_surgical_edit_handler
    result = await _dispatch_surgical_edit_handler(
        {"scope_description": "the second paragraph", "instruction": ""}, _ctx({}),
    )
    assert result.startswith("ERROR")
    assert "instruction" in result


@pytest.mark.asyncio
async def test_dispatch_surgical_edit_passes_config_models_through(monkeypatch):
    """When state.payload carries `surgical_edit_config`, the handler must
    forward the configured translator/subagent models to dispatch_surgical_edit
    rather than letting it fall back to module-level defaults. Regression for
    the bug where surgical-edit models were hard-coded to sonnet/haiku with no
    config plumbing."""
    from owtn.models.stage_4 import Stage4SurgicalEditConfig
    from owtn.stage_4 import tools as tools_mod
    from owtn.stage_4.tools import _dispatch_surgical_edit_handler

    captured: dict = {}

    async def fake_dispatch(**kwargs):
        captured.update(kwargs)
        return '{"ok": true}'

    monkeypatch.setattr("owtn.stage_4.surgical_edit.dispatch_surgical_edit", fake_dispatch)

    se_cfg = Stage4SurgicalEditConfig(
        translator_model="claude-haiku-4-5-20251001",
        subagent_model="claude-opus-4-7",
    )
    ctx = _ctx({"surgical_edit_config": se_cfg, "run_dir": "/tmp"})
    out = await _dispatch_surgical_edit_handler(
        {"scope_description": "p2", "instruction": "tighten"}, ctx,
    )
    assert out == '{"ok": true}'
    assert captured["translator_model"] == "claude-haiku-4-5-20251001"
    assert captured["surgical_edit_model"] == "claude-opus-4-7"


def test_light_yaml_surgical_edit_models_load():
    """Sanity check: configs/stage_4/light.yaml exposes surgical_edit
    translator and subagent models so they can be tuned per-run."""
    from owtn.models.stage_4 import Stage4Config

    cfg = Stage4Config.from_yaml("configs/stage_4/light.yaml")
    assert cfg.surgical_edit.translator_model == "claude-haiku-4-5-20251001"
    assert cfg.surgical_edit.subagent_model == "claude-sonnet-4-6"
