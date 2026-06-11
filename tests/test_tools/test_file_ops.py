"""Tests for owtn.tools.file_ops — sandbox resolver + handlers."""

from __future__ import annotations

from pathlib import Path

import pytest

from owtn.orchestration import ToolContext
from owtn.tools.file_ops import (
    EDIT_FILE,
    READ_FILE,
    WRITE_FILE,
    edit_file_handler,
    read_file_handler,
    resolve_sandbox_path,
    write_file_handler,
)


def _ctx(state_view: dict) -> ToolContext:
    return ToolContext(
        session_id="sess_test", phase_id="p", agent_id="a", state_view=state_view,
    )


def _state(tmp_path: Path) -> dict:
    sandbox = tmp_path / "sandbox" / "writer"
    sandbox.mkdir(parents=True)
    return {"run_dir": str(tmp_path), "sandbox_dir": str(sandbox)}


# ─── ToolSpec sanity ─────────────────────────────────────────────────────


def test_specs_export_expected_names():
    assert READ_FILE.name == "read_file"
    assert WRITE_FILE.name == "write_file"
    assert EDIT_FILE.name == "edit_file"


# ─── resolve_sandbox_path ───────────────────────────────────────────────


def test_resolve_returns_path_for_relative(tmp_path: Path):
    state = _state(tmp_path)
    resolved = resolve_sandbox_path(state, "story.md")
    assert isinstance(resolved, Path)
    assert resolved == Path(state["sandbox_dir"]).resolve() / "story.md"


def test_resolve_rejects_absolute(tmp_path: Path):
    state = _state(tmp_path)
    err = resolve_sandbox_path(state, "/etc/passwd")
    assert isinstance(err, str)
    assert "absolute paths" in err


def test_resolve_rejects_escape(tmp_path: Path):
    state = _state(tmp_path)
    err = resolve_sandbox_path(state, "../escaped.md")
    assert isinstance(err, str)
    assert "outside your sandbox" in err


def test_resolve_rejects_missing_sandbox_dir():
    err = resolve_sandbox_path({}, "story.md")
    assert isinstance(err, str)
    assert "sandbox_dir" in err


def test_resolve_rejects_nonexistent_sandbox(tmp_path: Path):
    err = resolve_sandbox_path({"sandbox_dir": str(tmp_path / "nope")}, "story.md")
    assert isinstance(err, str)
    assert "does not exist" in err


# ─── Handlers — happy path ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_then_read_round_trips(tmp_path: Path):
    state = _state(tmp_path)
    ctx = _ctx(state)
    written = await write_file_handler({"path": "story.md", "content": "hello\n"}, ctx)
    assert "Wrote" in written
    text = await read_file_handler({"path": "story.md"}, ctx)
    assert text == "hello\n"


@pytest.mark.asyncio
async def test_edit_unique_match(tmp_path: Path):
    state = _state(tmp_path)
    ctx = _ctx(state)
    await write_file_handler({"path": "x.md", "content": "the brown fox"}, ctx)
    edited = await edit_file_handler(
        {"path": "x.md", "find": "brown", "replace": "russet"}, ctx,
    )
    assert "1 replacement" in edited
    assert (await read_file_handler({"path": "x.md"}, ctx)) == "the russet fox"


@pytest.mark.asyncio
async def test_edit_replace_all(tmp_path: Path):
    state = _state(tmp_path)
    ctx = _ctx(state)
    await write_file_handler({"path": "x.md", "content": "fox fox fox"}, ctx)
    edited = await edit_file_handler(
        {"path": "x.md", "find": "fox", "replace": "cat", "replace_all": True}, ctx,
    )
    assert "3 replacement" in edited
    assert (await read_file_handler({"path": "x.md"}, ctx)) == "cat cat cat"


@pytest.mark.asyncio
async def test_read_with_offset_and_limit(tmp_path: Path):
    state = _state(tmp_path)
    ctx = _ctx(state)
    await write_file_handler(
        {"path": "long.md", "content": "\n".join(f"line {i}" for i in range(10)) + "\n"},
        ctx,
    )
    chunk = await read_file_handler({"path": "long.md", "offset": 3, "limit": 2}, ctx)
    assert chunk == "line 3\nline 4\n"


@pytest.mark.asyncio
async def test_write_creates_parent_dirs(tmp_path: Path):
    """`write_file` should mkdir the path's parents inside the sandbox."""
    state = _state(tmp_path)
    ctx = _ctx(state)
    result = await write_file_handler(
        {"path": "subdir/deeper/note.md", "content": "ok"}, ctx,
    )
    assert "Wrote" in result
    assert (Path(state["sandbox_dir"]) / "subdir" / "deeper" / "note.md").read_text() == "ok"


# ─── Handlers — error paths ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_missing_file(tmp_path: Path):
    ctx = _ctx(_state(tmp_path))
    err = await read_file_handler({"path": "missing.md"}, ctx)
    assert err.startswith("ERROR")
    assert "does not exist" in err


@pytest.mark.asyncio
async def test_read_empty_path(tmp_path: Path):
    ctx = _ctx(_state(tmp_path))
    err = await read_file_handler({"path": ""}, ctx)
    assert err.startswith("ERROR")
    assert "non-empty" in err


@pytest.mark.asyncio
async def test_write_rejects_non_string_content(tmp_path: Path):
    ctx = _ctx(_state(tmp_path))
    err = await write_file_handler({"path": "x.md", "content": 42}, ctx)
    assert err.startswith("ERROR")
    assert "must be a string" in err


@pytest.mark.asyncio
async def test_edit_rejects_when_file_missing(tmp_path: Path):
    ctx = _ctx(_state(tmp_path))
    err = await edit_file_handler(
        {"path": "nope.md", "find": "x", "replace": "y"}, ctx,
    )
    assert err.startswith("ERROR")
    assert "create it with write_file first" in err


@pytest.mark.asyncio
async def test_edit_non_unique_returns_remediation(tmp_path: Path):
    ctx = _ctx(_state(tmp_path))
    await write_file_handler({"path": "x.md", "content": "fox fox fox"}, ctx)
    err = await edit_file_handler(
        {"path": "x.md", "find": "fox", "replace": "cat"}, ctx,
    )
    assert err.startswith("ERROR")
    assert "matches 3 places" in err
    assert "replace_all" in err


# ─── Sandbox isolation ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_cannot_read_files_above_sandbox(tmp_path: Path):
    """run_dir contains orchestrator-owned files (critic JSONs, plateau
    logs, parent_log.yaml). The agent's read_file must not reach them."""
    state = _state(tmp_path)
    (tmp_path / "parent_log.yaml").write_text("orchestrator-owned\n")
    (tmp_path / "critiques").mkdir()
    (tmp_path / "critiques" / "cycle_0.json").write_text("{}\n")
    ctx = _ctx(state)
    for path in ("../parent_log.yaml", "../critiques/cycle_0.json"):
        err = await read_file_handler({"path": path}, ctx)
        assert err.startswith("ERROR")
        assert "outside" in err


@pytest.mark.asyncio
async def test_agent_cannot_write_outside_sandbox(tmp_path: Path):
    state = _state(tmp_path)
    ctx = _ctx(state)
    err = await write_file_handler(
        {"path": "../naughty.md", "content": "should not land"}, ctx,
    )
    assert err.startswith("ERROR")
    assert "outside" in err
    assert not (tmp_path / "naughty.md").exists()
