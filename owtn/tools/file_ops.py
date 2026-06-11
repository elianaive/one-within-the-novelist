"""Generic file-operation tools — read_file, write_file, edit_file.

Stage-agnostic. The sandbox shape is:

    state.payload["sandbox_dir"] = Path  # the calling agent's writable area

All three tools resolve relative paths against `sandbox_dir`. Absolute
paths and paths that escape `sandbox_dir` (via `..`) are rejected with
an actionable error string back to the LLM.

The sandbox is per-agent on disk (`run_dir/sandbox/{agent_id}/`); the
composer is responsible for scaffolding it and pointing
`state.payload["sandbox_dir"]` at the active agent's directory before
phases run. Critics share the writer's sandbox for reads since the
manuscript they evaluate lives there.

The sandbox layer keeps the agent's tool surface narrow: critic JSONs,
plateau logs, parent_log.yaml, and the session log tree all live OUTSIDE
the sandbox at `run_dir/`, where the orchestrator writes them and the
agent never sees them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from owtn.orchestration import ToolContext, ToolSpec


# ─── Atomic file primitives ──────────────────────────────────────────────


def _read_text(path: Path, *, offset: int | None = None, limit: int | None = None) -> str:
    """Read a text file. `offset` and `limit` are 0-indexed lines so the
    LLM can page through long files without loading everything."""
    text = path.read_text(encoding="utf-8")
    if offset is None and limit is None:
        return text
    lines = text.splitlines(keepends=True)
    start = max(0, offset or 0)
    if start >= len(lines):
        return ""
    end = len(lines) if limit is None else min(len(lines), start + limit)
    return "".join(lines[start:end])


def _write_text(path: Path, content: str) -> None:
    """Atomic full-file overwrite. Writes to a sibling tempfile and
    renames; either the new file or the prior content is visible to
    readers, never a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


class EditError(Exception):
    """Raised when an `edit_file` call cannot be applied unambiguously."""


def _apply_edit(path: Path, find: str, replace: str, *, replace_all: bool = False) -> int:
    """Find/replace edit. A unique `find` is replaced once; non-unique
    matches require `replace_all=True` or raise so the caller can refine
    `find`. Returns the number of replacements made."""
    if not find:
        raise EditError("`find` must be non-empty")
    text = path.read_text(encoding="utf-8")
    occurrences = text.count(find)
    if occurrences == 0:
        raise EditError(f"`find` not found in {path}")
    if occurrences > 1 and not replace_all:
        raise EditError(
            f"`find` matches {occurrences} places in {path}; pass replace_all=True "
            f"or extend `find` to make it unique"
        )
    if replace_all:
        new_text = text.replace(find, replace)
    else:
        new_text = text.replace(find, replace, 1)
    _write_text(path, new_text)
    return occurrences if replace_all else 1


# ─── Sandbox path resolution ─────────────────────────────────────────────


def resolve_sandbox_path(state_view: Mapping[str, Any], path: str) -> Path | str:
    """Resolve `path` against `state.payload["sandbox_dir"]`.

    Returns a `Path` on success or an error string the handler can ship
    back to the agent. Absolute paths and paths that resolve outside
    the sandbox are rejected — the agent shouldn't reach for arbitrary
    filesystem paths and the sandbox keeps misbehavior contained.
    """
    sandbox_dir = state_view.get("sandbox_dir")
    if sandbox_dir is None:
        return "ERROR: sandbox_dir not set in session state; cannot resolve paths"
    base = Path(sandbox_dir).resolve()
    if not base.exists():
        return f"ERROR: sandbox_dir {base} does not exist"
    p = Path(path)
    if p.is_absolute():
        return "ERROR: absolute paths not allowed; pass a path relative to your sandbox"
    resolved = (base / p).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        return f"ERROR: path {path!r} resolves outside your sandbox"
    return resolved


# ─── Handlers ────────────────────────────────────────────────────────────


async def read_file_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    path = params.get("path", "")
    if not path:
        return "ERROR: read_file requires non-empty `path`"
    resolved = resolve_sandbox_path(ctx.state_view, path)
    if isinstance(resolved, str):
        return resolved
    if not resolved.exists():
        return f"ERROR: {path} does not exist (yet); write_file creates it"
    if not resolved.is_file():
        return f"ERROR: {path} is not a regular file"
    offset = params.get("offset")
    limit = params.get("limit")
    try:
        return _read_text(resolved, offset=offset, limit=limit)
    except OSError as e:
        return f"ERROR: read failed ({type(e).__name__}: {e})"


async def write_file_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    path = params.get("path", "")
    content = params.get("content", "")
    if not path:
        return "ERROR: write_file requires non-empty `path`"
    if not isinstance(content, str):
        return "ERROR: write_file `content` must be a string"
    resolved = resolve_sandbox_path(ctx.state_view, path)
    if isinstance(resolved, str):
        return resolved
    try:
        _write_text(resolved, content)
    except OSError as e:
        return f"ERROR: write failed ({type(e).__name__}: {e})"
    return f"Wrote {len(content)} chars to {path}."


async def edit_file_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    path = params.get("path", "")
    find = params.get("find", "")
    replace = params.get("replace", "")
    replace_all = bool(params.get("replace_all", False))
    if not path:
        return "ERROR: edit_file requires non-empty `path`"
    if not isinstance(find, str) or not isinstance(replace, str):
        return "ERROR: edit_file `find` and `replace` must be strings"
    resolved = resolve_sandbox_path(ctx.state_view, path)
    if isinstance(resolved, str):
        return resolved
    if not resolved.exists():
        return f"ERROR: {path} does not exist; create it with write_file first"
    try:
        n = _apply_edit(resolved, find, replace, replace_all=replace_all)
    except EditError as e:
        return f"ERROR: {e}"
    except OSError as e:
        return f"ERROR: edit failed ({type(e).__name__}: {e})"
    return f"Edited {path}: {n} replacement(s)."


# ─── ToolSpec definitions ────────────────────────────────────────────────


READ_FILE = ToolSpec(
    name="read_file",
    description=(
        "Read a file in your sandbox. Pass a path relative to your "
        "sandbox root (e.g. `story.md`, `pre_think.md`). Optional "
        "`offset` and `limit` page through long files line-by-line. "
        "Returns the file's text, or an ERROR string when the path is "
        "missing / outside your sandbox / unreadable."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to your sandbox root."},
            "offset": {"type": "integer", "description": "0-indexed start line.", "minimum": 0},
            "limit": {"type": "integer", "description": "Max lines to return.", "minimum": 1},
        },
        "required": ["path"],
    },
    handler=read_file_handler,
)


WRITE_FILE = ToolSpec(
    name="write_file",
    description=(
        "Write a file in your sandbox, full overwrite. Atomic — either "
        "the new content or the prior content is visible, never a "
        "partial write. Creates parent directories as needed. Use "
        "write_file when the file is new or you want to replace its "
        "whole contents; use edit_file for find/replace inside an "
        "existing file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to your sandbox root."},
            "content": {"type": "string", "description": "Full new contents."},
        },
        "required": ["path", "content"],
    },
    handler=write_file_handler,
)


EDIT_FILE = ToolSpec(
    name="edit_file",
    description=(
        "Find/replace inside an existing file in your sandbox. `find` "
        "must match exactly once unless `replace_all=true`; non-unique "
        "matches return an ERROR so you can extend `find` to make it "
        "unique. Atomic. This is a mechanical substitution — it does "
        "the swap and nothing else. For changes that need a reader "
        "holding the surrounding prose in mind while reshaping a "
        "passage, dispatch a focused-passage edit instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to your sandbox root."},
            "find": {"type": "string", "description": "Verbatim string to find."},
            "replace": {"type": "string", "description": "Replacement string."},
            "replace_all": {
                "type": "boolean",
                "description": "When true, replace every occurrence; default false.",
                "default": False,
            },
        },
        "required": ["path", "find", "replace"],
    },
    handler=edit_file_handler,
)
