"""File operations on `story.md`.

The manuscript is the canonical state for Stage 4 prose. The agent reads
and writes it via generic file tools (Claude-Code-shape); this module
provides the parser, scaffolding, and atomic edit primitives those tools
delegate to.

File format:
- Optional YAML frontmatter (`---` ... `---`) at the very top of the file
- Each scene is a level-2 heading (`## scene_id`) whose text is the
  Stage 2 DAG node id
- Scene body is the text between this heading and the next (or end of
  file), normalized of leading/trailing blank lines

Convention only — the agent isn't blocked from restructuring. Helpers
that walk scenes are tolerant of headings the parser doesn't recognize
as scene_ids (they get rolled into the previous scene's body).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from owtn.models.stage_2.dag import DAG
from owtn.models.stage_4.manuscript import Manuscript, Scene


SCENE_HEADING_RE = re.compile(r"^##\s+(\S.*?)\s*$", re.MULTILINE)
"""Match level-2 markdown headings. The captured group is the heading
text — taken verbatim as the scene_id. Trailing whitespace is stripped;
leading whitespace must be absent (`##` at column 0)."""

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
"""YAML frontmatter at the very top of the file, between `---` fences."""


# ─── Parsing ─────────────────────────────────────────────────────────────


def parse(text: str) -> Manuscript:
    """Parse `story.md` text into a Manuscript.

    Tolerant of an empty file, missing frontmatter, and content before
    the first scene heading (treated as preamble and dropped — the agent
    can re-add it via write_file if needed). Scene bodies are stripped
    of leading and trailing blank lines but internal whitespace is
    preserved verbatim.
    """
    frontmatter: dict | None = None
    body = text

    fm_match = FRONTMATTER_RE.match(body)
    if fm_match:
        try:
            parsed_fm = yaml.safe_load(fm_match.group(1))
            if isinstance(parsed_fm, dict):
                frontmatter = parsed_fm
        except yaml.YAMLError:
            frontmatter = None
        body = body[fm_match.end():]

    scenes: list[Scene] = []
    matches = list(SCENE_HEADING_RE.finditer(body))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        scene_body = body[start:end].strip("\n")
        scenes.append(Scene(id=m.group(1), body=scene_body))

    return Manuscript(frontmatter=frontmatter, scenes=scenes)


def render(manuscript: Manuscript) -> str:
    """Render a Manuscript back to file text. Inverse of `parse` modulo
    blank-line normalization in scene bodies (round-tripping is exact for
    files this module produced; hand-edited files may drop trailing blank
    lines on round-trip)."""
    parts: list[str] = []
    if manuscript.frontmatter is not None:
        fm_text = yaml.safe_dump(
            manuscript.frontmatter, sort_keys=False, allow_unicode=True,
        ).rstrip()
        parts.append(f"---\n{fm_text}\n---\n")
    for s in manuscript.scenes:
        parts.append(f"## {s.id}\n\n{s.body}\n" if s.body else f"## {s.id}\n\n")
    return "\n".join(parts).rstrip() + "\n"


# ─── Scaffolding ─────────────────────────────────────────────────────────


def scaffold_from_dag(dag: DAG, *, frontmatter: dict | None = None) -> str:
    """Initial `story.md` text — one empty scene per DAG node, in
    topological order.

    The agent isn't required to follow the order or even keep the
    scaffolding; this is the starting position the orchestrator writes
    before Phase 2 begins. Topological order matches DAG.\\_check\\_acyclic\\_and\\_topo
    so disclosure / motivates / causal edges land in their natural
    forward order for the down draft.
    """
    topo = dag._check_acyclic_and_topo()
    ordered_ids = sorted(topo.keys(), key=lambda nid: topo[nid])
    manuscript = Manuscript(
        frontmatter=frontmatter,
        scenes=[Scene(id=nid) for nid in ordered_ids],
    )
    return render(manuscript)


# ─── Atomic file ops ─────────────────────────────────────────────────────


def read_text(path: Path | str, *, offset: int | None = None, limit: int | None = None) -> str:
    """Read the manuscript file. `offset` and `limit` are line-based — the
    same shape the agent's `read_file` tool exposes — so an agent can
    page through a long manuscript.

    `offset` is 0-indexed line count; `limit` caps lines returned.
    Out-of-range offsets return an empty string rather than raising.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if offset is None and limit is None:
        return text
    lines = text.splitlines(keepends=True)
    start = max(0, offset or 0)
    if start >= len(lines):
        return ""
    end = len(lines) if limit is None else min(len(lines), start + limit)
    return "".join(lines[start:end])


def write_text(path: Path | str, content: str) -> None:
    """Atomic full-file overwrite. Writes to a sibling tempfile and
    renames; either the new file or the old file is visible to readers,
    never a partial write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, p)


class EditError(Exception):
    """Raised when an `edit_file` call cannot be applied unambiguously."""


def apply_edit(
    path: Path | str,
    find: str,
    replace: str,
    *,
    replace_all: bool = False,
) -> int:
    """Find/replace edit on the manuscript.

    Mirrors the project's editor-tool semantics: a unique `find` string
    is replaced once; non-unique matches require `replace_all=True` or
    an `EditError` is raised so the caller can refine the find string.
    Empty `find` and zero matches both raise — the agent gets actionable
    feedback rather than a silent no-op.

    Returns the number of replacements made.
    """
    if not find:
        raise EditError("`find` must be non-empty")
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    occurrences = text.count(find)
    if occurrences == 0:
        raise EditError(f"`find` not found in {p}")
    if occurrences > 1 and not replace_all:
        raise EditError(
            f"`find` matches {occurrences} places in {p}; pass replace_all=True "
            f"or extend `find` to make it unique"
        )
    if replace_all:
        new_text = text.replace(find, replace)
    else:
        new_text = text.replace(find, replace, 1)
    write_text(p, new_text)
    return occurrences if replace_all else 1
