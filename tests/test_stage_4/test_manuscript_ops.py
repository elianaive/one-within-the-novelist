"""Tests for owtn.stage_4.manuscript — file ops, scaffolding, parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from owtn.models.stage_4 import Manuscript, Scene
from owtn.stage_4.manuscript import (
    EditError,
    apply_edit,
    parse,
    read_text,
    render,
    scaffold_from_dag,
    write_text,
)


# ─── Parser ──────────────────────────────────────────────────────────────


def test_parse_empty_text():
    ms = parse("")
    assert ms.scenes == []
    assert ms.frontmatter is None


def test_parse_single_scene():
    ms = parse("## opening\n\nThe room was quiet.\n")
    assert len(ms.scenes) == 1
    assert ms.scenes[0].id == "opening"
    assert ms.scenes[0].body == "The room was quiet."


def test_parse_multiple_scenes():
    text = "## alpha\n\nFirst scene body.\n\n## beta\n\nSecond scene body.\n"
    ms = parse(text)
    assert ms.scene_ids() == ["alpha", "beta"]
    assert ms.scenes[0].body == "First scene body."
    assert ms.scenes[1].body == "Second scene body."


def test_parse_strips_blank_lines_around_body_keeps_intra_line_whitespace():
    """Leading / trailing blank lines around a scene body are dropped, but
    intra-line whitespace (intentional indentation, em-dashes etc.) is
    preserved verbatim."""
    text = "## a\n\n\n  Body line.  \n\n\n## b\n\n"
    ms = parse(text)
    assert ms.scenes[0].body == "  Body line.  "
    assert ms.scenes[1].body == ""


def test_parse_with_frontmatter():
    text = "---\nrun: 42\nmodel: opus\n---\n## opening\n\nHello.\n"
    ms = parse(text)
    assert ms.frontmatter == {"run": 42, "model": "opus"}
    assert ms.scenes[0].body == "Hello."


def test_parse_drops_preamble_before_first_scene():
    """Content before the first ## heading isn't a scene; we drop it
    rather than smuggle it into the first scene's body."""
    text = "Some preamble text.\n\n## scene-one\n\nReal body.\n"
    ms = parse(text)
    assert len(ms.scenes) == 1
    assert ms.scenes[0].body == "Real body."


def test_parse_ignores_deeper_headings():
    """Level-3 headings inside a scene are part of the body."""
    text = "## scene-one\n\nIntro.\n\n### subhead\n\nMore body.\n"
    ms = parse(text)
    assert len(ms.scenes) == 1
    assert "subhead" in ms.scenes[0].body
    assert "More body" in ms.scenes[0].body


def test_parse_scene_ids_with_kebab_case():
    """Stage 2 node ids use kebab-case; the parser must preserve them
    verbatim including hyphens and digits."""
    text = "## legislative-testimony-2028\n\nbody\n"
    ms = parse(text)
    assert ms.scenes[0].id == "legislative-testimony-2028"


# ─── Render / round-trip ─────────────────────────────────────────────────


def test_render_round_trip_preserves_content():
    text = "## a\n\nFirst.\n\n## b\n\nSecond.\n"
    ms = parse(text)
    rendered = render(ms)
    re_parsed = parse(rendered)
    assert re_parsed.scenes == ms.scenes


def test_render_with_frontmatter_round_trips():
    ms = Manuscript(
        frontmatter={"version": 1},
        scenes=[Scene(id="a", body="body")],
    )
    rendered = render(ms)
    assert rendered.startswith("---\n")
    re_parsed = parse(rendered)
    assert re_parsed.frontmatter == {"version": 1}
    assert re_parsed.scenes == ms.scenes


def test_render_empty_scene_has_blank_body():
    ms = Manuscript(scenes=[Scene(id="a"), Scene(id="b")])
    rendered = render(ms)
    assert "## a\n" in rendered
    assert "## b\n" in rendered


# ─── Scaffolding from DAG ────────────────────────────────────────────────


def test_scaffold_from_dag_uses_topological_order(canonical_lottery):
    text = scaffold_from_dag(canonical_lottery)
    ms = parse(text)
    # Every node id appears
    dag_ids = {n.id for n in canonical_lottery.nodes}
    assert set(ms.scene_ids()) == dag_ids
    # All scenes are empty (scaffold only)
    assert all(s.is_empty for s in ms.scenes)
    # Order respects topological order — each scene id's position equals
    # its DAG topological index (or earlier when ties allow).
    topo = canonical_lottery._check_acyclic_and_topo()
    positions = {sid: i for i, sid in enumerate(ms.scene_ids())}
    # No edge dst comes before its src in the rendered order
    for e in canonical_lottery.edges:
        assert positions[e.src] < positions[e.dst], f"{e.src} → {e.dst} reversed"


def test_scaffold_with_frontmatter_round_trips(canonical_hemingway):
    fm = {"run_id": "test", "model": "deepseek-v4-pro"}
    text = scaffold_from_dag(canonical_hemingway, frontmatter=fm)
    ms = parse(text)
    assert ms.frontmatter == fm


# ─── read_text / write_text ──────────────────────────────────────────────


def test_write_then_read_text_round_trips(tmp_path: Path):
    path = tmp_path / "story.md"
    content = "## a\n\nbody\n"
    write_text(path, content)
    assert read_text(path) == content


def test_write_text_atomic_no_tempfile_left_behind(tmp_path: Path):
    path = tmp_path / "story.md"
    write_text(path, "first")
    write_text(path, "second")
    assert read_text(path) == "second"
    # The temp-suffix file from the atomic write must not survive
    assert not (tmp_path / "story.md.tmp").exists()


def test_write_text_creates_parent_directories(tmp_path: Path):
    path = tmp_path / "subdir" / "deeper" / "story.md"
    write_text(path, "ok")
    assert path.exists()
    assert read_text(path) == "ok"


def test_read_text_with_offset_and_limit(tmp_path: Path):
    path = tmp_path / "long.md"
    write_text(path, "\n".join(f"line {i}" for i in range(10)) + "\n")
    chunk = read_text(path, offset=3, limit=2)
    assert chunk == "line 3\nline 4\n"


def test_read_text_offset_beyond_end_returns_empty(tmp_path: Path):
    path = tmp_path / "short.md"
    write_text(path, "only line\n")
    assert read_text(path, offset=10) == ""


# ─── apply_edit ──────────────────────────────────────────────────────────


def test_apply_edit_unique_match(tmp_path: Path):
    path = tmp_path / "story.md"
    write_text(path, "The quick brown fox.")
    n = apply_edit(path, "brown", "russet")
    assert n == 1
    assert read_text(path) == "The quick russet fox."


def test_apply_edit_non_unique_without_replace_all_raises(tmp_path: Path):
    path = tmp_path / "story.md"
    write_text(path, "fox fox fox")
    with pytest.raises(EditError) as exc:
        apply_edit(path, "fox", "cat")
    # Error names the count and the remediation
    assert "matches 3 places" in str(exc.value)
    assert "replace_all=True" in str(exc.value)


def test_apply_edit_with_replace_all(tmp_path: Path):
    path = tmp_path / "story.md"
    write_text(path, "fox fox fox")
    n = apply_edit(path, "fox", "cat", replace_all=True)
    assert n == 3
    assert read_text(path) == "cat cat cat"


def test_apply_edit_no_match_raises(tmp_path: Path):
    path = tmp_path / "story.md"
    write_text(path, "the brown fox")
    with pytest.raises(EditError) as exc:
        apply_edit(path, "missing", "x")
    assert "not found" in str(exc.value)


def test_apply_edit_empty_find_raises(tmp_path: Path):
    path = tmp_path / "story.md"
    write_text(path, "any content")
    with pytest.raises(EditError):
        apply_edit(path, "", "anything")
