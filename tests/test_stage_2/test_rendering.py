"""Rendering tests: structural-marker checks + snapshot equality.

Two layers:
1. **Structural markers** — the rendered output contains the required headers,
   sections, and edge verbs. Catches regressions in the renderer's contract
   without being fragile to whitespace or formatting tweaks.
2. **Snapshot equality** — current rendering matches the byte-stable snapshot
   in `snapshots/canonical_<name>.txt`. Catches any change in output, intended
   or accidental.

Updating snapshots intentionally: regenerate via the CLI:

    for n in lottery hemingway chiang oconnor; do
        uv run python -m owtn.stage_2.rendering \
            tests/test_stage_2/fixtures/canonical_${n}.json \
            > tests/test_stage_2/snapshots/canonical_${n}.txt
    done

A diff in `git status` after that exposes exactly what changed. If the change
was intentional, commit; otherwise revert and fix the renderer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from owtn.models.stage_2.dag import DAG
from owtn.stage_2.rendering import render


SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


# ----- Structural markers -----

class TestRenderingMarkers:
    """Checks that don't tie to specific text — catches structural breakage."""

    def test_header_includes_label(self, canonical_lottery: DAG) -> None:
        out = render(canonical_lottery, label="A")
        assert out.startswith("STORY STRUCTURE A\n=================\n")

    def test_label_b_renders_with_b_header(self, canonical_lottery: DAG) -> None:
        out = render(canonical_lottery, label="B")
        assert out.startswith("STORY STRUCTURE B")

    def test_every_node_appears_with_brackets(self, any_canonical: DAG) -> None:
        out = render(any_canonical)
        for node in any_canonical.nodes:
            assert f"[{node.id}" in out, (
                f"node id {node.id!r} not present in rendering"
            )

    def test_motif_threads_section_present(self, any_canonical: DAG) -> None:
        out = render(any_canonical)
        assert "MOTIF THREADS" in out
        for thread in any_canonical.motif_threads:
            assert thread in out

    def test_character_arcs_section_when_arcs_present(self, any_canonical: DAG) -> None:
        out = render(any_canonical)
        if any_canonical.character_arcs:
            assert "CHARACTER ARCS" in out
            for arc in any_canonical.character_arcs:
                assert arc.agent in out
        else:
            assert "CHARACTER ARCS" not in out

    def test_story_constraints_section_when_present(self, any_canonical: DAG) -> None:
        out = render(any_canonical)
        if any_canonical.story_constraints:
            assert "STORY CONSTRAINTS" in out
        else:
            assert "STORY CONSTRAINTS" not in out

    def test_concept_demands_section_only_when_non_empty(self, any_canonical: DAG) -> None:
        out = render(any_canonical)
        if any_canonical.concept_demands:
            assert "CONCEPT DEMANDS" in out
            for demand in any_canonical.concept_demands:
                assert demand in out
        else:
            assert "CONCEPT DEMANDS" not in out

    def test_structural_tensions_appendix_present_when_disclosure_or_motivates(
        self, any_canonical: DAG,
    ) -> None:
        has_disclosure_or_motivates = any(
            e.type in ("disclosure", "motivates") for e in any_canonical.edges
        )
        out = render(any_canonical)
        if has_disclosure_or_motivates:
            assert "STRUCTURAL TENSIONS" in out

    def test_role_annotation_in_node_header(self, canonical_oconnor: DAG) -> None:
        # O'Connor's grace is multi-role: climax+pivot
        out = render(canonical_oconnor)
        assert "[grace, role=climax+pivot]" in out

    def test_disclosed_to_reader_visible(self, canonical_lottery: DAG) -> None:
        out = render(canonical_lottery)
        assert "disclosed_to: reader" in out

    def test_disclosed_to_multi_audience_visible(self, canonical_oconnor: DAG) -> None:
        # opening → arrival is the dual-audience case
        out = render(canonical_oconnor)
        assert "disclosed_to: reader, the grandmother" in out

    def test_motif_modes_rendered_inline(self, canonical_oconnor: DAG) -> None:
        out = render(canonical_oconnor)
        # All 5 modes present in O'Connor: introduced, performed, agent, echoed, inverted
        assert "[introduced]" in out
        assert "[performed]" in out
        assert "[agent]" in out
        assert "[echoed]" in out
        assert "[inverted]" in out

    def test_edge_verbs_used(self, any_canonical: DAG) -> None:
        out = render(any_canonical)
        edge_types = {e.type for e in any_canonical.edges}
        verb_for = {
            "causal": "causes",
            "disclosure": "discloses to",
            "implication": "entails",
            "constraint": "prevents",
            "motivates": "motivates",
        }
        for et in edge_types:
            assert verb_for[et] in out, (
                f"edge type {et!r} present but verb {verb_for[et]!r} missing"
            )


# ----- Snapshot equality -----

@pytest.mark.parametrize("canonical_name", [
    "lottery", "hemingway", "chiang", "oconnor",
])
def test_snapshot_match(canonical_name: str) -> None:
    fixture = Path(__file__).parent / "fixtures" / f"canonical_{canonical_name}.json"
    snapshot = SNAPSHOTS_DIR / f"canonical_{canonical_name}.txt"

    assert snapshot.exists(), (
        f"snapshot {snapshot} missing — regenerate with: "
        f"uv run python -m owtn.stage_2.rendering {fixture} > {snapshot}"
    )

    dag = DAG.model_validate_json(fixture.read_text())
    expected = snapshot.read_text()
    actual = render(dag)

    assert actual == expected, (
        f"rendering of canonical_{canonical_name} differs from snapshot.\n"
        f"To accept the change: regenerate the snapshot via the CLI command in "
        f"the test docstring. To investigate: diff stdout from the CLI against "
        f"{snapshot}."
    )


# ----- Determinism -----

def test_rendering_is_deterministic(canonical_chiang: DAG) -> None:
    """Same DAG renders identically across two calls (no hidden randomness)."""
    a = render(canonical_chiang)
    b = render(canonical_chiang)
    assert a == b


def test_label_only_changes_header(canonical_lottery: DAG) -> None:
    """A and B renderings differ only in the header — important for pairwise
    judging where formatting differences would confound position-bias."""
    out_a = render(canonical_lottery, label="A")
    out_b = render(canonical_lottery, label="B")
    body_a = out_a[out_a.index("\n\n"):]
    body_b = out_b[out_b.index("\n\n"):]
    assert body_a == body_b
