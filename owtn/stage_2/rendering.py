"""Incident-encoded DAG rendering for judges and prompts.

Produces the format specified in `docs/stage-2/evaluation.md` §DAG Rendering:
beats listed in topological order with inline edges, a STRUCTURAL TENSIONS
appendix duplicating long-range and disclosure/motivates edges, and trailing
sections for character_arcs, story_constraints, motif_threads, and
concept_demands.

Format chosen because incident encoding beats flat adjacency by 34 accuracy
points on relational reasoning (Fatemi et al., ICLR 2024) and application-domain
edge labels beat abstract labels by 18 points.

Standalone CLI:
    uv run python -m owtn.stage_2.rendering <path-to-genome.json>
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

from pydantic import ValidationError

from owtn.models.stage_2.dag import DAG, Edge, EdgeType, Node


# ----- Topological order (recomputed; same algorithm as DAG validator) -----

def _topological_order(dag: DAG) -> list[str]:
    """Return node IDs in a stable topological order (Kahn's algorithm).

    DAG has already been validated as acyclic; this recomputes the order
    so the renderer is self-contained. Stable: ties broken by node-list
    position so identical DAGs render identically.
    """
    in_degree: dict[str, int] = {n.id: 0 for n in dag.nodes}
    out_edges: dict[str, list[str]] = defaultdict(list)
    for e in dag.edges:
        in_degree[e.dst] += 1
        out_edges[e.src].append(e.dst)

    node_index = {n.id: i for i, n in enumerate(dag.nodes)}
    ready = deque(sorted(
        (nid for nid, d in in_degree.items() if d == 0),
        key=lambda nid: node_index[nid],
    ))
    order: list[str] = []
    while ready:
        nid = ready.popleft()
        order.append(nid)
        for child in out_edges[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
    return order


# ----- Edge formatting helpers -----

_EDGE_VERBS: dict[EdgeType, str] = {
    "causal": "causes",
    "disclosure": "discloses to",
    "implication": "entails",
    "constraint": "prevents",
    "motivates": "motivates",
}


def _edge_payload(edge: Edge) -> str:
    """Format an edge's type-specific payload as `field1: value1; field2: value2`."""
    if edge.type == "causal":
        fields = [("realizes", edge.realizes)]
    elif edge.type == "disclosure":
        audience = edge.disclosed_to or ["reader"]
        fields = [
            ("reframes", edge.reframes),
            ("withheld", edge.withheld),
            ("disclosed_to", ", ".join(audience)),
        ]
    elif edge.type == "implication":
        fields = [("entails", edge.entails)]
    elif edge.type == "constraint":
        fields = [("prohibits", edge.prohibits)]
    elif edge.type == "motivates":
        fields = [("agent", edge.agent), ("goal", edge.goal)]
        if edge.stakes:
            fields.append(("stakes", edge.stakes))
    else:  # pragma: no cover — Pydantic enforces EdgeType
        raise ValueError(f"unknown edge type: {edge.type}")
    return "; ".join(f"{k}: {v}" for k, v in fields if v is not None)


def _sketch_snippet(sketch: str, *, max_chars: int = 80) -> str:
    """First ~80 chars of a sketch, for inline edge target reference.

    Truncates at a word boundary near `max_chars` (no sentence detection —
    abbreviation periods like "Mr." would confuse it). Adds ellipsis when
    truncated. Renderer consumers (LLMs and snapshot tests) don't need the
    snippet to be a complete sentence.
    """
    if len(sketch) <= max_chars:
        return sketch.strip()
    cutoff = sketch[:max_chars]
    # Back up to the last whitespace so we don't slice mid-word.
    space_idx = cutoff.rfind(" ")
    if space_idx > max_chars // 2:  # only back up if there's a reasonable break
        cutoff = cutoff[:space_idx]
    return cutoff.rstrip(" .,;") + "..."


def _node_header(node: Node) -> str:
    """`[id]` or `[id, role=climax]` or `[id, role=climax+pivot]`."""
    if node.role:
        role_str = "+".join(node.role)
        label = f"[{node.id}, role={role_str}]"
    else:
        label = f"[{node.id}]"
    return f"{label} {node.sketch}"


# ----- Section renderers -----

def _render_node_section(
    node: Node,
    outgoing: list[Edge],
    nodes_by_id: dict[str, Node],
) -> list[str]:
    """One node's listing: header, motif annotations, inline outgoing edges."""
    lines: list[str] = [_node_header(node)]

    if node.motifs:
        motif_strs = [f"{m.motif} [{m.mode}]" for m in node.motifs]
        lines.append(f"  motifs: {'; '.join(motif_strs)}")

    for edge in outgoing:
        verb = _EDGE_VERBS[edge.type]
        target = nodes_by_id[edge.dst]
        snippet = _sketch_snippet(target.sketch)
        lines.append(f"  -> {verb} [{edge.dst}] {snippet}")
        lines.append(f"      ({_edge_payload(edge)})")

    return lines


def _is_long_range(edge: Edge, topo_index: dict[str, int]) -> bool:
    return abs(topo_index[edge.dst] - topo_index[edge.src]) > 1


def _render_structural_tensions(
    dag: DAG,
    topo_index: dict[str, int],
) -> list[str]:
    """Appendix duplicating disclosure, motivates, and any long-range edges.

    Per docs/stage-2/evaluation.md §Rendering rules: any edge spanning >1
    topological position OR any disclosure/motivates edge regardless of
    distance is repeated here. The duplication surfaces long-range structural
    relationships that would otherwise be buried inside per-node listings.
    """
    appendix: list[Edge] = []
    for edge in dag.edges:
        if edge.type in ("disclosure", "motivates") or _is_long_range(edge, topo_index):
            appendix.append(edge)
    if not appendix:
        return []

    lines = ["", "STRUCTURAL TENSIONS (long-range and reframing edges)"]
    # Sort by source topo position for stable rendering.
    appendix.sort(key=lambda e: (topo_index[e.src], topo_index[e.dst]))
    for edge in appendix:
        verb = _EDGE_VERBS[edge.type]
        lines.append(f"  [{edge.src}] -> {verb} [{edge.dst}]")
        lines.append(f"      ({_edge_payload(edge)})")
    return lines


def _render_character_arcs(dag: DAG) -> list[str]:
    if not dag.character_arcs:
        return []
    lines = ["", "CHARACTER ARCS (whole-story trajectories)"]
    for arc in dag.character_arcs:
        lines.append(f"  - agent: {arc.agent}")
        lines.append(f"    goal: {arc.goal}")
        if arc.stakes is not None:
            lines.append(f"    stakes: {arc.stakes}")
    return lines


def _render_story_constraints(dag: DAG) -> list[str]:
    if not dag.story_constraints:
        return []
    lines = ["", "STORY CONSTRAINTS (diegetic rules holding across the story)"]
    for sc in dag.story_constraints:
        lifts = sc.lifts_at if sc.lifts_at is not None else "(never lifts)"
        lines.append(f"  - prohibits: {sc.prohibits}")
        lines.append(f"    lifts at: {lifts}")
    return lines


def _render_motif_threads(dag: DAG) -> list[str]:
    if not dag.motif_threads:
        return []
    lines = ["", "MOTIF THREADS (inventory; per-node mode tags shown inline above)"]
    for thread in dag.motif_threads:
        lines.append(f"  - {thread}")
    return lines


def _render_concept_demands(dag: DAG) -> list[str]:
    if not dag.concept_demands:
        return []
    lines = ["", "CONCEPT DEMANDS (predicates the structure must satisfy — Tier 3)"]
    for demand in dag.concept_demands:
        lines.append(f"  - {demand}")
    return lines


def _render_progress_line(dag: DAG, topo_index: dict[str, int]) -> list[str]:
    """Two-line DAG-progress hint for the expansion-call header.

    Tells the LLM how much budget it has left and how the current beats are
    distributed around the anchor — without those signals, the model has no
    way to tell "almost done, write resolution material" from "still building
    rising action." Anchor partition is computed via topological order: the
    anchor's index splits the DAG into upstream / anchor / downstream.
    """
    n = len(dag.nodes)
    target = dag.target_node_count
    anchor_id = next((nd.id for nd in dag.nodes if nd.role), None)
    if anchor_id is None or anchor_id not in topo_index:
        partition = "(no anchor present yet)"
    else:
        anchor_pos = topo_index[anchor_id]
        upstream = sum(1 for nid, idx in topo_index.items() if idx < anchor_pos)
        downstream = sum(1 for nid, idx in topo_index.items() if idx > anchor_pos)
        partition = f"{upstream} upstream / 1 anchor / {downstream} downstream"
    return [
        f"DAG SIZE: {n} of {target} target nodes  ({partition})",
    ]


# ----- Public API -----

def render(dag: DAG, *, label: str = "A", include_progress: bool = False) -> str:
    """Render a DAG as an incident-encoded outline.

    `label` controls the header (`STORY STRUCTURE A` or `STORY STRUCTURE B`).
    Used during pairwise comparison so the judge sees two structurally
    identical renderings differing only in their A/B label.

    `include_progress=True` adds a `DAG SIZE` line showing current node count
    relative to the concept's `target_node_count`, plus a partition count
    (upstream / anchor / downstream). Expansion calls pass True so the LLM
    knows how much room it has left; judge calls leave it False so the panel
    evaluates structural quality without anchor-progress framing biasing the
    "is this DAG too small/too large" judgment.
    """
    topo_order = _topological_order(dag)
    topo_index = {nid: i for i, nid in enumerate(topo_order)}
    nodes_by_id = {n.id: n for n in dag.nodes}

    # Group outgoing edges by source, preserving DAG.edges ordering within source.
    outgoing_by_src: dict[str, list[Edge]] = defaultdict(list)
    for edge in dag.edges:
        outgoing_by_src[edge.src].append(edge)

    header = f"STORY STRUCTURE {label}"
    lines: list[str] = [
        header,
        "=" * len(header),
        "",
    ]
    if include_progress:
        lines.extend(_render_progress_line(dag, topo_index))
        lines.append("")
    lines.extend([
        "Beats listed in causal order. Each beat shows what it causes or reveals,",
        "with load-bearing details in parentheses. Per-node motifs annotated under",
        "the beat header; whole-story arcs and constraints listed in their own",
        "sections at the end.",
        "",
    ])

    for nid in topo_order:
        node = nodes_by_id[nid]
        lines.extend(_render_node_section(node, outgoing_by_src[nid], nodes_by_id))
        lines.append("")  # blank line between nodes

    # Drop trailing blank line before appending sections.
    if lines and lines[-1] == "":
        lines.pop()

    lines.extend(_render_structural_tensions(dag, topo_index))
    lines.extend(_render_character_arcs(dag))
    lines.extend(_render_story_constraints(dag))
    lines.extend(_render_motif_threads(dag))
    lines.extend(_render_concept_demands(dag))

    return "\n".join(lines) + "\n"


# ----- Standalone CLI -----

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a Stage 2 genome JSON file as incident-encoded outline.",
    )
    parser.add_argument("genome_path", type=Path, help="Path to genome.json")
    parser.add_argument(
        "--label", default="A",
        help="STRUCTURE label (default 'A'); use 'B' for the right-hand side of pairwise.",
    )
    args = parser.parse_args(argv)

    if not args.genome_path.exists():
        print(f"error: {args.genome_path} not found", file=sys.stderr)
        return 1

    try:
        dag = DAG.model_validate_json(args.genome_path.read_text())
    except ValidationError as e:
        print(f"validation failed for {args.genome_path}:\n{e}", file=sys.stderr)
        return 1

    sys.stdout.write(render(dag, label=args.label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
