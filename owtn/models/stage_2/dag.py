"""Stage 2 typed-edge DAG genome models.

The DAG is the genome Stage 2 evolves under MCTS. This module defines the
data shape (Pydantic v2 BaseModels) plus Tier 0 deterministic validators —
the structural checks that need no LLM call. Tier 1/2/3 LLM-based checks
live in `owtn.evaluation.stage_2` (Phase 4).

Design-doc note: docs/stage-2/implementation.md sketches these as
`@dataclass`. We use Pydantic v2 BaseModel instead, mirroring Stage 1's
convention (`owtn.models.stage_1.concept_genome`). The shape matches the
design doc; the framework is just stronger for our needs (free
JSON parsing, ValidationError on bad input, integrated `model_validate`).

Authoritative schema reference: docs/stage-2/overview.md §The Genome.
Mode glossary: docs/stage-2/overview.md §Per-node motifs §Mode glossary.

Standalone CLI:
    uv run python -m owtn.models.stage_2.dag <path-to-genome.json>
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


RoleName = Literal["climax", "reveal", "pivot"]
EdgeType = Literal["causal", "disclosure", "implication", "constraint", "motivates"]
MotifMode = Literal[
    "introduced",
    "embodied",
    "performed",
    "agent",
    "echoed",
    "inverted",
]
RETURN_MODES: frozenset[MotifMode] = frozenset({"echoed", "inverted"})


def _is_substantive(value: str | None, *, min_tokens: int = 4) -> bool:
    """Tier 0 surface check for "non-empty, >3 tokens" (per evaluation.md).

    Whitespace-split tokens; a 4-token field counts as substantive. Generic
    fillers like "things happen" pass token count but should fail Tier 2's
    LLM plausibility check (Phase 4).
    """
    if value is None:
        return False
    return len(value.split()) >= min_tokens


class MotifMention(BaseModel):
    """One motif's appearance at one node, with its mode."""
    motif: str = Field(min_length=1)
    mode: MotifMode


class Node(BaseModel):
    """A story beat. Sketch is the 1-2 sentence plan; prose is Stage 4's job."""
    id: str = Field(min_length=1)
    sketch: str = Field(min_length=20)
    role: list[RoleName] | None = None
    motifs: list[MotifMention] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_role_uniqueness(self) -> Node:
        if self.role is not None:
            if len(self.role) == 0:
                raise ValueError(
                    f"node {self.id!r}: role must be None or a non-empty list"
                )
            if len(self.role) != len(set(self.role)):
                raise ValueError(
                    f"node {self.id!r}: role list contains duplicates: {self.role}"
                )
        return self


class Edge(BaseModel):
    """A typed relationship between two nodes. Payload fields are
    type-specific; per-type required fields are enforced after validation."""
    src: str
    dst: str
    type: EdgeType
    realizes: str | None = None       # causal
    reframes: str | None = None       # disclosure
    withheld: str | None = None       # disclosure
    disclosed_to: list[str] | None = None  # disclosure: defaults to ["reader"] when None
    entails: str | None = None        # implication
    prohibits: str | None = None      # constraint (local; story-scoped → DAG.story_constraints)
    agent: str | None = None          # motivates (local; whole-story arcs → DAG.character_arcs)
    goal: str | None = None           # motivates
    stakes: str | None = None         # motivates (optional payload)

    @model_validator(mode="after")
    def _validate_payload(self) -> Edge:
        if self.src == self.dst:
            raise ValueError(f"edge {self.src!r} → {self.dst!r}: self-loop")

        if self.type == "causal":
            if not _is_substantive(self.realizes):
                raise ValueError(
                    f"edge {self.src!r} → {self.dst!r} (causal): "
                    f"realizes must be non-empty (>3 tokens)"
                )
        elif self.type == "disclosure":
            if not _is_substantive(self.reframes):
                raise ValueError(
                    f"edge {self.src!r} → {self.dst!r} (disclosure): "
                    f"reframes must be non-empty (>3 tokens)"
                )
            if not _is_substantive(self.withheld):
                raise ValueError(
                    f"edge {self.src!r} → {self.dst!r} (disclosure): "
                    f"withheld must be non-empty (>3 tokens)"
                )
            if self.disclosed_to is None:
                # default audience is the reader; matches docs/stage-2/overview.md
                object.__setattr__(self, "disclosed_to", ["reader"])
            elif len(self.disclosed_to) == 0:
                raise ValueError(
                    f"edge {self.src!r} → {self.dst!r} (disclosure): "
                    f"disclosed_to must be non-empty (use ['reader'] for authorial reveal)"
                )
        elif self.type == "implication":
            if not _is_substantive(self.entails):
                raise ValueError(
                    f"edge {self.src!r} → {self.dst!r} (implication): "
                    f"entails must be non-empty (>3 tokens)"
                )
        elif self.type == "constraint":
            if not _is_substantive(self.prohibits):
                raise ValueError(
                    f"edge {self.src!r} → {self.dst!r} (constraint): "
                    f"prohibits must be non-empty (>3 tokens)"
                )
        elif self.type == "motivates":
            if not (self.agent and self.agent.strip()):
                raise ValueError(
                    f"edge {self.src!r} → {self.dst!r} (motivates): "
                    f"agent must be non-empty"
                )
            if not _is_substantive(self.goal):
                raise ValueError(
                    f"edge {self.src!r} → {self.dst!r} (motivates): "
                    f"goal must be non-empty (>3 tokens)"
                )
        return self


class CharacterArc(BaseModel):
    """A whole-story trajectory for one character.

    No `touches` list — agent presence is derivable from beat sketches.
    See docs/stage-2/overview.md §Character arcs for scoping rules.
    """
    agent: str = Field(min_length=1)
    goal: str = Field(min_length=10)
    stakes: str | None = None


class StoryConstraint(BaseModel):
    """A diegetic rule that holds across the whole story.

    Use when a prohibition is wholesale (Hemingway's not-naming, Jackson's
    ritual silence) rather than between two specific beats. Local constraints
    stay as `constraint` edges.
    """
    prohibits: str = Field(min_length=10)
    lifts_at: str | None = None  # node id where rule breaks; None = never lifts


class DAG(BaseModel):
    """A Stage 2 genome: nodes + typed edges + arcs + story_constraints + motifs.

    Cross-field invariants enforced by `_validate_dag`:
    - 1 ≤ |nodes| ≤ 18 (1 = a fresh seed_root output; complete DAGs
      reaching judges/tournament have ≥3 nodes per evaluation.md Tier 0,
      but that "completeness" check is enforced in Phase 4, not here)
    - node ids are unique
    - edges reference existing nodes
    - acyclic (Kahn topological sort succeeds)
    - all nodes reachable from at least one in-degree-0 node (connected)
    - exactly one node carries a non-None role (the anchor)
    - Node.motifs[].motif is a subset of motif_threads
    - return-mode motifs (echoed/inverted) require an earlier non-return-mode appearance
    - story_constraints[].lifts_at is None or a real node id
    - disclosure edges' non-"reader" audiences appear in the target beat's sketch
    - motivates edges are local (within 2 topological positions)
    - character_arcs[].agent appears (case-insensitive substring) in some beat sketch
    """
    concept_id: str = Field(min_length=1)
    preset: str = Field(min_length=1)
    motif_threads: list[str]
    concept_demands: list[str] = Field(default_factory=list)
    nodes: list[Node]
    edges: list[Edge]
    character_arcs: list[CharacterArc] = Field(default_factory=list)
    story_constraints: list[StoryConstraint] = Field(default_factory=list)
    target_node_count: int = Field(ge=3, le=18)

    # ----- Validators -----

    @model_validator(mode="after")
    def _validate_dag(self) -> DAG:
        self._check_node_count()
        node_ids = self._check_unique_node_ids()
        self._check_edge_endpoints(node_ids)
        topo_order = self._check_acyclic_and_topo()
        self._check_connected(topo_order)
        self._check_role_cardinality()
        self._check_motif_membership()
        self._check_motif_temporal_sanity(topo_order)
        self._check_story_constraint_lifts(node_ids)
        self._check_disclosure_audiences(node_ids)
        self._check_motivates_local(topo_order)
        self._check_character_arc_presence()
        return self

    def _check_node_count(self) -> None:
        n = len(self.nodes)
        if n < 1 or n > 18:
            raise ValueError(f"node count {n} not in [1, 18]")

    def _check_unique_node_ids(self) -> set[str]:
        ids = [n.id for n in self.nodes]
        seen: set[str] = set()
        for nid in ids:
            if nid in seen:
                raise ValueError(f"duplicate node id: {nid!r}")
            seen.add(nid)
        return seen

    def _check_edge_endpoints(self, node_ids: set[str]) -> None:
        for e in self.edges:
            if e.src not in node_ids:
                raise ValueError(f"edge src references unknown node: {e.src!r}")
            if e.dst not in node_ids:
                raise ValueError(f"edge dst references unknown node: {e.dst!r}")

    def _check_acyclic_and_topo(self) -> dict[str, int]:
        """Kahn's algorithm. Returns {node_id: topological_index}."""
        in_degree: dict[str, int] = {n.id: 0 for n in self.nodes}
        out_edges: dict[str, list[str]] = defaultdict(list)
        for e in self.edges:
            in_degree[e.dst] += 1
            out_edges[e.src].append(e.dst)

        # Stable topo order: sort sources by node-list order (matches author intent)
        node_index = {n.id: i for i, n in enumerate(self.nodes)}
        ready = deque(sorted(
            (nid for nid, d in in_degree.items() if d == 0),
            key=lambda nid: node_index[nid],
        ))
        order: dict[str, int] = {}
        idx = 0
        while ready:
            nid = ready.popleft()
            order[nid] = idx
            idx += 1
            for child in out_edges[nid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(child)
        if len(order) != len(self.nodes):
            unreached = [n.id for n in self.nodes if n.id not in order]
            raise ValueError(f"DAG contains a cycle (unreached: {unreached})")
        return order

    def _check_connected(self, topo_order: dict[str, int]) -> None:
        # Every node must be reachable from at least one in-degree-0 node.
        # Topo order being defined (no cycle) guarantees this is well-posed;
        # the topo sort itself reaches every node iff DAG is fully reachable
        # from sources, which is identical to "every node has a predecessor
        # path to some source". Topo success above already proves it.
        # This check is here as a documented intent stamp.
        if len(topo_order) != len(self.nodes):  # pragma: no cover
            raise ValueError("DAG connectivity violated (unreachable nodes)")

    def _check_role_cardinality(self) -> None:
        role_bearers = [n.id for n in self.nodes if n.role is not None]
        if len(role_bearers) != 1:
            raise ValueError(
                f"exactly one node must carry a non-None role (anchor); "
                f"found {len(role_bearers)}: {role_bearers}"
            )

    def _check_motif_membership(self) -> None:
        thread_set = set(self.motif_threads)
        for n in self.nodes:
            for m in n.motifs:
                if m.motif not in thread_set:
                    raise ValueError(
                        f"node {n.id!r}: motif {m.motif!r} not in motif_threads "
                        f"(known: {sorted(thread_set)})"
                    )

    def _check_motif_temporal_sanity(self, topo_order: dict[str, int]) -> None:
        # For each motif, every echoed/inverted appearance must be preceded
        # (topologically) by at least one non-return-mode appearance.
        first_introduction: dict[str, int] = {}
        # Sort node mentions by topological order to walk forward in story time.
        mentions: list[tuple[int, str, MotifMode, str]] = []
        for n in self.nodes:
            for m in n.motifs:
                mentions.append((topo_order[n.id], m.motif, m.mode, n.id))
        mentions.sort(key=lambda t: t[0])
        for _, motif, mode, node_id in mentions:
            if mode in RETURN_MODES:
                if motif not in first_introduction:
                    raise ValueError(
                        f"node {node_id!r}: motif {motif!r} tagged {mode!r} "
                        f"with no earlier non-return-mode appearance "
                        f"(introduced/embodied/performed/agent)"
                    )
            else:
                first_introduction.setdefault(motif, topo_order[node_id])

    def _check_story_constraint_lifts(self, node_ids: set[str]) -> None:
        for sc in self.story_constraints:
            if sc.lifts_at is not None and sc.lifts_at not in node_ids:
                raise ValueError(
                    f"story_constraint lifts_at={sc.lifts_at!r} is not a real node id"
                )

    def _check_disclosure_audiences(self, node_ids: set[str]) -> None:
        sketch_by_id = {n.id: n.sketch for n in self.nodes}
        for e in self.edges:
            if e.type != "disclosure" or e.disclosed_to is None:
                continue
            target_sketch = sketch_by_id[e.dst].lower()
            for audience in e.disclosed_to:
                if audience == "reader":
                    continue
                if audience.lower() not in target_sketch:
                    raise ValueError(
                        f"disclosure edge {e.src!r} → {e.dst!r}: "
                        f"audience {audience!r} not present in target beat sketch"
                    )

    def _check_motivates_local(self, topo_order: dict[str, int]) -> None:
        # Per docs/stage-2/operators.md: motivates edges must span ≤ 2
        # topological positions. Whole-story arcs go in character_arcs.
        for e in self.edges:
            if e.type != "motivates":
                continue
            span = abs(topo_order[e.dst] - topo_order[e.src])
            if span > 2:
                raise ValueError(
                    f"motivates edge {e.src!r} → {e.dst!r} spans {span} topological "
                    f"positions (max 2). Whole-story arcs belong in character_arcs."
                )

    def _check_character_arc_presence(self) -> None:
        # arc.agent should appear (case-insensitive substring) in at least
        # one beat sketch. Heuristic; Tier 1 (LLM) refines in Phase 4.
        sketches_lower = [n.sketch.lower() for n in self.nodes]
        for arc in self.character_arcs:
            agent_lower = arc.agent.lower()
            if not any(agent_lower in s for s in sketches_lower):
                raise ValueError(
                    f"character_arc agent {arc.agent!r} does not appear in any beat sketch "
                    f"(case-insensitive substring match)"
                )


# ----- Standalone CLI -----

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Stage 2 genome JSON file (Tier 0 checks only).",
    )
    parser.add_argument("genome_path", type=Path, help="Path to the genome.json file")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress success summary",
    )
    args = parser.parse_args(argv)

    try:
        text = args.genome_path.read_text()
    except FileNotFoundError:
        print(f"error: {args.genome_path} not found", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: cannot read {args.genome_path}: {e}", file=sys.stderr)
        return 1

    try:
        dag = DAG.model_validate_json(text)
    except ValidationError as e:
        print(f"validation failed for {args.genome_path}:\n{e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"valid: concept_id={dag.concept_id} preset={dag.preset} "
            f"nodes={len(dag.nodes)} edges={len(dag.edges)} "
            f"arcs={len(dag.character_arcs)} "
            f"story_constraints={len(dag.story_constraints)} "
            f"demands={len(dag.concept_demands)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
