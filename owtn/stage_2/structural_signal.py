"""Zero-LLM structural side-signal for MCTS UCB augmentation.

Per `docs/stage-2/mcts.md` §Selection §Structural Side-Signal: a deterministic
score in [0, 1] computed from a DAG's structure alone. Multiplied by β=0.1
and added to UCB so MCTS routes around clearly-broken branches before
expensive judge calls fire on them.

Five components, all reduced to numeric scores in [0, 1]; final value is
their unweighted mean. Components named in the design doc:

1. **Orphan beats** — penalize disconnected nodes.
2. **Edge-type entropy** — penalize >80% single-type DAGs.
3. **Anchor reachability** — penalize anchors disconnected from opening.
4. **Arc density** — penalize too-sparse or too-dense graphs.
5. **Payload completeness** — safety net (operator validation already
   rejects empty/generic payloads, so this is ~1.0 on in-tree DAGs; kept
   for rare validation-bypass cases).

The 5 components are deliberately equal-weighted because none has been
empirically shown to dominate the others. Empirical reweighting is a v1.5
concern after pilot data accumulates.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

from owtn.models.stage_2.dag import DAG


def orphan_score(dag: DAG) -> float:
    """1 if no node has both zero in-edges AND zero out-edges (except in
    a 1-node seed DAG), else 0.

    Single-node seed DAGs (post-`seed_root`, pre-MCTS expansion) have one
    node with zero edges by design — they pass with score 1.
    """
    if len(dag.nodes) == 1:
        return 1.0
    in_degree: dict[str, int] = {n.id: 0 for n in dag.nodes}
    out_degree: dict[str, int] = {n.id: 0 for n in dag.nodes}
    for e in dag.edges:
        in_degree[e.dst] += 1
        out_degree[e.src] += 1
    for nid in in_degree:
        if in_degree[nid] == 0 and out_degree[nid] == 0:
            return 0.0
    return 1.0


def edge_type_entropy_score(dag: DAG) -> float:
    """Normalized Shannon entropy of edge-type distribution.

    Score is `H(types) / log2(5)` where 5 = number of edge types. A single-
    type DAG scores 0 (max homogeneity); uniformly-distributed types score 1.
    Returns 1.0 on empty edge sets — no penalty for a 1-node seed.
    """
    if not dag.edges:
        return 1.0
    counts: dict[str, int] = defaultdict(int)
    for e in dag.edges:
        counts[e.type] += 1
    total = len(dag.edges)
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    h_max = math.log2(5)
    return h / h_max


def anchor_reachability_score(dag: DAG) -> float:
    """1 if the anchor is reachable from at least one in-degree-0 node
    within ≤ ⌈n_beats × 0.8⌉ hops. Else 0.

    The anchor is the unique role-bearing node (validated at DAG construction).
    Seed DAGs (1 node, trivially the anchor) pass.
    """
    role_bearers = [n for n in dag.nodes if n.role is not None]
    if not role_bearers:
        return 0.0  # invariant violation; should never reach here on valid DAGs
    anchor_id = role_bearers[0].id

    if len(dag.nodes) == 1:
        return 1.0  # the seed: anchor IS the only node

    # Build adjacency from src -> dst.
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {n.id: 0 for n in dag.nodes}
    for e in dag.edges:
        adj[e.src].append(e.dst)
        in_degree[e.dst] += 1

    sources = [nid for nid, d in in_degree.items() if d == 0]
    max_hops = math.ceil(len(dag.nodes) * 0.8)

    for source in sources:
        # BFS up to max_hops.
        visited: set[str] = {source}
        frontier: deque[tuple[str, int]] = deque([(source, 0)])
        while frontier:
            nid, depth = frontier.popleft()
            if nid == anchor_id:
                return 1.0
            if depth >= max_hops:
                continue
            for nbr in adj[nid]:
                if nbr not in visited:
                    visited.add(nbr)
                    frontier.append((nbr, depth + 1))
    return 0.0


def arc_density_score(dag: DAG) -> float:
    """1 if `edges / (n*(n-1))` ∈ [0.1, 0.5]; else linear penalty.

    The density measures the fraction of possible directed pairs realized
    as edges. Below 0.1 = too skeletal; above 0.5 = too dense. Within the
    band, score 1.0; outside, the penalty falls linearly to 0 at density 0
    or density 1.

    1-node DAGs and 2-node DAGs return 1.0 — too small for the metric to
    be meaningful (n*(n-1) is 0 or 2; the formula degenerates).
    """
    n = len(dag.nodes)
    if n < 3:
        return 1.0
    max_edges = n * (n - 1)
    density = len(dag.edges) / max_edges
    if 0.1 <= density <= 0.5:
        return 1.0
    if density < 0.1:
        # Linear from 0 at density=0 to 1 at density=0.1.
        return density / 0.1
    # density > 0.5: linear from 1 at density=0.5 to 0 at density=1.
    return max(0.0, (1.0 - density) / 0.5)


def payload_completeness_score(dag: DAG) -> float:
    """Fraction of edges with all required payload fields populated and >3 tokens.

    Operator validation already rejects edges with empty or generic payloads,
    so in-tree DAGs typically score 1.0. Kept as a safety net for cases
    where operator validation is bypassed (e.g. hand-constructed DAGs in
    tests, or v1.5 operators that skip validation for speed).
    """
    if not dag.edges:
        return 1.0
    valid_count = 0
    for e in dag.edges:
        if _payload_substantive(e):
            valid_count += 1
    return valid_count / len(dag.edges)


def _payload_substantive(edge) -> bool:
    """Per-edge-type payload check. Mirrors the validator in dag.py — kept
    in sync via the same `_is_substantive(text, ≥4 tokens)` rule."""
    def ok(value: str | None) -> bool:
        return value is not None and len(value.split()) >= 4

    if edge.type == "causal":
        return ok(edge.realizes)
    if edge.type == "disclosure":
        return ok(edge.reframes) and ok(edge.withheld)
    if edge.type == "implication":
        return ok(edge.entails)
    if edge.type == "constraint":
        return ok(edge.prohibits)
    if edge.type == "motivates":
        return bool(edge.agent and edge.agent.strip()) and ok(edge.goal)
    return False  # pragma: no cover — Pydantic enforces edge_type literal


def structural_score(dag: DAG) -> float:
    """Combined structural side-signal in [0, 1]. Mean of 5 components.

    Used by MCTS UCB selection: `UCB(child) = W̃/Ñ + c√(ln Ñ_p / Ñ_c) + β × structural_score(child)`.
    With β=0.1 (per `docs/stage-2/mcts.md`), this is a secondary signal —
    routes around broken branches without overwhelming the pairwise judge
    on close comparisons.
    """
    components = [
        orphan_score(dag),
        edge_type_entropy_score(dag),
        anchor_reachability_score(dag),
        arc_density_score(dag),
        payload_completeness_score(dag),
    ]
    return sum(components) / len(components)


def structural_score_breakdown(dag: DAG) -> dict[str, float]:
    """Per-component scores. Same calculation as `structural_score`; useful
    for diagnostics + tests that want to assert which component flagged."""
    return {
        "orphan": orphan_score(dag),
        "edge_type_entropy": edge_type_entropy_score(dag),
        "anchor_reachability": anchor_reachability_score(dag),
        "arc_density": arc_density_score(dag),
        "payload_completeness": payload_completeness_score(dag),
    }
