"""Borda no-self-vote — Phase 5 aggregation.

Each agent ranks the OTHER N-1 proposals best-to-worst on overall voice
quality. Borda gives `(N-2)` points to first, `(N-3)` to second, ...,
0 to last (so a 4-agent panel produces 2/1/0 per ranker as in the
overview's "rank-1 = 2, rank-2 = 1, rank-3 = 0" prescription).

Borda is preferred over Condorcet/Schulze for small panels — Yang et al.
AIES 2024 + LLM-Council practitioners both find Schulze methods amplify
noise and produce cycles with 3–5 voters; Borda gives a stable
average-rank signal.
"""

from __future__ import annotations


def borda_no_self_vote(
    rankings: dict[str, list[str]],
) -> dict[str, int]:
    """Aggregate per-agent rankings into Borda totals.

    `rankings[agent_id]` is the agent's best-to-worst ordering of the
    other agents' proposals. Each agent must rank every other agent
    exactly once and must not include itself.

    Points awarded per ranker: position-0 → (N-2), position-1 → (N-3), ...,
    last → 0. A 4-agent panel produces 2/1/0 per ranker, summed over
    rankers; max possible total per agent is `(N-1)*(N-2)`.

    Raises:
        ValueError: any ranker includes itself; any ranker omits or
            duplicates a peer; rankers disagree on the agent set.
    """
    if not rankings:
        return {}

    agent_ids = set(rankings.keys())
    n = len(agent_ids)
    if n < 2:
        raise ValueError(f"borda needs ≥2 agents; got {n}")

    points: dict[str, int] = {aid: 0 for aid in agent_ids}

    for ranker, ordering in rankings.items():
        if ranker in ordering:
            raise ValueError(f"ranker {ranker!r} included self in ranking")
        expected = agent_ids - {ranker}
        actual = set(ordering)
        if actual != expected:
            missing = expected - actual
            extra = actual - expected
            raise ValueError(
                f"ranker {ranker!r} ranking mismatch; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if len(ordering) != len(actual):
            raise ValueError(f"ranker {ranker!r} produced duplicate entries: {ordering}")

        for pos, target in enumerate(ordering):
            points[target] += (n - 2) - pos

    return points
