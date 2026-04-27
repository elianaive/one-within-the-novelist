"""MCTS tree-node bookkeeping wrapper.

Tree state — visit counts, cumulative return, parent/child references — that
lives only in memory during MCTS exploration (Phase 5+) and is never
serialized. dataclass instead of Pydantic: no JSON contract, no need for
validation, and `parent`/`children` form a self-referential graph that
Pydantic's validation walk doesn't handle gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from owtn.models.stage_2.actions import Action
from owtn.models.stage_2.dag import DAG


Phase = Literal["forward", "backward", "refinement"]


@dataclass
class MCTSNode:
    """One node in the MCTS tree, wrapping a partial DAG plus selection state.

    `pending_visits` is the AlphaZero virtual-loss counter: incremented when a
    parallel worker traverses through this node during selection, decremented
    when that worker's backprop reaches it. Drives the virtual-loss term in
    UCB so concurrent workers spread to different children. Always 0 in
    sequential mode (selection and backprop alternate atomically).

    `expansion_in_flight` deduplicates expand-fn calls per node: when a worker
    needs to populate the action cache, it sets this flag and fires the LLM;
    other workers reaching the same uncached node skip expansion to avoid a
    redundant proposal call. Cleared when the cache is set.
    """
    dag: DAG
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    cumulative_return: float = 0.0
    fully_expanded: bool = False
    phase: Phase = "forward"
    cached_candidate_actions: list[Action] | None = None
    pending_visits: int = 0
    expansion_in_flight: bool = False
