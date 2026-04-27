"""MCTS tree + selection + backprop with optional AlphaZero-style parallel
worker pool.

The algorithm is the standard select → expand → rollout → backprop loop with
two design choices specific to LLM-driven narrative search:

- **Cached expansion**: a leaf's first visit fires expand_fn ONCE for K
  candidate actions; subsequent visits to that leaf consume the cached
  candidates one per iteration. Avoids re-querying the LLM for every visit
  to the same partial DAG.
- **Structural side-signal augmentation**: the UCB exploitation term gets a
  small additive bonus (β × structural_score(child.dag)) so the search
  doesn't waste rollouts on structurally-malformed children. Mechanically
  identical to a value-network prior in AlphaZero.

Parallelism (when MCTSConfig.parallel_workers > 1) follows AlphaZero's
virtual-loss approach: when a worker selects a child during traversal, that
child's `pending_visits` increments, and the UCB score treats those pending
visits as completed-with-zero-reward (modulated by `virtual_loss`). Concurrent
workers reading the same parent see the elevated pending count and prefer
different children. On rollout completion, backprop decrements the pending
counter and increments the real `visits` + `cumulative_return`.

Architecture: `MCTS` takes `expand_fn` and `rollout_fn` as callables. This
keeps the algorithm decoupled from LLMs and the structural side-signal —
tests inject mock callables; production wires `propose_actions_via_llm` and
the `evaluate_rollout`-driven rollout closure from `tree_runtime`.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import ValidationError

from owtn.models.stage_2.actions import Action
from owtn.models.stage_2.dag import DAG
from owtn.models.stage_2.mcts_node import MCTSNode, Phase
from owtn.stage_2.operators import apply_action
from owtn.stage_2.structural_signal import structural_score


logger = logging.getLogger(__name__)


ExpandFn = Callable[[DAG], Awaitable[list[Action]]]
RolloutFn = Callable[[DAG], Awaitable[float]]


# ----- UCB selection (with virtual-loss accounting) -----


def ucb_score(
    child: MCTSNode,
    parent: MCTSNode,
    *,
    c: float = 0.5,
    beta: float = 0.1,
    virtual_loss: float = 1.0,
    structural_score_fn: Callable[[DAG], float] = structural_score,
) -> float:
    """UCB1 + structural-bonus + virtual-loss-aware accounting.

    Effective visit/return account for in-flight workers: a worker that has
    selected this child but not yet backpropped contributes one to
    `pending_visits`, and the score treats that pending visit as a
    completed-with-`-virtual_loss`-reward visit. This pushes concurrent
    workers off the same child during parallel selection.

    Returns +∞ for unvisited-and-unpending children so newly-instantiated
    children get visited at least once before any UCB calculus kicks in.
    """
    n_eff = child.visits + child.pending_visits
    if n_eff == 0:
        return math.inf
    w_eff = child.cumulative_return - virtual_loss * child.pending_visits
    parent_n_eff = max(parent.visits + parent.pending_visits, 1)
    exploitation = w_eff / n_eff
    exploration = c * math.sqrt(math.log(parent_n_eff) / n_eff)
    structural_bonus = beta * structural_score_fn(child.dag)
    return exploitation + exploration + structural_bonus


# ----- Config -----


@dataclass
class MCTSConfig:
    """Hyperparameters for one MCTS run.

    - `c=0.5`: BiT-MCTS finding for narrative-reward variance.
    - `gamma=1.0`: standard UCB1 (γ=0.93 D-UCB is a future option per
      `mcts.md`).
    - `beta=0.1`: structural side-signal weight.
    - `k_candidates_per_expansion=4`: BiT-MCTS optimum cache size per leaf.
    - `parallel_workers=1`: number of concurrent worker tasks running step()
      against the same tree. Selection applies virtual loss to in-flight
      paths so workers spread across children. K LLM calls (expansion +
      rollout) can be in-flight simultaneously.
    - `virtual_loss=1.0`: magnitude of virtual loss per pending visit;
      AlphaGo Zero uses values in the 1.0-3.0 range. Higher values push
      workers harder toward unexplored children at the cost of worse
      exploitation while rollouts are in flight.
    - `no_improvement_cutoff_iterations`: when set, the run halts early if
      best-leaf-mean-reward hasn't improved within this many iterations.
    """
    c: float = 0.5
    gamma: float = 1.0
    beta: float = 0.1
    k_candidates_per_expansion: int = 4
    parallel_workers: int = 1
    virtual_loss: float = 1.0
    no_improvement_cutoff_iterations: int | None = None
    seed: int | None = None


# ----- MCTS engine -----


class MCTS:
    """One MCTS tree. Drives selection, expansion, rollout, backprop —
    optionally across multiple concurrent workers per `MCTSConfig.parallel_workers`.

    Tree mutations (selection's virtual-loss application, expansion's child
    addition, backprop) are serialized via an asyncio lock. LLM calls
    (expand_fn, rollout_fn) execute outside the lock so multiple workers
    can have requests in flight simultaneously — that's where the wall-time
    win comes from.

    Use `step()` for one iteration or `run(n)` for `n` total iterations
    (with the configured worker pool).
    """

    def __init__(
        self,
        root_dag: DAG,
        *,
        expand_fn: ExpandFn,
        rollout_fn: RolloutFn,
        phase: Phase = "forward",
        config: MCTSConfig | None = None,
    ) -> None:
        self.config = config or MCTSConfig()
        self.expand_fn = expand_fn
        self.rollout_fn = rollout_fn
        self.root = MCTSNode(dag=root_dag, phase=phase)
        self.iterations = 0
        self._rng = random.Random(self.config.seed)
        self._tree_lock = asyncio.Lock()

    # ----- public API -----

    async def step(self) -> MCTSNode | None:
        """One MCTS iteration: select → expand → rollout → backprop. Returns
        the node the rollout was run against (or None if no expansion was
        possible). Safe to call concurrently from multiple worker tasks.
        """
        # SELECTION + virtual-loss application — under tree lock.
        async with self._tree_lock:
            self.iterations += 1
            path = self._select_path_applying_virtual_loss()

        leaf = path[-1]
        rollout_target = leaf

        # EXPANSION — releases lock during the LLM call so concurrent workers
        # can have multiple expand_fn invocations in flight on different leaves.
        if not leaf.fully_expanded:
            child = await self._expand_one(leaf)
            if child is None:
                # No child instantiated; undo virtual loss on the path and exit.
                async with self._tree_lock:
                    self._undo_virtual_loss(path)
                return None
            path.append(child)
            rollout_target = child

        # ROLLOUT — no lock. The rollout_fn closure is responsible for any
        # state coordination it needs (e.g., champion updates in tree_runtime).
        reward = await self.rollout_fn(rollout_target.dag)

        # BACKPROP — under lock. Removes virtual loss as it walks up.
        async with self._tree_lock:
            self._backprop_path(path, reward)

        return rollout_target

    async def run(self, n: int) -> None:
        """Run `n` MCTS iterations across the configured worker pool. Halts
        early if `no_improvement_cutoff_iterations` is configured and the
        best-leaf-mean-reward hasn't improved within that window.
        """
        workers = max(1, self.config.parallel_workers)
        if workers == 1:
            await self._run_sequential(n)
            return

        cutoff = self.config.no_improvement_cutoff_iterations
        completed = 0
        best_so_far = -math.inf
        last_improvement_at = 0
        halt = asyncio.Event()

        async def worker() -> None:
            nonlocal completed, best_so_far, last_improvement_at
            while not halt.is_set():
                async with self._tree_lock:
                    if completed >= n:
                        return
                    completed += 1
                    my_iter = completed
                await self.step()
                if cutoff is None:
                    continue
                async with self._tree_lock:
                    current = self._best_leaf_score_locked()
                    if current > best_so_far:
                        best_so_far = current
                        last_improvement_at = my_iter
                    elif my_iter - last_improvement_at >= cutoff:
                        halt.set()

        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(workers)])

    async def _run_sequential(self, n: int) -> None:
        cutoff = self.config.no_improvement_cutoff_iterations
        best_so_far = -math.inf
        last_improvement_iter = 0
        for i in range(n):
            await self.step()
            if cutoff is None:
                continue
            current_best = self._best_leaf_score_locked()
            if current_best > best_so_far:
                best_so_far = current_best
                last_improvement_iter = i
            elif i - last_improvement_iter >= cutoff:
                return

    def best_terminal(self) -> MCTSNode:
        """Return the highest-mean-reward visited LEAF (no children). Falls
        back to root only when no leaf has been visited (degenerate)."""
        best, _ = self._best_leaf()
        return best if best is not None else self.root

    # ----- selection -----

    def _select_path_applying_virtual_loss(self) -> list[MCTSNode]:
        """Walk from root via UCB, applying virtual loss to each node on the
        path. Returns the path (root-first) so backprop can undo the loss.
        """
        path: list[MCTSNode] = []
        node = self.root
        while True:
            path.append(node)
            node.pending_visits += 1
            if not (node.fully_expanded and node.children):
                break
            next_node = self._best_ucb_child(node)
            if next_node is None:
                break  # all children dead leaves; caller re-rollouts here
            node = next_node
        return path

    def _best_ucb_child(self, parent: MCTSNode) -> MCTSNode | None:
        """Pick the child with the highest UCB score, skipping dead leaves
        (fully_expanded with no children — every proposed action invalid).
        Returns None when no live child exists. Ties broken by random pick
        from the seeded RNG."""
        candidates = [c for c in parent.children if not _is_dead_leaf(c)]
        if not candidates:
            return None
        best_score = -math.inf
        best_children: list[MCTSNode] = []
        for child in candidates:
            score = ucb_score(
                child, parent,
                c=self.config.c, beta=self.config.beta,
                virtual_loss=self.config.virtual_loss,
            )
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
        return self._rng.choice(best_children)

    # ----- expansion -----

    async def _expand_one(self, node: MCTSNode) -> MCTSNode | None:
        """Instantiate one new child of `node`. Concurrent-safe:

        - First worker to reach an uncached node sets `expansion_in_flight`
          and fires the LLM. Other workers seeing the flag skip expansion
          (they'll either retry next select or hit a different path).
        - Cache population and child-slot claiming are both atomic under
          the tree lock; the LLM call itself releases the lock.
        - Invalid actions (validator-rejected) are dropped from the cache
          so subsequent expansions advance through remaining candidates.
        """
        # Phase A: claim responsibility for population if needed.
        async with self._tree_lock:
            if node.fully_expanded:
                return None
            need_to_populate = (
                node.cached_candidate_actions is None and not node.expansion_in_flight
            )
            if need_to_populate:
                node.expansion_in_flight = True

        if need_to_populate:
            try:
                actions = await self.expand_fn(node.dag)
            except Exception as e:  # noqa: BLE001 — never crash the search loop
                logger.warning(
                    "MCTS expand_fn raised at iteration %d: %s; cache empty",
                    self.iterations, e,
                )
                actions = []
            async with self._tree_lock:
                node.cached_candidate_actions = list(actions)[: self.config.k_candidates_per_expansion]
                node.expansion_in_flight = False

        # Phase B: claim a slot from the cache and instantiate a child. Drop
        # any invalid candidates encountered along the way.
        async with self._tree_lock:
            if node.cached_candidate_actions is None:
                # Another worker is still populating; treat as no expansion
                # available this turn.
                return None
            while True:
                next_idx = len(node.children)
                if next_idx >= len(node.cached_candidate_actions):
                    node.fully_expanded = True
                    return None
                action = node.cached_candidate_actions[next_idx]
                try:
                    child_dag = apply_action(node.dag, action)
                except (ValidationError, ValueError) as e:
                    logger.info(
                        "MCTS skipping invalid action at iteration %d (action_type=%s): %s",
                        self.iterations,
                        getattr(action, "action_type", type(action).__name__),
                        e,
                    )
                    node.cached_candidate_actions.pop(next_idx)
                    continue
                # The worker is now "in flight" on this new child — it will
                # rollout against child.dag next and backprop will decrement
                # the pending count. Initialize pending_visits=1 to match.
                child = MCTSNode(
                    dag=child_dag, parent=node, phase=node.phase,
                    pending_visits=1,
                )
                node.children.append(child)
                if len(node.children) >= len(node.cached_candidate_actions):
                    node.fully_expanded = True
                return child

    # ----- backprop -----

    def _backprop_path(self, path: list[MCTSNode], reward: float) -> None:
        """Walk the path from leaf to root: decrement virtual loss, increment
        real visits + reward. Caller holds the tree lock.
        """
        for n in path:
            n.pending_visits -= 1
            n.visits += 1
            n.cumulative_return += reward

    def _undo_virtual_loss(self, path: list[MCTSNode]) -> None:
        """Remove pending visits along a path that did NOT produce a rollout
        (e.g., expansion failed). Caller holds the tree lock."""
        for n in path:
            n.pending_visits -= 1

    # ----- helpers -----

    def _best_leaf(self) -> tuple[MCTSNode | None, float]:
        best: MCTSNode | None = None
        best_score = -math.inf
        for leaf in self._iter_leaves():
            score = self._mean_reward(leaf)
            if score > best_score:
                best = leaf
                best_score = score
        return best, best_score

    def _best_leaf_score_locked(self) -> float:
        """Score of the best leaf; -inf when no leaf has been visited.
        Caller is expected to hold (or not need) the tree lock — reads only."""
        return self._best_leaf()[1]

    def _iter_tree(self):
        stack: list[MCTSNode] = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children)

    def _iter_leaves(self):
        for node in self._iter_tree():
            if node is self.root:
                continue
            if node.children:
                continue
            if node.visits == 0:
                continue
            yield node

    @staticmethod
    def _mean_reward(node: MCTSNode) -> float:
        if node.visits == 0:
            return 0.0
        return node.cumulative_return / node.visits


def _is_dead_leaf(node: MCTSNode) -> bool:
    """A leaf whose expansion produced no live children — usually the LLM
    proposed K candidate actions and every one failed validation. Filtered
    out of UCB selection so siblings absorb the budget."""
    return node.fully_expanded and not node.children
