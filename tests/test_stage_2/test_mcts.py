"""MCTS core tests. No real LLM; expand_fn and rollout_fn are mock callables.

Phase 5 exit criteria covered:
- Handcrafted tree with known UCB values matches computed selection
  (TestUCBMath, TestSelection)
- One full iteration runs end-to-end with mocked expansion + mocked rollout
  (TestEndToEndIteration)
- D-UCB γ=1 produces identical trees to UCB1 when champion is stable
  (TestGammaSanity — Phase 5 ships γ=1 by default; the test confirms the
  baseline behavior the Phase 7 D-UCB ablation will compare against)
- python -c "from owtn.stage_2 import mcts" works (import smoke at top
  of every other Stage 2 test file; explicitly verified in test_no_circular_imports)
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable

import pytest

from owtn.models.stage_2.actions import AddBeatAction, Action
from owtn.models.stage_2.dag import DAG
from owtn.models.stage_2.mcts_node import MCTSNode
from owtn.stage_2.mcts import MCTS, MCTSConfig, ucb_score


# ----- Helpers -----


def _seed_dag(concept_id: str = "seed") -> DAG:
    """A minimum-valid 1-node seed DAG — what `seed_root` produces."""
    return DAG(
        concept_id=concept_id,
        preset="cassandra_ish",
        motif_threads=["motif a", "motif b"],
        concept_demands=[],
        nodes=[{
            "id": "anchor",
            "sketch": "A complete anchor sketch with enough words to validate.",
            "role": ["climax"],
            "motifs": [],
        }],
        edges=[],
        character_arcs=[],
        story_constraints=[],
        target_node_count=5,
    )


def _make_add_beat(node_id: str, *, anchor_id: str = "anchor") -> AddBeatAction:
    """Build an AddBeat action that produces a valid downstream beat."""
    return AddBeatAction(
        anchor_id=anchor_id,
        direction="downstream",
        new_node_id=node_id,
        sketch=f"A new beat sketch for {node_id} with enough words to validate.",
        edge_type="causal",
        edge_payload={
            "realizes": "this is a substantive realizes field with at least four tokens",
        },
    )


def _make_expand_fn(per_call: list[list[Action]]) -> Callable[[DAG], Awaitable[list[Action]]]:
    """Return an expand_fn that produces a different list of actions per call.

    Useful when a test needs the second-level expansion to differ from the
    first. Each list_index n is consumed on the n-th call. After the queue
    is exhausted, returns [].
    """
    queue = list(per_call)
    counter = [0]

    async def expand_fn(dag: DAG) -> list[Action]:
        if counter[0] < len(queue):
            actions = queue[counter[0]]
        else:
            actions = []
        counter[0] += 1
        return actions

    expand_fn.call_count = lambda: counter[0]  # type: ignore[attr-defined]
    return expand_fn


def _constant_rollout(value: float) -> Callable[[DAG], Awaitable[float]]:
    counter = [0]

    async def rollout_fn(dag: DAG) -> float:
        counter[0] += 1
        return value

    rollout_fn.call_count = lambda: counter[0]  # type: ignore[attr-defined]
    return rollout_fn


# ----- UCB math -----

class TestUCBMath:
    def test_unvisited_child_returns_infinity(self) -> None:
        seed = _seed_dag()
        child = MCTSNode(dag=seed)  # visits=0
        score = ucb_score(child, MCTSNode(dag=child.dag, visits=10), c=0.5, beta=0.0)
        assert score == math.inf

    def test_pure_exploitation(self) -> None:
        """With c=0 and beta=0, UCB reduces to W/N (mean reward)."""
        seed = _seed_dag()
        child = MCTSNode(dag=seed, visits=4, cumulative_return=2.0)
        parent = MCTSNode(dag=seed, visits=10)
        score = ucb_score(child, parent, c=0.0, beta=0.0)
        assert score == pytest.approx(0.5)

    def test_exploration_decays_with_visits(self) -> None:
        """Exploration term: c × sqrt(ln(parent_visits) / visits) decreases
        as a child gets visited more."""
        seed = _seed_dag()
        parent = MCTSNode(dag=seed, visits=200)
        child_low = MCTSNode(dag=seed, visits=1, cumulative_return=0.5)
        child_high = MCTSNode(dag=seed, visits=100, cumulative_return=50.0)
        score_low = ucb_score(child_low, parent, c=0.5, beta=0.0)
        score_high = ucb_score(child_high, parent, c=0.5, beta=0.0)
        # Same exploitation (0.5 each), exploration favors the less-visited.
        assert score_low > score_high

    def test_structural_bonus_applied(self) -> None:
        """beta × structural_score gets added. Use a stub structural_score_fn
        to isolate the math."""
        seed = _seed_dag()
        child = MCTSNode(dag=seed, visits=4, cumulative_return=2.0)
        parent = MCTSNode(dag=seed, visits=10)
        score_with_bonus = ucb_score(
            child, parent, c=0.5, beta=0.1,
            structural_score_fn=lambda dag: 1.0,
        )
        score_without_bonus = ucb_score(
            child, parent, c=0.5, beta=0.1,
            structural_score_fn=lambda dag: 0.0,
        )
        assert score_with_bonus - score_without_bonus == pytest.approx(0.1)


# ----- Selection on handcrafted tree -----

class TestSelection:
    def test_picks_higher_mean_reward_when_visits_equal(self) -> None:
        """Two children with equal visit counts; higher cumulative_return
        wins under UCB (exploitation dominates when exploration is equal)."""
        seed = _seed_dag()
        mcts = MCTS(seed, expand_fn=_make_expand_fn([]),
                    rollout_fn=_constant_rollout(0.5),
                    config=MCTSConfig(beta=0.0))  # disable structural bonus for clarity
        # Manually build root with two children, both visited equally.
        a = MCTSNode(dag=seed, parent=mcts.root, visits=10, cumulative_return=8.0)
        b = MCTSNode(dag=seed, parent=mcts.root, visits=10, cumulative_return=3.0)
        mcts.root.children.extend([a, b])
        mcts.root.visits = 20
        mcts.root.fully_expanded = True

        chosen = mcts._best_ucb_child(mcts.root)
        assert chosen is a

    def test_unvisited_child_picked_first(self) -> None:
        """Unvisited child has UCB=∞, beats any visited child."""
        seed = _seed_dag()
        mcts = MCTS(seed, expand_fn=_make_expand_fn([]),
                    rollout_fn=_constant_rollout(0.5),
                    config=MCTSConfig(beta=0.0))
        a = MCTSNode(dag=seed, parent=mcts.root, visits=10, cumulative_return=10.0)  # mean 1.0
        b = MCTSNode(dag=seed, parent=mcts.root, visits=0, cumulative_return=0.0)  # ∞
        mcts.root.children.extend([a, b])
        mcts.root.visits = 10
        mcts.root.fully_expanded = True

        chosen = mcts._best_ucb_child(mcts.root)
        assert chosen is b


# ----- Cached expansion -----

class TestCachedExpansion:
    def test_first_visit_calls_expand_fn(self) -> None:
        """expand_fn fires on the first visit to a leaf, populating the cache."""
        seed = _seed_dag()
        actions = [_make_add_beat(f"n{i}") for i in range(4)]
        expand_fn = _make_expand_fn([actions])
        rollout = _constant_rollout(0.5)
        mcts = MCTS(seed, expand_fn=expand_fn, rollout_fn=rollout)

        asyncio.run(mcts.step())

        assert expand_fn.call_count() == 1  # type: ignore[attr-defined]
        assert mcts.root.cached_candidate_actions is not None
        assert len(mcts.root.cached_candidate_actions) == 4
        assert len(mcts.root.children) == 1

    def test_subsequent_visits_consume_cache_one_at_a_time(self) -> None:
        """After cache is populated, each iteration adds one more child until
        the cache is exhausted, then the leaf is fully_expanded."""
        seed = _seed_dag()
        actions = [_make_add_beat(f"n{i}") for i in range(4)]
        expand_fn = _make_expand_fn([actions, [], [], [], []])  # only first list non-empty
        rollout = _constant_rollout(0.5)
        mcts = MCTS(seed, expand_fn=expand_fn, rollout_fn=rollout)

        # Iterations 1-4 each instantiate one child.
        for i in range(4):
            asyncio.run(mcts.step())

        # expand_fn was called exactly once (the cache lasted 4 iterations).
        assert expand_fn.call_count() == 1  # type: ignore[attr-defined]
        assert len(mcts.root.children) == 4
        assert mcts.root.fully_expanded is True

    def test_invalid_action_skipped_not_fatal(self) -> None:
        """If an action would produce an invalid DAG (Pydantic validation
        error), MCTS skips it and the iteration produces no expansion. The
        node is not crashed; the next iteration tries the next cached action."""
        seed = _seed_dag()
        # Action with a bogus parent_id — won't apply.
        bad_action = AddBeatAction(
            anchor_id="nonexistent_node",
            direction="downstream",
            new_node_id="x",
            sketch="A new beat sketch with enough words to validate.",
            edge_type="causal",
            edge_payload={"realizes": "substantive realizes payload of four tokens"},
        )
        good_action = _make_add_beat("good")
        expand_fn = _make_expand_fn([[bad_action, good_action]])
        rollout = _constant_rollout(0.5)
        mcts = MCTS(seed, expand_fn=expand_fn, rollout_fn=rollout)

        asyncio.run(mcts.step())

        # The bad action was skipped; the good action produced a child.
        assert len(mcts.root.children) == 1
        assert mcts.root.children[0].dag.nodes[-1].id == "good"

    def test_expand_fn_failure_marks_node_fully_expanded(self) -> None:
        """expand_fn raising stops MCTS from looping on this node forever."""
        seed = _seed_dag()

        async def failing_expand(dag: DAG) -> list[Action]:
            raise RuntimeError("simulated LLM outage")

        rollout = _constant_rollout(0.5)
        mcts = MCTS(seed, expand_fn=failing_expand, rollout_fn=rollout)

        result = asyncio.run(mcts.step())
        assert result is None  # no expansion happened
        assert mcts.root.fully_expanded is True


# ----- Backprop -----

class TestBackprop:
    def test_increments_visits_along_chain(self) -> None:
        seed = _seed_dag()
        mcts = MCTS(seed, expand_fn=_make_expand_fn([]),
                    rollout_fn=_constant_rollout(0.7))
        # Build a chain: root → a → b → leaf. Selection-time would have
        # incremented `pending_visits` along this path; backprop decrements
        # them and increments real `visits`. Simulate the pending state.
        a = MCTSNode(dag=seed, parent=mcts.root, pending_visits=1)
        b = MCTSNode(dag=seed, parent=a, pending_visits=1)
        leaf = MCTSNode(dag=seed, parent=b, pending_visits=1)
        mcts.root.children.append(a)
        a.children.append(b)
        b.children.append(leaf)
        mcts.root.pending_visits = 1
        path = [mcts.root, a, b, leaf]

        mcts._backprop_path(path, 0.7)

        assert leaf.visits == 1
        assert b.visits == 1
        assert a.visits == 1
        assert mcts.root.visits == 1
        assert leaf.cumulative_return == pytest.approx(0.7)
        assert mcts.root.cumulative_return == pytest.approx(0.7)
        # pending_visits cleared by backprop
        for n in path:
            assert n.pending_visits == 0

    def test_multiple_backprops_accumulate(self) -> None:
        seed = _seed_dag()
        mcts = MCTS(seed, expand_fn=_make_expand_fn([]),
                    rollout_fn=_constant_rollout(0.5))
        leaf = MCTSNode(dag=seed, parent=mcts.root)
        mcts.root.children.append(leaf)

        # Each backprop call simulates one rollout: bump pending in advance.
        for reward in (0.5, 0.7, 0.3):
            mcts.root.pending_visits += 1
            leaf.pending_visits += 1
            mcts._backprop_path([mcts.root, leaf], reward)

        assert leaf.visits == 3
        assert leaf.cumulative_return == pytest.approx(1.5)
        assert mcts.root.visits == 3
        assert mcts.root.cumulative_return == pytest.approx(1.5)


# ----- End-to-end iteration -----

class TestEndToEndIteration:
    """Phase 5 exit criterion: one full iteration runs end-to-end with
    mocked expansion + mocked cheap judge."""

    def test_one_iteration_select_expand_rollout_backprop(self) -> None:
        """Single step on a 1-node seed: expand to a child, rollout from child,
        backprop reward up to root."""
        seed = _seed_dag()
        actions = [_make_add_beat("first")]
        expand_fn = _make_expand_fn([actions])
        rollout = _constant_rollout(0.8)

        mcts = MCTS(seed, expand_fn=expand_fn, rollout_fn=rollout)
        leaf = asyncio.run(mcts.step())

        # The new child was returned.
        assert leaf is not None
        assert leaf.dag.nodes[-1].id == "first"
        # Rollout fired once on the child.
        assert rollout.call_count() == 1  # type: ignore[attr-defined]
        # Backprop incremented visits at child + root.
        assert leaf.visits == 1
        assert mcts.root.visits == 1
        assert leaf.cumulative_return == pytest.approx(0.8)
        assert mcts.root.cumulative_return == pytest.approx(0.8)

    def test_run_n_iterations(self) -> None:
        """Smoke: run(n) walks n iterations without crashing."""
        seed = _seed_dag()
        actions_per_node = [
            [_make_add_beat(f"n{i}") for i in range(4)],   # root expansion
            [_make_add_beat(f"m{i}", anchor_id="n0") for i in range(4)],  # n0 expansion
        ]
        expand_fn = _make_expand_fn(actions_per_node)
        rollout = _constant_rollout(0.5)
        mcts = MCTS(seed, expand_fn=expand_fn, rollout_fn=rollout)

        asyncio.run(mcts.run(8))

        assert mcts.iterations == 8
        # Root has 4 children (cache exhausted after 4 iterations); n0 has
        # children too once selection descended into it.
        assert len(mcts.root.children) == 4
        assert mcts.root.fully_expanded is True

    def test_best_terminal_returns_visited_node(self) -> None:
        seed = _seed_dag()
        actions = [_make_add_beat(f"n{i}") for i in range(4)]
        expand_fn = _make_expand_fn([actions, [], [], [], []])
        # Rollout returns higher reward when a specific node is in the DAG.
        async def conditional_rollout(dag: DAG) -> float:
            if any(n.id == "n2" for n in dag.nodes):
                return 0.95
            return 0.3
        mcts = MCTS(seed, expand_fn=expand_fn, rollout_fn=conditional_rollout)

        asyncio.run(mcts.run(4))

        best = mcts.best_terminal()
        assert best.visits >= 1


# ----- Leaf-only best_terminal + dead-leaf selection skip -----

class TestBestTerminalLeavesOnly:
    """Issue 1 from `2026-04-26-stage-2-mcts-bit-comparison.md`. Internal-
    node mean reward is a subtree average, not the node's own DAG quality;
    `best_terminal` must compare like with like."""

    def test_internal_node_not_returned_on_tied_means(self) -> None:
        """Construct a tied-mean tree where iter order would have surfaced
        the internal node. New behavior: internal nodes are excluded so the
        winner is a leaf with the same mean."""
        seed = _seed_dag()
        mcts = MCTS(seed, expand_fn=_make_expand_fn([]),
                    rollout_fn=_constant_rollout(0.5),
                    config=MCTSConfig(beta=0.0))
        # Order matters: place the internal node FIRST so the old pre-order
        # traversal would have returned it on tied scores.
        internal = MCTSNode(dag=seed, parent=mcts.root, visits=1, cumulative_return=0.5)
        deep_leaf = MCTSNode(dag=seed, parent=internal, visits=1, cumulative_return=0.5)
        internal.children.append(deep_leaf)
        sibling_leaf = MCTSNode(dag=seed, parent=mcts.root, visits=1, cumulative_return=0.5)
        mcts.root.children.extend([internal, sibling_leaf])
        mcts.root.visits = 2
        mcts.root.cumulative_return = 1.0

        best = mcts.best_terminal()
        assert best is not internal
        assert best in (deep_leaf, sibling_leaf)


class TestDeadLeafSkipped:
    """Issue 2 from the same. A dead leaf (fully_expanded, no children — all
    proposed actions were invalid) shouldn't keep absorbing rollout budget
    when siblings still have live exploration potential."""

    def test_dead_sibling_skipped_alive_sibling_expanded(self) -> None:
        seed = _seed_dag()
        expand_fn = _make_expand_fn([[_make_add_beat("from_alive")]])
        rollout = _constant_rollout(0.5)
        mcts = MCTS(seed, expand_fn=expand_fn, rollout_fn=rollout,
                    config=MCTSConfig(beta=0.0, seed=42))

        dead = MCTSNode(dag=seed, parent=mcts.root, visits=1, cumulative_return=0.5,
                        fully_expanded=True, children=[])
        alive = MCTSNode(dag=seed, parent=mcts.root, visits=1, cumulative_return=0.5)
        mcts.root.children.extend([dead, alive])
        mcts.root.visits = 2
        mcts.root.cumulative_return = 1.0
        mcts.root.fully_expanded = True

        asyncio.run(mcts.step())

        # Alive was selected and expanded; dead was skipped (visits unchanged).
        assert dead.visits == 1
        assert len(alive.children) == 1
        assert alive.children[0].dag.nodes[-1].id == "from_alive"

    def test_all_dead_children_falls_through_to_parent_rollout(self) -> None:
        """Edge case: every child of the parent is a dead leaf. Walk should
        stop at the parent and fall into the existing re-rollout branch.
        UCB will self-regulate the cost as the parent's visits accrue."""
        seed = _seed_dag()
        expand_fn = _make_expand_fn([])
        rollout = _constant_rollout(0.5)
        mcts = MCTS(seed, expand_fn=expand_fn, rollout_fn=rollout,
                    config=MCTSConfig(beta=0.0, seed=42))

        dead_a = MCTSNode(dag=seed, parent=mcts.root, visits=1, cumulative_return=0.5,
                          fully_expanded=True, children=[])
        dead_b = MCTSNode(dag=seed, parent=mcts.root, visits=1, cumulative_return=0.5,
                          fully_expanded=True, children=[])
        mcts.root.children.extend([dead_a, dead_b])
        mcts.root.visits = 2
        mcts.root.cumulative_return = 1.0
        mcts.root.fully_expanded = True

        result = asyncio.run(mcts.step())

        # Walk broke at root; step's else branch re-rolled root's DAG.
        assert result is mcts.root
        assert mcts.root.visits == 3  # one new rollout backpropped
        # Dead children unchanged (never selected).
        assert dead_a.visits == 1
        assert dead_b.visits == 1


# ----- No-improvement cutoff -----


class TestNoImprovementCutoff:
    """Issue 4 from `2026-04-26-stage-2-mcts-bit-comparison.md`.
    `Stage2Config.no_improvement_cutoff_iterations` was previously a config
    field with no effect; it now halts `MCTS.run` early when the best leaf
    score plateaus."""

    def test_run_halts_when_best_leaf_plateaus(self) -> None:
        """Constant rollout never improves best leaf; run halts at cutoff+1
        iteration regardless of the requested budget."""
        seed = _seed_dag()
        actions = [_make_add_beat(f"n{i}") for i in range(4)]
        expand_fn = _make_expand_fn([actions, [], [], [], []])
        rollout = _constant_rollout(0.5)
        mcts = MCTS(
            seed,
            expand_fn=expand_fn,
            rollout_fn=rollout,
            config=MCTSConfig(no_improvement_cutoff_iterations=3, seed=42),
        )

        asyncio.run(mcts.run(50))

        # First iteration installs a leaf at mean=0.5 (improvement vs -inf).
        # Subsequent iterations never improve; cutoff fires after 3 stale iters.
        assert mcts.iterations < 50
        assert mcts.iterations <= 5

    def test_no_cutoff_runs_full_budget(self) -> None:
        """Default config (cutoff=None) preserves Phase 5 behavior — the
        full requested budget runs even when scores never improve."""
        seed = _seed_dag()
        actions = [_make_add_beat(f"n{i}") for i in range(4)]
        expand_fn = _make_expand_fn([actions, [], [], [], []])
        rollout = _constant_rollout(0.5)
        mcts = MCTS(seed, expand_fn=expand_fn, rollout_fn=rollout)

        asyncio.run(mcts.run(8))

        assert mcts.iterations == 8


# ----- D-UCB γ=1 sanity -----

class TestGammaSanity:
    """Phase 5 exit: D-UCB with γ=1 produces identical trees to UCB1 when
    champion is stable. Phase 5 ships γ=1 only; this test confirms the
    baseline so Phase 7's γ=0.93 swap can be A/B-compared against it.
    """

    def test_default_gamma_is_one(self) -> None:
        config = MCTSConfig()
        assert config.gamma == 1.0

    def test_gamma_zero_point_nine_three_accepted(self) -> None:
        """Phase 7 will set γ=0.93 default; for now the field accepts it
        but the implementation doesn't apply time-discount yet (Phase 5
        runs the formula as standard UCB1 regardless of γ value)."""
        config = MCTSConfig(gamma=0.93)
        assert config.gamma == 0.93
        # Phase 5 doesn't read gamma in the UCB calc; it's API-forward
        # for Phase 7. Verify that running with γ=0.93 produces the same
        # tree as γ=1.0 (since the implementation is UCB1 either way).
        seed = _seed_dag()
        actions = [_make_add_beat(f"n{i}") for i in range(4)]
        rollout = _constant_rollout(0.5)

        mcts_a = MCTS(seed, expand_fn=_make_expand_fn([actions, [], [], [], []]),
                      rollout_fn=rollout, config=MCTSConfig(gamma=1.0, seed=42))
        mcts_b = MCTS(seed, expand_fn=_make_expand_fn([actions, [], [], [], []]),
                      rollout_fn=rollout, config=MCTSConfig(gamma=0.93, seed=42))

        asyncio.run(mcts_a.run(4))
        asyncio.run(mcts_b.run(4))

        assert len(mcts_a.root.children) == len(mcts_b.root.children)
        # Same iteration count → same visit counts at root.
        assert mcts_a.root.visits == mcts_b.root.visits


# ----- Virtual loss + parallel selection -----


class TestVirtualLoss:
    def test_pending_visits_reduce_ucb_score(self) -> None:
        """A child with pending_visits=2 (virtual loss) should score lower than
        the same child with pending_visits=0, given identical real visits."""
        seed = _seed_dag()
        parent = MCTSNode(dag=seed, visits=10)
        clean = MCTSNode(dag=seed, visits=4, cumulative_return=2.0, pending_visits=0)
        loaded = MCTSNode(dag=seed, visits=4, cumulative_return=2.0, pending_visits=2)
        score_clean = ucb_score(clean, parent, c=0.0, beta=0.0, virtual_loss=1.0)
        score_loaded = ucb_score(loaded, parent, c=0.0, beta=0.0, virtual_loss=1.0)
        # Pure exploitation isolated: clean = 2.0/4 = 0.5;
        # loaded = (2.0 - 1.0*2) / (4+2) = 0.0/6 = 0.0.
        assert score_clean == pytest.approx(0.5)
        assert score_loaded == pytest.approx(0.0)
        assert score_clean > score_loaded

    def test_unvisited_with_pending_still_unbounded(self) -> None:
        """A child with visits=0 and pending_visits>0 is still effectively
        unvisited from a "should we try it" standpoint — but its score is
        finite (not +∞) because n_eff > 0. Ensures parallel workers don't
        all dogpile the same fresh child."""
        seed = _seed_dag()
        parent = MCTSNode(dag=seed, visits=5)
        fresh = MCTSNode(dag=seed, visits=0, cumulative_return=0.0, pending_visits=0)
        in_flight = MCTSNode(dag=seed, visits=0, cumulative_return=0.0, pending_visits=1)
        score_fresh = ucb_score(fresh, parent, c=0.5, beta=0.0)
        score_in_flight = ucb_score(in_flight, parent, c=0.5, beta=0.0)
        assert score_fresh == math.inf
        assert score_in_flight < math.inf  # has pending visit, no longer treated as +∞


class TestParallelRun:
    def test_parallel_workers_produce_concurrent_rollouts(self) -> None:
        """Three workers running 3 iterations against a pre-cached root: each
        worker claims a different action from the cache, expands into its own
        child, and rolls out concurrently. The slow rollout proves multiple
        rollouts are in flight simultaneously, not serialized."""
        seed = _seed_dag()
        actions = [
            AddBeatAction(
                action_type="add_beat", anchor_id="anchor", direction="downstream",
                new_node_id=f"branch_{i}",
                sketch="A specific subsequent beat with detail.",
                edge_type="causal",
                edge_payload={"realizes": f"branch {i} continues from the anchor"},
            )
            for i in range(3)
        ]
        rollout_started: list[asyncio.Event] = []
        rollout_release: asyncio.Event = asyncio.Event()

        async def slow_rollout(dag):
            evt = asyncio.Event()
            rollout_started.append(evt)
            evt.set()
            await rollout_release.wait()
            return 0.5

        # expand_fn never fires because the cache is pre-populated below.
        async def unused_expand(dag):
            return []

        mcts = MCTS(
            seed, expand_fn=unused_expand, rollout_fn=slow_rollout,
            config=MCTSConfig(parallel_workers=3, virtual_loss=1.0, seed=42),
        )
        mcts.root.cached_candidate_actions = list(actions)

        async def driver():
            run_task = asyncio.create_task(mcts.run(3))
            for _ in range(30):
                if len(rollout_started) >= 3:
                    break
                await asyncio.sleep(0.01)
            assert len(rollout_started) == 3, (
                f"expected 3 rollouts in flight; got {len(rollout_started)}"
            )
            rollout_release.set()
            await run_task

        asyncio.run(driver())
        # Each worker created a distinct child via cache claim; all backprop'd.
        assert len(mcts.root.children) == 3
        assert mcts.root.visits == 3
        assert mcts.root.pending_visits == 0
        for child in mcts.root.children:
            assert child.visits == 1
            assert child.pending_visits == 0

    def test_sequential_behavior_unchanged_when_workers_one(self) -> None:
        """parallel_workers=1 is the legacy path; virtual loss never kicks in
        because pending_visits stays 0 between select and backprop."""
        seed = _seed_dag()
        actions = [
            AddBeatAction(
                action_type="add_beat", anchor_id="anchor", direction="downstream",
                new_node_id=f"b{i}", sketch="A specific subsequent beat with detail.",
                edge_type="causal",
                edge_payload={"realizes": f"the {i}-th continuation happens"},
            )
            for i in range(1, 3)
        ]
        mcts = MCTS(
            seed,
            # Root cache holds both actions; root is the leaf for both iters.
            expand_fn=_make_expand_fn([actions]),
            rollout_fn=_constant_rollout(0.7),
            config=MCTSConfig(parallel_workers=1, seed=42),
        )
        asyncio.run(mcts.run(2))
        assert mcts.root.pending_visits == 0
        assert mcts.root.visits == 2
        for child in mcts.root.children:
            assert child.pending_visits == 0


# ----- Import smoke -----

class TestImports:
    def test_no_circular_imports(self) -> None:
        """Phase 5 exit: `python -c "from owtn.stage_2 import mcts"` works."""
        from owtn.stage_2 import mcts as _mcts_module  # noqa: F401
        from owtn.stage_2.mcts import MCTS, MCTSConfig, ucb_score  # noqa: F401
