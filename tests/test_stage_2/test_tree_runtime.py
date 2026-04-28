"""Unit tests for `owtn.stage_2.tree_runtime`.

Covers the rollout closure's bookkeeping invariants:
- Records critiques only when full panel ran.
- Promotes challenger only when `outcome.promoted` is True.
- Force-resummarizes the champion brief on promotion (per `mcts.md`
  §Forced re-render) so the next expansion sees a brief that reflects
  the new champion's accumulated critiques.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from owtn.evaluation.models import (
    STAGE_2_DIMENSION_NAMES,
    PairwiseResult,
)
from owtn.evaluation.stage_2 import CheapJudgeOutcome, RolloutEvaluation
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG
from owtn.evaluation.scalar.types import ScoreCard
from owtn.stage_2.tree_runtime import (
    TreeRuntimeState,
    _make_rollout_fn,
    _make_rollout_fn_scalar,
)
from tests.conftest import HILLS_GENOME


def _judge() -> JudgePersona:
    return JudgePersona(
        id="t", name="t", identity="t", values=["v"], exemplars=["e"],
        lean_in_signals=["s"], harshness="standard", priority="primary",
        model=["claude-sonnet-4-6"], temperature=0.0,
    )


def _seed_dag(preset: str = "p") -> DAG:
    concept = ConceptGenome.model_validate(HILLS_GENOME)
    return DAG(
        concept_id="c1", preset=preset,
        motif_threads=["m1"], concept_demands=[],
        nodes=[{
            "id": "anchor",
            "sketch": concept.anchor_scene.sketch,
            "role": [concept.anchor_scene.role],
            "motifs": [],
        }],
        edges=[], character_arcs=[], story_constraints=[],
        target_node_count=5,
    )


def _full_panel_outcome(winner: str = "a") -> PairwiseResult:
    return PairwiseResult(
        winner=winner,
        dimension_wins={d: winner for d in STAGE_2_DIMENSION_NAMES},
        a_wins=8 if winner == "a" else 0,
        b_wins=8 if winner == "b" else 0,
        ties=0,
        a_weighted=8.0 if winner == "a" else 0.0,
        b_weighted=8.0 if winner == "b" else 0.0,
        tie_weighted=0.0,
        judgments=[{
            "judge_id": "t", "forward_reasoning": "fwd",
            "reverse_reasoning": "rev",
        }],
    )


def _outcome(
    *, promoted: bool, with_full_panel: bool, verified_partial: DAG | None = None,
) -> RolloutEvaluation:
    return RolloutEvaluation(
        backprop_reward=0.7 if promoted else 0.0,
        cheap_outcome=CheapJudgeOutcome(
            challenger_wins=promoted, reward=0.7 if promoted else 0.0,
            resolved_votes={d: "a_clear" if promoted else "b_clear"
                             for d in STAGE_2_DIMENSION_NAMES},
            cost=0.001,
        ),
        verified_partial=verified_partial if verified_partial is not None else _seed_dag(),
        full_panel_outcome=_full_panel_outcome("a" if promoted else "b") if with_full_panel else None,
        promoted=promoted,
        cheap_full_agreement=True if with_full_panel else None,
        total_cost=0.001,
        notes=[],
    )


class TestRolloutClosure:
    def test_promotion_with_full_panel_records_critique_and_force_resummarizes(self) -> None:
        seed = _seed_dag()
        challenger = seed.model_copy(update={"preset": "challenger_marker"})
        state = TreeRuntimeState(running_champion=seed)
        concept = ConceptGenome.model_validate(HILLS_GENOME)

        async def fake_eval(*args, **kwargs):
            # No simulator in the closure → verified_partial == input challenger.
            return _outcome(
                promoted=True, with_full_panel=True, verified_partial=challenger,
            )

        force_calls = []

        async def fake_brief(state_arg, *, classifier_model, force_resummarize=False, **kwargs):
            force_calls.append(force_resummarize)
            return "(rendered)"

        with (
            patch("owtn.stage_2.tree_runtime.evaluate_rollout", side_effect=fake_eval),
            patch("owtn.stage_2.tree_runtime.get_or_compute_brief", side_effect=fake_brief),
        ):
            rollout = _make_rollout_fn(
                concept=concept,
                cheap_judge=_judge(),
                full_panel=[_judge()],
                rejection_backprop=0.5,
                classifier_model="dummy-model",
                state=state,
            )
            reward = asyncio.run(rollout(challenger))

        assert reward == pytest.approx(0.7)
        assert state.running_champion.preset == "challenger_marker"
        assert len(state.brief_state.full_panel_critiques) == 1
        # Force-resummarize fired exactly once on the promotion event.
        assert force_calls == [True]

    def test_no_promotion_no_force_resummarize(self) -> None:
        seed = _seed_dag()
        challenger = seed.model_copy(update={"preset": "challenger_marker"})
        state = TreeRuntimeState(running_champion=seed)
        concept = ConceptGenome.model_validate(HILLS_GENOME)

        async def fake_eval(*args, **kwargs):
            return _outcome(promoted=False, with_full_panel=True)

        force_calls = []

        async def fake_brief(*args, force_resummarize=False, **kwargs):
            force_calls.append(force_resummarize)
            return "(rendered)"

        with (
            patch("owtn.stage_2.tree_runtime.evaluate_rollout", side_effect=fake_eval),
            patch("owtn.stage_2.tree_runtime.get_or_compute_brief", side_effect=fake_brief),
        ):
            rollout = _make_rollout_fn(
                concept=concept,
                cheap_judge=_judge(),
                full_panel=[_judge()],
                rejection_backprop=0.5,
                classifier_model="dummy-model",
                state=state,
            )
            asyncio.run(rollout(challenger))

        # Champion did NOT advance — same DAG instance.
        assert state.running_champion.preset == seed.preset
        # Critique still recorded (full_panel ran).
        assert len(state.brief_state.full_panel_critiques) == 1
        # No force-resummarize because no promotion.
        assert force_calls == []

    def test_cheap_only_path_records_no_critique_no_force_resummarize(self) -> None:
        """Cheap-judge-only path (full_panel=None): no critique recorded,
        and a non-promotion outcome doesn't trigger force-resummarize."""
        seed = _seed_dag()
        challenger = seed.model_copy(update={"preset": "challenger_marker"})
        state = TreeRuntimeState(running_champion=seed)
        concept = ConceptGenome.model_validate(HILLS_GENOME)

        async def fake_eval(*args, **kwargs):
            return _outcome(promoted=False, with_full_panel=False)

        force_calls = []

        async def fake_brief(*args, force_resummarize=False, **kwargs):
            force_calls.append(force_resummarize)
            return "(rendered)"

        with (
            patch("owtn.stage_2.tree_runtime.evaluate_rollout", side_effect=fake_eval),
            patch("owtn.stage_2.tree_runtime.get_or_compute_brief", side_effect=fake_brief),
        ):
            rollout = _make_rollout_fn(
                concept=concept,
                cheap_judge=_judge(),
                full_panel=None,
                rejection_backprop=0.5,
                classifier_model="dummy-model",
                state=state,
            )
            asyncio.run(rollout(challenger))

        assert len(state.brief_state.full_panel_critiques) == 0
        assert force_calls == []


class TestRolloutWithSimulation:
    """The bounded simulation walk happens inside `evaluate_rollout`; here
    we verify that the rollout closure honors the post-evaluation contract:
    on promotion, the new champion is the `verified_partial` returned in the
    `RolloutEvaluation` (the walked state, not the raw input challenger), and
    the simulator is plumbed through to `evaluate_rollout` when configured."""

    def test_promotion_uses_verified_partial_as_new_champion(self) -> None:
        """When the walk extends the partial and the panel confirms, the new
        champion is the walked partial — not the original input challenger."""
        seed = _seed_dag()
        original_challenger = seed.model_copy(update={"preset": "input_challenger"})
        walked_partial = seed.model_copy(update={"preset": "walked_state"})
        state = TreeRuntimeState(running_champion=seed)
        concept = ConceptGenome.model_validate(HILLS_GENOME)

        async def fake_eval(inputs, **kwargs):
            return _outcome(
                promoted=True, with_full_panel=True,
                verified_partial=walked_partial,
            )

        async def fake_brief(*args, **kwargs):
            return "(rendered)"

        with (
            patch("owtn.stage_2.tree_runtime.evaluate_rollout", side_effect=fake_eval),
            patch("owtn.stage_2.tree_runtime.get_or_compute_brief", side_effect=fake_brief),
        ):
            rollout = _make_rollout_fn(
                concept=concept, cheap_judge=_judge(), full_panel=[_judge()],
                rejection_backprop=0.5, classifier_model="dummy-model",
                state=state, simulator=lambda i: None,  # opaque sentinel
            )
            asyncio.run(rollout(original_challenger))

        assert state.running_champion.preset == "walked_state"

    def test_critique_record_uses_verified_partial(self) -> None:
        """Brief summarizer should reason about what the panel actually saw —
        the walked state — not the unextended input challenger."""
        seed = _seed_dag()
        original = seed.model_copy(update={"preset": "input_challenger"})
        walked = seed.model_copy(update={"preset": "walked_state"})
        state = TreeRuntimeState(running_champion=seed)
        concept = ConceptGenome.model_validate(HILLS_GENOME)

        async def fake_eval(inputs, **kwargs):
            return _outcome(
                promoted=True, with_full_panel=True, verified_partial=walked,
            )

        async def fake_brief(*args, **kwargs):
            return "(rendered)"

        with (
            patch("owtn.stage_2.tree_runtime.evaluate_rollout", side_effect=fake_eval),
            patch("owtn.stage_2.tree_runtime.get_or_compute_brief", side_effect=fake_brief),
        ):
            rollout = _make_rollout_fn(
                concept=concept, cheap_judge=_judge(), full_panel=[_judge()],
                rejection_backprop=0.5, classifier_model="dummy-model",
                state=state, simulator=lambda i: None,
            )
            asyncio.run(rollout(original))

        critique = state.brief_state.full_panel_critiques[0]
        assert critique["self_dag"]["preset"] == "walked_state"

    def test_simulator_passed_through_to_evaluate_rollout(self) -> None:
        """The configured simulator must reach `evaluate_rollout` as a kwarg
        so the bounded walk actually fires inside the evaluator."""
        seed = _seed_dag()
        challenger = seed.model_copy(update={"preset": "ch"})
        state = TreeRuntimeState(running_champion=seed)
        concept = ConceptGenome.model_validate(HILLS_GENOME)

        captured = {}
        sentinel_simulator = object()  # opaque marker

        async def fake_eval(inputs, **kwargs):
            captured["simulator"] = kwargs.get("simulator")
            return _outcome(promoted=False, with_full_panel=False)

        async def fake_brief(*args, **kwargs):
            return "(rendered)"

        with (
            patch("owtn.stage_2.tree_runtime.evaluate_rollout", side_effect=fake_eval),
            patch("owtn.stage_2.tree_runtime.get_or_compute_brief", side_effect=fake_brief),
        ):
            rollout = _make_rollout_fn(
                concept=concept, cheap_judge=_judge(), full_panel=None,
                rejection_backprop=0.5, classifier_model="dummy-model",
                state=state, simulator=sentinel_simulator,
            )
            asyncio.run(rollout(challenger))

        assert captured["simulator"] is sentinel_simulator

    def test_no_simulator_evaluator_receives_none(self) -> None:
        seed = _seed_dag()
        challenger = seed.model_copy(update={"preset": "ch"})
        state = TreeRuntimeState(running_champion=seed)
        concept = ConceptGenome.model_validate(HILLS_GENOME)

        captured = {}

        async def fake_eval(inputs, **kwargs):
            captured["simulator"] = kwargs.get("simulator")
            return _outcome(promoted=False, with_full_panel=False)

        async def fake_brief(*args, **kwargs):
            return "(rendered)"

        with (
            patch("owtn.stage_2.tree_runtime.evaluate_rollout", side_effect=fake_eval),
            patch("owtn.stage_2.tree_runtime.get_or_compute_brief", side_effect=fake_brief),
        ):
            rollout = _make_rollout_fn(
                concept=concept, cheap_judge=_judge(), full_panel=None,
                rejection_backprop=0.5, classifier_model="dummy-model",
                state=state,
                # simulator omitted
            )
            asyncio.run(rollout(challenger))

        assert captured["simulator"] is None


class TestRolloutClosureScalar:
    """Scalar-mode rollout closure: no champion, no full panel.

    The closure delegates to a Scorer; whatever ScoreCard.aggregate the
    scorer returns IS the backprop reward. Reasoning is recorded into the
    tree's brief feed for cadence-based summarization.
    """

    def test_returns_aggregate_as_reward(self) -> None:
        seed = _seed_dag()
        challenger = seed.model_copy(update={"preset": "ch"})
        state = TreeRuntimeState(running_champion=seed)

        class FakeScorer:
            rubric = None  # not exercised by the closure

            async def score(self, artifact):
                return ScoreCard(
                    dim_scores={"x": 12.0},
                    aggregate=0.625,
                    n_calls=1,
                    judge_label="fake",
                    raw_responses=["because reasons"],
                )

        rollout = _make_rollout_fn_scalar(scorer=FakeScorer(), state=state.brief_state)
        reward = asyncio.run(rollout(challenger))

        assert reward == pytest.approx(0.625)
        # No champion mutation
        assert state.running_champion is seed
        # Reasoning fed to the brief queue
        assert state.brief_state.rollout_reasonings == ["because reasons"]
        # No full-panel critiques recorded
        assert state.brief_state.full_panel_critiques == []

    def test_no_reasoning_skips_brief_recording(self) -> None:
        seed = _seed_dag()
        state = TreeRuntimeState(running_champion=seed)

        class SilentScorer:
            rubric = None

            async def score(self, artifact):
                return ScoreCard(
                    dim_scores={}, aggregate=0.0, n_calls=0,
                    judge_label="silent", raw_responses=[],
                )

        rollout = _make_rollout_fn_scalar(scorer=SilentScorer(), state=state.brief_state)
        asyncio.run(rollout(seed))
        assert state.brief_state.rollout_reasonings == []
