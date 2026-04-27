"""Bounded MCTS rollout simulation (BiT-MCTS Algorithm 1, Section 2.3).

Why the rollout needs simulation: a freshly-expanded MCTS leaf is a partial
DAG with (typically) one more beat than its parent. Cheap-judge comparison
of partial-vs-partial at this granularity is noisy — adjacent leaves differ
by a single beat, so the judge's signal-to-noise ratio is poor.

How simulation fixes that: from the new leaf, walk up to `s_max` one-step
extensions using the same expansion machinery the search itself uses.
After each accepted step, evaluate the new partial against the running
champion via cheap-judge. Halt as soon as reward fails to improve. The
backpropagated reward is the best seen during the walk; the partial state
that achieved it is what the full panel verifies on a declared win.

Walk extensions are EPHEMERAL: they don't enter the tree. The MCTS leaf
keeps its post-expansion state regardless of what the walk produced.
Promotion replaces the running champion with the walked partial — the
state the panel actually verified — not the leaf's state.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import ValidationError

from owtn.evaluation.stage_2 import CheapJudgeOutcome, CompareInputs, cheap_judge_compare
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.actions import Action
from owtn.models.stage_2.dag import DAG
from owtn.stage_2.operators import apply_action


logger = logging.getLogger(__name__)


ExpandFn = Callable[[DAG], Awaitable[list[Action]]]
"""The same expansion seam MCTS uses to grow the tree. Returns up to K
candidate actions; the simulator takes the first that applies cleanly."""


@dataclass
class SimulationResult:
    """Outcome of a bounded simulation walk."""
    outcome: CheapJudgeOutcome
    """Cheap-judge result for the best partial state reached during the walk."""
    walked_partial: DAG
    """The partial DAG that achieved `outcome.reward` — equal to the input
    partial when the walk took zero accepted steps."""
    steps_accepted: int
    """How many extension steps were accepted before the walk halted."""


async def simulate_bounded(
    partial: DAG,
    *,
    concept: ConceptGenome,
    champion: DAG,
    cheap_judge: JudgePersona,
    expand_fn: ExpandFn,
    s_max: int = 3,
) -> SimulationResult:
    """Walk up to `s_max` one-step extensions from `partial`, halting when
    cheap-judge reward stops improving (BiT-MCTS Algorithm 1).

    Each step calls `expand_fn` to produce candidate extensions, takes the
    first that applies cleanly (validators run on construction; invalid
    candidates are skipped), evaluates the new partial against `champion`,
    and accepts the step iff reward weakly improves over the current best.

    Returns the best partial reached and its cheap-judge outcome — both
    used downstream for full-panel verification and (on promotion) as the
    new champion.
    """
    base_inputs = CompareInputs(challenger=partial, champion=champion, concept=concept)
    best_outcome = await cheap_judge_compare(base_inputs, cheap_judge=cheap_judge)
    best_partial = partial
    accepted = 0

    for step in range(1, s_max + 1):
        try:
            candidates = await expand_fn(best_partial)
        except Exception as e:  # noqa: BLE001 — never crash a rollout
            logger.warning("simulation: expand_fn raised at step %d (%s: %s); halting walk",
                           step, type(e).__name__, e)
            break

        next_partial = _apply_first_valid(best_partial, candidates)
        if next_partial is None:
            logger.info("simulation: no valid extension at step %d; halting walk", step)
            break

        next_outcome = await cheap_judge_compare(
            CompareInputs(challenger=next_partial, champion=champion, concept=concept),
            cheap_judge=cheap_judge,
        )
        if next_outcome.reward >= best_outcome.reward:
            best_partial = next_partial
            best_outcome = next_outcome
            accepted += 1
        else:
            logger.info(
                "simulation: step %d non-improving (reward %.3f → %.3f); early stop",
                step, best_outcome.reward, next_outcome.reward,
            )
            break

    logger.info(
        "simulation: walk %d/%d steps accepted, final reward=%.3f, %d-node partial",
        accepted, s_max, best_outcome.reward, len(best_partial.nodes),
    )
    return SimulationResult(
        outcome=best_outcome, walked_partial=best_partial, steps_accepted=accepted,
    )


def _apply_first_valid(dag: DAG, candidates: list[Action]) -> DAG | None:
    """Return the first candidate that applies cleanly, or None if all fail.
    Mirrors MCTS's expansion-side validator handling: invalid candidates are
    silently skipped (the proposal model is not asked to retry)."""
    for action in candidates:
        try:
            return apply_action(dag, action)
        except (ValidationError, ValueError):
            continue
    return None


# ----- Closure builder for the rollout pipeline -----


SimulatorFn = Callable[[CompareInputs], Awaitable[SimulationResult]]


def make_simulator(
    *,
    cheap_judge: JudgePersona,
    expand_fn: ExpandFn,
    s_max: int = 3,
    min_partial_size: int = 0,
) -> SimulatorFn:
    """Build a per-rollout simulator closure.

    Captures `cheap_judge`, `expand_fn`, and walk hyperparameters; the
    returned callable accepts the rollout's `CompareInputs` and returns
    a `SimulationResult`.

    `min_partial_size` gates simulation: when the input partial has fewer
    nodes, the walk is skipped and the cheap-judge runs against the
    untouched partial. This avoids burning compute on early-iteration
    rollouts where any walked extension would dominate the comparison
    signal anyway. Set to 0 to always simulate.
    """
    async def simulator(inputs: CompareInputs) -> SimulationResult:
        if len(inputs.challenger.nodes) < min_partial_size:
            outcome = await cheap_judge_compare(inputs, cheap_judge=cheap_judge)
            return SimulationResult(
                outcome=outcome, walked_partial=inputs.challenger, steps_accepted=0,
            )
        return await simulate_bounded(
            inputs.challenger,
            concept=inputs.concept, champion=inputs.champion,
            cheap_judge=cheap_judge, expand_fn=expand_fn, s_max=s_max,
        )
    return simulator
