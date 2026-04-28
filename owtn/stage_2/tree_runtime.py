"""Per-tree MCTS runtime state and rollout/expansion glue.

What lives here:
- `TreeRuntimeState`: per-tree mutable state — running champion, brief
  accumulator. The MCTS rollout closure mutates this; the expansion
  factory's brief_fetcher reads from it.
- `_make_rollout_fn`: builds an MCTS rollout closure that calls
  `evaluate_rollout` (cheap-judge tiered against the running champion,
  optionally preceded by a bounded simulation walk), records full-panel
  critiques into the brief state, and promotes the verified partial
  when the full panel confirms a win.
- `_build_critique_record`: PairwiseResult → MatchCritique-shaped dict
  for the brief summarizer.
- `run_one_preset_tree`: forward + backward + Phase 3 for one preset;
  returns a tournament entry. Closures share a single `TreeRuntimeState`
  so brief accumulation is continuous across phases.

Why split from `orchestration.py`: per-tree mechanics (closures, rollout
bookkeeping, refinement wiring) are mechanically distinct from
per-concept orchestration (seed → fan out across presets → tournament →
handoff). Keeping them apart makes both files easier to read and keeps
each under the codebase's ~300-line readability budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from owtn.evaluation.scalar import Scorer, build_scorer_from_config
from owtn.evaluation.scalar.renderers import render_stage2_partial
from owtn.evaluation.stage_2 import (
    CompareInputs,
    RolloutEvaluation,
    evaluate_rollout,
)
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.config import Stage2Config
from owtn.models.stage_2.dag import DAG
from owtn.stage_2.bidirectional import (
    PhaseContext,
    forward_phase_context,
    run_bidirectional,
)
from owtn.stage_2.champion_brief import (
    TreeBriefState,
    get_or_compute_brief,
    record_full_panel_critique,
    record_rollout_reasoning,
)
from owtn.stage_2.mcts import MCTSConfig, RolloutFn
from owtn.stage_2.operators import make_expand_factory
from owtn.stage_2.refinement import render_topology_context, run_refinement
from owtn.stage_2.simulation import SimulatorFn, make_simulator
from owtn.stage_2.tournament import TournamentEntry


logger = logging.getLogger(__name__)


@dataclass
class TreeRuntimeState:
    """Per-tree mutable state threaded through MCTS via closures.

    The rollout closure updates `running_champion` on full-panel confirmed
    wins; the expand factory's brief_fetcher reads `brief_state.cached_render`.
    Used as a mutable container so the MCTS rollout / expand callables
    can share state without arguments.
    """
    running_champion: DAG
    brief_state: TreeBriefState = field(default_factory=TreeBriefState)


def _make_rollout_fn(
    *,
    concept: ConceptGenome,
    cheap_judge: JudgePersona,
    full_panel: list[JudgePersona] | None,
    rejection_backprop: float,
    classifier_model: str,
    state: TreeRuntimeState,
    simulator: SimulatorFn | None = None,
) -> RolloutFn:
    """Build the MCTS rollout closure.

    Each rollout: tiered evaluation (cheap-judge always, full-panel on
    declared wins) against the tree's running champion. On full-panel
    confirmation, promote the verified partial; record the critique
    either way so the brief sees the full-panel signal. On promotion,
    force a fresh brief render so the next expansion's `brief_fetcher`
    reads guidance aware of the new champion (`mcts.md` §Forced re-render).

    When `simulator` is provided, the rollout runs a bounded walk before
    cheap-judge fires (`mcts.md` §Simulation). The walk produces the
    `verified_partial` — the best partial state reached during the walk —
    which both the cheap-judge and full panel evaluate, and which becomes
    the new champion on confirmed promotion. Walk extensions are
    ephemeral; the MCTS leaf retains its post-expansion state regardless
    of what the walk produced.
    """
    async def rollout(challenger: DAG) -> float:
        outcome = await evaluate_rollout(
            CompareInputs(
                challenger=challenger,
                champion=state.running_champion,
                concept=concept,
            ),
            cheap_judge=cheap_judge,
            full_panel=full_panel,
            rejection_backprop=rejection_backprop,
            simulator=simulator,
        )
        if outcome.full_panel_outcome is not None:
            critique = _build_critique_record(
                challenger=outcome.verified_partial,
                champion=state.running_champion,
                outcome=outcome,
            )
            record_full_panel_critique(state.brief_state, critique)
        if outcome.promoted:
            state.running_champion = outcome.verified_partial
            await get_or_compute_brief(
                state.brief_state,
                classifier_model=classifier_model,
                force_resummarize=True,
            )
        return outcome.backprop_reward

    return rollout


def _make_rollout_fn_scalar(
    *,
    scorer: Scorer,
    state: TreeBriefState,
) -> RolloutFn:
    """Build the scalar-mode MCTS rollout closure.

    Each rollout: score the partial DAG absolutely via `scorer.score()`. The
    scorer's aggregate IS the backprop reward — no champion comparison, no
    promotion gate. Rollout reasoning is recorded into the tree's brief state
    for cadence-based summarization (full-panel critiques don't fire in
    scalar mode because there's no full panel).
    """
    async def rollout(challenger: DAG) -> float:
        card = await scorer.score(challenger)
        if card.raw_responses:
            record_rollout_reasoning(state, card.raw_responses[0])
        return card.aggregate

    return rollout


def _build_critique_record(
    *,
    challenger: DAG,
    champion: DAG,
    outcome: RolloutEvaluation,
) -> dict:
    """PairwiseResult → MatchCritique-shaped dict for the brief summarizer.

    The "subject" of every critique is the tree (per `mcts.md`). For
    promotion-gate matches: self_dag = challenger (the new candidate the
    tree just produced), opponent_genome = the running champion (also
    produced by this tree at an earlier iteration). self_was_champion is
    always False at promotion gates — the tree is challenging itself.
    """
    panel = outcome.full_panel_outcome
    assert panel is not None, "_build_critique_record requires full-panel outcome"
    dim_outcomes = {
        dim: ("won" if w == "a" else "lost" if w == "b" else "tied")
        for dim, w in panel.dimension_wins.items()
    }
    if panel.winner == "a":
        outcome_str = "won"
    elif panel.winner == "b":
        outcome_str = "lost"
    else:  # pragma: no cover — compare_stage2 doesn't currently emit "tie" winner
        outcome_str = "tied"

    judge_reasonings = []
    for j in panel.judgments:
        # Stage 2 records both forward and reverse reasoning; concatenate
        # for the summarizer with explicit headers so the summarizer can
        # see how each judge resolved the dual-ordering.
        judge_reasonings.append({
            "judge_id": j.get("judge_id", "?"),
            "harshness": "?",  # not currently propagated through PairwiseResult
            "reasoning": (
                f"FORWARD ORDERING:\n{j.get('forward_reasoning', '')}\n\n"
                f"REVERSE ORDERING:\n{j.get('reverse_reasoning', '')}"
            ),
        })

    return {
        "self_label": "a",
        "opponent_label": "b",
        "self_was_champion": False,
        "self_dag": challenger.model_dump(),
        "opponent_genome": champion.model_dump(),
        "outcome": outcome_str,
        "dim_outcomes": dim_outcomes,
        "judge_reasonings": judge_reasonings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def run_one_preset_tree(
    *,
    seed_dag: DAG,
    concept: ConceptGenome,
    preset: str,
    config: Stage2Config,
    cheap_judge: JudgePersona,
    full_panel: list[JudgePersona] | None,
    classifier_model: str,
) -> TournamentEntry:
    """Run forward + backward + Phase 3 for one preset; return a tournament entry.

    The expansion LLM call uses `make_expand_factory` with a brief-fetcher
    that closes over this tree's `TreeRuntimeState`; the rollout closure
    updates the same state on champion promotions. Forward and backward
    phases share the state, so brief accumulation is continuous across them.
    """
    state = TreeRuntimeState(running_champion=seed_dag)

    def brief_fetcher() -> str:
        # Sync read of the cached render; the brief is updated lazily by
        # `evaluate_rollout` recording critiques + the runner triggering
        # re-summarize. For this Phase 9 wiring, the brief stays stale
        # within a phase — Phase 7+ rechallenge fires after promotions
        # to re-summarize on demand. In scalar mode the brief sources from
        # rollout reasoning instead of full-panel critiques.
        return state.brief_state.cached_render or "(brief not yet available)"

    expand_factory = make_expand_factory(
        concept=concept,
        brief_fetcher=brief_fetcher,
        model_name=config.expansion_model,
        k=config.k_candidates_per_expansion,
    )
    # Refinement uses a separate factory that injects per-DAG topology
    # context (which nodes sit upstream vs. downstream of the anchor) so
    # the LLM proposes spanning edges directly instead of relying on the
    # post-hoc filter to reject ~110 candidate src/dst pairs blindly.
    refinement_expand_factory = make_expand_factory(
        concept=concept,
        brief_fetcher=brief_fetcher,
        extra_context_fn=render_topology_context,
        model_name=config.expansion_model,
        k=config.k_candidates_per_expansion,
    )
    # Bounded simulation gate: when enabled, build a per-tree simulator that
    # walks up to s_max forward extensions per rollout. Each extension is a
    # single beat addition, judge-filtered (only accepted on reward
    # improvement), with the walk halting on non-improvement — so the
    # extension model only needs to propose a reasonable single beat per
    # step. We build a SEPARATE expand_factory for simulation parameterized
    # by `simulation_model` (default: classifier_model, the cheap one) since
    # search-tier reasoning isn't needed for one judge-filtered step. The
    # FORWARD-phase context is used regardless of the calling phase —
    # simulation projects forward toward a complete-enough state for the
    # cheap-judge to discriminate, not about respecting add-direction.
    simulator = None
    if config.simulate_rollouts:
        sim_expand_factory = make_expand_factory(
            concept=concept,
            brief_fetcher=brief_fetcher,
            model_name=config.simulation_model,
            k=config.k_candidates_per_expansion,
        )
        anchor_id = next(n.id for n in seed_dag.nodes if n.role)
        forward_ctx = forward_phase_context(anchor_id=anchor_id, pacing_hint="")
        simulator = make_simulator(
            cheap_judge=cheap_judge,
            expand_fn=sim_expand_factory(forward_ctx),
            s_max=config.simulation_max_steps,
            min_partial_size=config.simulation_min_partial_size,
        )
    if config.scoring_mode == "scalar":
        scorer = build_scorer_from_config(
            config.scoring_rollout_composition,
            render_stage2_partial,
        )
        rollout_fn = _make_rollout_fn_scalar(scorer=scorer, state=state.brief_state)
    else:
        rollout_fn = _make_rollout_fn(
            concept=concept,
            cheap_judge=cheap_judge,
            full_panel=full_panel,
            rejection_backprop=config.full_panel_rejection_backprop,
            classifier_model=classifier_model,
            state=state,
            simulator=simulator,
        )
    mcts_config = MCTSConfig(
        c=config.exploration_constant,
        gamma=config.discount_gamma,
        k_candidates_per_expansion=config.k_candidates_per_expansion,
        no_improvement_cutoff_iterations=config.no_improvement_cutoff_iterations,
        parallel_workers=config.mcts_parallel_workers,
        virtual_loss=config.mcts_virtual_loss,
    )

    bidirectional_result = await run_bidirectional(
        seed_dag,
        expand_factory=expand_factory,
        rollout_fn=rollout_fn,
        config=mcts_config,
        forward_iterations=config.iterations_per_phase,
        backward_iterations=config.iterations_per_phase,
    )

    refined, refinement_metrics = await run_refinement(
        bidirectional_result.winner_dag,
        expand_factory=refinement_expand_factory,
        rollout_fn=rollout_fn,
        config=mcts_config,
        iterations=config.phase_3_iterations,
    )
    if refinement_metrics.improved:
        logger.info(
            "  preset %s: Phase 3 added %d edge(s)",
            preset,
            refinement_metrics.phase_3_edge_count - refinement_metrics.phase_2_edge_count,
        )

    best = bidirectional_result.backward_mcts.best_terminal()
    mcts_reward = best.cumulative_return / best.visits if best.visits else 0.0

    return TournamentEntry(
        preset=preset,
        dag=refined,
        mcts_reward=mcts_reward,
    )
