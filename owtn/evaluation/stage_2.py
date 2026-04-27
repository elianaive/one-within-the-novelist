"""Stage 2 pairwise comparison + tiered judge orchestration.

Two granularities of judging fire in Stage 2:

- **Cheap-judge rollout reward**: 1 judge × 2 orderings (parallel, no extra
  latency) per MCTS rollout terminal. Returns a score in [0, 1] from the
  challenger's perspective. ~$0.001-0.002 per call.
- **Full-panel commitment events**: the standard 4-judge × 2-ordering panel,
  fired only when the cheap judge declares a challenger wins. Verifies before
  the champion is actually promoted. ~$0.04-0.08 per comparison.

Both granularities share the same underlying mechanics: per-judge dual ordering
(forward + reverse), per-dimension vote resolution (both orderings must agree
on side; magnitude is the min), majority aggregation across judges (with
magnitude tiebreaker on tied-count splits), weighted-sum winner selection
with asymmetric tiebreaker.

The dim-set-agnostic aggregation helpers in `owtn.evaluation.pairwise` are
intentionally not imported here. They iterate Stage 1's `DIMENSION_NAMES`.
We re-implement the equivalent logic locally over Stage 2's 8 dimensions to
avoid touching Stage 1's contract for Stage 2's needs. Behavior is byte-
equivalent; if duplication grows in later phases, consolidate by parameter-
izing the Stage 1 helpers.

Standalone CLI: `python -m owtn.evaluation.stage_2 <champ.json> <challenger.json>`
runs cheap-judge dual-ordering against two canonical or fixture DAGs and
prints the resolved verdict + reward. Useful for hand-checking judges
before wiring into MCTS in Phase 5.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from owtn.evaluation.models import (
    STAGE_2_DIMENSION_NAMES,
    PairwiseResult,
    Stage2PairwiseJudgment,
    encode_vote,
    parse_vote,
)
from owtn.evaluation.pairwise import _MAGNITUDE_TIEBREAKER_THRESHOLD, _build_judge_kwargs
from owtn.llm.call_logger import llm_context
from owtn.llm.query import query_async
from owtn.models.judge import JudgePersona, load_panel
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG
from owtn.prompts.stage_2.registry import (
    build_stage2_pairwise_system,
    build_stage2_pairwise_user,
)
from owtn.stage_2.rendering import render


logger = logging.getLogger(__name__)


# Default per-dim weights: uniform 1.0. Tuned weights (parallel to Stage 1's
# `PairwiseAggregationConfig.dim_weights`) ship in Phase 9's run config. For
# Phase 4 — wiring evaluation before MCTS — uniform weights are correct since
# we don't yet have run data to calibrate against.
DEFAULT_DIM_WEIGHTS: dict[str, float] = {dim: 1.0 for dim in STAGE_2_DIMENSION_NAMES}


# ----- Vote handling (parallel to pairwise.py's private helpers) -----

def _flip_stage2_votes(judgment: Stage2PairwiseJudgment) -> dict[str, str]:
    """Flip a/b in votes for the reversed ordering.

    Parallel to `pairwise._flip_votes`; difference is only the dim set.
    Magnitude and tie are preserved — a judge who votes 'a_decisive' when
    A is labeled A should flip to 'b_decisive' when A is labeled B.
    """
    flipped: dict[str, str] = {}
    for dim, vote in judgment.votes().items():
        if vote == "tie":
            flipped[dim] = "tie"
        elif vote.startswith("a_"):
            flipped[dim] = "b_" + vote.split("_", 1)[1]
        else:
            flipped[dim] = "a_" + vote.split("_", 1)[1]
    return flipped


def _resolve_stage2_votes(
    forward_votes: dict[str, str],
    reverse_votes: dict[str, str],
) -> dict[str, str]:
    """Conservative dual-ordering vote resolution per Stage 2 dim set.

    For each dim: side wins iff both orderings agree on side; magnitude is
    the min of the two ordering magnitudes (judge who waffled between
    decisive and narrow expressed less confidence than either alone).
    Side disagreement or either-ordering tie collapses to 'tie'.

    Parallel to `pairwise._resolve_votes`; only the iterated dim set differs.
    """
    resolved: dict[str, str] = {}
    for dim in STAGE_2_DIMENSION_NAMES:
        fwd = forward_votes.get(dim, "tie")
        rev = reverse_votes.get(dim, "tie")
        fwd_side, fwd_mag = parse_vote(fwd)
        rev_side, rev_mag = parse_vote(rev)
        if fwd_side == rev_side and fwd_side != "tie":
            resolved[dim] = encode_vote(fwd_side, min(fwd_mag, rev_mag))
        else:
            resolved[dim] = "tie"
    return resolved


def _aggregate_stage2(
    all_resolved: list[dict[str, str]],
    dim_weights: dict[str, float],
) -> tuple[dict[str, str], int, int, int, float, float, float]:
    """Aggregate resolved votes across judges. Parallel to `pairwise._aggregate`.

    Per-dim majority-of-non-tie picks the side; magnitude tiebreaker resolves
    tied-count splits when the magnitude gap is ≥ _MAGNITUDE_TIEBREAKER_THRESHOLD.
    Weighted total = dim_weight × mean magnitude on the winning side.

    Returns (dim_winners, a_wins, b_wins, ties, a_weighted, b_weighted, tie_weighted).
    """
    dim_winners: dict[str, str] = {}
    a_wins = b_wins = tie_count = 0
    a_weighted = b_weighted = tie_weighted = 0.0

    for dim in STAGE_2_DIMENSION_NAMES:
        votes = [r[dim] for r in all_resolved]
        parsed = [parse_vote(v) for v in votes]
        sides = [s for s, _ in parsed]
        counts = Counter(sides)
        weight = dim_weights[dim]
        a_count = counts.get("a", 0)
        b_count = counts.get("b", 0)

        if a_count > b_count:
            dim_winners[dim] = "a"
            a_wins += 1
            a_mags = [m for s, m in parsed if s == "a"]
            a_weighted += weight * (sum(a_mags) / len(a_mags))
        elif b_count > a_count:
            dim_winners[dim] = "b"
            b_wins += 1
            b_mags = [m for s, m in parsed if s == "b"]
            b_weighted += weight * (sum(b_mags) / len(b_mags))
        elif a_count > 0:
            # Tied count, magnitude signal on both sides (e.g. 2-2).
            a_mags = [m for s, m in parsed if s == "a"]
            b_mags = [m for s, m in parsed if s == "b"]
            a_mean = sum(a_mags) / len(a_mags)
            b_mean = sum(b_mags) / len(b_mags)
            if a_mean - b_mean >= _MAGNITUDE_TIEBREAKER_THRESHOLD:
                dim_winners[dim] = "a"
                a_wins += 1
                a_weighted += weight * a_mean
            elif b_mean - a_mean >= _MAGNITUDE_TIEBREAKER_THRESHOLD:
                dim_winners[dim] = "b"
                b_wins += 1
                b_weighted += weight * b_mean
            else:
                dim_winners[dim] = "tie"
                tie_count += 1
                tie_weighted += weight
        else:
            dim_winners[dim] = "tie"
            tie_count += 1
            tie_weighted += weight

    return dim_winners, a_wins, b_wins, tie_count, a_weighted, b_weighted, tie_weighted


# ----- Per-judge call -----

async def _judge_one_ordering(
    judge: JudgePersona,
    *,
    concept: ConceptGenome,
    structure_a_rendering: str,
    structure_b_rendering: str,
) -> tuple[Stage2PairwiseJudgment, float]:
    """One judge, one ordering. Returns (judgment, cost)."""
    # Set inside the coroutine so each task writes its own judge_id into
    # its own Task-local context (matches Stage 1's pattern in pairwise.py).
    llm_context.set({"role": "stage2_pairwise_judge", "judge_id": judge.id})
    system_msg = build_stage2_pairwise_system(judge)
    user_msg = build_stage2_pairwise_user(
        concept, structure_a_rendering, structure_b_rendering,
    )
    model_name, judge_kwargs = _build_judge_kwargs(judge)

    # Stable system prefix for caching, same as Stage 1's pairwise.
    result = await query_async(
        model_name=model_name,
        msg=user_msg,
        system_msg="",
        system_prefix=system_msg,
        output_model=Stage2PairwiseJudgment,
        **judge_kwargs,
    )
    return result.content, result.cost


# ----- Public comparison API -----


@dataclass
class CompareInputs:
    """Bundle of arguments for a Stage 2 comparison. Used by both the cheap-
    judge path and the full-panel path so call sites stay symmetric."""
    challenger: DAG
    champion: DAG
    concept: ConceptGenome


async def compare_stage2(
    inputs: CompareInputs,
    *,
    panel: list[JudgePersona],
    dim_weights: dict[str, float] | None = None,
) -> PairwiseResult:
    """Run a panel of judges over the (challenger vs. champion) comparison.

    Each judge runs both orderings (forward: challenger=A, champion=B;
    reverse: challenger=B, champion=A), votes are resolved per-dim, judges'
    resolved votes are aggregated across the panel.

    Returns a `PairwiseResult` with `winner` field stating "a" (= challenger
    wins) or "b" (= champion wins). The result reuses Stage 1's `PairwiseResult`
    schema since the structure is dim-set-agnostic.
    """
    if dim_weights is None:
        dim_weights = DEFAULT_DIM_WEIGHTS

    structure_a = render(inputs.challenger, label="A")
    structure_b = render(inputs.champion, label="B")
    structure_a_reversed = render(inputs.challenger, label="B")
    structure_b_reversed = render(inputs.champion, label="A")

    # Each judge: 2 calls (forward + reverse). All judges in parallel.
    forward_tasks: list[Any] = []
    reverse_tasks: list[Any] = []
    for judge in panel:
        forward_tasks.append(_judge_one_ordering(
            judge, concept=inputs.concept,
            structure_a_rendering=structure_a,
            structure_b_rendering=structure_b,
        ))
        # Reverse: swap which DAG goes into the A slot vs. the B slot.
        reverse_tasks.append(_judge_one_ordering(
            judge, concept=inputs.concept,
            structure_a_rendering=structure_b_reversed,
            structure_b_rendering=structure_a_reversed,
        ))

    forward_results = await asyncio.gather(*forward_tasks, return_exceptions=True)
    reverse_results = await asyncio.gather(*reverse_tasks, return_exceptions=True)

    all_resolved: list[dict[str, str]] = []
    judgments: list[dict] = []
    total_cost = 0.0

    for judge, fwd, rev in zip(panel, forward_results, reverse_results):
        if isinstance(fwd, BaseException) or isinstance(rev, BaseException):
            err = fwd if isinstance(fwd, BaseException) else rev
            logger.warning(
                "Stage 2 judge %r failed (%s: %s); skipping for this comparison.",
                judge.id, type(err).__name__, err,
            )
            continue

        fwd_judgment, fwd_cost = fwd
        rev_judgment, rev_cost = rev
        total_cost += fwd_cost + rev_cost

        forward_votes = fwd_judgment.votes()
        # Reverse-ordering votes are with A/B swapped; flip them so they're
        # comparable to forward-ordering votes (challenger always = "a" side).
        reverse_votes = _flip_stage2_votes(rev_judgment)
        resolved = _resolve_stage2_votes(forward_votes, reverse_votes)
        all_resolved.append(resolved)

        judgments.append({
            "judge_id": judge.id,
            "forward_votes": forward_votes,
            "reverse_votes": reverse_votes,
            "resolved": resolved,
            "forward_reasoning": fwd_judgment.reasoning,
            "reverse_reasoning": rev_judgment.reasoning,
            "cost": fwd_cost + rev_cost,
        })

    if not all_resolved:
        # Every judge failed. Return a tie with an empty result.
        return PairwiseResult(
            winner="b",  # champion holds on every-judge-failure
            dimension_wins={d: "tie" for d in STAGE_2_DIMENSION_NAMES},
            a_wins=0, b_wins=0, ties=len(STAGE_2_DIMENSION_NAMES),
            a_weighted=0.0, b_weighted=0.0,
            tie_weighted=sum(dim_weights.values()),
            judgments=[],
        )

    dim_winners, a_wins, b_wins, ties, a_weighted, b_weighted, tie_weighted = (
        _aggregate_stage2(all_resolved, dim_weights)
    )

    # Winner: simpler than Stage 1's asymmetric tiebreaker (no champion-leans
    # rule for Stage 2 in v1; champion holds only when weighted totals tie
    # exactly). Phase 8's tournament uses richer tiebreaking; Phase 4's
    # rollout reward only needs the binary direction.
    if a_weighted > b_weighted:
        winner = "a"
    elif b_weighted > a_weighted:
        winner = "b"
    else:
        winner = "b"  # champion holds on exact tie

    return PairwiseResult(
        winner=winner,
        dimension_wins=dim_winners,
        a_wins=a_wins, b_wins=b_wins, ties=ties,
        a_weighted=a_weighted, b_weighted=b_weighted, tie_weighted=tie_weighted,
        judgments=judgments,
    )


# ----- Reward + tiered orchestration -----


def reward_from_resolved(resolved: dict[str, str]) -> float:
    """Reward score in [0, 1] from the challenger's perspective.

    `(dim_wins + 0.5 * ties) / 8`, computed on the resolved per-dim votes.
    Mirrors Stage 1's `a_score` semantics but at dim granularity rather than
    weighted. Used as the rollout reward signal MCTS backpropagates.
    """
    a_wins = sum(1 for v in resolved.values() if v.startswith("a_"))
    ties = sum(1 for v in resolved.values() if v == "tie")
    total = len(resolved) or 1
    return (a_wins + 0.5 * ties) / total


@dataclass
class CheapJudgeOutcome:
    """Result of a single cheap-judge dual-ordering comparison."""
    challenger_wins: bool   # convenience: True if reward > 0.5 strictly
    reward: float           # in [0, 1]
    resolved_votes: dict[str, str]
    cost: float
    judgment_forward: Stage2PairwiseJudgment | None = None
    judgment_reverse: Stage2PairwiseJudgment | None = None


async def cheap_judge_compare(
    inputs: CompareInputs,
    *,
    cheap_judge: JudgePersona,
) -> CheapJudgeOutcome:
    """One cheap judge × 2 orderings in parallel. Used for MCTS rollout reward.

    Total LLM calls: 2 (forward + reverse). On either-call failure, returns
    an all-tie outcome with reward 0.5 — MCTS backpropagates the neutral
    score rather than trying again.
    """
    structure_a = render(inputs.challenger, label="A")
    structure_b = render(inputs.champion, label="B")
    structure_a_reversed = render(inputs.challenger, label="B")
    structure_b_reversed = render(inputs.champion, label="A")

    forward_task = _judge_one_ordering(
        cheap_judge, concept=inputs.concept,
        structure_a_rendering=structure_a,
        structure_b_rendering=structure_b,
    )
    reverse_task = _judge_one_ordering(
        cheap_judge, concept=inputs.concept,
        structure_a_rendering=structure_b_reversed,
        structure_b_rendering=structure_a_reversed,
    )

    fwd_pair, rev_pair = await asyncio.gather(
        forward_task, reverse_task, return_exceptions=True,
    )

    if isinstance(fwd_pair, BaseException) or isinstance(rev_pair, BaseException):
        err = fwd_pair if isinstance(fwd_pair, BaseException) else rev_pair
        logger.warning(
            "Cheap-judge call failed (%s: %s); returning neutral reward 0.5.",
            type(err).__name__, err,
        )
        return CheapJudgeOutcome(
            challenger_wins=False,
            reward=0.5,
            resolved_votes={d: "tie" for d in STAGE_2_DIMENSION_NAMES},
            cost=0.0,
        )

    fwd_judgment, fwd_cost = fwd_pair
    rev_judgment, rev_cost = rev_pair
    forward_votes = fwd_judgment.votes()
    reverse_votes = _flip_stage2_votes(rev_judgment)
    resolved = _resolve_stage2_votes(forward_votes, reverse_votes)
    reward = reward_from_resolved(resolved)

    return CheapJudgeOutcome(
        challenger_wins=reward > 0.5,
        reward=reward,
        resolved_votes=resolved,
        cost=fwd_cost + rev_cost,
        judgment_forward=fwd_judgment,
        judgment_reverse=rev_judgment,
    )


@dataclass
class RolloutEvaluation:
    """Result of one MCTS rollout's tiered evaluation.

    Fields:
        backprop_reward: the score MCTS backpropagates up the tree. Per
            `docs/stage-2/mcts.md` §Reward Function: cheap-judge score
            normally; the configured rejection-backprop value (default 0.5)
            when the cheap judge says win but the full panel rejects.
        cheap_outcome: the cheap judge's dual-ordering result.
        verified_partial: the partial DAG the full panel evaluated AND that
            becomes the new champion on promotion. When simulation is on,
            this is the BEST partial reached during the bounded walk (which
            may have 0-3 more beats than the rollout's input challenger);
            when simulation is off, this is the input challenger unchanged.
        full_panel_outcome: present only when the cheap judge said win and
            the full panel was triggered. None otherwise.
        promoted: True iff the challenger should become the new champion
            (cheap says win AND full panel confirmed, OR cheap says win
            and no full panel was provided).
        cheap_full_agreement: present when full panel ran. True iff full
            panel's winner matches cheap judge's winner. Used for the
            cheap-judge-drift monitoring per `mcts.md`.
        total_cost: sum of LLM costs across cheap + (optional) full panel.
    """
    backprop_reward: float
    cheap_outcome: CheapJudgeOutcome
    verified_partial: DAG
    full_panel_outcome: PairwiseResult | None = None
    promoted: bool = False
    cheap_full_agreement: bool | None = None
    total_cost: float = 0.0
    notes: list[str] = field(default_factory=list)


# Type imported lazily inside the function to avoid a cyclic import at module
# load: simulation.py imports from this module (CompareInputs, cheap_judge_compare).
SimulatorFn = Any  # Callable[[CompareInputs], Awaitable[SimulationResult]]


async def evaluate_rollout(
    inputs: CompareInputs,
    *,
    cheap_judge: JudgePersona,
    full_panel: list[JudgePersona] | None = None,
    rejection_backprop: float = 0.5,
    dim_weights: dict[str, float] | None = None,
    simulator: SimulatorFn | None = None,
) -> RolloutEvaluation:
    """Tiered evaluation: cheap judge always; full panel only on declared win.

    Per `docs/stage-2/mcts.md` §Reward Function §Tiered judge design:
    - Cheap judge runs every rollout.
    - If cheap says challenger wins AND `full_panel` is provided, the full
      panel verifies before promotion.
    - On full-panel rejection: backprop = `rejection_backprop` (default 0.5),
      NOT the cheap judge's win-score. Per the design, this corrects the
      false-positive signal without mixing full-panel statistics into the
      tree's UCB. Logged so the cheap-vs-full agreement rate can be tracked.
    - On full-panel confirmation: backprop = cheap judge's score; challenger
      promoted to champion.

    `full_panel=None` disables the full-panel verification step. In that mode,
    promotion is decided by the cheap judge alone (cheap says win → promote).
    Useful for tests and for runs that disable full-panel verification.

    `simulator`, when provided, runs a bounded walk per `mcts.md` §Simulation
    before the cheap judge fires. The walk extends the partial up to s_max
    times against the running champion, halting when reward stops improving.
    The cheap-judge outcome and the full-panel verification both run on the
    walked partial — the state the walk actually accepted — so what the panel
    confirms matches what cheap-judge approved. Without simulation, the
    cheap judge runs once on the unwalked challenger.
    """
    if simulator is not None:
        result = await simulator(inputs)
        cheap_outcome = result.outcome
        verified_partial = result.walked_partial
        verification_inputs = CompareInputs(
            challenger=verified_partial,
            champion=inputs.champion,
            concept=inputs.concept,
        )
    else:
        cheap_outcome = await cheap_judge_compare(inputs, cheap_judge=cheap_judge)
        verified_partial = inputs.challenger
        verification_inputs = inputs

    notes: list[str] = []
    total_cost = cheap_outcome.cost

    if not cheap_outcome.challenger_wins:
        notes.append("cheap_says_no_win:no_full_panel_call")
        return RolloutEvaluation(
            backprop_reward=cheap_outcome.reward,
            cheap_outcome=cheap_outcome,
            verified_partial=verified_partial,
            full_panel_outcome=None,
            promoted=False,
            cheap_full_agreement=None,
            total_cost=total_cost,
            notes=notes,
        )

    # Cheap says challenger wins. If we have a full panel, verify.
    if full_panel is None:
        notes.append("cheap_says_win:no_full_panel_configured:promoted")
        return RolloutEvaluation(
            backprop_reward=cheap_outcome.reward,
            cheap_outcome=cheap_outcome,
            verified_partial=verified_partial,
            full_panel_outcome=None,
            promoted=True,
            cheap_full_agreement=None,
            total_cost=total_cost,
            notes=notes,
        )

    panel_outcome = await compare_stage2(
        verification_inputs, panel=full_panel, dim_weights=dim_weights,
    )
    panel_cost = sum(j["cost"] for j in panel_outcome.judgments)
    total_cost += panel_cost

    panel_says_challenger_wins = panel_outcome.winner == "a"
    cheap_full_agreement = panel_says_challenger_wins  # cheap said yes; agreement iff panel also says yes

    if panel_says_challenger_wins:
        notes.append("cheap_says_win:full_confirmed:promoted")
        return RolloutEvaluation(
            backprop_reward=cheap_outcome.reward,
            cheap_outcome=cheap_outcome,
            verified_partial=verified_partial,
            full_panel_outcome=panel_outcome,
            promoted=True,
            cheap_full_agreement=True,
            total_cost=total_cost,
            notes=notes,
        )

    # Cheap said win, full panel rejected. Backprop the configured neutral
    # value, not the cheap judge's original win-score.
    notes.append(
        f"cheap_says_win:full_rejected:backprop={rejection_backprop}:not_promoted"
    )
    return RolloutEvaluation(
        backprop_reward=rejection_backprop,
        cheap_outcome=cheap_outcome,
        verified_partial=verified_partial,
        full_panel_outcome=panel_outcome,
        promoted=False,
        cheap_full_agreement=False,
        total_cost=total_cost,
        notes=notes,
    )


# ----- Standalone CLI -----

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a cheap-judge dual-ordering comparison between two Stage 2 "
            "DAG fixtures. Makes 2 real LLM calls (~$0.005-0.02 total). "
            "Prints the resolved verdict and reward in [0, 1]."
        ),
    )
    parser.add_argument("challenger_path", type=Path,
                        help="Path to challenger DAG JSON")
    parser.add_argument("champion_path", type=Path,
                        help="Path to champion DAG JSON")
    parser.add_argument("--concept", type=Path, required=True,
                        help="Path to a Stage 1 concept JSON (the genome both "
                             "DAGs realize). Format: same shape as Stage 1's "
                             "ConceptGenome (premise, anchor_scene, ...).")
    parser.add_argument("--judge", required=True,
                        help="Judge id (filename without .yaml). e.g. 'gwern'.")
    parser.add_argument("--judges-dir", default="configs/judges",
                        help="Dir containing judge YAML configs")
    args = parser.parse_args(argv)

    for path in (args.challenger_path, args.champion_path, args.concept):
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 1

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        challenger = DAG.model_validate_json(args.challenger_path.read_text())
        champion = DAG.model_validate_json(args.champion_path.read_text())
        concept = ConceptGenome.model_validate(json.loads(args.concept.read_text()))
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"error parsing inputs: {e}", file=sys.stderr)
        return 1

    panel = load_panel(args.judges_dir, [args.judge])
    if not panel:
        print(f"error: judge {args.judge!r} not found in {args.judges_dir}", file=sys.stderr)
        return 1
    cheap_judge = panel[0]

    inputs = CompareInputs(challenger=challenger, champion=champion, concept=concept)
    outcome = asyncio.run(cheap_judge_compare(inputs, cheap_judge=cheap_judge))

    print(f"reward (challenger perspective):  {outcome.reward:.3f}")
    print(f"challenger declared winner:       {outcome.challenger_wins}")
    print(f"total cost:                       ${outcome.cost:.4f}")
    print("resolved per-dim votes:")
    for dim, vote in outcome.resolved_votes.items():
        print(f"  {dim:32s}  {vote}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
