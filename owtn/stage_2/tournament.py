"""Within-concept tournament: round-robin ranking of preset winners.

Per `docs/stage-2/evaluation.md` §Within-Concept Tournament:
- After all 4 MCTS trees complete for one concept, the 4 preset-winner DAGs
  compete via round-robin (every pair compared once).
- Each match uses the full per-criterion pairwise protocol — 4 judges × 2
  orderings × 8 dimensions, via `owtn.evaluation.stage_2.compare_stage2`.
- Round-robin (not Swiss) is appropriate for 4 entries: only C(4,2)=6 matches,
  no need for the pairing heuristics that justify Swiss at larger pool sizes.

Ranking (tiebreaker chain per `evaluation.md`):
1. Most DAG-level wins (overall match wins).
2. Most dimension-level wins across all matches.
3. Higher mean judge-reasoning length (proxy for engagement).

Why a parallel module instead of reusing `owtn.evaluation.tournament`:
The Stage 1 tournament is Swiss-system, takes `ConceptGenome`, and calls
Stage 1's `pairwise.compare`. Stage 2 differs in pool size (4 fixed),
matching topology (round-robin), input type (DAG), match function
(`compare_stage2`), and tiebreaker chain. The conceptually shared bit is
just the entry dataclass shape — duplication is small enough that
parameterizing the Stage 1 module would be more disruptive than parallel
code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from owtn.evaluation.stage_2 import CompareInputs, compare_stage2
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG


logger = logging.getLogger(__name__)


@dataclass
class TournamentMatch:
    """One pairing's outcome."""
    opponent_preset: str
    result: str            # "win" / "loss" / "tie" from this entry's perspective
    dimension_wins: dict[str, str]  # {dim_name: "a"/"b"/"tie"} from this entry's perspective
    self_dim_wins: int     # count of dims this entry won
    self_dim_losses: int
    self_dim_ties: int
    mean_reasoning_length: float


@dataclass
class TournamentEntry:
    """One preset's place in the within-concept tournament.

    Fields after `mcts_reward` accumulate during `run_within_concept_tournament`;
    populated as matches resolve.
    """
    preset: str
    dag: DAG
    mcts_reward: float = 0.0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    dim_wins_total: int = 0
    dim_losses_total: int = 0
    dim_ties_total: int = 0
    matches: list[TournamentMatch] = field(default_factory=list)

    @property
    def mean_reasoning_length(self) -> float:
        """Mean of per-match mean-reasoning-lengths. Used as tiebreaker 3."""
        if not self.matches:
            return 0.0
        return sum(m.mean_reasoning_length for m in self.matches) / len(self.matches)

    def sort_key(self) -> tuple[int, int, float]:
        """Primary sort key. Higher is better; we negate so default ascending
        sort produces best-first ordering.

        Order:
        1. Most DAG-level wins (`-wins`)
        2. Most dimension-level wins across all matches (`-dim_wins_total`)
        3. Higher mean reasoning length (proxy for judge engagement)
        """
        return (-self.wins, -self.dim_wins_total, -self.mean_reasoning_length)


def _mean_reasoning_length(judgments: list[dict]) -> float:
    """Average length (in chars) of `forward_reasoning` + `reverse_reasoning`
    fields across the judgments. Returns 0.0 if no reasoning text is present."""
    lengths: list[int] = []
    for j in judgments:
        for key in ("forward_reasoning", "reverse_reasoning"):
            text = j.get(key) or ""
            lengths.append(len(text))
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


async def run_within_concept_tournament(
    entries: list[TournamentEntry],
    *,
    concept: ConceptGenome,
    panel: list[JudgePersona],
) -> list[TournamentEntry]:
    """Run round-robin among the entries; return ranked list (best first).

    Mutates each entry's `wins` / `losses` / `ties` / `dim_*` / `matches`
    fields. The list returned is a re-sorted copy of the input list with
    the same `TournamentEntry` instances.

    Raises ValueError if fewer than 2 entries — a tournament with one
    competitor is undefined.
    """
    n = len(entries)
    if n < 2:
        raise ValueError(
            f"within-concept tournament requires ≥2 entries; got {n}"
        )

    logger.info(
        "Stage 2 within-concept tournament: %d entries, %d matches",
        n, n * (n - 1) // 2,
    )

    for i in range(n):
        for j in range(i + 1, n):
            entry_a = entries[i]
            entry_b = entries[j]
            await _run_one_match(entry_a, entry_b, concept=concept, panel=panel)

    ranked = sorted(entries, key=lambda e: e.sort_key())
    for rank, entry in enumerate(ranked, 1):
        logger.info(
            "  #%d: %s (W%d-L%d-T%d, dim=%d/%d/%d)",
            rank, entry.preset, entry.wins, entry.losses, entry.ties,
            entry.dim_wins_total, entry.dim_losses_total, entry.dim_ties_total,
        )
    return ranked


async def _run_one_match(
    entry_a: TournamentEntry,
    entry_b: TournamentEntry,
    *,
    concept: ConceptGenome,
    panel: list[JudgePersona],
) -> None:
    """Run one comparison; update both entries' state based on the result.

    The tournament is symmetric: A is challenger arbitrarily, B is champion
    arbitrarily. `compare_stage2`'s "winner" field is "a" (= entry_a) or
    "b" (= entry_b); we translate to wins/losses on each entry.
    """
    inputs = CompareInputs(
        challenger=entry_a.dag, champion=entry_b.dag, concept=concept,
    )
    result = await compare_stage2(inputs, panel=panel)

    # Per-entry dim tallies (from THIS entry's perspective, not "a"/"b").
    a_dim_wins = result.a_wins
    b_dim_wins = result.b_wins
    dim_ties = result.ties
    mean_len = _mean_reasoning_length(result.judgments)

    # Translate result.dimension_wins into per-entry "win/loss/tie" dicts.
    a_dim_winners = {
        dim: ("win" if w == "a" else "loss" if w == "b" else "tie")
        for dim, w in result.dimension_wins.items()
    }
    b_dim_winners = {
        dim: ("win" if w == "b" else "loss" if w == "a" else "tie")
        for dim, w in result.dimension_wins.items()
    }

    # Match-level outcome.
    if result.winner == "a":
        a_match_result, b_match_result = "win", "loss"
        entry_a.wins += 1
        entry_b.losses += 1
    elif result.winner == "b":
        a_match_result, b_match_result = "loss", "win"
        entry_a.losses += 1
        entry_b.wins += 1
    else:  # pragma: no cover — compare_stage2 currently always picks a/b
        a_match_result = b_match_result = "tie"
        entry_a.ties += 1
        entry_b.ties += 1

    # Dim-level totals across all matches.
    entry_a.dim_wins_total += a_dim_wins
    entry_a.dim_losses_total += b_dim_wins
    entry_a.dim_ties_total += dim_ties
    entry_b.dim_wins_total += b_dim_wins
    entry_b.dim_losses_total += a_dim_wins
    entry_b.dim_ties_total += dim_ties

    entry_a.matches.append(TournamentMatch(
        opponent_preset=entry_b.preset,
        result=a_match_result,
        dimension_wins=a_dim_winners,
        self_dim_wins=a_dim_wins,
        self_dim_losses=b_dim_wins,
        self_dim_ties=dim_ties,
        mean_reasoning_length=mean_len,
    ))
    entry_b.matches.append(TournamentMatch(
        opponent_preset=entry_a.preset,
        result=b_match_result,
        dimension_wins=b_dim_winners,
        self_dim_wins=b_dim_wins,
        self_dim_losses=a_dim_wins,
        self_dim_ties=dim_ties,
        mean_reasoning_length=mean_len,
    ))

    logger.info(
        "  %s vs %s → %s wins (%d-%d-%d dim)",
        entry_a.preset, entry_b.preset,
        "TIE" if result.winner == "tie" else (
            entry_a.preset if result.winner == "a" else entry_b.preset
        ),
        a_dim_wins, b_dim_wins, dim_ties,
    )
