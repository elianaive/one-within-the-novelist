"""Swiss-system pairwise tournament for ranking concepts.

Run after evolution completes to produce a final ranking of island
champions (or any set of concepts). Each round pairs concepts with
similar records; the per-criteria pairwise protocol handles each match.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from owtn.evaluation.pairwise import compare as pairwise_compare
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_1.config import StageConfig

logger = logging.getLogger(__name__)


@dataclass
class TournamentEntry:
    """A concept's state within the tournament."""

    program_id: str
    genome: ConceptGenome
    wins: int = 0
    losses: int = 0
    # Buchholz score: sum of opponents' wins. Higher = faced tougher opponents.
    opponent_wins: list[int] = field(default_factory=list)
    match_history: list[dict] = field(default_factory=list)

    @property
    def buchholz(self) -> int:
        return sum(self.opponent_wins)

    def sort_key(self) -> tuple:
        """Sort by wins DESC, then Buchholz DESC for tiebreaking."""
        return (-self.wins, -self.buchholz)


def _pair_by_record(entries: list[TournamentEntry]) -> list[tuple[int, int]]:
    """Pair entries with similar records (standard Swiss pairing).

    Sort by wins descending, then pair adjacent entries. If odd number,
    the last entry gets a bye (automatic win).
    """
    sorted_indices = sorted(range(len(entries)), key=lambda i: -entries[i].wins)
    pairs = []
    used = set()

    for i in range(0, len(sorted_indices) - 1, 2):
        a = sorted_indices[i]
        b = sorted_indices[i + 1]
        pairs.append((a, b))
        used.add(a)
        used.add(b)

    # Bye for unpaired entry (odd count).
    for idx in sorted_indices:
        if idx not in used:
            entries[idx].wins += 1
            entries[idx].match_history.append({"opponent": "bye", "result": "win"})
            logger.info("Tournament: %s gets a bye", entries[idx].program_id[:8])

    return pairs


async def run_tournament(
    participants: list[tuple[str, ConceptGenome]],
    config: StageConfig,
) -> list[TournamentEntry]:
    """Run a Swiss-system tournament on a set of concepts.

    Args:
        participants: List of (program_id, genome) tuples.
        config: Stage config for judge panel loading.

    Returns:
        Entries sorted by final ranking (best first).
    """
    n = len(participants)
    if n < 2:
        return [TournamentEntry(program_id=pid, genome=g) for pid, g in participants]

    entries = [TournamentEntry(program_id=pid, genome=g) for pid, g in participants]
    num_rounds = max(1, math.ceil(math.log2(n)))

    logger.info("")
    logger.info("Swiss tournament: %d participants, %d rounds", n, num_rounds)

    for round_num in range(num_rounds):
        pairs = _pair_by_record(entries)
        logger.info("--- Round %d/%d (%d matches) ---", round_num + 1, num_rounds, len(pairs))

        for idx_a, idx_b in pairs:
            entry_a = entries[idx_a]
            entry_b = entries[idx_b]

            result = await pairwise_compare(
                genome_a=entry_a.genome,
                genome_b=entry_b.genome,
                config=config,
                champion_label="a",  # No incumbent advantage in tournament
            )

            if result.winner == "a":
                entry_a.wins += 1
                entry_b.losses += 1
                winner_id = entry_a.program_id
            else:
                entry_b.wins += 1
                entry_a.losses += 1
                winner_id = entry_b.program_id

            # Track opponent strength for Buchholz tiebreaking.
            entry_a.opponent_wins.append(entry_b.wins)
            entry_b.opponent_wins.append(entry_a.wins)

            entry_a.match_history.append({
                "opponent": entry_b.program_id,
                "result": "win" if result.winner == "a" else "loss",
                "dimension_wins": result.dimension_wins,
                "score": f"{result.a_wins}-{result.b_wins}-{result.ties}",
            })
            entry_b.match_history.append({
                "opponent": entry_a.program_id,
                "result": "win" if result.winner == "b" else "loss",
                "dimension_wins": result.dimension_wins,
                "score": f"{result.b_wins}-{result.a_wins}-{result.ties}",
            })

            logger.info(
                "  %s vs %s → %s wins (%d-%d-%d)",
                entry_a.program_id[:8], entry_b.program_id[:8],
                winner_id[:8], result.a_wins, result.b_wins, result.ties,
            )

    # Final ranking.
    entries.sort(key=lambda e: e.sort_key())
    for rank, entry in enumerate(entries, 1):
        logger.info(
            "  #%d: %s (W%d-L%d, Buchholz=%d)",
            rank, entry.program_id[:8], entry.wins, entry.losses, entry.buchholz,
        )

    return entries
