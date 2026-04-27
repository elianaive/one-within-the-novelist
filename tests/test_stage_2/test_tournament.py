"""Stage 2 within-concept tournament tests. Mocked judges.

Phase 8 exit criterion covered:
- 4-preset within-concept tournament produces a complete ranking
  (`TestRoundRobinRanking`).

Plus tiebreaker chain coverage and per-match accumulation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from owtn.evaluation.models import STAGE_2_DIMENSION_NAMES, Stage2PairwiseJudgment
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG
from owtn.stage_2.tournament import (
    TournamentEntry,
    TournamentMatch,
    run_within_concept_tournament,
)
from tests.conftest import HILLS_GENOME


# ----- Test scaffolding -----


@dataclass
class _FakeQueryResult:
    content: object
    cost: float = 0.005
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    thought: str = ""
    model_name: str = "claude-sonnet-4-6"


def _make_judge(judge_id: str = "gwern") -> JudgePersona:
    return JudgePersona(
        id=judge_id,
        name=f"Judge {judge_id}",
        identity="A test judge",
        values=["v"],
        exemplars=["e"],
        lean_in_signals=["s"],
        harshness="standard",
        priority="primary",
        model=["claude-sonnet-4-6"],
        temperature=0.0,
    )


def _make_entry(
    preset: str, dag: DAG, mcts_reward: float = 0.5,
) -> TournamentEntry:
    return TournamentEntry(preset=preset, dag=dag, mcts_reward=mcts_reward)


def _all_a_judgment(reasoning: str = "test") -> Stage2PairwiseJudgment:
    return Stage2PairwiseJudgment(
        reasoning=reasoning,
        edge_logic="a_clear",
        motivational_coherence="a_clear",
        tension_information_arch="a_clear",
        post_dictability="a_clear",
        arc_integrity_ending="a_clear",
        structural_coherence="a_clear",
        beat_quality="a_clear",
        concept_fidelity_thematic="a_clear",
    )


def _all_b_judgment(reasoning: str = "test") -> Stage2PairwiseJudgment:
    return Stage2PairwiseJudgment(
        reasoning=reasoning,
        edge_logic="b_clear",
        motivational_coherence="b_clear",
        tension_information_arch="b_clear",
        post_dictability="b_clear",
        arc_integrity_ending="b_clear",
        structural_coherence="b_clear",
        beat_quality="b_clear",
        concept_fidelity_thematic="b_clear",
    )


def _all_tie_judgment(reasoning: str = "test") -> Stage2PairwiseJudgment:
    return Stage2PairwiseJudgment(
        reasoning=reasoning,
        edge_logic="tie", motivational_coherence="tie",
        tension_information_arch="tie", post_dictability="tie",
        arc_integrity_ending="tie", structural_coherence="tie",
        beat_quality="tie", concept_fidelity_thematic="tie",
    )


@pytest.fixture
def hills_concept() -> ConceptGenome:
    return ConceptGenome.model_validate(HILLS_GENOME)


# ----- Round-robin produces a complete ranking -----


class TestRoundRobinRanking:
    """Phase 8 exit: 4-preset tournament produces a complete ranking."""

    def test_unanimous_dominator_wins_all_matches(
        self,
        hills_concept: ConceptGenome,
        canonical_lottery: DAG,
        canonical_hemingway: DAG,
        canonical_chiang: DAG,
        canonical_oconnor: DAG,
    ) -> None:
        """One entry's DAG is always treated as the winning side. After the
        round-robin, that entry has 3 wins / 0 losses; the others all have
        1 win / 2 losses (each beats the others when paired together).

        Setup: in every match, regardless of side, the FIRST DAG in the
        pair becomes the winner. Tournament dispatches pairs in order
        (i, j) for i<j, so entry 0 always wins, entry 1 wins against 2,3,
        entry 2 wins against 3, entry 3 has 0 wins.
        """
        # All forward calls return a-winning, all reverse return b-winning →
        # resolves to A wins for whichever entry is in the A slot.
        # Each judge × ordering call = 1 LLM call.

        panel = [_make_judge("g"), _make_judge("r")]
        n_panel = len(panel)

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            # Per-match: forward calls are 1..n_panel, reverse calls are
            # n_panel+1..2*n_panel (compare_stage2 dispatches forward batch
            # first, then reverse batch). Cycle every 2*n_panel calls.
            n = call_count[0]
            in_match = (n - 1) % (2 * n_panel) + 1
            judgment = (
                _all_a_judgment() if in_match <= n_panel else _all_b_judgment()
            )
            return _FakeQueryResult(content=judgment)

        entries = [
            _make_entry("cassandra_ish", canonical_lottery, mcts_reward=0.7),
            _make_entry("phoebe_ish", canonical_hemingway, mcts_reward=0.6),
            _make_entry("randy_ish", canonical_chiang, mcts_reward=0.5),
            _make_entry("winston_ish", canonical_oconnor, mcts_reward=0.4),
        ]

        with patch("owtn.evaluation.stage_2.query_async", side_effect=fake_query):
            ranked = asyncio.run(run_within_concept_tournament(
                entries, concept=hills_concept, panel=panel,
            ))

        assert len(ranked) == 4
        # Ranked best-to-worst.
        assert ranked[0].preset == "cassandra_ish"
        assert ranked[0].wins == 3
        assert ranked[0].losses == 0
        assert ranked[3].preset == "winston_ish"
        assert ranked[3].wins == 0
        assert ranked[3].losses == 3

        # Each entry has exactly 3 matches recorded.
        for entry in ranked:
            assert len(entry.matches) == 3

    def test_tournament_with_two_entries_runs_one_match(
        self,
        hills_concept: ConceptGenome,
        canonical_lottery: DAG,
        canonical_hemingway: DAG,
    ) -> None:
        panel = [_make_judge("g")]
        n_panel = 1

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            judgment = (
                _all_a_judgment() if call_count[0] <= n_panel else _all_b_judgment()
            )
            return _FakeQueryResult(content=judgment)

        entries = [
            _make_entry("cassandra_ish", canonical_lottery),
            _make_entry("phoebe_ish", canonical_hemingway),
        ]

        with patch("owtn.evaluation.stage_2.query_async", side_effect=fake_query):
            ranked = asyncio.run(run_within_concept_tournament(
                entries, concept=hills_concept, panel=panel,
            ))

        assert len(ranked) == 2
        assert ranked[0].wins == 1
        assert ranked[1].wins == 0

    def test_single_entry_raises(
        self,
        hills_concept: ConceptGenome,
        canonical_lottery: DAG,
    ) -> None:
        panel = [_make_judge("g")]
        with pytest.raises(ValueError, match="≥2 entries"):
            asyncio.run(run_within_concept_tournament(
                [_make_entry("cassandra_ish", canonical_lottery)],
                concept=hills_concept,
                panel=panel,
            ))


# ----- Tiebreaker chain -----


class TestTiebreakers:
    """Per `evaluation.md` §Ranking mechanism:
    1. Most DAG-level wins (overall match wins).
    2. Most dim-level wins across all matches.
    3. Higher mean reasoning length."""

    def test_dim_wins_total_breaks_match_win_ties(self) -> None:
        # Two entries, both with the same wins/losses count (1-1-0).
        # Higher dim_wins_total wins the tiebreaker.
        e1 = TournamentEntry(preset="a", dag=DAG.model_construct(), mcts_reward=0.5,
                             wins=1, losses=1, dim_wins_total=10, dim_losses_total=6)
        e2 = TournamentEntry(preset="b", dag=DAG.model_construct(), mcts_reward=0.5,
                             wins=1, losses=1, dim_wins_total=6, dim_losses_total=10)
        assert e1.sort_key() < e2.sort_key()  # e1 ranks first

    def test_mean_reasoning_length_breaks_dim_ties(self) -> None:
        # Same wins, same dim totals — falls through to mean reasoning length.
        long_match = TournamentMatch(
            opponent_preset="x", result="win",
            dimension_wins={d: "win" for d in STAGE_2_DIMENSION_NAMES},
            self_dim_wins=8, self_dim_losses=0, self_dim_ties=0,
            mean_reasoning_length=2000.0,
        )
        short_match = TournamentMatch(
            opponent_preset="x", result="win",
            dimension_wins={d: "win" for d in STAGE_2_DIMENSION_NAMES},
            self_dim_wins=8, self_dim_losses=0, self_dim_ties=0,
            mean_reasoning_length=500.0,
        )
        e1 = TournamentEntry(preset="a", dag=DAG.model_construct(), mcts_reward=0.5,
                             wins=1, losses=0, dim_wins_total=8, dim_losses_total=0)
        e1.matches.append(long_match)
        e2 = TournamentEntry(preset="b", dag=DAG.model_construct(), mcts_reward=0.5,
                             wins=1, losses=0, dim_wins_total=8, dim_losses_total=0)
        e2.matches.append(short_match)
        assert e1.sort_key() < e2.sort_key()  # e1 has higher mean reasoning length

    def test_mean_reasoning_length_zero_when_no_matches(self) -> None:
        e = TournamentEntry(preset="x", dag=DAG.model_construct(), mcts_reward=0.5)
        assert e.mean_reasoning_length == 0.0
