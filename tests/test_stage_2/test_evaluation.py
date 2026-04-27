"""Stage 2 evaluation tests. All judges mocked.

Phase 4 wires up tiered judging before MCTS exists. Tests verify the *plumbing*
end-to-end: dual-ordering collapse, magnitude resolution, panel aggregation,
cheap-judge → full-panel orchestration, and the rejection-backprop rule.

What this file does NOT test:
- LLM output content. Judges return canned `Stage2PairwiseJudgment` instances
  via mocked `query_async`.
- Real prompt assembly. Stage 2's prompt registry is unit-tested in
  `test_operators.py` for seed_root; the same patterns apply here. The
  mocks intercept `query_async`, so prompt building isn't on the test path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from owtn.evaluation.models import STAGE_2_DIMENSION_NAMES, Stage2PairwiseJudgment
from owtn.evaluation.stage_2 import (
    CompareInputs,
    DEFAULT_DIM_WEIGHTS,
    _aggregate_stage2,
    _flip_stage2_votes,
    _resolve_stage2_votes,
    cheap_judge_compare,
    compare_stage2,
    evaluate_rollout,
    reward_from_resolved,
)
from owtn.models.judge import JudgePersona
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG
from tests.conftest import HILLS_GENOME


# ----- Fakes -----

@dataclass
class _FakeQueryResult:
    """Mirrors `_FakeQueryResult` in test_operators.py. With output_model set,
    `result.content` is the parsed Pydantic instance, not a JSON string."""
    content: object
    cost: float = 0.005
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    thought: str = ""
    model_name: str = "claude-sonnet-4-6"


def _make_judge(judge_id: str = "gwern", harshness: str = "standard") -> JudgePersona:
    """Build a minimal valid JudgePersona for tests. We don't need real
    persona content because prompts go through mocked query_async."""
    return JudgePersona(
        id=judge_id,
        name=f"Judge {judge_id.capitalize()}",
        identity="A judge",
        values=["values matter"],
        exemplars=["an exemplar"],
        lean_in_signals=["a signal"],
        harshness=harshness,
        priority="primary",
        model=["claude-sonnet-4-6"],
        temperature=0.0,
    )


def _make_judgment(
    *,
    edge_logic: str = "tie",
    motivational_coherence: str = "tie",
    tension_information_arch: str = "tie",
    post_dictability: str = "tie",
    arc_integrity_ending: str = "tie",
    structural_coherence: str = "tie",
    beat_quality: str = "tie",
    concept_fidelity_thematic: str = "tie",
    reasoning: str = "test reasoning",
) -> Stage2PairwiseJudgment:
    """Build a Stage2PairwiseJudgment with explicit per-dim votes.

    Default is all-tie; override individual dims by name. Vote strings must
    be in the canonical 7-value Vote literal."""
    return Stage2PairwiseJudgment(
        reasoning=reasoning,
        edge_logic=edge_logic,
        motivational_coherence=motivational_coherence,
        tension_information_arch=tension_information_arch,
        post_dictability=post_dictability,
        arc_integrity_ending=arc_integrity_ending,
        structural_coherence=structural_coherence,
        beat_quality=beat_quality,
        concept_fidelity_thematic=concept_fidelity_thematic,
    )


# ----- Fixtures -----

@pytest.fixture
def hills_concept() -> ConceptGenome:
    return ConceptGenome.model_validate(HILLS_GENOME)


@pytest.fixture
def compare_inputs(
    canonical_lottery: DAG,
    canonical_hemingway: DAG,
    hills_concept: ConceptGenome,
) -> CompareInputs:
    """Compare two real canonicals with a real concept. Mocked judges decide
    the actual votes, but we use real DAGs so prompt rendering can run."""
    return CompareInputs(
        challenger=canonical_lottery,
        champion=canonical_hemingway,
        concept=hills_concept,
    )


# ----- Vote handling -----

class TestFlipVotes:
    def test_a_flips_to_b(self) -> None:
        j = _make_judgment(edge_logic="a_decisive", beat_quality="a_narrow")
        flipped = _flip_stage2_votes(j)
        assert flipped["edge_logic"] == "b_decisive"
        assert flipped["beat_quality"] == "b_narrow"

    def test_b_flips_to_a(self) -> None:
        j = _make_judgment(post_dictability="b_clear")
        flipped = _flip_stage2_votes(j)
        assert flipped["post_dictability"] == "a_clear"

    def test_tie_stays_tie(self) -> None:
        j = _make_judgment(structural_coherence="tie")
        flipped = _flip_stage2_votes(j)
        assert flipped["structural_coherence"] == "tie"

    def test_magnitude_preserved(self) -> None:
        j = _make_judgment(arc_integrity_ending="a_decisive")
        flipped = _flip_stage2_votes(j)
        assert flipped["arc_integrity_ending"] == "b_decisive"  # not b_narrow


class TestResolveVotes:
    def test_both_orderings_agree_resolves_to_min_magnitude(self) -> None:
        # Forward says a_decisive, reverse-flipped also says a (judge waffled
        # to a_clear in the reverse). Min magnitude wins.
        forward = {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES}
        reverse = {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES}
        forward["edge_logic"] = "a_decisive"
        reverse["edge_logic"] = "a_clear"
        resolved = _resolve_stage2_votes(forward, reverse)
        assert resolved["edge_logic"] == "a_clear"  # min(decisive, clear) = clear

    def test_disagreement_collapses_to_tie(self) -> None:
        # Position bias: forward says A wins, reverse-flipped says B wins.
        # Collapse to tie — that's the dual-ordering catch.
        forward = {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES}
        reverse = {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES}
        forward["beat_quality"] = "a_decisive"
        reverse["beat_quality"] = "b_decisive"
        resolved = _resolve_stage2_votes(forward, reverse)
        assert resolved["beat_quality"] == "tie"

    def test_one_ordering_tie_collapses_to_tie(self) -> None:
        forward = {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES}
        reverse = {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES}
        forward["motivational_coherence"] = "a_clear"
        # reverse stays tie
        resolved = _resolve_stage2_votes(forward, reverse)
        assert resolved["motivational_coherence"] == "tie"


class TestAggregate:
    def test_majority_picks_winner(self) -> None:
        # 3 judges: 2 vote a, 1 votes b on edge_logic.
        all_resolved = [
            {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES},
            {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES},
            {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES},
        ]
        all_resolved[0]["edge_logic"] = "a_clear"
        all_resolved[1]["edge_logic"] = "a_clear"
        all_resolved[2]["edge_logic"] = "b_clear"
        dim_winners, a_wins, b_wins, ties, a_w, b_w, tie_w = _aggregate_stage2(
            all_resolved, DEFAULT_DIM_WEIGHTS,
        )
        assert dim_winners["edge_logic"] == "a"
        assert a_wins == 1  # only edge_logic is non-tie

    def test_tied_count_with_magnitude_breaks_to_higher_mag(self) -> None:
        # 2-2 split with A votes decisive+decisive (mean 1.0), B votes
        # narrow+narrow (mean 0.5). Gap 0.5 > threshold 0.25, so A wins.
        all_resolved = [
            {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES} for _ in range(4)
        ]
        all_resolved[0]["beat_quality"] = "a_decisive"
        all_resolved[1]["beat_quality"] = "a_decisive"
        all_resolved[2]["beat_quality"] = "b_narrow"
        all_resolved[3]["beat_quality"] = "b_narrow"
        dim_winners, *_ = _aggregate_stage2(all_resolved, DEFAULT_DIM_WEIGHTS)
        assert dim_winners["beat_quality"] == "a"

    def test_tied_count_same_magnitude_stays_tie(self) -> None:
        # 2-2 with a_clear + a_clear vs. b_clear + b_clear → still tie.
        all_resolved = [
            {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES} for _ in range(4)
        ]
        all_resolved[0]["beat_quality"] = "a_clear"
        all_resolved[1]["beat_quality"] = "a_clear"
        all_resolved[2]["beat_quality"] = "b_clear"
        all_resolved[3]["beat_quality"] = "b_clear"
        dim_winners, *_ = _aggregate_stage2(all_resolved, DEFAULT_DIM_WEIGHTS)
        assert dim_winners["beat_quality"] == "tie"


class TestRewardFromResolved:
    def test_all_a_gives_one(self) -> None:
        resolved = {dim: "a_clear" for dim in STAGE_2_DIMENSION_NAMES}
        assert reward_from_resolved(resolved) == 1.0

    def test_all_b_gives_zero(self) -> None:
        resolved = {dim: "b_clear" for dim in STAGE_2_DIMENSION_NAMES}
        assert reward_from_resolved(resolved) == 0.0

    def test_all_tie_gives_half(self) -> None:
        resolved = {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES}
        assert reward_from_resolved(resolved) == 0.5

    def test_4_a_4_tie_gives_three_quarters(self) -> None:
        resolved = {dim: "tie" for dim in STAGE_2_DIMENSION_NAMES}
        # 4 dim wins for A, 4 ties → (4 + 0.5*4) / 8 = 6/8 = 0.75
        for dim in STAGE_2_DIMENSION_NAMES[:4]:
            resolved[dim] = "a_clear"
        assert reward_from_resolved(resolved) == 0.75


# ----- End-to-end with mocked judges -----

class TestCompareStage2EndToEnd:
    """End-to-end: render two real DAGs, dispatch a 2-judge mocked panel,
    aggregate, return PairwiseResult."""

    def test_panel_runs_and_aggregates(
        self, compare_inputs: CompareInputs,
    ) -> None:
        # Two judges, both vote A wins decisively across all dims in both
        # orderings (they're consistent, so resolution preserves the verdict).
        a_winning_judgment = _make_judgment(
            edge_logic="a_decisive",
            motivational_coherence="a_decisive",
            tension_information_arch="a_decisive",
            post_dictability="a_decisive",
            arc_integrity_ending="a_decisive",
            structural_coherence="a_decisive",
            beat_quality="a_decisive",
            concept_fidelity_thematic="a_decisive",
        )
        # In the reverse ordering the judge sees A and B swapped, so the
        # *flipped* judgment must be A-winning. The judge call returns a
        # judgment as-presented; if presented (B, A) and judge says B wins,
        # the flip in our code produces A wins. So the reverse-call judgment
        # should literally say "B wins" to flip to "A wins".
        b_winning_judgment = _make_judgment(
            edge_logic="b_decisive",
            motivational_coherence="b_decisive",
            tension_information_arch="b_decisive",
            post_dictability="b_decisive",
            arc_integrity_ending="b_decisive",
            structural_coherence="b_decisive",
            beat_quality="b_decisive",
            concept_fidelity_thematic="b_decisive",
        )

        # `compare_stage2` dispatches all forward tasks first, then all
        # reverse tasks. With panel size N: calls 1..N are forward (return
        # a-winning); calls N+1..2N are reverse (return b-winning, which
        # flips to a-winning during resolution).
        panel = [_make_judge("gwern"), _make_judge("roon")]
        n = len(panel)
        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            judgment = a_winning_judgment if call_count[0] <= n else b_winning_judgment
            return _FakeQueryResult(content=judgment)

        with patch("owtn.evaluation.stage_2.query_async", side_effect=fake_query):
            result = asyncio.run(compare_stage2(compare_inputs, panel=panel))

        assert call_count[0] == 4  # 2 judges × 2 orderings
        assert result.winner == "a"  # challenger wins
        assert result.a_wins == 8
        assert result.b_wins == 0
        for dim in STAGE_2_DIMENSION_NAMES:
            assert result.dimension_wins[dim] == "a"
        # A weighted score: 8 dims × weight 1.0 × mean magnitude 1.0 = 8.0
        assert result.a_weighted == 8.0

    def test_judge_failure_skipped_not_fatal(
        self, compare_inputs: CompareInputs,
    ) -> None:
        """One judge raising mid-call is logged and skipped; remaining judges
        still aggregate. Important so a single provider hiccup doesn't kill
        a whole MCTS rollout."""
        a_winning = _make_judgment(
            edge_logic="a_clear",
            motivational_coherence="a_clear",
            tension_information_arch="a_clear",
            post_dictability="a_clear",
            arc_integrity_ending="a_clear",
            structural_coherence="a_clear",
            beat_quality="a_clear",
            concept_fidelity_thematic="a_clear",
        )
        b_winning = _make_judgment(
            edge_logic="b_clear",
            motivational_coherence="b_clear",
            tension_information_arch="b_clear",
            post_dictability="b_clear",
            arc_integrity_ending="b_clear",
            structural_coherence="b_clear",
            beat_quality="b_clear",
            concept_fidelity_thematic="b_clear",
        )

        # Judge 0: succeeds both orderings (votes A wins).
        # Judge 1: forward succeeds, reverse raises.
        # Judge 0's two calls are scheduled before Judge 1's; gather order
        # matches dispatch order (deterministic with our explicit gather).
        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            n = call_count[0]
            # Pattern: j0_fwd=1, j0_rev=2, j1_fwd=3, j1_rev=4 — reverse calls
            # come AFTER all forward calls because compare_stage2 gathers
            # forward_tasks first, then reverse_tasks.
            # Forward: judges 0, 1 → calls 1, 2.
            # Reverse: judges 0, 1 → calls 3, 4.
            if n == 4:  # judge 1 reverse
                raise RuntimeError("simulated provider hiccup")
            if n in (1, 2):  # forward calls
                return _FakeQueryResult(content=a_winning)
            else:  # reverse calls (n==3 only)
                return _FakeQueryResult(content=b_winning)

        panel = [_make_judge("gwern"), _make_judge("roon")]
        with patch("owtn.evaluation.stage_2.query_async", side_effect=fake_query):
            result = asyncio.run(compare_stage2(compare_inputs, panel=panel))

        # Only judge 0 contributed (judge 1's reverse call failed).
        assert result.winner == "a"
        assert len(result.judgments) == 1

    def test_all_judges_failing_returns_champion_holds(
        self, compare_inputs: CompareInputs,
    ) -> None:
        """Total panel failure is non-fatal — champion holds, all dims tie."""
        async def failing_query(**kwargs):
            raise RuntimeError("everything is on fire")

        panel = [_make_judge("gwern"), _make_judge("roon")]
        with patch("owtn.evaluation.stage_2.query_async", side_effect=failing_query):
            result = asyncio.run(compare_stage2(compare_inputs, panel=panel))

        assert result.winner == "b"  # champion holds
        for dim in STAGE_2_DIMENSION_NAMES:
            assert result.dimension_wins[dim] == "tie"


class TestCheapJudgeCompare:
    def test_returns_reward_in_unit_interval(
        self, compare_inputs: CompareInputs,
    ) -> None:
        # Symmetric setup: forward says all A wins; reverse says all B wins.
        # Resolution should yield all A (after flip on reverse).
        a_winning = _make_judgment(
            edge_logic="a_clear",
            motivational_coherence="a_clear",
            tension_information_arch="a_clear",
            post_dictability="a_clear",
            arc_integrity_ending="a_clear",
            structural_coherence="a_clear",
            beat_quality="a_clear",
            concept_fidelity_thematic="a_clear",
        )
        b_winning = _make_judgment(
            edge_logic="b_clear",
            motivational_coherence="b_clear",
            tension_information_arch="b_clear",
            post_dictability="b_clear",
            arc_integrity_ending="b_clear",
            structural_coherence="b_clear",
            beat_quality="b_clear",
            concept_fidelity_thematic="b_clear",
        )

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(
                content=a_winning if call_count[0] == 1 else b_winning
            )

        with patch("owtn.evaluation.stage_2.query_async", side_effect=fake_query):
            outcome = asyncio.run(cheap_judge_compare(
                compare_inputs, cheap_judge=_make_judge("cheap"),
            ))

        assert outcome.reward == 1.0  # all 8 dims went to A after dual-ordering
        assert outcome.challenger_wins is True
        assert outcome.cost > 0

    def test_failure_returns_neutral_reward(
        self, compare_inputs: CompareInputs,
    ) -> None:
        async def failing_query(**kwargs):
            raise RuntimeError("provider down")

        with patch("owtn.evaluation.stage_2.query_async", side_effect=failing_query):
            outcome = asyncio.run(cheap_judge_compare(
                compare_inputs, cheap_judge=_make_judge("cheap"),
            ))

        assert outcome.reward == 0.5
        assert outcome.challenger_wins is False
        assert outcome.cost == 0.0


# ----- Tiered orchestration: cheap → full panel rejection backprop -----

class TestEvaluateRolloutTiering:
    """The exit criterion: full_panel_rejection_backprop=0.5 path is unit-tested."""

    def test_cheap_says_no_win_no_full_panel_call(
        self, compare_inputs: CompareInputs,
    ) -> None:
        # Cheap judge votes all-tie → reward 0.5 → not a win → full panel skipped.
        all_tie = _make_judgment()  # all defaults are tie

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=all_tie)

        cheap = _make_judge("cheap")
        full_panel = [_make_judge("g"), _make_judge("r")]

        call_count = [0]

        async def counting_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=all_tie)

        with patch("owtn.evaluation.stage_2.query_async", side_effect=counting_query):
            outcome = asyncio.run(evaluate_rollout(
                compare_inputs, cheap_judge=cheap, full_panel=full_panel,
            ))

        assert outcome.promoted is False
        assert outcome.backprop_reward == 0.5
        assert outcome.full_panel_outcome is None
        assert outcome.cheap_full_agreement is None
        assert call_count[0] == 2  # only cheap judge × 2 orderings

    def test_cheap_says_win_full_panel_confirms_promotion(
        self, compare_inputs: CompareInputs,
    ) -> None:
        a_clear = _make_judgment(
            edge_logic="a_clear",
            motivational_coherence="a_clear",
            tension_information_arch="a_clear",
            post_dictability="a_clear",
            arc_integrity_ending="a_clear",
            structural_coherence="a_clear",
            beat_quality="a_clear",
            concept_fidelity_thematic="a_clear",
        )
        b_clear = _make_judgment(
            edge_logic="b_clear",
            motivational_coherence="b_clear",
            tension_information_arch="b_clear",
            post_dictability="b_clear",
            arc_integrity_ending="b_clear",
            structural_coherence="b_clear",
            beat_quality="b_clear",
            concept_fidelity_thematic="b_clear",
        )
        # Cheap judge: 2 calls (fwd=1, rev=2). Full panel (2 judges): 4 calls
        # (fwds=3,4 then revs=5,6). Within each block, forward=a_clear and
        # reverse=b_clear so the flip resolves to A wins consistently.
        cheap = _make_judge("cheap")
        full_panel = [_make_judge("g"), _make_judge("r")]
        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            n = call_count[0]
            # Cheap: call 1 forward, call 2 reverse.
            if n == 1:
                return _FakeQueryResult(content=a_clear)
            if n == 2:
                return _FakeQueryResult(content=b_clear)
            # Full panel forward (calls 3, 4): a_clear.
            # Full panel reverse (calls 5, 6): b_clear.
            if n in (3, 4):
                return _FakeQueryResult(content=a_clear)
            return _FakeQueryResult(content=b_clear)

        with patch("owtn.evaluation.stage_2.query_async", side_effect=fake_query):
            outcome = asyncio.run(evaluate_rollout(
                compare_inputs, cheap_judge=cheap, full_panel=full_panel,
            ))

        assert outcome.promoted is True
        assert outcome.cheap_full_agreement is True
        assert outcome.full_panel_outcome is not None
        assert outcome.full_panel_outcome.winner == "a"
        # Backprop is the cheap-judge's score, not 0.5.
        assert outcome.backprop_reward == 1.0  # all dims went to A

    def test_cheap_says_win_full_panel_rejects_backprop_neutral(
        self, compare_inputs: CompareInputs,
    ) -> None:
        """The exit-criterion test: cheap declares a win, full panel rejects;
        backprop is the configured neutral value (0.5), not the cheap score."""
        a_clear = _make_judgment(
            edge_logic="a_clear",
            motivational_coherence="a_clear",
            tension_information_arch="a_clear",
            post_dictability="a_clear",
            arc_integrity_ending="a_clear",
            structural_coherence="a_clear",
            beat_quality="a_clear",
            concept_fidelity_thematic="a_clear",
        )
        b_clear = _make_judgment(
            edge_logic="b_clear",
            motivational_coherence="b_clear",
            tension_information_arch="b_clear",
            post_dictability="b_clear",
            arc_integrity_ending="b_clear",
            structural_coherence="b_clear",
            beat_quality="b_clear",
            concept_fidelity_thematic="b_clear",
        )

        # Tracking which calls go to cheap vs. full panel:
        # - cheap forward (call 1): a_clear
        # - cheap reverse (call 2): b_clear → resolves to A wins → cheap says win
        # - full panel forward calls (3, 4 — for 2 judges): b_clear (B = champion = wins)
        # - full panel reverse calls (5, 6): a_clear (B wins after flip)
        # So full panel votes B wins; rejects the cheap judge's promotion.
        cheap = _make_judge("cheap")
        full_panel = [_make_judge("g"), _make_judge("r")]

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            n = call_count[0]
            if n == 1:
                return _FakeQueryResult(content=a_clear)
            elif n == 2:
                return _FakeQueryResult(content=b_clear)
            # Full-panel calls 3,4 are forward; full-panel calls 5,6 are reverse.
            elif n in (3, 4):
                return _FakeQueryResult(content=b_clear)
            else:  # 5, 6
                return _FakeQueryResult(content=a_clear)

        with patch("owtn.evaluation.stage_2.query_async", side_effect=fake_query):
            outcome = asyncio.run(evaluate_rollout(
                compare_inputs,
                cheap_judge=cheap,
                full_panel=full_panel,
                rejection_backprop=0.5,  # explicit for clarity
            ))

        assert outcome.cheap_outcome.challenger_wins is True
        assert outcome.full_panel_outcome is not None
        assert outcome.full_panel_outcome.winner == "b"  # full says champion wins
        assert outcome.cheap_full_agreement is False
        assert outcome.promoted is False
        # Backprop is the configured rejection value, NOT the cheap score.
        assert outcome.backprop_reward == 0.5
        assert any("full_rejected" in note for note in outcome.notes)

    def test_custom_rejection_backprop_value_honored(
        self, compare_inputs: CompareInputs,
    ) -> None:
        """rejection_backprop is configurable per the design (default 0.5,
        per docs/stage-2/implementation.md). Use 0.3 here to confirm the
        configured value flows through."""
        a_clear = _make_judgment(
            edge_logic="a_clear",
            motivational_coherence="a_clear",
            tension_information_arch="a_clear",
            post_dictability="a_clear",
            arc_integrity_ending="a_clear",
            structural_coherence="a_clear",
            beat_quality="a_clear",
            concept_fidelity_thematic="a_clear",
        )
        b_clear = _make_judgment(
            edge_logic="b_clear",
            motivational_coherence="b_clear",
            tension_information_arch="b_clear",
            post_dictability="b_clear",
            arc_integrity_ending="b_clear",
            structural_coherence="b_clear",
            beat_quality="b_clear",
            concept_fidelity_thematic="b_clear",
        )

        cheap = _make_judge("cheap")
        full_panel = [_make_judge("g")]

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            n = call_count[0]
            # cheap fwd=1 → a_clear; cheap rev=2 → b_clear → cheap says A wins.
            # full fwd=3 → b_clear; full rev=4 → a_clear → full says B wins.
            if n == 1 or n == 4:
                return _FakeQueryResult(content=a_clear)
            return _FakeQueryResult(content=b_clear)

        with patch("owtn.evaluation.stage_2.query_async", side_effect=fake_query):
            outcome = asyncio.run(evaluate_rollout(
                compare_inputs,
                cheap_judge=cheap,
                full_panel=full_panel,
                rejection_backprop=0.3,
            ))

        assert outcome.backprop_reward == 0.3

    def test_no_full_panel_promotes_on_cheap_win_alone(
        self, compare_inputs: CompareInputs,
    ) -> None:
        a_clear = _make_judgment(
            edge_logic="a_clear",
            motivational_coherence="a_clear",
            tension_information_arch="a_clear",
            post_dictability="a_clear",
            arc_integrity_ending="a_clear",
            structural_coherence="a_clear",
            beat_quality="a_clear",
            concept_fidelity_thematic="a_clear",
        )
        b_clear = _make_judgment(
            edge_logic="b_clear",
            motivational_coherence="b_clear",
            tension_information_arch="b_clear",
            post_dictability="b_clear",
            arc_integrity_ending="b_clear",
            structural_coherence="b_clear",
            beat_quality="b_clear",
            concept_fidelity_thematic="b_clear",
        )

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=a_clear if call_count[0] == 1 else b_clear)

        with patch("owtn.evaluation.stage_2.query_async", side_effect=fake_query):
            outcome = asyncio.run(evaluate_rollout(
                compare_inputs, cheap_judge=_make_judge("cheap"), full_panel=None,
            ))

        assert outcome.promoted is True
        assert outcome.full_panel_outcome is None
        assert outcome.cheap_full_agreement is None  # no full panel ran
        assert outcome.backprop_reward == 1.0


class TestAgreementLogging:
    """Per Phase 4 exit criterion: cheap-vs-full agreement logging is in place
    so cheap-judge drift can be tracked from run 1."""

    def test_promotion_event_records_agreement_flag(
        self, compare_inputs: CompareInputs,
    ) -> None:
        a_clear = _make_judgment(
            edge_logic="a_clear",
            motivational_coherence="a_clear",
            tension_information_arch="a_clear",
            post_dictability="a_clear",
            arc_integrity_ending="a_clear",
            structural_coherence="a_clear",
            beat_quality="a_clear",
            concept_fidelity_thematic="a_clear",
        )
        b_clear = _make_judgment(
            edge_logic="b_clear",
            motivational_coherence="b_clear",
            tension_information_arch="b_clear",
            post_dictability="b_clear",
            arc_integrity_ending="b_clear",
            structural_coherence="b_clear",
            beat_quality="b_clear",
            concept_fidelity_thematic="b_clear",
        )

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=a_clear if call_count[0] % 2 == 1 else b_clear)

        with patch("owtn.evaluation.stage_2.query_async", side_effect=fake_query):
            outcome = asyncio.run(evaluate_rollout(
                compare_inputs,
                cheap_judge=_make_judge("cheap"),
                full_panel=[_make_judge("g")],
            ))

        # Promotion confirmed → agreement flag is True (not None).
        assert outcome.cheap_full_agreement is True
        # Notes string carries the resolution path explicitly.
        assert any("full_confirmed" in n for n in outcome.notes)
