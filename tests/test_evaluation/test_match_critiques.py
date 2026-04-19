"""Tests for MatchCritique construction in pairwise compare().

See `lab/issues/2026-04-18-lazy-feedback-summarizer.md` — Phase 1.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from owtn.evaluation import pairwise
from owtn.evaluation.models import (
    DIMENSION_NAMES,
    MatchCritique,
    PairwiseJudgment,
)
from owtn.models.stage_1.concept_genome import ConceptGenome


class _FakeResult:
    def __init__(self, content: PairwiseJudgment) -> None:
        self.content = content
        self.cost = 0.0


def _all_votes(winner: str) -> PairwiseJudgment:
    return PairwiseJudgment(
        reasoning=f"All dims go to {winner}.",
        **{dim: winner for dim in DIMENSION_NAMES},
    )


@pytest.fixture
def fake_genomes():
    challenger = ConceptGenome(
        premise="Challenger: a cartographer who misremembers a forgotten country.",
        target_effect="The vertigo of reverse-inventing a homeland.",
    )
    champion = ConceptGenome(
        premise="Champion: a lighthouse keeper translates the sea's silences.",
        target_effect="Dread and complicity.",
    )
    return challenger, champion


@pytest.fixture
def config():
    return SimpleNamespace(
        judges=SimpleNamespace(
            judges_dir="configs/judges",
            panel=["mira-okonkwo", "tomas-varga", "sable-ahn"],
        )
    )


def _prefer_challenger_fake(challenger_premise: str):
    """Return a fake_query_async that consistently prefers the challenger
    across both orderings. In the forward ordering (challenger=A) it votes
    'a'; in the reverse ordering (challenger=B) it votes 'b'. This matches
    a position-consistent judge who genuinely prefers the challenger."""

    async def fake(*args, **kwargs):
        user = kwargs.get("msg", "")
        # Premise ordering in the user message determines A/B — the first
        # "Premise:" is concept A. If challenger's premise appears first,
        # challenger is labeled A.
        a_block = user.split("CONCEPT A:", 1)[-1].split("CONCEPT B:", 1)[0]
        if challenger_premise in a_block:
            vote = "a"  # challenger is A → prefer A
        else:
            vote = "b"  # challenger is B → prefer B
        return _FakeResult(_all_votes(vote))

    return fake


@pytest.mark.asyncio
async def test_challenger_wins_produces_both_critiques(
    monkeypatch, fake_genomes, config
):
    """When challenger (a) wins 9-0, both concepts get MatchCritiques with
    opposite outcomes and self/opponent labels correctly assigned."""
    challenger, champion = fake_genomes

    monkeypatch.setattr(
        pairwise, "query_async", _prefer_challenger_fake(challenger.premise)
    )

    result = await pairwise.compare(
        genome_a=challenger,
        genome_b=champion,
        config=config,
        champion_label="b",
    )

    assert result.winner == "a"
    assert set(result.critiques_by_label.keys()) == {"a", "b"}

    chall_c = result.critiques_by_label["a"]
    champ_c = result.critiques_by_label["b"]

    assert isinstance(chall_c, MatchCritique)
    assert chall_c.self_label == "a"
    assert chall_c.opponent_label == "b"
    assert chall_c.self_was_champion is False
    assert chall_c.outcome == "won"
    assert all(v == "won" for v in chall_c.dim_outcomes.values())
    assert chall_c.opponent_genome["premise"] == champion.premise

    assert champ_c.self_label == "b"
    assert champ_c.self_was_champion is True
    assert champ_c.outcome == "lost"
    assert all(v == "lost" for v in champ_c.dim_outcomes.values())
    assert champ_c.opponent_genome["premise"] == challenger.premise


@pytest.mark.asyncio
async def test_champion_retains_produces_tied_outcomes(
    monkeypatch, fake_genomes, config
):
    """All judges vote 'tie' → dim_outcomes all tied → match outcome is
    'tied' for both, and incumbent champion retains via champion_label rule."""
    challenger, champion = fake_genomes

    async def fake_query_async(*args, **kwargs):
        return _FakeResult(_all_votes("tie"))

    monkeypatch.setattr(pairwise, "query_async", fake_query_async)

    result = await pairwise.compare(
        genome_a=challenger,
        genome_b=champion,
        config=config,
        champion_label="b",
    )

    # All-ties → incumbent retains.
    assert result.winner == "b"

    chall_c = result.critiques_by_label["a"]
    champ_c = result.critiques_by_label["b"]

    # Match-level outcome reflects actual winner, not dim totals.
    assert chall_c.outcome == "lost"
    assert champ_c.outcome == "won"
    assert all(v == "tied" for v in chall_c.dim_outcomes.values())
    assert all(v == "tied" for v in champ_c.dim_outcomes.values())


@pytest.mark.asyncio
async def test_judge_reasonings_attached_to_both_critiques(
    monkeypatch, fake_genomes, config
):
    """Both critiques share the same judge reasonings (the reasoning text
    references A/B and the summarizer resolves the labels at prompt time)."""
    challenger, champion = fake_genomes

    call_count = {"n": 0}
    base_fake = _prefer_challenger_fake(challenger.premise)

    async def counting_fake(*args, **kwargs):
        call_count["n"] += 1
        return await base_fake(*args, **kwargs)

    monkeypatch.setattr(pairwise, "query_async", counting_fake)

    result = await pairwise.compare(
        genome_a=challenger, genome_b=champion, config=config, champion_label="b",
    )

    chall_c = result.critiques_by_label["a"]
    champ_c = result.critiques_by_label["b"]

    # 3 judges × 2 orderings = 6 calls; only forward used.
    assert call_count["n"] == 6
    assert len(chall_c.judge_reasonings) == 3
    assert len(champ_c.judge_reasonings) == 3
    # Both critiques carry the same verbatim reasonings.
    assert chall_c.judge_reasonings == champ_c.judge_reasonings

    # Each reasoning record has id/harshness/reasoning.
    for rec in chall_c.judge_reasonings:
        assert rec.judge_id in {"mira-okonkwo", "tomas-varga", "sable-ahn"}
        assert rec.harshness in {
            "advancing", "standard", "demanding", "failing_unless_exceptional",
        }
        assert "All dims go to a" in rec.reasoning


@pytest.mark.asyncio
async def test_match_critique_serializes_to_dict(monkeypatch, fake_genomes, config):
    """model_dump() produces plain dict ready for private_metrics / JSON."""
    challenger, champion = fake_genomes

    monkeypatch.setattr(
        pairwise, "query_async", _prefer_challenger_fake(challenger.premise)
    )

    result = await pairwise.compare(
        genome_a=challenger, genome_b=champion, config=config, champion_label="b",
    )

    as_dict = result.critiques_by_label["a"].model_dump()
    assert isinstance(as_dict, dict)
    assert as_dict["self_label"] == "a"
    assert isinstance(as_dict["opponent_genome"], dict)
    assert isinstance(as_dict["judge_reasonings"], list)
    assert as_dict["judge_reasonings"][0]["judge_id"]

    # Round-trip: JSON-serializable, no pydantic types remaining.
    import json
    roundtripped = json.loads(json.dumps(as_dict))
    assert roundtripped["self_label"] == "a"
