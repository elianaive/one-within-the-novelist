"""Regression test for judge_id context leakage across async tasks.

Bug: `owtn/evaluation/pairwise.py` used to call `llm_context.set(...)` inside
the loop that scheduled coroutines for `asyncio.gather`. Because Task creation
copies the caller's context at gather-time, every task saw the *last* judge's
`judge_id` instead of its own, making all logged calls tag the final panel
member. This test captures the judge_id value in each simulated LLM call and
asserts the full panel is represented.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from owtn.evaluation import pairwise
from owtn.evaluation.models import DIMENSION_NAMES, PairwiseJudgment
from owtn.llm.call_logger import llm_context
from owtn.models.stage_1.concept_genome import ConceptGenome


class _FakeResult:
    """Stand-in for the LLM query_async result shape used by pairwise.py."""
    def __init__(self, content: PairwiseJudgment) -> None:
        self.content = content
        self.cost = 0.0


def _judgment(winner: str) -> PairwiseJudgment:
    return PairwiseJudgment(
        reasoning="stub",
        **{dim: winner for dim in DIMENSION_NAMES},
    )


@pytest.mark.asyncio
async def test_compare_logs_correct_judge_id_per_call(monkeypatch):
    seen: list[str] = []

    async def fake_query_async(*args, **kwargs):
        ctx = llm_context.get({})
        seen.append(ctx.get("judge_id", "<missing>"))
        return _FakeResult(_judgment("a"))

    monkeypatch.setattr(pairwise, "query_async", fake_query_async)

    genome_a = ConceptGenome(
        premise="A first test concept premise long enough.",
        target_effect="first target effect.",
    )
    genome_b = ConceptGenome(
        premise="A second test concept premise long enough.",
        target_effect="second target effect.",
    )

    config = SimpleNamespace(
        judges=SimpleNamespace(
            judges_dir="configs/judges",
            panel=["mira-okonkwo", "tomas-varga", "sable-ahn"],
        )
    )

    await pairwise.compare(genome_a, genome_b, config)

    # 3 judges × 2 orderings = 6 calls, each should report its own judge_id.
    assert len(seen) == 6
    assert set(seen) == {"mira-okonkwo", "tomas-varga", "sable-ahn"}
    for judge_id in ("mira-okonkwo", "tomas-varga", "sable-ahn"):
        assert seen.count(judge_id) == 2
