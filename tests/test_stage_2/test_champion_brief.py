"""Champion brief tests. LLM mocked.

Phase 7 exit criteria covered:
- Cold-start path (zero events) renders raw fallback in the expansion prompt.
- Summarizer re-fires at N=3 events (configurable); intermediate calls
  hit the cache.
- Failed summarizer falls back to raw critiques, same as cold start.

The brief module reuses `LineageBrief` schema and supporting helpers from
`owtn.optimizer` (rather than duplicating them); these tests focus on what's
new in Stage 2: tree-level state, cache cadence, raw fallback wording.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from owtn.models.stage_2.dag import DAG
from owtn.optimizer.adapters import (
    _stage_2_build_summarizer_user_msg,
    _stage_2_format_dag_for_match,
    stage_2_render_raw_fallback,
)
from owtn.optimizer.models import LineageBrief
from owtn.stage_2.champion_brief import (
    TreeBriefState,
    get_or_compute_brief,
    record_full_panel_critique,
)


# ----- Test scaffolding -----


@dataclass
class _FakeQueryResult:
    """Mirrors `_FakeQueryResult` in test_operators.py — content is the
    parsed Pydantic instance when output_model is set, not a JSON string."""
    content: object
    cost: float = 0.005
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    thought: str = ""
    model_name: str = "claude-haiku-4-5"


def _canned_brief() -> LineageBrief:
    return LineageBrief(
        established_weaknesses=[
            "the tree keeps producing all-causal edge distributions with token disclosure at the climax",
        ],
        contested_strengths=[
            "judges split on whether the anchor's gravitational pull on edges is intentional or default",
        ],
        attractor_signature=[
            "anchor-cluster topology: most edges incident to the role-bearing node",
        ],
        divergence_directions=[
            "Do not propose structures whose edges all converge on the anchor — vary edge sources.",
        ],
    )


def _make_critique(
    *,
    self_dag_data: dict,
    opponent_genome: dict | None = None,
    self_was_champion: bool = False,
    outcome: str = "won",
) -> dict:
    """Build a minimal full-panel critique record matching the shape
    `summarize_lineage` and `_format_tree_match_block` consume."""
    return {
        "self_label": "a",
        "opponent_label": "b",
        "self_was_champion": self_was_champion,
        "self_dag": self_dag_data,
        "opponent_genome": opponent_genome or {},
        "outcome": outcome,
        "dim_outcomes": {
            "edge_logic": "won",
            "motivational_coherence": "tied",
            "tension_information_arch": "won",
            "post_dictability": "lost",
            "arc_integrity_ending": "won",
            "structural_coherence": "tied",
            "beat_quality": "won",
            "concept_fidelity_thematic": "tied",
        },
        "judge_reasonings": [
            {
                "judge_id": "gwern",
                "harshness": "demanding",
                "reasoning": "The structure deploys causal density without information architecture.",
            },
            {
                "judge_id": "roon",
                "harshness": "standard",
                "reasoning": "Beats land but the disclosure work is decorative — it doesn't reframe.",
            },
        ],
        "timestamp": "2026-04-24T12:00:00Z",
    }


@pytest.fixture
def lottery_dag_dict(canonical_lottery: DAG) -> dict:
    return canonical_lottery.model_dump()


# ----- Cold start -----


class TestColdStart:
    """Phase 7 exit: zero-event tree returns the raw-fallback placeholder
    without firing an LLM call."""

    def test_empty_state_returns_seed_placeholder(self) -> None:
        state = TreeBriefState()
        # Critically: no LLM patch needed — we're asserting NO call fires.
        result = asyncio.run(get_or_compute_brief(
            state, classifier_model="dummy",
        ))
        assert "Initial tree" in result
        assert state.cached_brief is None  # no cache populated

    def test_render_raw_fallback_stage2_on_empty(self) -> None:
        assert "Initial tree" in stage_2_render_raw_fallback([])


# ----- Cache cadence -----


class TestCacheCadence:
    """Phase 7 exit: summarizer re-fires at N=3 events; intermediate calls
    hit the cache."""

    def test_first_call_with_critique_invokes_summarizer(
        self, lottery_dag_dict: dict,
    ) -> None:
        state = TreeBriefState()
        record_full_panel_critique(state, _make_critique(self_dag_data=lottery_dag_dict))

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.stage_2.champion_brief.query_async", side_effect=fake_query, create=True):
            # The local import in `_call_summarizer` patches at use-site;
            # provide both patches so either resolution finds the mock.
            with patch("owtn.llm.query.query_async", side_effect=fake_query):
                rendered = asyncio.run(get_or_compute_brief(
                    state, classifier_model="dummy", re_summarize_every=3,
                ))

        assert call_count[0] == 1
        assert "This tree has been evaluated" in rendered
        assert state.cached_count == 1
        assert state.cached_brief is not None

    def test_intermediate_calls_hit_cache(self, lottery_dag_dict: dict) -> None:
        """After the first summarizer fires, subsequent calls with fewer
        than `re_summarize_every` new events read from cache (no LLM)."""
        state = TreeBriefState()
        record_full_panel_critique(state, _make_critique(self_dag_data=lottery_dag_dict))

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.llm.query.query_async", side_effect=fake_query):
            # 1st call: summarizer fires.
            asyncio.run(get_or_compute_brief(
                state, classifier_model="dummy", re_summarize_every=3,
            ))
            assert call_count[0] == 1

            # Add 1 more critique. Delta = 1, threshold = 3 → cache hit.
            record_full_panel_critique(state, _make_critique(self_dag_data=lottery_dag_dict))
            asyncio.run(get_or_compute_brief(
                state, classifier_model="dummy", re_summarize_every=3,
            ))
            assert call_count[0] == 1  # no new LLM call

            # Add 1 more. Delta = 2, threshold = 3 → still cached.
            record_full_panel_critique(state, _make_critique(self_dag_data=lottery_dag_dict))
            asyncio.run(get_or_compute_brief(
                state, classifier_model="dummy", re_summarize_every=3,
            ))
            assert call_count[0] == 1

    def test_threshold_crossed_triggers_resummarize(
        self, lottery_dag_dict: dict,
    ) -> None:
        state = TreeBriefState()
        record_full_panel_critique(state, _make_critique(self_dag_data=lottery_dag_dict))

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.llm.query.query_async", side_effect=fake_query):
            asyncio.run(get_or_compute_brief(
                state, classifier_model="dummy", re_summarize_every=3,
            ))
            assert call_count[0] == 1

            # Cross the threshold: add 3 more events. Delta = 3 == threshold → resummarize.
            for _ in range(3):
                record_full_panel_critique(state, _make_critique(self_dag_data=lottery_dag_dict))
            asyncio.run(get_or_compute_brief(
                state, classifier_model="dummy", re_summarize_every=3,
            ))
            assert call_count[0] == 2  # resummarized

    def test_force_resummarize_bypasses_cache(
        self, lottery_dag_dict: dict,
    ) -> None:
        """Per `mcts.md` "Forced re-render on champion promotion": callers
        can force a re-summarize even when the cache is fresh."""
        state = TreeBriefState()
        record_full_panel_critique(state, _make_critique(self_dag_data=lottery_dag_dict))

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.llm.query.query_async", side_effect=fake_query):
            asyncio.run(get_or_compute_brief(
                state, classifier_model="dummy", re_summarize_every=3,
            ))
            assert call_count[0] == 1

            asyncio.run(get_or_compute_brief(
                state, classifier_model="dummy",
                re_summarize_every=3, force_resummarize=True,
            ))
            assert call_count[0] == 2  # forced even though cache fresh


# ----- Summarizer failure -----


class TestSummarizerFailure:
    """Phase 7 exit: failed summarizer falls back to raw critiques rather
    than crashing the expansion prompt."""

    def test_provider_outage_falls_back_to_raw(
        self, lottery_dag_dict: dict,
    ) -> None:
        state = TreeBriefState()
        record_full_panel_critique(state, _make_critique(self_dag_data=lottery_dag_dict))

        async def failing_query(**kwargs):
            raise RuntimeError("simulated provider outage")

        with patch("owtn.llm.query.query_async", side_effect=failing_query):
            rendered = asyncio.run(get_or_compute_brief(
                state, classifier_model="dummy",
            ))

        # The raw fallback names the prior match's outcome and includes
        # judge reasoning text rather than the structured 4-field render.
        assert "Prior match" in rendered
        assert "Sample reasoning from judge" in rendered
        # Cache is NOT populated on failure — we want next call to retry.
        assert state.cached_brief is None
        assert state.cached_count is None

    def test_unexpected_content_type_falls_back_to_raw(
        self, lottery_dag_dict: dict,
    ) -> None:
        """If the provider returns something that isn't a LineageBrief,
        the helper raises a RuntimeError which gets caught and triggers
        fallback. Defensive against provider-wrapper drift."""
        state = TreeBriefState()
        record_full_panel_critique(state, _make_critique(self_dag_data=lottery_dag_dict))

        async def fake_query(**kwargs):
            return _FakeQueryResult(content="raw string instead of LineageBrief")

        with patch("owtn.llm.query.query_async", side_effect=fake_query):
            rendered = asyncio.run(get_or_compute_brief(
                state, classifier_model="dummy",
            ))

        assert "Prior match" in rendered  # raw fallback render


# ----- Format helpers -----


class TestFormatHelpers:
    def test_format_dag_for_match_renders_real_canonical(
        self, lottery_dag_dict: dict,
    ) -> None:
        rendered = _stage_2_format_dag_for_match(lottery_dag_dict)
        # Renderer's signature output starts with "STORY STRUCTURE A".
        assert "STORY STRUCTURE" in rendered
        assert "[gathering]" in rendered

    def test_format_dag_for_match_truncates_long_renderings(
        self, canonical_oconnor: DAG,
    ) -> None:
        # O'Connor's full rendering is ~3400 chars; truncated to ~2000.
        rendered = _stage_2_format_dag_for_match(canonical_oconnor.model_dump())
        assert len(rendered) <= 2050  # allow headroom for the truncation marker
        if len(rendered) > 2000:
            assert "(truncated)" in rendered

    def test_format_dag_for_match_handles_corrupt_dict(self) -> None:
        # An empty dict won't validate as a DAG. Helper returns a one-line
        # error rather than crashing the summarizer.
        rendered = _stage_2_format_dag_for_match({})
        assert "could not be reconstructed" in rendered

    def test_summarizer_user_msg_uses_tree_wording(
        self, lottery_dag_dict: dict,
    ) -> None:
        critique = _make_critique(self_dag_data=lottery_dag_dict)
        msg = _stage_2_build_summarizer_user_msg([critique])
        assert "THIS TREE" in msg  # not "THIS LINEAGE"
        assert "OPPONENT" in msg


# ----- Render wording -----


class TestRenderWording:
    """The renderer reuses `optimizer.render_lineage_brief` then patches
    'lineage' → 'tree' wording. Verify the patch lands."""

    def test_rendered_brief_uses_tree_wording(
        self, lottery_dag_dict: dict,
    ) -> None:
        state = TreeBriefState()
        record_full_panel_critique(state, _make_critique(self_dag_data=lottery_dag_dict))

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.llm.query.query_async", side_effect=fake_query):
            rendered = asyncio.run(get_or_compute_brief(
                state, classifier_model="dummy",
            ))

        assert "This tree has been evaluated" in rendered
        # No leftover lineage wording.
        assert "This lineage" not in rendered
