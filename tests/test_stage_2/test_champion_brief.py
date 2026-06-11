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
    TREE_SUBJECT,
    _stage_2_extract_self_fn,
    _stage_2_format_dag_for_match,
    _stage_2_format_self_fn,
    stage_2_render_raw_fallback,
)
from owtn.optimizer.lineage_brief import _build_summarizer_user_msg
from owtn.optimizer.models import LineageBrief
from owtn.stage_2.champion_brief import (
    TreeBriefState,
    get_or_compute_brief,
    get_or_compute_scalar_brief,
    get_or_compute_scalar_lineage_brief,
    record_full_panel_critique,
    record_rollout_outcome,
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
            with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
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

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
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

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
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

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
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

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=failing_query):
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

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
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
        msg = _build_summarizer_user_msg(
            match_critiques=[critique],
            extract_self=_stage_2_extract_self_fn,
            format_self=_stage_2_format_self_fn,
            subject=TREE_SUBJECT,
        )
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

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
            rendered = asyncio.run(get_or_compute_brief(
                state, classifier_model="dummy",
            ))

        assert "This tree has been evaluated" in rendered
        # No leftover lineage wording.
        assert "This lineage" not in rendered


# ----- Scalar-mode brief -----


class TestScalarBrief:
    """Scalar-mode mirror of the pairwise tests above. Same caching semantics,
    different summarizer input shape (rollout records, not match critiques)."""

    def test_empty_state_returns_seed_placeholder(self) -> None:
        state = TreeBriefState()
        result = asyncio.run(get_or_compute_scalar_brief(
            state, classifier_model="dummy",
        ))
        assert "Initial tree" in result
        assert state.cached_brief is None

    def test_first_call_invokes_summarizer(self, lottery_dag_dict: dict) -> None:
        state = TreeBriefState()
        record_rollout_outcome(
            state, score=0.42, reasoning="anchor cluster", dag=lottery_dag_dict,
        )

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
            rendered = asyncio.run(get_or_compute_scalar_brief(
                state, classifier_model="dummy", re_summarize_every=1,
            ))

        assert call_count[0] == 1
        # Stats line uses scalar-specific wording (rollouts + score range).
        assert "rollout" in rendered
        assert "scores:" in rendered
        # The four-section body shape is shared with the pairwise render.
        assert "Established weaknesses" in rendered
        assert "Divergence directions" in rendered
        assert state.cached_count == 1
        assert state.cached_brief is not None

    def test_intermediate_calls_hit_cache(self, lottery_dag_dict: dict) -> None:
        state = TreeBriefState()
        record_rollout_outcome(
            state, score=0.42, reasoning="r1", dag=lottery_dag_dict,
        )

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
            asyncio.run(get_or_compute_scalar_brief(
                state, classifier_model="dummy", re_summarize_every=5,
            ))
            assert call_count[0] == 1
            for _ in range(3):
                record_rollout_outcome(
                    state, score=0.5, reasoning="r", dag=lottery_dag_dict,
                )
                asyncio.run(get_or_compute_scalar_brief(
                    state, classifier_model="dummy", re_summarize_every=5,
                ))
            # 1 + 3 records, delta=3 < threshold=5 → still cached.
            assert call_count[0] == 1

    def test_threshold_crosses_resummarizes(self, lottery_dag_dict: dict) -> None:
        state = TreeBriefState()
        record_rollout_outcome(
            state, score=0.42, reasoning="r1", dag=lottery_dag_dict,
        )

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
            asyncio.run(get_or_compute_scalar_brief(
                state, classifier_model="dummy", re_summarize_every=3,
            ))
            assert call_count[0] == 1
            for _ in range(3):
                record_rollout_outcome(
                    state, score=0.7, reasoning="r", dag=lottery_dag_dict,
                )
            asyncio.run(get_or_compute_scalar_brief(
                state, classifier_model="dummy", re_summarize_every=3,
            ))
            assert call_count[0] == 2

    def test_provider_outage_falls_back_to_raw(
        self, lottery_dag_dict: dict,
    ) -> None:
        state = TreeBriefState()
        record_rollout_outcome(
            state, score=0.42,
            reasoning="anchor cluster — edges all converge on the role-bearing node",
            dag=lottery_dag_dict,
        )

        async def failing_query(**kwargs):
            raise RuntimeError("simulated provider outage")

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=failing_query):
            rendered = asyncio.run(get_or_compute_scalar_brief(
                state, classifier_model="dummy", re_summarize_every=1,
            ))

        # Scalar raw fallback names "Prior rollout" with the score.
        assert "Prior rollout (score 0.420)" in rendered
        assert "anchor cluster" in rendered
        assert state.cached_brief is None
        assert state.cached_count is None

    def test_pairwise_and_scalar_share_cache_slot(
        self, lottery_dag_dict: dict,
    ) -> None:
        """The two paths never run on the same tree (mode is a config field),
        so they share the (cached_count, cached_brief, cached_render) slot.
        Exercise that they don't collide when only one is populated."""
        state = TreeBriefState()
        # Scalar populates the cache.
        record_rollout_outcome(
            state, score=0.42, reasoning="r", dag=lottery_dag_dict,
        )

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
            asyncio.run(get_or_compute_scalar_brief(
                state, classifier_model="dummy", re_summarize_every=1,
            ))

        assert state.cached_count == 1
        assert state.cached_render is not None
        # full_panel_critiques untouched.
        assert state.full_panel_critiques == []


# ----- Scalar lineage (per-leaf) brief -----


def _ancestor_dag(dag_dict: dict, *, drop_last_node: bool = True) -> dict:
    """Build a DAG dict that is a structural prefix of `dag_dict`.

    Drops the last node + any edges incident to it; the resulting dict
    matches `lineage_records_for_target`'s ancestor predicate (subset of
    nodes, subset of edges)."""
    out = {**dag_dict}
    nodes = list(dag_dict.get("nodes", []))
    if drop_last_node and len(nodes) > 1:
        last_id = nodes[-1]["id"]
        out["nodes"] = nodes[:-1]
        out["edges"] = [
            e for e in dag_dict.get("edges", [])
            if e["src"] != last_id and e["dst"] != last_id
        ]
    return out


class TestScalarLineageBrief:
    """Per-leaf scalar lineage brief: records are filtered to ancestors of
    the target DAG before being summarized; cache is per-DAG-digest."""

    def test_no_ancestor_records_returns_seed_placeholder(
        self, lottery_dag_dict: dict,
    ) -> None:
        """A fresh leaf with no ancestor rollouts gets the seed placeholder
        without firing an LLM call — even if the tree has unrelated records."""
        state = TreeBriefState()
        # Record on an unrelated DAG (subset relation reversed — target is
        # smaller, so the recorded DAG is NOT an ancestor of target).
        ancestor = _ancestor_dag(lottery_dag_dict)
        record_rollout_outcome(
            state, score=0.5, reasoning="r", dag=lottery_dag_dict,
        )

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
            rendered = asyncio.run(get_or_compute_scalar_lineage_brief(
                state, target_dag=ancestor, classifier_model="dummy",
            ))

        assert "Initial path" in rendered
        assert call_count[0] == 0  # no LLM call

    def test_filters_to_ancestor_records(
        self, lottery_dag_dict: dict,
    ) -> None:
        """Only records whose DAG is a structural prefix of the target are
        fed to the summarizer."""
        state = TreeBriefState()
        ancestor = _ancestor_dag(lottery_dag_dict)
        # Two records: one is an ancestor of target, one is a divergent
        # sibling — same prefix as `ancestor`, then a *different* beat
        # added (a node id that doesn't appear in the target).
        record_rollout_outcome(
            state, score=0.4, reasoning="ancestor reasoning", dag=ancestor,
        )
        sibling = {**ancestor}
        sibling_nodes = list(ancestor.get("nodes", [])) + [{
            "id": "divergent_node_not_in_target",
            "sketch": "A divergent beat sketch with enough words to validate.",
            "role": None,
            "motifs": [],
        }]
        sibling["nodes"] = sibling_nodes
        record_rollout_outcome(
            state, score=0.6, reasoning="sibling reasoning", dag=sibling,
        )

        captured_user_msgs = []

        async def fake_query(**kwargs):
            captured_user_msgs.append(kwargs.get("msg", ""))
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
            rendered = asyncio.run(get_or_compute_scalar_lineage_brief(
                state, target_dag=lottery_dag_dict, classifier_model="dummy",
            ))

        assert len(captured_user_msgs) == 1
        # Only the ancestor reasoning made it into the summarizer input.
        msg = captured_user_msgs[0]
        assert "ancestor reasoning" in msg
        assert "sibling reasoning" not in msg
        assert "THIS PATH" in msg  # subject wording
        # Render uses the lineage-flavored stats line.
        assert "this path" in rendered.lower()

    def test_cache_per_leaf_avoids_recomputation(
        self, lottery_dag_dict: dict,
    ) -> None:
        """Repeat lookups for the same target DAG hit the per-leaf cache
        when the ancestor count is unchanged."""
        state = TreeBriefState()
        ancestor = _ancestor_dag(lottery_dag_dict)
        record_rollout_outcome(
            state, score=0.4, reasoning="r", dag=ancestor,
        )

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
            asyncio.run(get_or_compute_scalar_lineage_brief(
                state, target_dag=lottery_dag_dict, classifier_model="dummy",
            ))
            asyncio.run(get_or_compute_scalar_lineage_brief(
                state, target_dag=lottery_dag_dict, classifier_model="dummy",
            ))

        assert call_count[0] == 1  # second call was a cache hit

    def test_cache_invalidates_when_ancestor_count_grows(
        self, lottery_dag_dict: dict,
    ) -> None:
        """When a new ancestor record arrives, the lineage cache for the
        target re-summarizes."""
        state = TreeBriefState()
        ancestor_a = _ancestor_dag(lottery_dag_dict)
        record_rollout_outcome(
            state, score=0.4, reasoning="r1", dag=ancestor_a,
        )

        call_count = [0]

        async def fake_query(**kwargs):
            call_count[0] += 1
            return _FakeQueryResult(content=_canned_brief())

        with patch("owtn.optimizer.lineage_brief.query_async", side_effect=fake_query):
            asyncio.run(get_or_compute_scalar_lineage_brief(
                state, target_dag=lottery_dag_dict, classifier_model="dummy",
            ))
            assert call_count[0] == 1
            # Add a second ancestor record (the target's full DAG itself).
            record_rollout_outcome(
                state, score=0.5, reasoning="r2", dag=lottery_dag_dict,
            )
            asyncio.run(get_or_compute_scalar_lineage_brief(
                state, target_dag=lottery_dag_dict, classifier_model="dummy",
            ))
            assert call_count[0] == 2  # ancestor count grew → re-summarize


class TestStage2BriefPromptsHaveNoSchemaTrailer:
    """Lockfile guard — under `tool_choice: auto` the schema-restating
    trailer ("Respond with a single JSON object…") competed with the tool
    schema and caused haiku to skip tool_use; under forced `tool_choice` it
    is just deadweight tokens. Re-introducing it would silently regress.
    See `lab/issues/2026-04-30-stage-2-lineage-brief-tool-use-miss.md`."""

    def test_pairwise_brief_omits_trailer(self):
        from owtn.prompts.stage_2.registry import load_champion_brief_system

        prompt = load_champion_brief_system()
        assert "Respond with a single JSON object" not in prompt
        assert "Respond with a JSON object" not in prompt

    def test_scalar_tree_brief_omits_trailer(self):
        from owtn.prompts.stage_2.registry import load_champion_brief_scalar_system

        prompt = load_champion_brief_scalar_system()
        assert "Respond with a single JSON object" not in prompt
        assert "Respond with a JSON object" not in prompt

    def test_scalar_lineage_brief_omits_trailer(self):
        from owtn.prompts.stage_2.registry import (
            load_champion_brief_scalar_lineage_system,
        )

        prompt = load_champion_brief_scalar_lineage_system()
        assert "Respond with a single JSON object" not in prompt
        assert "Respond with a JSON object" not in prompt
