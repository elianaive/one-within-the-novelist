"""Tests for the lazy lineage-brief summarizer.

Moved from `tests/test_evaluation/test_feedback.py` as part of the
`owtn/optimizer/` refactor. See
`lab/issues/2026-04-24-refactor-feedback-to-optimizer-module.md`.

History: `lab/issues/closed/2026-04-18-lazy-feedback-summarizer.md` — Phase 2.
"""

from __future__ import annotations

import pytest

from owtn.optimizer import lineage_brief
from owtn.optimizer.adapters import (
    _stage_1_format_self,
    _stage_1_lineage_system_prompt,
    compute_stage_1_lineage_brief,
)
from owtn.optimizer.models import LineageBrief


def _sample_brief() -> LineageBrief:
    return LineageBrief(
        established_weaknesses=[
            "Hook relies on withheld information rather than an embodied image.",
        ],
        contested_strengths=[
            "Reviewers split on whether the voice constraint is load-bearing.",
        ],
        attractor_signature=["Archive/apparatus framing as thematic vehicle."],
        divergence_directions=[
            "Next attempt should replace withheld information with an embodied image.",
        ],
    )


def _sample_critique(
    *,
    self_label: str = "a",
    outcome: str = "lost",
    was_champion: bool = False,
    reasoning: str = "Both concepts withhold, but A's withholding feels thin...",
) -> dict:
    """Build a match_critique dict in the shape actually stored."""
    from owtn.evaluation.models import DIMENSION_NAMES

    opponent_label = "b" if self_label == "a" else "a"
    return {
        "self_label": self_label,
        "opponent_label": opponent_label,
        "self_was_champion": was_champion,
        "opponent_genome": {
            "premise": "A lighthouse keeper translates silences.",
            "target_effect": "Dread.",
        },
        "outcome": outcome,
        "dim_outcomes": {dim: "lost" for dim in DIMENSION_NAMES},
        "judge_reasonings": [
            {
                "judge_id": "mira-okonkwo",
                "harshness": "advancing",
                "reasoning": reasoning,
            },
            {
                "judge_id": "tomas-varga",
                "harshness": "demanding",
                "reasoning": reasoning + " (tomas view)",
            },
            {
                "judge_id": "sable-ahn",
                "harshness": "failing_unless_exceptional",
                "reasoning": reasoning + " (sable view)",
            },
        ],
        "timestamp": "2026-04-18T18:00:00+00:00",
    }


class TestRenderLineageBrief:
    def test_single_match_challenger_phrasing(self):
        brief = _sample_brief()
        critiques = [_sample_critique(was_champion=False)]
        out = lineage_brief.render_lineage_brief(brief, critiques)
        assert "1 match" in out
        assert "1 as challenger" in out
        assert "0 as defender" in out
        assert "Established weaknesses" in out
        assert "Archive/apparatus framing" in out

    def test_multiple_matches_plural_and_mixed_roles(self):
        brief = _sample_brief()
        critiques = [
            _sample_critique(was_champion=False),
            _sample_critique(was_champion=True),
            _sample_critique(was_champion=True),
        ]
        out = lineage_brief.render_lineage_brief(brief, critiques)
        assert "3 matches" in out
        assert "1 as challenger" in out
        assert "2 as defender" in out

    def test_empty_sections_render_none_marker(self):
        brief = LineageBrief(
            established_weaknesses=[],
            contested_strengths=[],
            attractor_signature=[],
            divergence_directions=[],
        )
        out = lineage_brief.render_lineage_brief(brief, [_sample_critique()])
        assert "(none identified)" in out


class TestCacheFreshness:
    def test_fresh_when_count_matches(self):
        pm = {
            "match_critiques": [_sample_critique()],
            "lineage_brief_cache": {
                "count": 1,
                "brief": _sample_brief().model_dump(),
            },
        }
        assert lineage_brief._cache_is_fresh(pm, 1) is True

    def test_stale_when_count_grew(self):
        pm = {
            "match_critiques": [_sample_critique(), _sample_critique()],
            "lineage_brief_cache": {
                "count": 1,
                "brief": _sample_brief().model_dump(),
            },
        }
        assert lineage_brief._cache_is_fresh(pm, 2) is False

    def test_stale_when_no_cache(self):
        pm = {"match_critiques": [_sample_critique()]}
        assert lineage_brief._cache_is_fresh(pm, 1) is False


class TestGetOrComputeBrief:
    @pytest.mark.asyncio
    async def test_empty_critiques_returns_seed_placeholder(self):
        render, payload = await compute_stage_1_lineage_brief(
            self_genome={"premise": "x", "target_effect": "y"},
            private_metrics={},
            classifier_model="irrelevant",
        )
        assert "Initial lineage" in render
        assert payload is None

    @pytest.mark.asyncio
    async def test_fresh_cache_returns_without_calling_llm(self, monkeypatch):
        async def explode(*args, **kwargs):
            raise RuntimeError("must not be called")

        monkeypatch.setattr(lineage_brief, "summarize_lineage", explode)

        pm = {
            "match_critiques": [_sample_critique()],
            "lineage_brief_cache": {
                "count": 1,
                "brief": _sample_brief().model_dump(),
            },
        }
        render, payload = await compute_stage_1_lineage_brief(
            self_genome={"premise": "x", "target_effect": "y"},
            private_metrics=pm,
            classifier_model="irrelevant",
        )
        assert "Established weaknesses" in render
        assert "Archive/apparatus" in render
        assert payload is None  # cache hit — nothing to persist

    @pytest.mark.asyncio
    async def test_stale_cache_recomputes_and_returns_payload(self, monkeypatch):
        calls = []

        async def fake_summarize(**kwargs):
            calls.append(kwargs)
            return _sample_brief()

        monkeypatch.setattr(lineage_brief, "summarize_lineage", fake_summarize)

        pm = {
            "match_critiques": [_sample_critique(), _sample_critique()],
            "lineage_brief_cache": {
                "count": 1,
                "brief": _sample_brief().model_dump(),
            },
        }
        render, payload = await compute_stage_1_lineage_brief(
            self_genome={"premise": "x", "target_effect": "y"},
            private_metrics=pm,
            classifier_model="gpt-4.1-mini",
        )
        assert len(calls) == 1
        assert calls[0]["classifier_model"] == "gpt-4.1-mini"
        assert "Established weaknesses" in render
        assert payload is not None
        assert payload["count"] == 2
        assert "established_weaknesses" in payload["brief"]

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_raw_render(self, monkeypatch):
        async def fail(**kwargs):
            raise RuntimeError("summarizer boom")

        monkeypatch.setattr(lineage_brief, "summarize_lineage", fail)

        pm = {"match_critiques": [_sample_critique()]}
        render, payload = await compute_stage_1_lineage_brief(
            self_genome={"premise": "x", "target_effect": "y"},
            private_metrics=pm,
            classifier_model="gpt-4.1-mini",
        )
        assert "Prior match" in render
        assert "Sample reasoning" in render
        assert payload is None


class TestSummarizerPromptAssembly:
    def test_user_msg_contains_match_header_and_labels(self):
        critique = _sample_critique(self_label="a", was_champion=False)
        self_genome = {"premise": "THIS genome", "target_effect": "x"}
        msg = lineage_brief._build_summarizer_user_msg(
            self_genome, [critique], _stage_1_format_self
        )
        # Per-match header disambiguates A/B.
        assert "THIS LINEAGE was labeled 'A'" in msg
        assert "opponent was labeled 'B'" in msg
        assert "THIS genome" in msg
        # Judge reasonings present per-judge.
        assert "Judge mira-okonkwo" in msg
        assert "harshness=advancing" in msg
        assert "harshness=demanding" in msg
        assert "harshness=failing_unless_exceptional" in msg

    def test_champion_match_header_shows_defending(self):
        critique = _sample_critique(self_label="b", was_champion=True)
        self_genome = {"premise": "x", "target_effect": "y"}
        msg = lineage_brief._build_summarizer_user_msg(
            self_genome, [critique], _stage_1_format_self
        )
        assert "champion (defending)" in msg
        assert "THIS LINEAGE was labeled 'B'" in msg


class TestStage1AdapterPlaceholders:
    """The Stage 1 adapter must fill the generic prompt template so none of
    the `{…}` placeholders leak through into the system message."""

    def test_no_unresolved_placeholders(self):
        prompt = _stage_1_lineage_system_prompt()
        assert "{" not in prompt or "}" not in prompt, (
            "Unresolved placeholder in Stage 1 lineage prompt:\n" + prompt
        )

    def test_stage_1_domain_hints_present(self):
        prompt = _stage_1_lineage_system_prompt()
        assert "fiction concept" in prompt
        assert "nine resonance dimensions" in prompt
        assert "LineageBrief" in prompt
