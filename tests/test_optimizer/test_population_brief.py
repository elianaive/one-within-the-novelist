"""Structural tests for the population-brief summarizer and Stage 1 adapter.

Mocks the LLM. Asserts structural validity only, not content (per CLAUDE.md
test principles and `lab/issues/2026-04-22-global-optimizer-state.md`).

RunBrief redesigned per
`lab/deep-research/runs/20260424_031609-avoidance-instruction-compliance/`:
- `divergence_pressure` → `exploration_directions` (positively framed)
- Rendered output split into `(population_context, exploration_directions)`
- Exploration directions injected at both top AND end of the user message
  (instruction sandwich)
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from owtn.optimizer import population_brief as population_brief_mod
from owtn.optimizer.adapters import (
    _format_stage_1_population_user_msg,
    _stage_1_gather_lineage_briefs,
    _stage_1_population_system_prompt,
    compute_stage_1_population_brief,
)
from owtn.optimizer.population_brief import (
    PopulationBrief,
    render_exploration_directions,
    render_population_context,
)
from owtn.prompts.stage_1.registry import build_operator_prompt, load_registry


def _sample_population_brief() -> PopulationBrief:
    return PopulationBrief(
        population_attractors=[
            "Stakes derive entirely from the behavior of a single defined "
            "system rather than from human contingency."
        ],
        per_judge_signals=[
            "Gwern rewards lineages where embodied consequence outweighs "
            "structural elegance.",
            "Wahls rewards lineages with sharper individual voice.",
        ],
        population_drift=[
            "Population is clustering on single-mechanism paradox structures."
        ],
        exploration_directions=[
            "Find suspense in contingency — situations where the outcome "
            "genuinely depends on accumulated small decisions, not the "
            "predetermined unfolding of a single system. "
            "Example: a welder's one-time decision to cut corners on a "
            "bridge joint, discovered thirty years later by her granddaughter."
        ],
    )


def _sample_lineage_brief_dict() -> dict:
    return {
        "established_weaknesses": ["Hook relies on withheld information."],
        "contested_strengths": ["Voice constraint is load-bearing."],
        "attractor_signature": ["Archive/apparatus framing."],
        "divergence_directions": ["Replace withheld info with embodied image."],
    }


class _FakeCursor:
    def __init__(self, rows: list[dict]):
        self._rows = rows
        self._last_query = None

    def execute(self, sql: str):
        self._last_query = sql

    def fetchall(self) -> list[dict]:
        return self._rows


class _FakeDB:
    def __init__(self, rows: list[dict]):
        self.cursor = _FakeCursor(rows)


class TestPopulationPromptPlaceholders:
    def test_all_placeholders_resolved(self):
        prompt = _stage_1_population_system_prompt(
            judge_names=["gwern", "jamie-wahls", "roon"]
        )
        assert "{" not in prompt or "}" not in prompt, (
            "Unresolved placeholder in Stage 1 population prompt:\n" + prompt
        )
        # Stage 1 domain hints and judge list both present.
        assert "concept evolution" in prompt
        assert "gwern" in prompt
        assert "jamie-wahls" in prompt

    def test_prompt_forbids_negation_phrasing(self):
        """Exploration directions must be positive; the prompt must not
        instruct the summarizer to use 'do not' / 'avoid' phrasing."""
        prompt = _stage_1_population_system_prompt(judge_names=["gwern"])
        # The prompt should explicitly tell the summarizer NEVER to phrase
        # exploration directions as negations.
        assert "Never phrase as" in prompt or "never phrase as" in prompt.lower()

    def test_prompt_caps_exploration_directions(self):
        """Research says the reliable compliance ceiling is ~3 items."""
        prompt = _stage_1_population_system_prompt(judge_names=["gwern"])
        assert "MAX 3" in prompt or "max 3" in prompt.lower()


class TestGatherLineageBriefs:
    def test_skips_programs_without_cache(self):
        rows = [
            {
                "id": "abc",
                "code": json.dumps({"premise": "A."}),
                "private_metrics": json.dumps({"match_critiques": []}),
            },
            {
                "id": "def",
                "code": json.dumps({"premise": "B."}),
                "private_metrics": json.dumps({
                    "match_critiques": [{}],
                    "lineage_brief_cache": {
                        "count": 1,
                        "brief": _sample_lineage_brief_dict(),
                    },
                }),
            },
        ]
        db = _FakeDB(rows)
        entries = _stage_1_gather_lineage_briefs(db)
        assert len(entries) == 1
        program_id, premise, brief = entries[0]
        assert program_id == "def"
        assert premise == "B."
        assert brief["attractor_signature"] == ["Archive/apparatus framing."]

    def test_empty_when_no_cached_briefs(self):
        db = _FakeDB([])
        assert _stage_1_gather_lineage_briefs(db) == []


class TestPopulationUserMsgFormatting:
    def test_includes_all_lineage_brief_fields(self):
        entries = [
            ("abc12345", "Premise A.", _sample_lineage_brief_dict()),
            ("def67890", "Premise B.", _sample_lineage_brief_dict()),
        ]
        msg = _format_stage_1_population_user_msg(entries)
        assert "Lineage 1" in msg
        assert "Lineage 2" in msg
        assert "abc12345" in msg
        assert "def67890" in msg
        assert "Premise A." in msg
        assert "Established weaknesses" in msg
        assert "Attractor signature" in msg


class TestComputeStage1PopulationBrief:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_cached_briefs(self, monkeypatch):
        db = _FakeDB([])

        async def must_not_call(self, *, system_prompt, user_msg):
            raise AssertionError("summarizer must not be called with no entries")

        monkeypatch.setattr(
            population_brief_mod.PopulationBriefSummarizer,
            "summarize",
            must_not_call,
        )

        result = await compute_stage_1_population_brief(
            db=db,
            run_brief_model="fake-model",
            judge_names=["gwern"],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_dispatches_and_splits_output(self, monkeypatch):
        rows = [
            {
                "id": "prog1",
                "code": json.dumps({"premise": "A lighthouse."}),
                "private_metrics": json.dumps({
                    "match_critiques": [{}],
                    "lineage_brief_cache": {
                        "count": 1,
                        "brief": _sample_lineage_brief_dict(),
                    },
                }),
            }
        ]
        db = _FakeDB(rows)

        captured: dict = {}

        async def fake_summarize(self, *, system_prompt, user_msg):
            captured["system_prompt"] = system_prompt
            captured["user_msg"] = user_msg
            captured["model_name"] = self.model_name
            return _sample_population_brief()

        monkeypatch.setattr(
            population_brief_mod.PopulationBriefSummarizer,
            "summarize",
            fake_summarize,
        )

        result = await compute_stage_1_population_brief(
            db=db,
            run_brief_model="fake-nano",
            judge_names=["gwern", "wahls"],
        )
        assert result is not None
        context, directions = result
        # Context block contains attractor shapes, per-judge signals, drift —
        # but NOT the exploration directions (those are rendered separately
        # for the instruction sandwich).
        assert "attractor shapes" in context.lower()
        assert "Per-judge signal" in context
        assert "Population drift" in context
        assert "Exploration directions" not in context
        # Directions block is the positively-framed actionable guidance.
        assert "Exploration directions" in directions
        assert "welder" in directions.lower()
        assert captured["model_name"] == "fake-nano"
        assert "Lineage 1" in captured["user_msg"]
        assert "gwern" in captured["system_prompt"]

    @pytest.mark.asyncio
    async def test_summarizer_failure_returns_none(self, monkeypatch):
        rows = [
            {
                "id": "prog1",
                "code": json.dumps({"premise": "A lighthouse."}),
                "private_metrics": json.dumps({
                    "match_critiques": [{}],
                    "lineage_brief_cache": {
                        "count": 1,
                        "brief": _sample_lineage_brief_dict(),
                    },
                }),
            }
        ]
        db = _FakeDB(rows)

        async def boom(self, *, system_prompt, user_msg):
            raise RuntimeError("llm down")

        monkeypatch.setattr(
            population_brief_mod.PopulationBriefSummarizer,
            "summarize",
            boom,
        )
        result = await compute_stage_1_population_brief(
            db=db, run_brief_model="fake", judge_names=[],
        )
        assert result is None


class TestRenderPopulationContext:
    def test_empty_lists_render_none_marker(self):
        brief = PopulationBrief(
            population_attractors=[],
            per_judge_signals=[],
            population_drift=[],
            exploration_directions=[],
        )
        out = render_population_context(brief)
        assert "(none identified)" in out

    def test_three_context_sections_present_exploration_absent(self):
        out = render_population_context(_sample_population_brief())
        assert "attractor shapes" in out.lower()
        assert "Per-judge signal" in out
        assert "Population drift" in out
        # Exploration directions are rendered by a separate function.
        assert "Exploration directions" not in out


class TestRenderExplorationDirections:
    def test_renders_directions_block(self):
        out = render_exploration_directions(_sample_population_brief())
        assert "Exploration directions" in out
        assert "contingency" in out.lower()

    def test_no_negation_phrasing_in_sample(self):
        """Guardrail: sample brief used in tests must use positive framing.
        If this fails, tests are drifting away from the research mandate."""
        out = render_exploration_directions(_sample_population_brief())
        lower = out.lower()
        # Our sample should not contain these negation patterns.
        assert "do not produce" not in lower
        assert "do not rely on" not in lower


class TestInstructionSandwichInjection:
    """Exploration directions must appear at BOTH the top and the end of the
    user message (primacy + recency). Population context sits in the middle.
    """

    def test_sandwich_both_ends(self):
        registry = load_registry()
        directions = render_exploration_directions(_sample_population_brief())
        context = render_population_context(_sample_population_brief())
        sys_msg, user_msg = build_operator_prompt(
            "collision",
            registry=registry,
            parent_genome='{"premise": "x"}',
            feedback="Lineage brief goes here.",
            population_context=context,
            exploration_directions=directions,
            is_initial=False,
        )
        # Exploration directions appear twice (top + end).
        assert user_msg.count("Exploration directions") == 2
        # Top exploration block appears before everything else in the user
        # message (no parent/feedback/work between).
        first_directions_idx = user_msg.index("Exploration directions")
        parent_idx = user_msg.index("# The parent")
        work_idx = user_msg.index("# The work")
        assert first_directions_idx < parent_idx < work_idx
        # End exploration block appears after operator instructions.
        last_directions_idx = user_msg.rindex("Exploration directions")
        assert last_directions_idx > work_idx

    def test_context_appears_in_middle_not_in_sandwich(self):
        registry = load_registry()
        directions = render_exploration_directions(_sample_population_brief())
        context = render_population_context(_sample_population_brief())
        sys_msg, user_msg = build_operator_prompt(
            "collision",
            registry=registry,
            parent_genome='{"premise": "x"}',
            feedback="Lineage brief.",
            population_context=context,
            exploration_directions=directions,
            is_initial=False,
        )
        assert "# Population signal" in user_msg
        # Population signal comes AFTER the first exploration block but
        # BEFORE the operator instructions.
        pop_idx = user_msg.index("# Population signal")
        first_directions_idx = user_msg.index("Exploration directions")
        work_idx = user_msg.index("# The work")
        assert first_directions_idx < pop_idx < work_idx

    def test_no_population_without_any_signal(self):
        registry = load_registry()
        sys_msg, user_msg = build_operator_prompt(
            "collision",
            registry=registry,
            parent_genome='{"premise": "x"}',
            feedback="Just lineage.",
            is_initial=False,
        )
        assert "# Population signal" not in user_msg
        assert "Exploration directions" not in user_msg

    def test_context_only_no_sandwich(self):
        """When the summarizer produced context but no exploration directions
        (e.g. evidence was too thin), inject context only — no empty
        sandwich wrappers."""
        registry = load_registry()
        context = render_population_context(_sample_population_brief())
        sys_msg, user_msg = build_operator_prompt(
            "collision",
            registry=registry,
            parent_genome='{"premise": "x"}',
            feedback="Lineage brief.",
            population_context=context,
            exploration_directions="",
            is_initial=False,
        )
        assert "# Population signal" in user_msg
        assert "Exploration directions" not in user_msg

    def test_ignored_in_genesis_path(self):
        """Genesis prompts have no parent and no iteration-template block —
        population signal and exploration directions must not leak in."""
        registry = load_registry()
        sys_msg, user_msg = build_operator_prompt(
            "collision",
            registry=registry,
            is_initial=True,
            population_context=render_population_context(_sample_population_brief()),
            exploration_directions=render_exploration_directions(
                _sample_population_brief()
            ),
        )
        assert "# Population signal" not in user_msg
        assert "Exploration directions" not in user_msg


class TestRunnerPopulationBriefLifecycle:
    """The runner stores both rendered blocks on both itself and the
    PromptSampler end-of-gen, so the next generation's mutations pick them up.
    """

    @pytest.mark.asyncio
    async def test_compute_population_brief_stores_on_runner_and_sampler(
        self, monkeypatch
    ):
        from owtn import runner as runner_mod

        async def fake_compute(**kwargs):
            return ("FAKE CONTEXT BLOCK", "FAKE DIRECTIONS BLOCK")

        monkeypatch.setattr(
            "owtn.optimizer.adapters.compute_stage_1_population_brief",
            fake_compute,
        )

        fake_runner = SimpleNamespace(
            stage_config=SimpleNamespace(
                llm=SimpleNamespace(run_brief_model="fake-model"),
                judges=SimpleNamespace(panel=["gwern", "wahls"]),
            ),
            db=object(),
            prompt_sampler=SimpleNamespace(
                population_context="", exploration_directions=""
            ),
            _latest_population_context=None,
            _latest_exploration_directions=None,
        )

        await runner_mod.ConceptEvolutionRunner._compute_population_brief(
            fake_runner
        )
        assert fake_runner._latest_population_context == "FAKE CONTEXT BLOCK"
        assert (
            fake_runner._latest_exploration_directions == "FAKE DIRECTIONS BLOCK"
        )
        assert fake_runner.prompt_sampler.population_context == "FAKE CONTEXT BLOCK"
        assert (
            fake_runner.prompt_sampler.exploration_directions
            == "FAKE DIRECTIONS BLOCK"
        )

    @pytest.mark.asyncio
    async def test_compute_skipped_when_run_brief_model_none(self, monkeypatch):
        from owtn import runner as runner_mod

        called = {"n": 0}

        async def fake_compute(**kwargs):
            called["n"] += 1
            return ("X", "Y")

        monkeypatch.setattr(
            "owtn.optimizer.adapters.compute_stage_1_population_brief",
            fake_compute,
        )

        fake_runner = SimpleNamespace(
            stage_config=SimpleNamespace(
                llm=SimpleNamespace(run_brief_model=None),
                judges=SimpleNamespace(panel=["gwern"]),
            ),
            db=object(),
            prompt_sampler=SimpleNamespace(
                population_context="", exploration_directions=""
            ),
            _latest_population_context=None,
            _latest_exploration_directions=None,
        )
        await runner_mod.ConceptEvolutionRunner._compute_population_brief(
            fake_runner
        )
        assert called["n"] == 0
        assert fake_runner._latest_population_context is None
        assert fake_runner._latest_exploration_directions is None
        assert fake_runner.prompt_sampler.population_context == ""
        assert fake_runner.prompt_sampler.exploration_directions == ""
