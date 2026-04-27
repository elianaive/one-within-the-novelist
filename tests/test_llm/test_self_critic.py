"""Contract tests for the self-critic critique-revise cycle.

These tests verify call sequencing + message-history shape, not output
content. The real cycle fires only for registered models on generation
roles.

See lab/issues/2026-04-24-self-critic-critique-revise.md.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest

from owtn.llm import query as query_mod
from owtn.llm.call_logger import llm_context


def _full_genome_output(label: str = "x") -> str:
    """Build a generator output that parses through apply_full_patch — a
    valid ``` ```json``` fence with a minimally-valid genome."""
    return (
        f"<NAME>\n{label}\n</NAME>\n"
        f'<CODE>\n```json\n{{"premise": "{label}"}}\n```\n</CODE>\n'
    )


@dataclass
class _FakeResult:
    """Minimal stand-in for QueryResult — _query_async_single's consumers
    read .content, .input_tokens, .output_tokens, etc. The wrapper logic
    only touches .content.
    """
    content: str = "OUTPUT"
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost: float = 0.0
    duration_s: float = 0.0
    thought: str = ""
    model_name: str = "claude-sonnet-4-6"


@pytest.fixture(autouse=True)
def _reset_self_critic_registry():
    """Each test starts with an empty self-critic registry and resets ctx."""
    original = dict(query_mod._self_critic_config)
    query_mod._self_critic_config.clear()
    token = llm_context.set({})
    yield
    llm_context.reset(token)
    query_mod._self_critic_config.clear()
    query_mod._self_critic_config.update(original)


class TestRegistration:
    def test_register_populates_config(self):
        query_mod.register_self_critic_models({"claude-sonnet-4-6": "disabled", "gpt-4o": "low"})
        assert query_mod._self_critic_config == {
            "claude-sonnet-4-6": "disabled",
            "gpt-4o": "low",
        }

    def test_register_replaces_prior_config(self):
        query_mod.register_self_critic_models({"model-a": "disabled"})
        query_mod.register_self_critic_models({"model-b": "low"})
        assert query_mod._self_critic_config == {"model-b": "low"}


class TestSelfCriticGating:
    """The cycle fires only when (role is a generation role) AND (model is
    registered). Anything else passes through with a single call.
    """

    @pytest.fixture
    def fake_single(self):
        """Patch _query_async_single so we can count calls + their args."""
        calls: list[dict] = []

        async def fake(**kwargs):
            calls.append(kwargs)
            return _FakeResult(content=f"OUTPUT-{len(calls)}")

        with patch.object(query_mod, "_query_async_single", side_effect=fake) as m:
            yield m, calls

    def _run(self, **kw):
        return asyncio.get_event_loop().run_until_complete(
            query_mod.query_async(**kw)
        )

    def test_non_generation_role_no_cycle(self, fake_single):
        query_mod.register_self_critic_models({"model-a": "disabled"})
        llm_context.set({"role": "pairwise_judge"})
        result = asyncio.run(
            query_mod.query_async(
                model_name="model-a",
                msg="user",
                system_msg="sys",
            )
        )
        _, calls = fake_single
        assert len(calls) == 1
        assert result.content == "OUTPUT-1"

    def test_unregistered_model_no_cycle(self, fake_single):
        # registry empty
        llm_context.set({"role": "generation"})
        result = asyncio.run(
            query_mod.query_async(
                model_name="unregistered",
                msg="user",
                system_msg="sys",
            )
        )
        _, calls = fake_single
        assert len(calls) == 1
        assert result.content == "OUTPUT-1"

    def test_generation_role_registered_fires_cycle(self, fake_single):
        """Initial output is a parseable full genome, so the cycle fires."""
        _, calls = fake_single
        calls.clear()

        async def fake(**kwargs):
            calls.append(kwargs)
            content = _full_genome_output("v1") if len(calls) == 1 else f"OUT-{len(calls)}"
            return _FakeResult(content=content)

        query_mod.register_self_critic_models({"model-a": "disabled"})
        llm_context.set({"role": "generation", "operator": "collision"})
        with patch.object(query_mod, "_query_async_single", side_effect=fake):
            result = asyncio.run(
                query_mod.query_async(
                    model_name="model-a", msg="user", system_msg="sys"
                )
            )

        # 3 calls: initial + critic + revise
        assert len(calls) == 3

        # Call 1: initial — user's msg, user's system
        assert calls[0]["msg"] == "user"
        assert calls[0]["system_msg"] == "sys"

        # Call 2: critic — system is the critic prompt; user-msg is the
        # extracted JSON genome. output_model=None.
        assert "Can you provide actionable feedback?" in calls[1]["system_msg"]
        assert calls[1]["msg"] == '{"premise": "v1"}'
        assert calls[1]["output_model"] is None

        # Call 3: revise — user's original system, history seeded with the
        # original exchange (FULL original output, not the extracted JSON),
        # new user msg begins with the revise preamble loaded from disk.
        assert calls[2]["system_msg"] == "sys"
        revise_preamble = query_mod._load_self_critic_prompt(
            "self_critic_revise.txt"
        ).strip()
        assert calls[2]["msg"].startswith(revise_preamble)
        assert calls[2]["msg_history"] == [
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": _full_genome_output("v1")},
        ]
        assert result.content == "OUT-3"

    def test_full_operator_critic_sees_extracted_genome(self):
        """For full-routing operators, the critic receives the JSON content
        of the ```json``` fence — same extraction as apply_full_patch."""
        full_output = (
            "## Review\nLong preamble.\n\n"
            "<NAME>\nthe_name\n</NAME>\n\n"
            "<CODE>\n```json\n{\"premise\": \"x\"}\n```\n</CODE>\n"
        )
        calls: list[dict] = []

        async def capture(**kwargs):
            calls.append(kwargs)
            return _FakeResult(content=full_output if len(calls) == 1 else "OK")

        query_mod.register_self_critic_models({"model-a": "disabled"})
        # operator=collision is full-routing in OPERATOR_DEFS
        llm_context.set({"role": "genesis", "operator": "collision"})
        with patch.object(query_mod, "_query_async_single", side_effect=capture):
            asyncio.run(
                query_mod.query_async(
                    model_name="model-a", msg="user", system_msg="sys"
                )
            )
        assert calls[1]["msg"] == '{"premise": "x"}'

    def test_diff_operator_critic_sees_applied_genome(self):
        """For diff-routing operators (inversion), the critic receives the
        full parent genome with all SEARCH/REPLACE blocks applied — the
        same JSON that becomes Program.code via apply_diff_patch.

        Uses a multi-block diff and a realistic-shape genome. Asserts the
        result is exactly equal to the parent with the targeted edits, and
        is still valid JSON.
        """
        import json
        parent_obj = {
            "premise": "old premise",
            "thematic_engine": "old engine",
            "target_effect": "stays",
            "anchor_scene": {"sketch": "old sketch", "role": "climax"},
        }
        parent = json.dumps(parent_obj, indent=2)
        # Two-block diff: rewrite the premise and the anchor sketch.
        diff_output = (
            "<<<<<<< SEARCH\n"
            '"premise": "old premise",\n'
            "=======\n"
            '"premise": "new premise",\n'
            ">>>>>>> REPLACE\n"
            "\n"
            "<<<<<<< SEARCH\n"
            '"sketch": "old sketch",\n'
            "=======\n"
            '"sketch": "new sketch",\n'
            ">>>>>>> REPLACE\n"
        )
        calls: list[dict] = []

        async def capture(**kwargs):
            calls.append(kwargs)
            return _FakeResult(content=diff_output if len(calls) == 1 else "OK")

        query_mod.register_self_critic_models({"model-a": "disabled"})
        llm_context.set({
            "role": "mutation",
            "operator": "inversion",  # diff-routed per OPERATOR_DEFS
            "parent_code": parent,
        })
        with patch.object(query_mod, "_query_async_single", side_effect=capture):
            asyncio.run(
                query_mod.query_async(
                    model_name="model-a", msg="user", system_msg="sys"
                )
            )

        # Critic call: msg is the parent with both diff blocks applied.
        # Re-parse and compare structurally so the test isn't fragile to
        # apply_diff_patch's whitespace/newline policy.
        critic_msg = calls[1]["msg"]
        applied = json.loads(critic_msg)
        assert applied == {
            "premise": "new premise",
            "thematic_engine": "old engine",
            "target_effect": "stays",
            "anchor_scene": {"sketch": "new sketch", "role": "climax"},
        }
        # And the critic system prompt is the critic — confirms we're in
        # the right call slot.
        assert "Can you provide actionable feedback?" in calls[1]["system_msg"]

    def test_diff_operator_skips_cycle_when_search_does_not_match(self):
        """If a SEARCH block doesn't match the parent, apply_diff_patch
        returns 0 patches applied. The pipeline will retry; self-critic
        should skip the cycle rather than show a malformed/empty critic
        message or fall back to raw content.
        """
        import json
        parent = json.dumps({"premise": "alpha"}, indent=2)
        diff_output = (
            "<<<<<<< SEARCH\n"
            '"premise": "this text does not exist in parent"\n'
            "=======\n"
            '"premise": "replacement"\n'
            ">>>>>>> REPLACE\n"
        )
        calls: list[dict] = []

        async def capture(**kwargs):
            calls.append(kwargs)
            return _FakeResult(content=diff_output if len(calls) == 1 else "OK")

        query_mod.register_self_critic_models({"model-a": "disabled"})
        llm_context.set({
            "role": "mutation",
            "operator": "inversion",
            "parent_code": parent,
        })
        with patch.object(query_mod, "_query_async_single", side_effect=capture):
            result = asyncio.run(
                query_mod.query_async(
                    model_name="model-a", msg="user", system_msg="sys"
                )
            )

        # Only the initial call fired — no critic, no revise.
        assert len(calls) == 1
        # Original result is returned unchanged.
        assert result.content == diff_output

    def test_diff_operator_skips_cycle_when_parent_code_missing(self):
        """A diff-routed call without parent_code in the context can't be
        applied. Skip the cycle rather than fall back to raw content."""
        diff_output = (
            "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE\n"
        )
        calls: list[dict] = []

        async def capture(**kwargs):
            calls.append(kwargs)
            return _FakeResult(content=diff_output if len(calls) == 1 else "OK")

        query_mod.register_self_critic_models({"model-a": "disabled"})
        # Note: no parent_code in context.
        llm_context.set({"role": "mutation", "operator": "inversion"})
        with patch.object(query_mod, "_query_async_single", side_effect=capture):
            asyncio.run(
                query_mod.query_async(
                    model_name="model-a", msg="user", system_msg="sys"
                )
            )

        assert len(calls) == 1

    def test_critic_cycle_skipped_when_genome_extraction_fails(self):
        """Malformed output (no fence, no SEARCH/REPLACE) means the pipeline
        will retry. Self-critic should not waste a critique on garbage —
        skip the cycle, return the original result."""
        garbage = "no fence, no markers, just prose"
        calls: list[dict] = []

        async def capture(**kwargs):
            calls.append(kwargs)
            return _FakeResult(content=garbage if len(calls) == 1 else "OK")

        query_mod.register_self_critic_models({"model-a": "disabled"})
        llm_context.set({"role": "genesis", "operator": "collision"})
        with patch.object(query_mod, "_query_async_single", side_effect=capture):
            result = asyncio.run(
                query_mod.query_async(
                    model_name="model-a", msg="user", system_msg="sys"
                )
            )
        # Only the initial call fired — no critic, no revise.
        assert len(calls) == 1
        # Original result is what comes back.
        assert result.content == garbage

    def test_cycle_does_not_recurse(self, fake_single):
        """The critic/revise sub-calls go through _query_async_single directly
        and must NOT re-enter query_async (which would loop forever).
        """
        _, calls = fake_single
        calls.clear()

        async def fake(**kwargs):
            calls.append(kwargs)
            content = _full_genome_output("g") if len(calls) == 1 else "OK"
            return _FakeResult(content=content)

        query_mod.register_self_critic_models({"model-a": "disabled"})
        llm_context.set({"role": "genesis", "operator": "collision"})
        with patch.object(query_mod, "_query_async_single", side_effect=fake):
            asyncio.run(
                query_mod.query_async(
                    model_name="model-a", msg="user", system_msg="sys"
                )
            )
        assert len(calls) == 3

    def test_ctx_role_restored_after_cycle(self, fake_single):
        """Outer role must survive the sub-call context manipulation."""
        _, calls = fake_single
        calls.clear()

        async def fake(**kwargs):
            calls.append(kwargs)
            content = _full_genome_output("m") if len(calls) == 1 else "OK"
            return _FakeResult(content=content)

        query_mod.register_self_critic_models({"model-a": "disabled"})
        llm_context.set({
            "role": "mutation",
            "operator": "collision",
            "extra": "preserve-me",
        })
        with patch.object(query_mod, "_query_async_single", side_effect=fake):
            asyncio.run(
                query_mod.query_async(
                    model_name="model-a", msg="user", system_msg="sys"
                )
            )
        ctx = llm_context.get({})
        assert ctx.get("role") == "mutation"
        assert ctx.get("extra") == "preserve-me"

    def test_reasoning_kwargs_stripped_on_critic_only_when_disabled(self, fake_single):
        """When the model is registered with the default 'disabled' effort,
        thinking/reasoning kwargs are removed from the *critic* sub-call
        (critique IS the thinking) but the *revise* call keeps them — that
        call is the structured genome regeneration and benefits from
        extended reasoning. Initial call is the caller's, untouched."""
        _, calls = fake_single
        calls.clear()

        async def fake(**kwargs):
            calls.append(kwargs)
            content = _full_genome_output("v") if len(calls) == 1 else "OK"
            return _FakeResult(content=content)

        query_mod.register_self_critic_models({"model-a": "disabled"})
        llm_context.set({"role": "generation", "operator": "collision"})
        with patch.object(query_mod, "_query_async_single", side_effect=fake):
            asyncio.run(
                query_mod.query_async(
                    model_name="model-a",
                    msg="user",
                    system_msg="sys",
                    reasoning_effort="high",
                    thinking={"type": "enabled", "budget_tokens": 8000},
                    extra_body={"thinking": {"type": "enabled"}, "reasoning": {"enabled": True}},
                    temperature=1.0,
                )
            )
        assert len(calls) == 3
        # Initial keeps everything — caller's kwargs untouched.
        assert calls[0].get("reasoning_effort") == "high"
        assert "thinking" in calls[0]
        # Critic call: stripped.
        critic = calls[1]
        assert "reasoning_effort" not in critic
        assert "thinking" not in critic
        assert "extra_body" not in critic
        assert critic.get("temperature") == 1.0  # non-thinking kwargs survive
        # Revise call: thinking kept, same as the original generator call.
        revise = calls[2]
        assert revise.get("reasoning_effort") == "high"
        assert revise.get("thinking") == {"type": "enabled", "budget_tokens": 8000}
        assert revise.get("extra_body") == {
            "thinking": {"type": "enabled"}, "reasoning": {"enabled": True}
        }

    def test_reasoning_kwargs_preserved_on_critic_when_effort_not_disabled(self, fake_single):
        """Opting out of the default: if self_critic_reasoning_effort is
        anything other than 'disabled', the critic keeps the generator's
        kwargs too. Revise has always kept them."""
        _, calls = fake_single
        calls.clear()

        async def fake(**kwargs):
            calls.append(kwargs)
            content = _full_genome_output("v") if len(calls) == 1 else "OK"
            return _FakeResult(content=content)

        query_mod.register_self_critic_models({"model-a": "low"})
        llm_context.set({"role": "generation", "operator": "collision"})
        with patch.object(query_mod, "_query_async_single", side_effect=fake):
            asyncio.run(
                query_mod.query_async(
                    model_name="model-a",
                    msg="user",
                    system_msg="sys",
                    reasoning_effort="high",
                    thinking={"type": "enabled"},
                )
            )
        for sub in (calls[1], calls[2]):
            assert sub.get("reasoning_effort") == "high"
            assert sub.get("thinking") == {"type": "enabled"}

    def test_sub_call_roles_tagged_correctly(self, fake_single):
        """Critic and revise calls must be logged under the right role so
        stats_report surfaces them separately from the initial call."""
        seen_roles: list[str] = []

        async def capture(**kwargs):
            seen_roles.append(llm_context.get({}).get("role"))
            content = _full_genome_output("v") if len(seen_roles) == 1 else "OK"
            return _FakeResult(content=content)

        query_mod.register_self_critic_models({"model-a": "disabled"})
        llm_context.set({"role": "generation", "operator": "collision"})
        with patch.object(query_mod, "_query_async_single", side_effect=capture):
            asyncio.run(
                query_mod.query_async(
                    model_name="model-a",
                    msg="user",
                    system_msg="sys",
                )
            )
        assert seen_roles == [
            "generation",          # initial
            "self_critic_review",  # critic
            "self_critic_revise",  # revise
        ]
