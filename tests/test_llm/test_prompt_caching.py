"""Tests for Anthropic prompt caching (cache_control content blocks)."""

from unittest.mock import MagicMock, patch

from owtn.llm.providers.anthropic import (
    _build_system,
    get_anthropic_costs,
    query_anthropic,
)


class TestBuildSystem:
    def test_with_prefix_returns_content_blocks(self):
        result = _build_system("suffix msg", "cached prefix")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {
            "type": "text",
            "text": "cached prefix",
            "cache_control": {"type": "ephemeral"},
        }
        assert result[1] == {"type": "text", "text": "suffix msg"}

    def test_without_prefix_wraps_with_cache_control(self):
        """No system_prefix: system_msg itself is the cached block. Engages
        prompt caching on repeat calls in the production code path."""
        result = _build_system("just a string", None)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == {
            "type": "text",
            "text": "just a string",
            "cache_control": {"type": "ephemeral"},
        }

    def test_empty_string_prefix_wraps_with_cache_control(self):
        """Empty-string prefix is treated like None — system_msg becomes
        the cached block."""
        result = _build_system("msg", "")
        assert isinstance(result, list)
        assert result[0]["text"] == "msg"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_empty_system_msg_no_prefix_returns_empty_string(self):
        """Truly empty system: pass through as the empty string rather than
        a content-block wrapper."""
        assert _build_system("", None) == ""


class TestGetAnthropicCosts:
    def test_includes_cache_tokens(self):
        response = MagicMock()
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50
        response.usage.cache_read_input_tokens = 80
        response.usage.cache_creation_input_tokens = 20

        with patch(
            "owtn.llm.providers.anthropic.calculate_cost", return_value=(0.01, 0.02)
        ):
            costs = get_anthropic_costs(response, "claude-sonnet-4-20250514")

        assert costs["cache_read_tokens"] == 80
        assert costs["cache_creation_tokens"] == 20
        assert costs["input_tokens"] == 100
        assert costs["output_tokens"] == 50

    def test_missing_cache_fields_default_to_zero(self):
        response = MagicMock(spec=[])
        response.usage = MagicMock(spec=[])
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50

        with patch(
            "owtn.llm.providers.anthropic.calculate_cost", return_value=(0.01, 0.02)
        ):
            costs = get_anthropic_costs(response, "model")

        assert costs["cache_read_tokens"] == 0
        assert costs["cache_creation_tokens"] == 0


class TestQueryAnthropicSystemPrefix:
    @patch("owtn.llm.providers.anthropic.calculate_cost", return_value=(0.01, 0.02))
    def test_system_prefix_passed_as_content_blocks(self, mock_cost):
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response text")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 8
        client.messages.create.return_value = mock_response

        query_anthropic(
            client=client,
            model="test-model",
            msg="hello",
            system_msg="per-call suffix",
            msg_history=[],
            output_model=None,
            system_prefix="cached prefix",
        )

        call_kwargs = client.messages.create.call_args
        system_arg = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        assert isinstance(system_arg, list)
        assert system_arg[0]["cache_control"] == {"type": "ephemeral"}
        assert system_arg[0]["text"] == "cached prefix"
        assert system_arg[1]["text"] == "per-call suffix"

    @patch("owtn.llm.providers.anthropic.calculate_cost", return_value=(0.01, 0.02))
    def test_no_prefix_wraps_system_msg_with_cache_control(self, mock_cost):
        """Without system_prefix, system_msg is wrapped as a single
        cache_control content block — caching engages on repeat calls."""
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        client.messages.create.return_value = mock_response

        query_anthropic(
            client=client,
            model="test-model",
            msg="hello",
            system_msg="system message",
            msg_history=[],
            output_model=None,
        )

        call_kwargs = client.messages.create.call_args
        system_arg = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        assert isinstance(system_arg, list)
        assert len(system_arg) == 1
        assert system_arg[0] == {
            "type": "text",
            "text": "system message",
            "cache_control": {"type": "ephemeral"},
        }


class TestToolDefinitionCaching:
    """The last tool entry carries cache_control so the tools array joins
    the cached prefix. Adds a 3rd cache breakpoint per request — calls that
    share system+tools but have different user_msg still hit the tools cache.
    """

    @patch("owtn.llm.providers.anthropic.calculate_cost", return_value=(0.01, 0.02))
    def test_structured_output_tool_carries_cache_control(self, _mock_cost):
        """Forced-tool-use structured-output path (judges, picker, casting)
        marks cache_control on the schema tool."""
        from pydantic import BaseModel
        from owtn.llm.providers.anthropic import AnthropicProvider

        class Verdict(BaseModel):
            sentiment: str

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "Verdict"
        tool_block.input = {"sentiment": "ok"}
        response = MagicMock()
        response.content = [tool_block]
        response.usage.input_tokens = 1
        response.usage.output_tokens = 1
        response.usage.cache_read_input_tokens = 0
        response.usage.cache_creation_input_tokens = 0
        client = MagicMock()
        client.messages.create.return_value = response

        AnthropicProvider().query(
            model="claude-sonnet-4-6", msg="x", system_msg="sys",
            msg_history=[], system_prefix=None,
            output_model=Verdict, kwargs={"max_tokens": 100},
            client=client,
        )
        tools_arg = client.messages.create.call_args.kwargs["tools"]
        assert len(tools_arg) == 1
        assert tools_arg[0]["cache_control"] == {"type": "ephemeral"}
        assert tools_arg[0]["name"] == "Verdict"

    @patch("owtn.llm.providers.anthropic.calculate_cost", return_value=(0.01, 0.02))
    def test_tool_use_loop_marks_last_tool_with_cache_control(self, _mock_cost):
        """Multi-tool agent path (Stage 3 phases, Stage 4 _explore/critics/
        surgical_edit) marks cache_control on the LAST tool definition only."""
        import asyncio
        from owtn.llm.providers.anthropic import AnthropicProvider

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "done"
        response = MagicMock()
        response.content = [text_block]
        response.usage.input_tokens = 10
        response.usage.output_tokens = 5
        response.usage.cache_read_input_tokens = 0
        response.usage.cache_creation_input_tokens = 0

        async def fake_create(**_kw):
            return response

        client = MagicMock()
        client.messages.create = fake_create

        async def dispatch(_name, _params):
            return "noop"

        tools = [
            {"name": "alpha", "description": "first", "input_schema": {"type": "object"}},
            {"name": "beta", "description": "second", "input_schema": {"type": "object"}},
            {"name": "gamma", "description": "third", "input_schema": {"type": "object"}},
        ]

        # Capture the kwargs that messages.create was called with by patching.
        captured: dict = {}

        async def capture_create(**kw):
            captured.update(kw)
            return response

        client.messages.create = capture_create

        asyncio.run(AnthropicProvider().query_async_with_tools(
            model="claude-sonnet-4-6", msg="m", system_msg="s",
            msg_history=[], system_prefix=None,
            tools=tools, dispatch=dispatch, max_iters=1,
            kwargs={"max_tokens": 100},
            client=client,
        ))

        tools_arg = captured["tools"]
        assert len(tools_arg) == 3
        # Only the last tool carries cache_control.
        assert "cache_control" not in tools_arg[0]
        assert "cache_control" not in tools_arg[1]
        assert tools_arg[2]["cache_control"] == {"type": "ephemeral"}
        assert tools_arg[2]["name"] == "gamma"
