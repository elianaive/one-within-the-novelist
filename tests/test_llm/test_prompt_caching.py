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

    def test_without_prefix_returns_string(self):
        result = _build_system("just a string", None)
        assert result == "just a string"

    def test_empty_string_prefix_returns_string(self):
        result = _build_system("msg", "")
        assert result == "msg"


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
    def test_no_prefix_passes_string(self, mock_cost):
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
        assert isinstance(system_arg, str)
        assert system_arg == "system message"
