"""Verify diagnostic yaml logging on structured-output validation failure.

Regression for issue 2026-04-30-llm-raw-payload-logging-on-validation-failure:
when a provider's structured-output call returns a 200 but the body fails
Pydantic validation, api.py must write a yaml capturing the raw payload,
tokens, cost, and the validation error before the exception propagates.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import BaseModel, Field

from owtn.llm.api import query
from owtn.llm.call_logger import llm_log_dir
from owtn.llm.errors import LLMValidationError


class StrictWidget(BaseModel):
    """A schema with constraints that arbitrary tool inputs won't satisfy."""
    widget_id: int  # mock will return a string here → ValidationError
    name: str = Field(min_length=1)


def _build_mock_anthropic_response(tool_input: dict) -> MagicMock:
    """A fake Anthropic response with a tool_use block carrying the given
    input dict. Token/cost fields populated so the LLMValidationError
    captures the same shape a real response would.
    """
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = StrictWidget.__name__
    tool_block.input = tool_input

    response = MagicMock()
    response.content = [tool_block]
    response.usage.input_tokens = 1234
    response.usage.output_tokens = 56
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = 0
    return response


@patch("owtn.llm.providers.anthropic.calculate_cost", return_value=(0.005, 0.001))
def test_validation_failure_writes_diagnostic_yaml(_mock_cost, tmp_path, monkeypatch):
    """A 200-OK structured-output call whose payload fails validation
    should: (a) raise LLMValidationError, (b) write a yaml under
    <log_dir>/llm/<model>/ containing status=validation_failed, the raw
    payload as `output:`, the original validation error in `error:`, and
    the token/cost telemetry from the call that already happened."""
    log_dir = tmp_path / "run_test"
    token = llm_log_dir.set(str(log_dir))
    try:
        bad_payload = {"widget_id": "not-an-int", "name": "ok"}
        mock_response = _build_mock_anthropic_response(bad_payload)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("owtn.llm.providers.anthropic.AnthropicProvider._sync",
                   return_value=mock_client):
            with pytest.raises(LLMValidationError) as exc_info:
                query(
                    model_name="claude-sonnet-4-6",
                    msg="give me a widget",
                    system_msg="be a widget factory",
                    output_model=StrictWidget,
                )

        err = exc_info.value
        assert "widget_id" in str(err.cause)
        assert err.input_tokens == 1234
        assert err.output_tokens == 56
        assert err.raw_output  # the JSON of bad_payload

        # A yaml landed under llm/<model>/.
        model_dir = log_dir / "llm" / "claude-sonnet-4-6"
        assert model_dir.exists()
        yaml_files = list(model_dir.glob("*.yaml"))
        assert len(yaml_files) == 1, f"expected 1 yaml, got {yaml_files}"

        record = yaml.safe_load(yaml_files[0].read_text())
        assert record["status"] == "validation_failed"
        assert record["model"] == "claude-sonnet-4-6"
        assert record["provider"] == "anthropic"
        assert record["tokens"]["input"] == 1234
        assert record["tokens"]["output"] == 56
        assert record["cost"] > 0  # billed even though parsing failed
        assert "widget_id" in record["error"]
        # Raw payload preserved as output.
        assert "not-an-int" in record["output"]
        # Prompts captured for repro.
        assert record["system_msg"] == "be a widget factory"
        assert record["user_msg"] == "give me a widget"
    finally:
        llm_log_dir.reset(token)


@patch("owtn.llm.providers.anthropic.calculate_cost", return_value=(0.005, 0.001))
def test_successful_structured_call_logs_raw_output(_mock_cost, tmp_path):
    """On success, the yaml's `output:` field should carry the raw JSON the
    model emitted, not the parsed-then-stringified Pydantic repr."""
    log_dir = tmp_path / "run_success"
    token = llm_log_dir.set(str(log_dir))
    try:
        good_payload = {"widget_id": 42, "name": "spinner"}
        mock_response = _build_mock_anthropic_response(good_payload)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("owtn.llm.providers.anthropic.AnthropicProvider._sync",
                   return_value=mock_client):
            result = query(
                model_name="claude-sonnet-4-6",
                msg="give me a widget",
                system_msg="be a widget factory",
                output_model=StrictWidget,
            )
        assert result.content.widget_id == 42
        assert "42" in result.raw_output
        assert "spinner" in result.raw_output

        yaml_files = list((log_dir / "llm" / "claude-sonnet-4-6").glob("*.yaml"))
        assert len(yaml_files) == 1
        record = yaml.safe_load(yaml_files[0].read_text())
        # output: is the raw JSON, not the Pydantic repr.
        assert "42" in record["output"]
        assert "spinner" in record["output"]
        assert "WidgetGenome" not in record["output"]  # no class repr
    finally:
        llm_log_dir.reset(token)


@patch("owtn.llm.providers.anthropic.calculate_cost", return_value=(0.005, 0.001))
def test_missing_tool_use_block_raises_validation_error_with_telemetry(
    _mock_cost, tmp_path,
):
    """A response with no tool_use block at all should also produce a
    diagnostic yaml — same logging path, different cause class."""
    log_dir = tmp_path / "run_missing"
    token = llm_log_dir.set(str(log_dir))
    try:
        # Response with only a text block, no tool_use.
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I refuse to use the tool"
        response = MagicMock()
        response.content = [text_block]
        response.usage.input_tokens = 100
        response.usage.output_tokens = 10
        response.usage.cache_read_input_tokens = 0
        response.usage.cache_creation_input_tokens = 0

        mock_client = MagicMock()
        mock_client.messages.create.return_value = response

        with patch("owtn.llm.providers.anthropic.AnthropicProvider._sync",
                   return_value=mock_client):
            with pytest.raises(LLMValidationError):
                query(
                    model_name="claude-sonnet-4-6",
                    msg="x",
                    system_msg="y",
                    output_model=StrictWidget,
                )

        yaml_files = list((log_dir / "llm" / "claude-sonnet-4-6").glob("*.yaml"))
        assert len(yaml_files) == 1
        record = yaml.safe_load(yaml_files[0].read_text())
        assert record["status"] == "validation_failed"
        assert "missing tool_use" in record["error"]
        assert record["tokens"]["input"] == 100
    finally:
        llm_log_dir.reset(token)
