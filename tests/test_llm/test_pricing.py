"""Tests for pricing lookup and cost calculation."""

from owtn.llm.providers.pricing import (
    calculate_cost,
    get_provider,
    is_reasoning_model,
    model_exists,
)


class TestModelExists:
    def test_known_model(self):
        assert model_exists("claude-4-sonnet-20250514")

    def test_unknown_model(self):
        assert not model_exists("nonexistent-model-xyz")


class TestGetProvider:
    def test_anthropic_model(self):
        assert get_provider("claude-4-sonnet-20250514") == "anthropic"

    def test_openai_model(self):
        provider = get_provider("gpt-4.1")
        assert provider == "openai"


class TestCalculateCost:
    def test_returns_two_floats(self):
        input_cost, output_cost = calculate_cost(
            "claude-4-sonnet-20250514", 1000, 500
        )
        assert isinstance(input_cost, float)
        assert isinstance(output_cost, float)
        assert input_cost >= 0
        assert output_cost >= 0

    def test_more_tokens_higher_cost(self):
        cost_small_in, cost_small_out = calculate_cost(
            "claude-4-sonnet-20250514", 100, 50
        )
        cost_large_in, cost_large_out = calculate_cost(
            "claude-4-sonnet-20250514", 10000, 5000
        )
        assert cost_large_in > cost_small_in
        assert cost_large_out > cost_small_out

    def test_zero_tokens_zero_cost(self):
        input_cost, output_cost = calculate_cost(
            "claude-4-sonnet-20250514", 0, 0
        )
        assert input_cost == 0.0
        assert output_cost == 0.0


class TestIsReasoningModel:
    def test_non_reasoning_model(self):
        # deepseek-chat is the canonical non-reasoning model in pricing.csv.
        # (claude-4-sonnet is a reasoning model with extended thinking.)
        assert not is_reasoning_model("deepseek-chat")

    def test_reasoning_model(self):
        assert is_reasoning_model("claude-4-sonnet-20250514")
