"""Unit tests for QueryResult serialization and display."""

from owtn.llm.providers.result import QueryResult


def _make_result(**overrides):
    defaults = dict(
        content="4",
        msg="What is 2+2?",
        system_msg="You are helpful.",
        new_msg_history=[
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        model_name="test-model",
        kwargs={},
        input_tokens=10,
        output_tokens=5,
        cost=0.001,
    )
    defaults.update(overrides)
    return QueryResult(**defaults)


class TestToDict:
    def test_roundtrip_fields(self):
        result = _make_result()
        d = result.to_dict()
        assert d["content"] == "4"
        assert d["model_name"] == "test-model"
        assert d["input_tokens"] == 10
        assert d["cache_read_tokens"] == 0
        assert d["cache_creation_tokens"] == 0

    def test_cache_tokens_present(self):
        result = _make_result(cache_read_tokens=50, cache_creation_tokens=20)
        d = result.to_dict()
        assert d["cache_read_tokens"] == 50
        assert d["cache_creation_tokens"] == 20


class TestStrRepresentation:
    def test_contains_model_and_cost(self):
        result = _make_result(model_name="claude-haiku-4-5-20251001")
        s = str(result)
        assert "Model:" in s
        assert "Total Cost:" in s
        assert "claude-haiku-4-5-20251001" in s


class TestMsgHistory:
    def test_history_preserved(self):
        result = _make_result()
        assert len(result.new_msg_history) >= 2
        assert result.new_msg_history[-1]["role"] == "assistant"
