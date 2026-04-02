"""Tests for _query_cache_key determinism and differentiation."""

from owtn.llm.query import _query_cache_key


class TestCacheKeyDeterminism:
    def test_same_inputs_same_key(self):
        args = ("claude-sonnet-4-20250514", "hello", "you are helpful", [])
        k1 = _query_cache_key(args, {})
        k2 = _query_cache_key(args, {})
        assert k1 == k2

    def test_different_model_different_key(self):
        k1 = _query_cache_key(("model-a", "msg", "sys", []), {})
        k2 = _query_cache_key(("model-b", "msg", "sys", []), {})
        assert k1 != k2

    def test_different_msg_different_key(self):
        k1 = _query_cache_key(("model", "msg1", "sys", []), {})
        k2 = _query_cache_key(("model", "msg2", "sys", []), {})
        assert k1 != k2

    def test_different_system_msg_different_key(self):
        k1 = _query_cache_key(("model", "msg", "sys1", []), {})
        k2 = _query_cache_key(("model", "msg", "sys2", []), {})
        assert k1 != k2

    def test_system_prefix_affects_key(self):
        k1 = _query_cache_key(("model", "msg", "sys", []), {})
        k2 = _query_cache_key(("model", "msg", "sys", []), {"system_prefix": "cached"})
        assert k1 != k2

    def test_msg_history_affects_key(self):
        k1 = _query_cache_key(("model", "msg", "sys", []), {})
        k2 = _query_cache_key(
            ("model", "msg", "sys", [{"role": "user", "content": "prior"}]), {}
        )
        assert k1 != k2

    def test_temperature_affects_key(self):
        k1 = _query_cache_key(("model", "msg", "sys", []), {"temperature": 0.0})
        k2 = _query_cache_key(("model", "msg", "sys", []), {"temperature": 1.0})
        assert k1 != k2

    def test_kwargs_fallback(self):
        """When args are empty, kwargs are used for key fields."""
        k1 = _query_cache_key((), {"model_name": "m", "msg": "hi", "system_msg": "s"})
        k2 = _query_cache_key((), {"model_name": "m", "msg": "hi", "system_msg": "s"})
        assert k1 == k2

    def test_returns_hex_string(self):
        key = _query_cache_key(("model", "msg", "sys", []), {})
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex
        int(key, 16)  # valid hex
