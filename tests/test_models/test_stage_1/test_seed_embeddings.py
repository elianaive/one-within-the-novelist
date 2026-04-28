"""Cache round-trip + invalidation tests for seed_embeddings.

The model itself is heavy and not fast to instantiate; we mock
`compute_embeddings` here and just exercise the cache plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from owtn.models.stage_1 import seed_embeddings as se
from owtn.models.stage_1.seed_bank import Seed


def _seeds(n: int = 3) -> list[Seed]:
    return [
        Seed(id=f"s{i}", type="real_world", content=f"content-{i}", tags=[])
        for i in range(n)
    ]


def _stub_compute(seeds: list[Seed]) -> dict[str, list[float]]:
    """Deterministic stub: index-based unit vector so we can assert exact values."""
    return {s.id: [float(i), 0.0, 0.0] for i, s in enumerate(seeds)}


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "seed-bank-embeddings.jsonl"


class TestCacheRoundTrip:
    def test_cold_cache_computes_and_writes(self, monkeypatch, cache_path: Path):
        seeds = _seeds(3)
        monkeypatch.setattr(se, "compute_embeddings", _stub_compute)
        out = se.load_or_compute(seeds, cache_path)
        assert set(out) == {"s0", "s1", "s2"}
        assert cache_path.exists()
        # JSONL format: meta line + one line per seed
        lines = cache_path.read_text().splitlines()
        assert len(lines) == 4
        meta = json.loads(lines[0])["_meta"]
        assert meta["model"] == se.MODEL_ID
        assert meta["instruction"] == se.INSTRUCTION
        for line in lines[1:]:
            entry = json.loads(line)
            assert {"seed_id", "hash", "vector"} <= set(entry)

    def test_warm_cache_does_not_recompute(self, monkeypatch, cache_path: Path):
        seeds = _seeds(3)
        monkeypatch.setattr(se, "compute_embeddings", _stub_compute)
        # Prime the cache.
        se.load_or_compute(seeds, cache_path)
        # Second call: every seed is fresh; compute_embeddings must not run.
        def _explode(_seeds):
            raise AssertionError("compute_embeddings called on warm cache")
        monkeypatch.setattr(se, "compute_embeddings", _explode)
        out = se.load_or_compute(seeds, cache_path)
        assert set(out) == {"s0", "s1", "s2"}

    def test_partial_invalidation_only_recomputes_changed(self, monkeypatch, cache_path: Path):
        seeds = _seeds(3)
        monkeypatch.setattr(se, "compute_embeddings", _stub_compute)
        se.load_or_compute(seeds, cache_path)
        # Edit s1's content; s0 and s2 stay valid.
        seeds[1] = Seed(id="s1", type="real_world", content="edited", tags=[])
        called_with: list[Seed] = []

        def _track(input_seeds):
            called_with.extend(input_seeds)
            return {s.id: [99.0, 99.0, 99.0] for s in input_seeds}

        monkeypatch.setattr(se, "compute_embeddings", _track)
        out = se.load_or_compute(seeds, cache_path)
        assert [s.id for s in called_with] == ["s1"]
        assert out["s1"] == [99.0, 99.0, 99.0]
        # The unchanged entries should keep their original (stub-computed) vectors.
        assert out["s0"] == [0.0, 0.0, 0.0]
        assert out["s2"] == [2.0, 0.0, 0.0]


class TestInvalidation:
    def test_meta_mismatch_rewrites_full_cache(self, monkeypatch, cache_path: Path):
        # Write a cache that looks like it was made with a different instruction.
        cache_path.write_text(
            json.dumps({"_meta": {"model": "old-model", "instruction": "old"}}) + "\n"
            + json.dumps({"seed_id": "s0", "hash": "stale", "vector": [-1.0]}) + "\n"
        )
        seeds = _seeds(2)
        called_ids: list[str] = []

        def _track(input_seeds):
            called_ids.extend(s.id for s in input_seeds)
            return {s.id: [42.0] for s in input_seeds}

        monkeypatch.setattr(se, "compute_embeddings", _track)
        out = se.load_or_compute(seeds, cache_path)
        # Both seeds got recomputed (stale entry didn't survive meta mismatch).
        assert sorted(called_ids) == ["s0", "s1"]
        # And the cache was rewritten with the current meta.
        first_line = cache_path.read_text().splitlines()[0]
        assert json.loads(first_line)["_meta"]["model"] == se.MODEL_ID
        assert out["s0"] == [42.0]

    def test_corrupt_rows_get_recomputed(self, monkeypatch, cache_path: Path):
        # Valid meta, but s0's row is corrupt and s1 is missing.
        cache_path.write_text(
            json.dumps({"_meta": {"model": se.MODEL_ID, "instruction": se.INSTRUCTION}}) + "\n"
            + "this is not json\n"
        )
        seeds = _seeds(2)
        monkeypatch.setattr(se, "compute_embeddings", _stub_compute)
        out = se.load_or_compute(seeds, cache_path)
        # Both got recomputed since neither had a valid cached entry.
        assert set(out) == {"s0", "s1"}
        # Cache now well-formed.
        lines = cache_path.read_text().splitlines()
        assert len(lines) == 3  # meta + 2 seeds
        for ln in lines[1:]:
            json.loads(ln)  # should not raise
