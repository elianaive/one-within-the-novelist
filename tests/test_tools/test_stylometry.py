"""Tests for owtn.tools — stylometry tool surface and corpus loader."""

import json
import textwrap

import pytest
import yaml

from owtn.tools import _corpus as corpus_module
from owtn.tools._corpus import ReferenceCorpus, ReferenceEntry, load_corpus
from owtn.tools import StylometricToolReport, stylometry


# ─── Test corpus fixture (small synthetic) ────────────────────────────────

@pytest.fixture
def synthetic_corpus(tmp_path, monkeypatch):
    """Build a tiny on-disk corpus and patch the loader to point at it."""
    passages_dir = tmp_path / "passages"
    passages_dir.mkdir()

    samples = {
        "lit-1": (
            "The hills across the valley of the Ebro were long and white. "
            "On this side there was no shade and no trees. "
            "Inside the bar, a man and a woman were drinking beer. "
            "It was very hot. The express from Barcelona would come in forty minutes."
        ),
        "lit-2": (
            "Snow had begun to fall. He stood at the window. "
            "The world outside was white, and quiet, and seemed to be waiting for something."
        ),
        "llm-sonnet-1": (
            "Walking down the path, she felt the weight of everything she had been carrying. "
            "The morning air was crisp and clean. The first rays of sunlight filtered through the branches above. "
            "She smiled to herself. It was going to be a good day."
        ),
        "llm-sonnet-2": (
            "He opened the door slowly, almost reverently. The room was exactly as he remembered it. "
            "Dust motes danced in the afternoon light. Memories came flooding back."
        ),
        "expo-1": (
            "Photosynthesis converts light energy into chemical energy. "
            "The light-dependent reactions occur in the thylakoid membranes. "
            "The Calvin cycle then fixes carbon dioxide into organic molecules."
        ),
    }
    for name, content in samples.items():
        (passages_dir / f"{name}.txt").write_text(content)

    yaml_data = {
        "references": [
            {"id": "lit-1", "tags": ["literary", "minimalist"], "source": "test", "license": "test", "text_file": "passages/lit-1.txt"},
            {"id": "lit-2", "tags": ["literary", "lyric"], "source": "test", "license": "test", "text_file": "passages/lit-2.txt"},
            {"id": "llm-sonnet-1", "tags": ["llm_default", "model:sonnet-test"], "source": "test", "license": "test", "text_file": "passages/llm-sonnet-1.txt"},
            {"id": "llm-sonnet-2", "tags": ["llm_default", "model:sonnet-test"], "source": "test", "license": "test", "text_file": "passages/llm-sonnet-2.txt"},
            {"id": "expo-1", "tags": ["expository"], "source": "test", "license": "test", "text_file": "passages/expo-1.txt"},
        ]
    }
    yaml_path = tmp_path / "voice-references-stylometric.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_data))

    monkeypatch.setattr(corpus_module, "CORPUS_YAML", yaml_path)
    monkeypatch.setattr(corpus_module, "PASSAGES_DIR", tmp_path)
    monkeypatch.setattr(corpus_module, "CACHE_PATH", tmp_path / "cache.json")
    # Reset the module-level singleton so the next load_corpus call rebuilds
    monkeypatch.setattr(corpus_module, "_CORPUS_CACHE", None)

    return load_corpus(force_reload=True)


# ─── Corpus loader ────────────────────────────────────────────────────────

class TestCorpusLoader:
    def test_loads_all_entries(self, synthetic_corpus):
        ids = {e.id for e in synthetic_corpus.entries}
        assert ids == {"lit-1", "lit-2", "llm-sonnet-1", "llm-sonnet-2", "expo-1"}

    def test_entries_have_signals(self, synthetic_corpus):
        for entry in synthetic_corpus.entries:
            assert entry.signals.word_count > 0
            assert 0.0 <= entry.signals.mattr <= 1.0

    def test_by_tag_filters(self, synthetic_corpus):
        literary = synthetic_corpus.by_tag("literary")
        assert {e.id for e in literary} == {"lit-1", "lit-2"}
        llm = synthetic_corpus.by_tag("llm_default")
        assert {e.id for e in llm} == {"llm-sonnet-1", "llm-sonnet-2"}

    def test_by_model_tag(self, synthetic_corpus):
        sonnet = synthetic_corpus.by_model_tag("sonnet-test")
        assert {e.id for e in sonnet} == {"llm-sonnet-1", "llm-sonnet-2"}
        absent = synthetic_corpus.by_model_tag("nonexistent-model")
        assert absent == []

    def test_aggregate_signals(self, synthetic_corpus):
        agg = synthetic_corpus.aggregate_signals(synthetic_corpus.by_tag("literary"))
        assert agg["n_samples"] == 2
        assert "burstiness" in agg
        assert "mattr" in agg

    def test_missing_passage_skipped(self, tmp_path, monkeypatch):
        passages_dir = tmp_path / "passages"
        passages_dir.mkdir()
        (passages_dir / "exists.txt").write_text("Some prose here. More prose follows.")
        yaml_data = {
            "references": [
                {"id": "exists", "tags": ["literary"], "source": "test", "license": "test", "text_file": "passages/exists.txt"},
                {"id": "missing", "tags": ["literary"], "source": "test", "license": "test", "text_file": "passages/missing.txt"},
            ]
        }
        yaml_path = tmp_path / "voice-references-stylometric.yaml"
        yaml_path.write_text(yaml.safe_dump(yaml_data))
        monkeypatch.setattr(corpus_module, "CORPUS_YAML", yaml_path)
        monkeypatch.setattr(corpus_module, "PASSAGES_DIR", tmp_path)
        monkeypatch.setattr(corpus_module, "CACHE_PATH", tmp_path / "cache.json")
        monkeypatch.setattr(corpus_module, "_CORPUS_CACHE", None)
        c = load_corpus(force_reload=True)
        assert {e.id for e in c.entries} == {"exists"}


# ─── Cache invalidation ───────────────────────────────────────────────────

class TestCacheInvalidation:
    def test_cache_persists_between_loads(self, synthetic_corpus, tmp_path, monkeypatch):
        cache_path = tmp_path / "cache.json"
        assert cache_path.exists()
        cache = json.loads(cache_path.read_text())
        assert cache["version"] == corpus_module.CACHE_VERSION
        assert "lit-1" in cache["passages"]
        # Re-load and verify hash-equal entries are not recomputed (no error means OK)
        monkeypatch.setattr(corpus_module, "_CORPUS_CACHE", None)
        load_corpus(force_reload=True)

    def test_cache_invalidates_on_text_change(self, synthetic_corpus, tmp_path, monkeypatch):
        # Change a passage; reload; cache entry for that passage should be regenerated.
        passage = tmp_path / "passages" / "lit-1.txt"
        original_text = passage.read_text()
        passage.write_text(original_text + " New sentence appended for test.")

        monkeypatch.setattr(corpus_module, "_CORPUS_CACHE", None)
        c2 = load_corpus(force_reload=True)
        new_lit1 = next(e for e in c2.entries if e.id == "lit-1")
        # Word count should reflect the new content
        assert new_lit1.signals.word_count > synthetic_corpus.entries[0].signals.word_count

    def test_cache_drops_removed_entries(self, synthetic_corpus, tmp_path, monkeypatch):
        # Remove an entry from the YAML; reload; cache should drop it.
        cache_path = tmp_path / "cache.json"
        yaml_path = tmp_path / "voice-references-stylometric.yaml"
        yaml_data = yaml.safe_load(yaml_path.read_text())
        yaml_data["references"] = [r for r in yaml_data["references"] if r["id"] != "expo-1"]
        yaml_path.write_text(yaml.safe_dump(yaml_data))

        monkeypatch.setattr(corpus_module, "_CORPUS_CACHE", None)
        load_corpus(force_reload=True)
        cache = json.loads(cache_path.read_text())
        assert "expo-1" not in cache["passages"]


# ─── stylometry() tool surface ────────────────────────────────────────────

class TestStylometryTool:
    def test_returns_report_with_required_fields(self, synthetic_corpus):
        report = stylometry(
            "She walked. He waited. The room held its breath.",
            corpus=synthetic_corpus,
        )
        assert isinstance(report, StylometricToolReport)
        assert "burstiness" in report.candidate
        assert "mattr" in report.candidate
        assert "fw_cosine_from_llm_centroid" in report.candidate
        assert "fw_cosine_from_human_literary" in report.candidate
        assert "llm_default_centroid" in report.references
        assert "human_literary_centroid" in report.references
        assert report.interpretation_notes

    def test_caller_model_known_includes_own_default(self, synthetic_corpus):
        report = stylometry(
            "Some prose here. More prose follows.",
            caller_model="sonnet-test",
            corpus=synthetic_corpus,
        )
        assert "fw_cosine_from_own_model_default" in report.candidate
        assert report.references["caller_model_default"]["model"] == "sonnet-test"
        assert report.references["caller_model_default"]["n_samples"] == 2

    def test_caller_model_unknown_falls_back(self, synthetic_corpus):
        report = stylometry(
            "Some prose here.",
            caller_model="not-in-corpus",
            corpus=synthetic_corpus,
        )
        assert "fw_cosine_from_own_model_default" not in report.candidate
        assert report.references["caller_model_default"]["n_samples"] == 0

    def test_neutral_baseline_adds_distance(self, synthetic_corpus):
        report = stylometry(
            "She walked. He waited. The room held its breath.",
            neutral_baseline="The man and the woman walked into the room together.",
            corpus=synthetic_corpus,
        )
        assert "fw_cosine_from_neutral_baseline" in report.candidate
        assert 0.0 <= report.candidate["fw_cosine_from_neutral_baseline"] <= 1.0

    def test_response_under_token_budget(self, synthetic_corpus):
        report = stylometry(
            "Some test prose. " * 20,
            caller_model="sonnet-test",
            corpus=synthetic_corpus,
        )
        # JSON-serialized report should comfortably fit under the 4kB target
        as_json = json.dumps(report.to_dict())
        assert len(as_json.encode("utf-8")) < 4096

    def test_to_dict_serializes_cleanly(self, synthetic_corpus):
        report = stylometry("Test prose here.", corpus=synthetic_corpus)
        d = report.to_dict()
        # Must be JSON-serializable
        json.dumps(d)

    def test_caller_model_includes_calibration_fields(self, synthetic_corpus):
        report = stylometry(
            "Some test prose.",
            caller_model="sonnet-test",
            corpus=synthetic_corpus,
        )
        own = report.references["caller_model_default"]
        # Calibration fields should be present
        assert "intra_dist_mean" in own
        assert "intra_dist_p50" in own
        assert "intra_dist_p95" in own
        assert "intra_dist_max" in own
        assert "calibration_reliable" in own
        # synthetic_corpus has 2 sonnet samples; calibration is unreliable
        assert own["calibration_reliable"] is False

    def test_interpretation_uses_calibrated_thresholds(self, synthetic_corpus):
        report = stylometry(
            "She walked. He waited. The room held its breath.",
            caller_model="sonnet-test",
            corpus=synthetic_corpus,
        )
        # With n=2 samples, calibration is unreliable; notes should say so
        assert "advisory" in report.interpretation_notes.lower() \
            or "only" in report.interpretation_notes.lower() \
            or "calibration" in report.interpretation_notes.lower()


class TestLookupExemplar:
    def test_lookup_by_author(self, synthetic_corpus):
        from owtn.tools import lookup_exemplar
        # synthetic corpus has lit-1, lit-2 entries (literary)
        r = lookup_exemplar("lit", n=2, corpus=synthetic_corpus)
        assert r["match"] == "authors_only"
        assert r["n_returned"] == 2
        assert all(p["category"] == "exemplars" for p in r["passages"])

    def test_lookup_by_tag(self, synthetic_corpus):
        from owtn.tools import lookup_exemplar
        r = lookup_exemplar("literary", n=5, corpus=synthetic_corpus)
        assert r["match"] == "tags_only"
        # Both lit-1 and lit-2 carry the "literary" tag
        assert r["n_returned"] == 2

    def test_lookup_by_exact_id(self, synthetic_corpus):
        from owtn.tools import lookup_exemplar
        r = lookup_exemplar("lit-1", n=1, corpus=synthetic_corpus)
        assert r["match"] == "id"
        assert r["passages"][0]["id"] == "lit-1"

    def test_lookup_blocks_default(self, synthetic_corpus):
        from owtn.tools import lookup_exemplar
        # llm-sonnet-1 is in the synthetic corpus with llm_default tag
        r = lookup_exemplar("llm-sonnet-1", n=1, corpus=synthetic_corpus)
        assert r["match"] == "blocked"
        assert r["n_returned"] == 0
        assert "default" in r["note"].lower()

    def test_lookup_unknown(self, synthetic_corpus):
        from owtn.tools import lookup_exemplar
        r = lookup_exemplar("totally-nonexistent-thing", corpus=synthetic_corpus)
        assert r["match"] == "none"

    def test_lookup_truncates_long_passages(self, synthetic_corpus):
        from owtn.tools import lookup_exemplar
        r = lookup_exemplar("lit-1", n=1, max_words=5, corpus=synthetic_corpus)
        # synthetic lit-1 has more than 5 words; should be truncated
        p = r["passages"][0]
        assert p["truncated"] is True
        assert "[…]" in p["text"]


class TestStyleDistances:
    def test_target_styles_returns_distances(self, synthetic_corpus):
        # synthetic corpus has lit-1 + lit-2 entries with literary tag
        report = stylometry(
            "Some prose here.",
            target_styles=["minimalist", "literary"],
            corpus=synthetic_corpus,
        )
        sd = report.references.get("style_distances")
        assert sd is not None
        assert "literary" in sd
        assert "fw_cosine" in sd["literary"]
        assert "fw_delta" in sd["literary"]
        assert "intra_dist_p95" in sd["literary"]
        assert sd["literary"]["kind"] == "tag"

    def test_target_style_unknown_marks_not_found(self, synthetic_corpus):
        report = stylometry(
            "Some prose here.",
            target_styles=["nonexistent-author-xxxx"],
            corpus=synthetic_corpus,
        )
        sd = report.references["style_distances"]
        assert sd["nonexistent-author-xxxx"]["status"] == "not_found"

    def test_target_styles_omitted_no_section(self, synthetic_corpus):
        report = stylometry("Some prose here.", corpus=synthetic_corpus)
        assert "style_distances" not in report.references


class TestIntraClusterSpread:
    def test_spread_for_n2(self, synthetic_corpus):
        sonnet = synthetic_corpus.by_model_tag("sonnet-test")
        spread = synthetic_corpus.intra_cluster_spread(sonnet)
        assert spread["n_samples"] == 2
        assert spread["calibration_reliable"] is False
        assert 0.0 <= spread["intra_dist_p50"] <= 1.0
        assert spread["intra_dist_p95"] >= spread["intra_dist_p50"]

    def test_spread_for_too_few(self, synthetic_corpus):
        # An empty group
        spread = synthetic_corpus.intra_cluster_spread([])
        assert spread["n_samples"] == 0
        assert spread["calibration_reliable"] is False
