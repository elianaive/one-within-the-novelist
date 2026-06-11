"""Scene + Manuscript Pydantic shape tests."""

from __future__ import annotations

from owtn.models.stage_4 import Manuscript, Scene


def test_scene_word_count_basic():
    scene = Scene(id="opening", body="Three quick words.")
    assert scene.word_count == 3
    assert scene.is_empty is False


def test_scene_empty_body():
    scene = Scene(id="opening")
    assert scene.body == ""
    assert scene.word_count == 0
    assert scene.is_empty is True


def test_scene_whitespace_only_is_empty():
    scene = Scene(id="opening", body="   \n\n  \t  ")
    assert scene.is_empty is True


def test_manuscript_scene_lookup():
    ms = Manuscript(scenes=[
        Scene(id="alpha", body="One two three."),
        Scene(id="beta", body="Four five."),
    ])
    assert ms.scene_ids() == ["alpha", "beta"]
    assert ms.find("alpha").word_count == 3
    assert ms.find("missing") is None


def test_manuscript_total_word_count_sums_scenes():
    ms = Manuscript(scenes=[
        Scene(id="a", body="one two"),
        Scene(id="b", body="three four five"),
    ])
    assert ms.total_word_count == 5
