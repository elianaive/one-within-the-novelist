"""Tests for owtn.stage_3.voting — Borda no-self-vote."""

from __future__ import annotations

import pytest

from owtn.stage_3.voting import borda_no_self_vote


def test_four_agent_unanimous_winner():
    """Three rankers all put 'a' first → a sweeps."""
    rankings = {
        "a": ["b", "c", "d"],
        "b": ["a", "c", "d"],
        "c": ["a", "b", "d"],
        "d": ["a", "b", "c"],
    }
    points = borda_no_self_vote(rankings)
    # Per the overview: rank-1=2, rank-2=1, rank-3=0
    # a is ranked first by b/c/d → 2+2+2 = 6
    # a is not in a's own ranking
    assert points["a"] == 6
    # d is ranked last by a/b/c → 0+0+0 = 0
    assert points["d"] == 0


def test_three_agent_panel():
    """3 agents → 1/0 per ranker."""
    rankings = {
        "a": ["b", "c"],
        "b": ["c", "a"],
        "c": ["a", "b"],
    }
    points = borda_no_self_vote(rankings)
    # a: ranked-1 by b (0), ranked-0 by c (1) → 1
    # b: ranked-0 by a (1), ranked-1 by c (0) → 1
    # c: ranked-1 by a (0), ranked-0 by b (1) → 1
    assert points == {"a": 1, "b": 1, "c": 1}


def test_split_votes_resolve_correctly():
    """Two top contenders split first-place; third is consistent loser."""
    rankings = {
        "a": ["b", "c", "d"],
        "b": ["a", "c", "d"],
        "c": ["b", "a", "d"],
        "d": ["a", "b", "c"],
    }
    points = borda_no_self_vote(rankings)
    # a: by b=2, by c=1, by d=2 → 5
    # b: by a=2, by c=2, by d=1 → 5
    # c: by a=1, by b=1, by d=0 → 2
    # d: by a=0, by b=0, by c=0 → 0
    assert points == {"a": 5, "b": 5, "c": 2, "d": 0}


def test_self_in_ranking_raises():
    rankings = {
        "a": ["a", "b", "c"],  # 'a' included self
        "b": ["a", "c"],
        "c": ["a", "b"],
    }
    with pytest.raises(ValueError, match="included self"):
        borda_no_self_vote(rankings)


def test_missing_peer_in_ranking_raises():
    rankings = {
        "a": ["b", "c"],     # missing d
        "b": ["a", "c", "d"],
        "c": ["a", "b", "d"],
        "d": ["a", "b", "c"],
    }
    with pytest.raises(ValueError, match="missing"):
        borda_no_self_vote(rankings)


def test_duplicate_peer_in_ranking_raises():
    rankings = {
        "a": ["b", "b", "c"],  # 'b' twice
        "b": ["a", "c"],
        "c": ["a", "b"],
    }
    with pytest.raises(ValueError):
        borda_no_self_vote(rankings)


def test_empty_rankings_returns_empty():
    assert borda_no_self_vote({}) == {}


def test_single_agent_raises():
    with pytest.raises(ValueError, match="≥2"):
        borda_no_self_vote({"a": []})
