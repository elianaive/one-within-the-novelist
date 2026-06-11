"""CriticPersona model + loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from owtn.models.stage_4 import CriticPersona
from owtn.stage_4.personas import load_critic_pool


def test_model_rejects_short_mechanism():
    with pytest.raises(ValidationError):
        CriticPersona(
            id="x",
            tier="tier_a",
            mechanism="short",
            focus_areas=["one"],
            model="any",
        )


def test_model_rejects_empty_focus_areas():
    with pytest.raises(ValidationError):
        CriticPersona(
            id="x",
            tier="tier_a",
            mechanism="a" * 100,
            focus_areas=[],
            model="any",
        )


def test_tier_a_helper():
    p = CriticPersona(
        id="continuity",
        tier="tier_a",
        persona=False,
        mechanism="a" * 100,
        focus_areas=["one"],
        model="deepseek-v4-pro",
    )
    assert p.is_tier_a is True
    assert p.is_tier_b is False


def test_load_critic_pool_finds_committed_yamls():
    pool = load_critic_pool()
    ids = sorted(p.id for p in pool)
    # The two starter critics must be present.
    assert "continuity" in ids
    assert "payload_enactment" in ids


def test_load_critic_pool_skips_template():
    pool = load_critic_pool()
    assert all(not p.id.startswith("_") for p in pool)


def test_load_critic_pool_returns_sorted():
    pool = load_critic_pool()
    ids = [p.id for p in pool]
    assert ids == sorted(ids)


def test_load_critic_pool_missing_dir_returns_empty(tmp_path: Path):
    pool = load_critic_pool(tmp_path / "nonexistent")
    assert pool == []


def test_load_critic_pool_skips_malformed_yaml(tmp_path: Path):
    """A bad YAML is logged-and-skipped; the good ones still load."""
    (tmp_path / "good.yaml").write_text(yaml.safe_dump({
        "id": "good",
        "tier": "tier_a",
        "persona": False,
        "mechanism": "x" * 100,
        "focus_areas": ["one focus area"],
        "model": "deepseek-v4-pro",
    }))
    (tmp_path / "bad.yaml").write_text("id: bad\n  this isn't valid yaml: : :\n")
    pool = load_critic_pool(tmp_path)
    assert [p.id for p in pool] == ["good"]


def test_continuity_yaml_parses_with_no_persona():
    """Continuity is criteria-direct — persona=false, identity empty."""
    pool = load_critic_pool()
    cont = next(p for p in pool if p.id == "continuity")
    assert cont.persona is False
    assert cont.identity == ""
    assert cont.tier == "tier_a"


def test_payload_enactment_yaml_parses_with_persona():
    """payload_enactment is persona-driven — identity is non-empty prose."""
    pool = load_critic_pool()
    pe = next(p for p in pool if p.id == "payload_enactment")
    assert pe.persona is True
    assert pe.name == "The Developmental Editor"
    # Ordinary-specific persona, not famous-author archetype
    assert "developmental editor" in pe.identity.lower()
    # Severity calibration set
    assert "severe" in pe.severity_calibration
    assert "moderate" in pe.severity_calibration
    assert "minor" in pe.severity_calibration
