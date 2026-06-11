"""domain_expert factory tests + session integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from owtn.models.stage_4 import (
    CriticPersona,
    ExpertNeed,
    ExpertNeedsList,
)
from owtn.stage_4 import CriticRegistry
from owtn.stage_4.critics import CRITIC_TOOL_REGISTRY
from owtn.stage_4.domain_expert import (
    DOMAIN_EXPERT_ID_PREFIX,
    domain_expert_id,
    instantiate_domain_expert,
    instantiate_domain_experts,
)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _spec(*, domain="quantum optics", web_search: bool = True) -> ExpertNeed:
    return ExpertNeed(
        domain=domain,
        expertise_focus=[
            "measurement back-action and entanglement protocols",
            "the practical pacing of bench-side optical work",
        ],
        persona_hint=(
            "A postdoc in atomic physics who reviews fiction manuscripts as a "
            "side gig. Spent the last six years inside an experimental optics "
            "lab and reads novels in spare moments between data runs."
        ),
        web_search_recommended=web_search,
    )


# ─── Slug + id ───────────────────────────────────────────────────────────


def test_id_is_stable_kebab_to_snake():
    assert domain_expert_id("quantum optics") == "domain_expert_quantum_optics"


def test_id_handles_punctuation_and_capitals():
    assert domain_expert_id("Roman Military History (late Republic)") == "domain_expert_roman_military_history_late_republic"


def test_id_collapses_consecutive_separators():
    assert domain_expert_id("a — b") == "domain_expert_a_b"


def test_id_falls_back_when_domain_is_pure_punctuation():
    """Edge case — a malformed domain string still produces a valid id."""
    assert domain_expert_id("!!!").startswith(DOMAIN_EXPERT_ID_PREFIX)


# ─── Single instantiation ──────────────────────────────────────────────


def test_instantiate_carries_persona_hint_as_identity():
    persona = instantiate_domain_expert(_spec())
    assert persona.identity.startswith("A postdoc in atomic physics")
    # `name` is generated from the domain
    assert "quantum optics" in persona.name


def test_instantiate_includes_web_search_when_recommended():
    persona = instantiate_domain_expert(_spec(web_search=True))
    assert "web_search" in persona.tools
    assert "fetch_page" in persona.tools  # paired with web_search
    assert "read_file" in persona.tools


def test_instantiate_omits_web_search_when_not_recommended():
    persona = instantiate_domain_expert(_spec(web_search=False))
    assert "web_search" not in persona.tools
    assert "fetch_page" not in persona.tools  # paired with web_search
    assert "read_file" in persona.tools


def test_instantiate_is_tool_using():
    persona = instantiate_domain_expert(_spec())
    assert persona.is_tool_using is True


def test_instantiate_tier_is_domain():
    persona = instantiate_domain_expert(_spec())
    assert persona.tier == "domain"
    assert persona.is_tier_a is False
    assert persona.is_tier_b is False


def test_instantiate_focus_areas_include_spec_focus_first():
    """Spec-supplied focus comes first; the generic factory focus follows."""
    persona = instantiate_domain_expert(_spec())
    assert persona.focus_areas[0].startswith("measurement back-action")
    # Generic factory bullets land afterward
    assert any("verisimilitude break" in fa for fa in persona.focus_areas)


def test_instantiate_severity_mapping_in_place():
    persona = instantiate_domain_expert(_spec())
    assert "severe" in persona.severity_calibration
    assert "verisimilitude break" in persona.severity_calibration["severe"]


def test_instantiate_mechanism_interpolates_domain():
    persona = instantiate_domain_expert(_spec(domain="competitive bridge"))
    assert "competitive bridge" in persona.mechanism


def test_instantiate_passes_model_through():
    persona = instantiate_domain_expert(
        _spec(), model="claude-sonnet-4-6", reasoning_effort="high",
    )
    assert persona.model == "claude-sonnet-4-6"
    assert persona.reasoning_effort == "high"


# ─── Plural instantiation ──────────────────────────────────────────────


def test_instantiate_many_handles_empty_list():
    assert instantiate_domain_experts(ExpertNeedsList()) == []


def test_instantiate_many_returns_one_per_spec():
    needs = ExpertNeedsList(experts=[
        _spec(domain="quantum optics"),
        _spec(domain="competitive bridge", web_search=False),
    ])
    personas = instantiate_domain_experts(needs)
    assert len(personas) == 2
    ids = [p.id for p in personas]
    assert "domain_expert_quantum_optics" in ids
    assert "domain_expert_competitive_bridge" in ids
    # Each has its own tool set per the spec
    by_id = {p.id: p for p in personas}
    assert "web_search" in by_id["domain_expert_quantum_optics"].tools
    assert "fetch_page" in by_id["domain_expert_quantum_optics"].tools
    assert "web_search" not in by_id["domain_expert_competitive_bridge"].tools
    assert "fetch_page" not in by_id["domain_expert_competitive_bridge"].tools


# ─── Registry interaction ──────────────────────────────────────────────


def test_registry_with_replaced_inserts_new_id():
    """`with_replaced` inserts when the id isn't present — domain_expert
    ids are session-time additions, not stub overrides."""
    starting = CriticRegistry([
        CriticPersona(
            id="continuity", tier="tier_a", persona=False,
            mechanism="x" * 100, focus_areas=["x"],
            model="deepseek-v4-pro",
        ),
    ])
    persona = instantiate_domain_expert(_spec())
    new_registry = starting.with_replaced(persona)
    assert "continuity" in new_registry
    assert persona.id in new_registry
    assert len(new_registry) == 2


def test_web_search_in_critic_tool_registry():
    """domain_expert critics that opt in find `web_search` available."""
    assert "web_search" in CRITIC_TOOL_REGISTRY


def test_fetch_page_in_critic_tool_registry():
    """`fetch_page` is registered alongside `web_search` so domain_expert
    critics that opt into web verification can also follow up on hits."""
    assert "fetch_page" in CRITIC_TOOL_REGISTRY


# ─── Session integration ───────────────────────────────────────────────


def test_chained_with_replaced_lands_multiple_experts():
    """The session composer's loop pattern: start with a base registry,
    apply `with_replaced` for each instantiated expert. Verify each
    one is reachable on the final registry."""
    base = CriticRegistry([
        CriticPersona(
            id=cid, tier="tier_a", persona=False,
            mechanism="x" * 100, focus_areas=["x"],
            model="deepseek-v4-pro",
        )
        for cid in ("voice_fidelity", "payload_enactment", "continuity", "motif_fidelity")
    ])
    needs = ExpertNeedsList(experts=[
        _spec(domain="historical Rome", web_search=False),
        _spec(domain="quantum optics", web_search=True),
    ])

    registry = base
    for expert in instantiate_domain_experts(needs):
        registry = registry.with_replaced(expert)

    # Originals still present
    for cid in ("voice_fidelity", "payload_enactment", "continuity", "motif_fidelity"):
        assert cid in registry
    # Both experts landed
    assert "domain_expert_historical_rome" in registry
    assert "domain_expert_quantum_optics" in registry
    # Tool sets per spec
    assert "web_search" not in registry.get("domain_expert_historical_rome").tools
    assert "web_search" in registry.get("domain_expert_quantum_optics").tools
