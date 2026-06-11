"""domain_expert factory — runtime instantiation from `ExpertNeed`.

The pre-stage filter classifies whether a story demands domain expertise
to land convincingly and, if so, returns 0-N `ExpertNeed` specs naming
the domain, what the prose will need to get right, who has the
expertise, and whether web verification is recommended. This module
turns each spec into a `CriticPersona` the registry can dispatch the
same way it dispatches any other tool-using critic.

The pattern is a *factory*, not a registry: critics are parameterized
at runtime from upstream classification rather than hardcoded per
domain. Handles the open-ended expertise space (competitive bridge,
deaf culture, late-Republic Roman, in-universe canon, anything the
filter identifies) without code changes per new domain.
"""

from __future__ import annotations

import re

from owtn.models.stage_4 import CriticPersona, ExpertNeed, ExpertNeedsList


DOMAIN_EXPERT_ID_PREFIX = "domain_expert_"


_DOMAIN_EXPERT_MECHANISM_TEMPLATE = (
    "You read this manuscript for one thing: where the prose meets the {domain} "
    "and where it doesn't. Fiction set inside a domain accumulates moments where "
    "the writing has to get something right — a procedure, a piece of jargon, a "
    "pattern of practice, a temporal or material fact. When the prose lands those "
    "moments, the work earns the reader's trust on the harder claims that come "
    "later. When the prose gets one wrong, it costs the reader's trust in the "
    "rest of the work. Your job is to read for the moments that make this trade, "
    "not to fact-check every sentence. Some inaccuracies are a writer's "
    "deliberate compression; others break the work."
)


_DOMAIN_EXPERT_BASE_FOCUS = [
    "Specific factual claims the prose makes — dates, procedures, names, technical specifics, in-universe canon",
    "Tacit-knowledge moments — how a practice actually feels and sounds at the bench, in the field, at the table",
    "Distinguish a verisimilitude break (a wrong fact a domain reader would catch and lose trust over) from a dramatic license acceptable to the work (a deliberate compression or invention that the work has earned)",
    "Flag any claim where you would want to verify but cannot from the prose alone — those go in the report as moderate-severity items so the writer knows where to dig further",
]


_DOMAIN_EXPERT_SEVERITY: dict[str, str] = {
    "severe": (
        "A clear verisimilitude break — a wrong fact, procedure, or convention "
        "that a domain reader would catch and that costs the work credibility. "
        "Multiple breaks in adjacent passages also count as severe; they "
        "compound."
    ),
    "moderate": (
        "Either a single verisimilitude break that lands in a non-critical "
        "passage, or a claim where you would want to verify but the prose "
        "alone doesn't give you enough. Recommends review by the writer or a "
        "deeper-domain reader."
    ),
    "minor": (
        "Small surface inaccuracies the work can absorb — a minor period "
        "anachronism, a piece of jargon used adjacent to its real meaning, "
        "a procedural shortcut a practitioner would forgive."
    ),
}


def _slug(domain: str) -> str:
    """Conservative kebab-to-snake slug for domain ids. Lowercase, ascii
    alphanumerics, underscores, no leading/trailing underscore."""
    s = "".join(c if c.isalnum() else "_" for c in domain.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "expert"


def domain_expert_id(domain: str) -> str:
    """Stable id for a domain_expert critic. Used by the registry and the
    cycle's `critic_calls` list."""
    return f"{DOMAIN_EXPERT_ID_PREFIX}{_slug(domain)}"


def instantiate_domain_expert(
    spec: ExpertNeed,
    *,
    model: str = "deepseek-v4-pro",
    reasoning_effort: str = "medium",
) -> CriticPersona:
    """Build one `CriticPersona` from an `ExpertNeed` spec.

    The persona's identity is the spec's `persona_hint` (ordinary-
    specific practitioner). The mechanism, focus, and severity
    calibration are templates parameterized by the domain — the *kind*
    of work doesn't change per domain, only the subject matter.
    """
    tools = ["read_file"]
    if spec.web_search_recommended:
        # `web_search` finds candidate sources; `fetch_page` reads them.
        # Snippet-only verification is often inadequate for doctrinal
        # passages or quoted material, so opting into search means
        # opting into follow-up fetches as well.
        tools.append("web_search")
        tools.append("fetch_page")

    focus_areas = [
        f.strip() for f in spec.expertise_focus if f.strip()
    ] + list(_DOMAIN_EXPERT_BASE_FOCUS)

    return CriticPersona(
        id=domain_expert_id(spec.domain),
        name=f"Domain expert — {spec.domain}",
        tier="domain",
        persona=True,
        mechanism=_DOMAIN_EXPERT_MECHANISM_TEMPLATE.format(domain=spec.domain),
        identity=spec.persona_hint.strip(),
        focus_areas=focus_areas,
        severity_calibration=dict(_DOMAIN_EXPERT_SEVERITY),
        model=model,
        reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
        tools=tools,
    )


def instantiate_domain_experts(
    needs: ExpertNeedsList,
    *,
    model: str = "deepseek-v4-pro",
    reasoning_effort: str = "medium",
) -> list[CriticPersona]:
    """Instantiate one `CriticPersona` per `ExpertNeed`. Empty input
    returns an empty list — the common case (most stories don't demand
    domain expertise)."""
    return [
        instantiate_domain_expert(spec, model=model, reasoning_effort=reasoning_effort)
        for spec in needs.experts
    ]
