"""Stage 4 pre-stage classification outputs.

Two cheap haiku-class calls fire at session setup before the Stage 4
agent spawns. They produce structured specs that condition critic
instantiation and the agent's system prompt.

`ExpertNeed` / `ExpertNeedsList` parameterize the dynamic `domain_expert`
critic factory — a generic critic class instantiated per-concept rather
than a fixed registry, so the open-ended expertise space (competitive
bridge, deaf culture, late-Republic Roman military) is handled by
runtime parameterization rather than hardcoded critics.

`AudienceFraming` is Rabinowitz's authorial audience — who the work is
*for*, not who's writing it. Populated into the agent's system prompt;
complements the voice spec (HOW to write) without competing with it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExpertNeed(BaseModel):
    """One domain a `domain_expert` critic should be instantiated for.

    `persona_hint` is a one-paragraph description of who has this
    expertise, in the project's ordinary-specific persona register
    ("a postdoc in atomic physics who reviews fiction manuscripts as a
    side gig", not "a Nobel-winning physicist"). Famous-author or
    famous-figure archetypes collapse to caricature (CoMPosT EMNLP 2023;
    Mikros 2025).

    `web_search_recommended` is True when factual verification matters
    more than tacit understanding — historical detail, technical claims,
    in-universe canon. False for tacit-knowledge domains where web
    search adds noise (jazz musicians talking shop; competitive-bridge
    table feel).
    """
    domain: str = Field(min_length=4)
    expertise_focus: list[str] = Field(min_length=1)
    persona_hint: str = Field(min_length=40)
    web_search_recommended: bool = False


class ExpertNeedsList(BaseModel):
    """Pre-stage filter output.

    Empty `experts` means the concept doesn't demand domain expertise
    for verisimilitude — most stories. Capped at 2 in v1 to keep cost
    bounded; YAML override can suppress entirely or force-instantiate
    additional experts.
    """
    experts: list[ExpertNeed] = Field(default_factory=list, max_length=2)


class AudienceFraming(BaseModel):
    """Implied authorial audience for the work.

    `description` is a paragraph-shape rendering of the implied reader —
    their tastes, prior literary exposure, what they bring to the text,
    what they recognize. Distinct from voice (writer aesthetic): the
    voice spec says HOW to write; this says WHO it's for.

    `recognizes` and `tolerates` are short bullet-form lists the agent's
    PreThink reader-state model can layer on top: who the reader is
    (audience framing) → what they feel/know/hold open at scene
    boundaries (reader-state model). Different scales; consistent.
    """
    description: str = Field(min_length=80)
    recognizes: list[str] = Field(default_factory=list)
    tolerates: list[str] = Field(default_factory=list)
