"""Voice genome — Stage 3's load-bearing handoff to Stage 4.

Schema matches `docs/stage-3/overview.md` §"The Genome": a thin discrete
shell on the dimensions LLMs systematically default badly on (POV/tense,
five lit-theory fields drawn from Genette/Cohn/Bakhtin/Booth/Le Guin),
free-form prose for register/diction/description, asymmetric constraints
(positive primary, prohibitions secondary), and three rendered passages
on the adjacent-scene test bench.

The renderings are not exemplars — they are the genome's load-bearing
field. A mutation that produces a new verbal description without new
renderings is a mutation in the *labeling* of a voice, not in voice space.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ─── Lit-theory sub-models ───────────────────────────────────────────────


class ConsciousnessRendering(BaseModel):
    """Cohn 1978 — how prose handles the character-narrator boundary.

    Modes:
    - `psycho_narration` — narrator summarizes interior ("she felt the weight").
      LLM default; usually wrong unless the voice WANTS that distance.
    - `narrated_monologue` — free indirect discourse ("How could she possibly
      be ready?"). Wood's "novelist's instrument of compassion."
    - `quoted_monologue` — interior speech surfaced as character-voiced thought.
    - `mixed` — voice moves between the above modes.
    - `external_focalization` — purely external surface; no interior access
      at all. Use when the narrator reports observable behavior and physical
      facts only — no interiority, no even-narrated interior summary.

    `fid_depth` is the deepest single discriminator between literary styles
    per Wood and the lit-theory report. Should be `none` whenever
    `mode='external_focalization'`.
    """
    mode: Literal[
        "psycho_narration",
        "narrated_monologue",
        "quoted_monologue",
        "mixed",
        "external_focalization",
    ]
    fid_depth: Literal["none", "shallow", "deep", "dominant"]


class ImpliedAuthor(BaseModel):
    """Booth 1961 — the second-self constructed from aesthetic choices.

    Distinct from the narrator. Sets moral temperature. LLMs default to
    affectively neutral narrators absent an explicit stance.
    """
    stance_toward_characters: Literal[
        "compassionate", "ironic", "contemptuous",
        "elegiac", "satirical", "neutral",
    ]
    moral_temperature: Literal["warm", "cool", "cold", "ambiguous"]


class DialogicMode(BaseModel):
    """Bakhtin — whether the narrator's language carries social registers.

    Orthogonal to focalization. A narrator's voice can be socially infected
    regardless of how many characters are focalized.
    """
    type: Literal["monologic", "heteroglossic", "polyphonic"]


class Craft(BaseModel):
    """Le Guin 1998 — practitioner dimensions absent from academic narratology.

    `sentence_rhythm` is auto-evaluable via burstiness. `crowding_leaping`
    is the dimension on which voice transformation produces the most
    measurable departure from neutral baseline.
    """
    sentence_rhythm: Literal[
        "short_dominant", "varied", "long_dominant",
        "staccato", "sinuous",
    ]
    crowding_leaping: Literal["crowded", "balanced", "leaping"]


# ─── Multi-mode support ──────────────────────────────────────────────────


class VoiceMode(BaseModel):
    """One alternative voice configuration for stories with explicit shifts.

    Munro's tense alternation, Calvino's nested registers, Cloud Atlas's
    six prose registers. The `overrides` dict carries field-level changes
    for this mode (e.g. `{"pov": "first", "tense": "present"}`) — kept
    free-shape so we don't have to mirror the full genome schema.
    """
    name: str = Field(min_length=2)
    condition: str = Field(min_length=10)
    overrides: dict[str, str] = Field(default_factory=dict)


# ─── Renderings ──────────────────────────────────────────────────────────


class Rendering(BaseModel):
    """One voice-transformed adjacent scene — the genome's proof field."""
    scene_id: str = Field(min_length=2)
    text: str = Field(min_length=80)


# ─── The genome ──────────────────────────────────────────────────────────


class SignatureRisk(BaseModel):
    """The structural move a voice commits to that the model would not
    take by default for this concept. Required before `finalize_voice_genome`
    can be called — the gate exists to surface mid-shaped voices early,
    when the agent can still go back and find a real risk rather than
    submitting the literary-fluent default in literary clothing.

    The three fields force the agent to think across two axes:
    1. The move itself, named at the sentence-level / structural level
       (not as a "stance" or "register" — too abstract to verify in prose).
    2. The model's near-default for this concept — what a competent but
       unrisky voice would produce. Named explicitly so the agent has
       articulated what they're committing *against*.
    3. Why this concept demands the move. A risk that doesn't serve
       the concept's load-bearing demands is decoration, not signature.
    """
    move: str = Field(
        min_length=20,
        description=(
            "The structural / syntactic / formal commitment, named at the "
            "sentence-shape or paragraph-architecture level. NOT 'incantatory' "
            "or 'restrained' (too abstract); rather 'long anaphoric and-chains "
            "as default sentence shape, 50+ words, broken only by Saunders "
            "beats' or 'editorial-apparatus framing — bracketed headnotes, "
            "footnotes, transcribed-document register'. The kind of move "
            "another voice in the panel would not take."
        ),
    )
    model_default_alternative: str = Field(
        min_length=20,
        description=(
            "What a competent literary-fluent voice would do for this "
            "concept by default — the version of this voice the model "
            "would produce if it were not committed against it. Naming "
            "this surfaces the gap between the move and the safe choice. "
            "If you can't name a concrete default to commit against, the "
            "move is either trivial or imaginary."
        ),
    )
    concept_demand_justification: str = Field(
        min_length=30,
        description=(
            "Why THIS concept's load-bearing demands need this specific "
            "move (not just 'good prose'). Reference the concept's "
            "target_effect / thematic_engine / constraints; tie the move "
            "to a specific demand the prose has to deliver. A risk that "
            "doesn't serve a concept-specific demand is decoration."
        ),
    )


class VoiceGenomeBody(BaseModel):
    """The voice fields an agent produces — identifiers attached separately.

    The agent's structured-output target. `pair_id` / `agent_id` are known
    to the orchestrator at session start; we don't ask the model to echo
    them — we attach them post-commit when constructing the full
    `VoiceGenome`. This avoids id-fabrication risk and keeps the agent's
    output focused on the voice work.
    """
    pov: Literal["first", "second", "third"]
    pov_notes: str = ""
    tense: Literal["past", "present"]
    tense_notes: str = ""

    voice_modes: list[VoiceMode] = Field(default_factory=list)
    voice_dynamics: str = ""

    consciousness_rendering: ConsciousnessRendering
    implied_author: ImpliedAuthor
    dialogic_mode: DialogicMode
    craft: Craft

    description: str = Field(min_length=40)
    diction: str = Field(min_length=20)

    positive_constraints: list[str] = Field(min_length=1)
    prohibitions: list[str] = Field(default_factory=list)

    signature_risk: SignatureRisk

    renderings: list[Rendering] = Field(min_length=3, max_length=3)


class VoiceGenome(VoiceGenomeBody):
    """A complete voice specification — body + session identifiers.

    Produced by combining a `VoiceGenomeBody` (from one voice agent in
    Phase 1, possibly revised in Phase 4) with the orchestrator-known
    `pair_id` and `agent_id`. Selection operates on the renderings;
    verbal fields document the voice but do not constitute it.
    """
    pair_id: str = Field(min_length=2)
    agent_id: str = Field(min_length=2)


# ─── Phase 3 / 4 / 5 artifacts ───────────────────────────────────────────


class Critique(BaseModel):
    """One agent's structured single-pass critique of another's proposal.

    Fixed shape, no follow-ups. Two strengths force the critic to find
    real value before raising the concern; the concern is singular so
    the writer-target gets a single load-bearing thing to address in
    Phase 4 revision rather than a flood.
    """
    critic_id: str = Field(min_length=2)
    target_id: str = Field(min_length=2)
    strengths: list[str] = Field(min_length=2, max_length=2)
    concern: str = Field(min_length=20)


class CritiqueBody(BaseModel):
    """One critique fields the agent produces — `critic_id` is attached
    by the orchestrator from `agent.id` post-commit (the agent doesn't
    need to know its own id; the orchestrator does)."""
    target_id: str = Field(min_length=2)
    strengths: list[str] = Field(min_length=2, max_length=2)
    concern: str = Field(min_length=20)


class CritiqueSet(BaseModel):
    """Phase 3 structured-output target — one critique per OTHER agent.

    The model produces one CritiqueBody per target. The orchestrator
    attaches `critic_id` and validates target_id coverage post-parse.
    """
    critiques: list[CritiqueBody] = Field(min_length=1)


class BordaRanking(BaseModel):
    """Phase 5 structured-output target — best-to-worst ordering of OTHER agents.

    `ranking[0]` is the agent the ranker thinks produced the strongest
    voice; `ranking[-1]` is the weakest. Self must not appear; the
    orchestrator validates against the panel ids before scoring.
    """
    ranking: list[str] = Field(min_length=1)


class VoiceSessionResult(BaseModel):
    """Final session output — winner + all proposals + Borda metadata.

    `winner` is selected by writers' Borda in v0.1; the readers' verdict
    (pairwise judge tournament on renderings) lands in v0.2 and will
    take selection precedence with Borda demoted to tie-breaker metadata.
    """
    pair_id: str
    winner: VoiceGenome
    proposals: list[VoiceGenome]
    critiques: list[Critique]
    borda_points: dict[str, int]
    cost_usd: float
    session_log_dir: str
