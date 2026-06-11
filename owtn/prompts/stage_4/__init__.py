"""Stage 4 prompt assembly.

Mirrors `owtn.prompts.stage_3` — template files loaded from disk,
substitutions performed at call time. Builders return ready-to-send
strings; LLM-fed files in this dir follow the project's token-hygiene
rules (no hard-wrapping; folded scalars in YAML; structural blank lines
only where they help the model parse).
"""

from __future__ import annotations

from pathlib import Path

from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_3 import VoiceGenome
from owtn.models.stage_4 import AudienceFraming, CriticPersona


_PROMPTS_DIR = Path(__file__).resolve().parent


def _load(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


# ─── Critic prompts ──────────────────────────────────────────────────────


def load_critic_base() -> str:
    """Fixed scaffolding loaded into every critic system prompt — workshop
    frame, observations-not-rewrites contract, severity taxonomy, valid
    `not_load_bearing` output framing."""
    return _load("critic_base.txt")


def load_writer_base() -> str:
    """Fixed scaffolding for the writer agent's system prompt — workshop
    frame, file-as-source-of-truth, no writer-persona (voice spec carries
    the aesthetic stance)."""
    return _load("base_system.txt")


def build_prethink_user_prompt() -> str:
    """Phase 1 user prompt. The DAG, voice spec, concept, and audience
    are already in the agent's system prompt — this prompt only carries
    the per-phase task description."""
    return _load("prethink.txt")


def build_downdraft_user_prompt() -> str:
    """Phase 2 user prompt. Per-scene write loop with read-as-reader
    micro-loop instructions; tools listed for clarity but the orchestrator
    enforces the allowlist."""
    return _load("drafter.txt")


def build_revise_gather_user_prompt(
    *,
    critic_lines: list[str],
) -> str:
    """Sub-phase A user prompt. `critic_lines` is the registry-derived
    list of available critics rendered as bullet items; the composer
    builds the list once at session start since the registry is fixed
    across cycles."""
    return _load("revise_gather.txt").replace("{CRITIC_LIST}", "\n".join(critic_lines))


def build_revise_apply_user_prompt(
    *,
    plan_summary: str,
    intended_revisions: list[str],
) -> str:
    """Sub-phase B user prompt. The plan the agent committed in
    sub-phase A is rendered back so the agent has it inline."""
    revisions_block = (
        "\n".join(f"- {r}" for r in intended_revisions)
        if intended_revisions
        else "- (none — the plan summary carries the work)"
    )
    return (
        _load("revise_apply.txt")
        .replace("{PLAN_SUMMARY}", plan_summary.strip())
        .replace("{INTENDED_REVISIONS}", revisions_block)
    )


def _render_persona_block(persona: CriticPersona) -> str:
    """Mechanism + (optional) identity + focus areas + (optional)
    severity calibration + (optional) exemplars.

    Persona-driven critics get the full block; criteria-direct critics
    skip the identity and exemplar sections (they don't earn their place
    when the criteria are mechanical)."""
    lines: list[str] = []

    if persona.name:
        lines.append(f"You are: {persona.name} (critic id: {persona.id}).")
    else:
        lines.append(f"Critic id: {persona.id}.")
    lines.append("")

    if persona.persona and persona.identity:
        lines.append(persona.identity.strip())
        lines.append("")

    lines.append("What you are reading for:")
    lines.append(persona.mechanism.strip())
    lines.append("")

    lines.append("Where to look:")
    for area in persona.focus_areas:
        lines.append(f"- {area}")
    lines.append("")

    if persona.severity_calibration:
        lines.append("How to tag severity:")
        for sev in ("severe", "moderate", "minor"):
            note = persona.severity_calibration.get(sev)
            if note:
                lines.append(f"- {sev}: {note}")
        lines.append("")

    if persona.persona and persona.exemplars:
        lines.append("Passages where the work you are looking for is operating well:")
        for ex in persona.exemplars:
            lines.append(f"- `{ex.id}` — {ex.note}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _render_voice_block(voice_genome: VoiceGenome) -> str:
    """Voice context the critic reads against. The three renderings are
    the load-bearing field — they exemplify what 'in voice' looks like
    on this story's adjacent scenes — so they're surfaced verbatim."""
    lines = ["Voice the manuscript is in (the writing target):", ""]
    lines.append(f"POV: {voice_genome.pov}, tense: {voice_genome.tense}")
    if voice_genome.pov_notes:
        lines.append(f"POV notes: {voice_genome.pov_notes}")
    if voice_genome.tense_notes:
        lines.append(f"Tense notes: {voice_genome.tense_notes}")
    lines.append("")
    lines.append("Description:")
    lines.append(voice_genome.description.strip())
    lines.append("")
    lines.append("Diction:")
    lines.append(voice_genome.diction.strip())
    lines.append("")
    if voice_genome.positive_constraints:
        lines.append("Positive constraints (what the voice IS):")
        for c in voice_genome.positive_constraints:
            lines.append(f"- {c}")
        lines.append("")
    if voice_genome.prohibitions:
        lines.append("Prohibitions:")
        for p in voice_genome.prohibitions:
            lines.append(f"- {p}")
        lines.append("")
    lines.append("Three sample passages in this voice — these are NOT scenes from this manuscript; they show the voice operating well in adjacent scenes:")
    lines.append("")
    for r in voice_genome.renderings:
        lines.append(f"### Rendering: {r.scene_id}")
        lines.append("")
        lines.append(r.text.strip())
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_concept_block(concept: ConceptGenome) -> str:
    """Concept context. Goal fields (`target_effect`, `thematic_engine`)
    are surfaced verbatim — critics read them as evaluation context, not
    as prose-generation targets, so the goals-cannot-ride-in-plaintext
    risk doesn't apply here."""
    lines = ["Concept:", ""]
    lines.append(f"Premise: {concept.premise.strip()}")
    if concept.target_effect:
        lines.append(f"Target effect: {concept.target_effect.strip()}")
    if concept.thematic_engine:
        lines.append(f"Thematic engine: {concept.thematic_engine.strip()}")
    if concept.character_seeds:
        lines.append("")
        lines.append("Character seeds:")
        for cs in concept.character_seeds:
            label = cs.label if hasattr(cs, "label") else cs.get("label", "")
            sketch = cs.sketch if hasattr(cs, "sketch") else cs.get("sketch", "")
            lines.append(f"- {label}: {sketch}")
    if concept.setting_seeds:
        lines.append("")
        lines.append(f"Setting: {concept.setting_seeds.strip()}")
    if concept.constraints:
        lines.append("")
        lines.append("Constraints — what the world refuses or requires (these are about the story, not about the voice):")
        for c in concept.constraints:
            lines.append(f"- {c}")
    if concept.style_hint:
        lines.append("")
        lines.append(f"Style hint (background only; the voice above carries the writing target): {concept.style_hint.strip()}")
    return "\n".join(lines).rstrip()


def _render_audience_block(audience: AudienceFraming) -> str:
    lines = ["Implied audience — who the work is for; what they bring to the page:", ""]
    lines.append(audience.description.strip())
    if audience.recognizes:
        lines.append("")
        lines.append("Recognizes:")
        for r in audience.recognizes:
            lines.append(f"- {r}")
    if audience.tolerates:
        lines.append("")
        lines.append("Tolerates:")
        for t in audience.tolerates:
            lines.append(f"- {t}")
    return "\n".join(lines).rstrip()


def build_critic_system_prompt(
    persona: CriticPersona,
    *,
    voice_genome: VoiceGenome,
    concept: ConceptGenome,
    dag_rendering: str,
    audience_framing: AudienceFraming | None = None,
) -> str:
    """Compose a critic's system prompt.

    Layout: base scaffolding → persona/mechanism block → work context
    (voice + concept + DAG + audience). Stable across calls so prompt-
    cache hit rates are maximal — only the per-call user prompt
    (manuscript + focus) varies."""
    sections: list[str] = [load_critic_base().strip(), "", _render_persona_block(persona)]

    sections.append("")
    sections.append("─" * 8 + " WORK CONTEXT " + "─" * 8)
    sections.append("")
    sections.append(_render_voice_block(voice_genome))
    sections.append("")
    sections.append(_render_concept_block(concept))
    sections.append("")
    sections.append("Structural plan — the typed-edge graph of beats and the work each one carries:")
    sections.append("")
    sections.append(dag_rendering.strip())
    if audience_framing is not None:
        sections.append("")
        sections.append(_render_audience_block(audience_framing))
    sections.append("")
    sections.append(
        "Output a CriticReport. Set `critic_id` to the id above and `cycle` "
        "to whatever the user prompt names — those are not yours to invent."
    )
    return "\n".join(sections).strip()


def build_writer_system_prompt(
    *,
    voice_genome: VoiceGenome,
    concept: ConceptGenome,
    dag_rendering: str,
    audience_framing: AudienceFraming | None = None,
) -> str:
    """Compose the writer agent's system prompt.

    Layout: base scaffolding → work context (voice + concept + structural
    plan + audience). Stable across all three phases so the prompt-cache
    prefix hits across PreThink, DownDraft, and Revise calls — only the
    per-phase user prompt varies.
    """
    sections: list[str] = [load_writer_base().strip()]
    sections.append("")
    sections.append("─" * 8 + " WHAT THE WRITING IS " + "─" * 8)
    sections.append("")
    sections.append(_render_voice_block(voice_genome))
    sections.append("")
    sections.append(_render_concept_block(concept))
    sections.append("")
    sections.append("Structural plan — the typed-edge graph of beats and the work each one carries:")
    sections.append("")
    sections.append(dag_rendering.strip())
    if audience_framing is not None:
        sections.append("")
        sections.append(_render_audience_block(audience_framing))
    return "\n".join(sections).strip()


def build_surgical_edit_subagent_prompt(
    *,
    voice_genome: VoiceGenome,
    region_text: str,
    instruction: str,
    anchor_before: str,
    anchor_after: str,
    audience_framing: AudienceFraming | None = None,
) -> str:
    """Compose the surgical-edit subagent's system prompt.

    Reuses `_render_voice_block` so the subagent sees the same voice
    context the writer does — the verbal description, structured
    fields (POV, diction, positive constraints), and the three
    renderings that show what the voice looks like operating well.
    Without the renderings the subagent is rewriting prose against
    a voice it's only been *described*.

    `audience_framing` is the implied-reader description from the
    pre-stage filter. The writer agent and critics see it; the
    subagent rewriting prose should too — it is reshaping prose for a
    specific implied audience, not in a vacuum.
    """
    template = _load("surgical_edit_subagent.txt")
    audience_block = (
        _render_audience_block(audience_framing) if audience_framing is not None
        else "(implied audience: not specified for this run)"
    )
    return (
        template
        .replace("{EDITABLE_REGION}", region_text)
        .replace("{INSTRUCTION}", instruction.strip())
        .replace("{VOICE_BLOCK}", _render_voice_block(voice_genome))
        .replace("{AUDIENCE_BLOCK}", audience_block)
        .replace("{ANCHOR_BEFORE}", anchor_before)
        .replace("{ANCHOR_AFTER}", anchor_after)
    )


def build_critic_user_prompt(
    manuscript_text: str,
    *,
    cycle: int,
    focus: str | None = None,
) -> str:
    """Per-call user prompt. The manuscript is the primary input; an
    optional `focus` hint sharpens which scene or pattern the critic
    should weight."""
    parts: list[str] = [
        f"Cycle: {cycle}.",
        "",
    ]
    if focus:
        parts.append(f"Focus: {focus}")
        parts.append("")
    parts.append("Manuscript (`story.md`):")
    parts.append("")
    parts.append(manuscript_text.strip() if manuscript_text.strip() else "(empty — no prose written yet)")
    parts.append("")
    parts.append(
        "Read the manuscript end-to-end first, then return your CriticReport."
    )
    return "\n".join(parts)
