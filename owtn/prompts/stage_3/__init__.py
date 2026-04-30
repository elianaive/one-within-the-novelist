"""Stage 3 prompt assembly.

Mirrors `owtn.prompts.stage_2.registry` — template files loaded from disk,
placeholders substituted at call time. Two prompt families currently:

- `adjacent_scene_*` — test-bench picker + neutral-voice drafter
- `casting_*` — voice-panel caster (filter + select)

Voice-spec generation, critique, and revision prompts will land alongside.

Placeholder convention: `{UPPER_SNAKE_CASE}` for substitutions performed
via `str.replace` (avoiding `.format` collisions with literal braces in
JSON dumps).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from owtn.models.stage_1.concept_genome import ConceptGenome

if TYPE_CHECKING:
    from owtn.stage_3.personas import VoicePersona


_PROMPTS_DIR = Path(__file__).resolve().parent


def _load(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def load_base_system() -> str:
    """Stage 3 base system message — adjacent-scene bench role.

    Workshop-frame voice-stage positioning (per `feedback_assistant_register`
    memory and `lab/deep-research/runs/2026-04-27-aestheticized-prompt-framing/`).
    """
    return _load("base_system.txt")


def load_casting_system() -> str:
    """Stage 3 system message for the casting classifier.

    Used by both `casting_filter` (Stage 1) and `casting_select` (Stage 2);
    the role frame is the same — the user message provides the per-stage
    task.
    """
    return _load("casting_system.txt")


def _render_drafter_concept_context(concept: ConceptGenome) -> str:
    """Slim, prose-rendered concept context for the neutral-voice drafter.

    Per `docs/prompting-guide.md` §"Goals Cannot Ride in Plaintext": goal
    fields (`target_effect`, `thematic_engine`, `style_hint`) get literalized
    when passed raw into a prose-generating prompt. The drafter only needs
    grounding facts — premise, characters, setting — to keep names and world
    coherent. Goal fields are omitted entirely.
    """
    lines = [f"Premise: {concept.premise}"]
    if concept.character_seeds:
        lines.append("")
        lines.append("Characters:")
        for cs in concept.character_seeds:
            lines.append(f"- {cs.label}: {cs.sketch}")
    if concept.setting_seeds:
        lines.append("")
        lines.append(f"Setting: {concept.setting_seeds}")
    return "\n".join(lines)


def _render_persona_block(persona: "VoicePersona") -> str:
    """Condensed persona view for the casting filter and argument prompts.

    Includes the fields the classifier needs to evaluate engagement and
    to argue picks — id, name, first commitment (load-bearing),
    aesthetic_commitments (full ranked list), aversions, identity,
    starved_by patterns. Skips the persona's demonstrations + exemplars
    (downstream phase content, not casting input).
    """
    lines = [f"### {persona.id} — {persona.name}"]
    lines.append(f"identity: {persona.identity.strip()}")
    lines.append("aesthetic_commitments (ranked, first is load-bearing):")
    for i, c in enumerate(persona.aesthetic_commitments, 1):
        lines.append(f"  {i}. {c}")
    if persona.aversions:
        lines.append("aversions:")
        for a in persona.aversions:
            lines.append(f"  - {a}")
    if persona.starved_by:
        lines.append("starved_by:")
        for sb in persona.starved_by:
            lines.append(f"  - tag: {sb.tag}")
            lines.append(f"    reason: {sb.reason}")
    return "\n".join(lines)


def build_casting_classify_prompt(
    concept: ConceptGenome,
    dag_rendering: str,
    vocabulary: Mapping[str, str],
) -> tuple[str, str]:
    """Stage 1 of casting — concept classification against the vocabulary.

    Single LLM call producing a TRUE/FALSE judgment per vocabulary tag
    with a per-tag reason. Persona engagement is computed downstream by
    deterministic intersection of each persona's `starved_by` tags
    against the TRUE concept-features.
    """
    system_msg = load_casting_system()
    template = _load("casting_classify.txt")
    concept_json = json.dumps(concept.model_dump(exclude_none=True), indent=2)
    vocab_lines = [
        f"  - tag: {tag}\n    meaning: {meaning}"
        for tag, meaning in vocabulary.items()
    ]
    vocab_block = "\n".join(vocab_lines)
    user_msg = (
        template
        .replace("{CONCEPT_JSON}", concept_json)
        .replace("{DAG_RENDERING}", dag_rendering)
        .replace("{VOCABULARY_BLOCK}", vocab_block)
    )
    return system_msg, user_msg


def build_casting_argue_prompt(
    concept: ConceptGenome,
    dag_rendering: str,
    engaged_personas: list["VoicePersona"],
) -> tuple[str, str]:
    """Stage 2a of casting — per-persona argument.

    Single LLM call producing a structured `case_for` / `risks` / `cast_role`
    per engaged persona. The Stage 2b selection follows up in additive
    context, so the model sees its own arguments when ranking.
    """
    system_msg = load_casting_system()
    template = _load("casting_argue.txt")
    concept_json = json.dumps(concept.model_dump(exclude_none=True), indent=2)
    engaged_block = "\n\n".join(_render_persona_block(p) for p in engaged_personas)
    user_msg = (
        template
        .replace("{CONCEPT_JSON}", concept_json)
        .replace("{DAG_RENDERING}", dag_rendering)
        .replace("{ENGAGED_BLOCK}", engaged_block)
    )
    return system_msg, user_msg


def build_casting_select_user_msg(panel_size: int) -> str:
    """Stage 2b of casting — top-N selection user message.

    Returned standalone (no system message — Stage 2b is a follow-up turn
    in the same conversation as Stage 2a; system_msg + msg_history come
    from the 2a call). The 2b user message is a continuation prompt;
    persona context lives in the prior turn's `msg_history`.

    `panel_size` is substituted into the prompt to set how many personas
    to cast.
    """
    return _load("casting_select.txt").replace("{PANEL_SIZE}", str(panel_size))


def build_adjacent_scene_picker_prompt(
    concept: ConceptGenome,
    dag_rendering: str,
) -> tuple[str, str]:
    """Assemble (system_msg, user_msg) for the AU-scene picker.

    `dag_rendering` is the incident-encoded outline from
    `owtn.stage_2.rendering.render(dag)` — same format Stage 2 expansion
    operators and judges see, so the picker has the full structural context.

    The picker is selecting scenes (structured-output task), not generating
    prose, so the full concept JSON is passed verbatim — literalization risk
    is low since the output is `scene_id`/`synopsis`/`demand`/`why_distinct`
    fields rather than story prose.
    """
    system_msg = load_base_system()
    template = _load("adjacent_scene_picker.txt")
    concept_json = json.dumps(concept.model_dump(exclude_none=True), indent=2)
    user_msg = (
        template
        .replace("{CONCEPT_JSON}", concept_json)
        .replace("{DAG_RENDERING}", dag_rendering)
    )
    return system_msg, user_msg


def build_voice_brief_prompt(
    concept: ConceptGenome,
    dag_rendering: str,
    bench_drafts: list[dict],
) -> str:
    """Phase 1 user message — develop a voice for this story.

    Bench drafts are inlined directly (not fetched via tool) — the agent
    needs all three for the entire session, every agent reads them, and
    they're cached in the prefix anyway. Tool-fetching them just burns
    explore-loop iterations on data the agent should have started with.
    """
    template = _load("voice_brief.txt")
    concept_json = json.dumps(concept.model_dump(exclude_none=True), indent=2)
    return (
        template
        .replace("{CONCEPT_JSON}", concept_json)
        .replace("{DAG_RENDERING}", dag_rendering)
        .replace("{BENCH_DRAFTS}", render_bench_drafts(bench_drafts))
    )


def render_proposal_for_review(genome: dict, *, header: str | None = None) -> str:
    """Format a VoiceGenome (as dict) for inclusion in critique/Borda prompts.

    Pulls the verbal fields and renderings into a compact shape; skips
    bookkeeping like pair_id since reviewers don't need it. Used by
    Phase 3 (critique) and Phase 5 (Borda) where readers need to see
    what the voice claims and how it lands on the bench scenes.
    """
    lines: list[str] = []
    if header:
        lines.append(header)
    lines.append(f"pov: {genome['pov']}; tense: {genome['tense']}")
    cr = genome["consciousness_rendering"]
    ia = genome["implied_author"]
    dm = genome["dialogic_mode"]
    cf = genome["craft"]
    lines.append(
        f"consciousness: {cr['mode']} (fid_depth={cr['fid_depth']}); "
        f"implied_author: {ia['stance_toward_characters']} / {ia['moral_temperature']}; "
        f"dialogic: {dm['type']}; craft: {cf['sentence_rhythm']} / {cf['crowding_leaping']}"
    )
    lines.append("")
    lines.append("description:")
    lines.append(genome["description"])
    lines.append("")
    lines.append("diction:")
    lines.append(genome["diction"])
    if genome.get("positive_constraints"):
        lines.append("")
        lines.append("positive_constraints:")
        for c in genome["positive_constraints"]:
            lines.append(f"- {c}")
    if genome.get("prohibitions"):
        lines.append("")
        lines.append("prohibitions:")
        for p in genome["prohibitions"]:
            lines.append(f"- {p}")
    lines.append("")
    lines.append("renderings:")
    for r in genome["renderings"]:
        lines.append(f"  [{r['scene_id']}]")
        lines.append(r["text"])
        lines.append("")
    return "\n".join(lines)


def build_voice_critique_prompt(
    *,
    your_agent_id: str,
    your_proposal: dict,
    other_proposals: dict[str, dict],
) -> str:
    """Phase 3 — single-pass structured critique of every other proposal.

    `other_proposals` is `{agent_id: VoiceGenome.model_dump()}` for every
    panel agent except `your_agent_id`. The agent produces one Critique
    per entry.
    """
    template = _load("voice_critique.txt")
    your_block = render_proposal_for_review(your_proposal)
    others_block = "\n\n".join(
        render_proposal_for_review(p, header=f"### {aid}")
        for aid, p in other_proposals.items()
    )
    return (
        template
        .replace("{YOUR_AGENT_ID}", your_agent_id)
        .replace("{YOUR_PROPOSAL}", your_block)
        .replace("{OTHER_PROPOSALS}", others_block)
        .replace("{N_TARGETS}", str(len(other_proposals)))
    )


def build_voice_revise_prompt(
    *,
    your_proposal: dict,
    critiques_received: list[dict],
    bench_drafts: list[dict],
    your_phase_1_notes: list[str] | None = None,
) -> str:
    """Phase 4 — revise under critiques + metric ensemble.

    `critiques_received` is a list of `{critic_id, target_id, strengths,
    concern}` dicts targeting this agent. `your_phase_1_notes` is the
    agent's own scratchpad from Phase 1 (cross-phase ICL channel); empty
    list or None means the agent didn't write notes earlier. Bench drafts
    are inlined so the agent can cross-reference their own renderings
    against the originals without a tool fetch.
    """
    template = _load("voice_revise.txt")
    your_block = render_proposal_for_review(your_proposal)
    crit_lines: list[str] = []
    for c in critiques_received:
        crit_lines.append(f"From {c['critic_id']}:")
        for s in c["strengths"]:
            crit_lines.append(f"  + {s}")
        crit_lines.append(f"  concern: {c['concern']}")
        crit_lines.append("")
    notes_block = render_prior_notes(
        {"phase 1": your_phase_1_notes or []}
    )
    return (
        template
        .replace("{YOUR_PROPOSAL}", your_block)
        .replace("{CRITIQUES_RECEIVED}", "\n".join(crit_lines).strip())
        .replace("{BENCH_DRAFTS}", render_bench_drafts(bench_drafts))
        .replace("{YOUR_PRIOR_NOTES}", notes_block)
    )


def build_voice_borda_prompt(
    *,
    your_agent_id: str,
    your_proposal: dict,
    other_proposals: dict[str, dict],
    your_phase_1_notes: list[str] | None = None,
    your_phase_4_notes: list[str] | None = None,
) -> str:
    """Phase 5 — rank other agents' final proposals best to worst.

    Phase 1 + Phase 4 notes are surfaced here as the agent's curated
    aesthetic working memory (cross-phase ICL channel). Phase 3 notes
    (where they critiqued others) are intentionally NOT surfaced — that
    history is from a different task (evaluating others) and would risk
    biasing this evaluation toward the same conclusions on what may be
    revised proposals.
    """
    template = _load("voice_borda.txt")
    your_block = render_proposal_for_review(your_proposal)
    others_block = "\n\n".join(
        render_proposal_for_review(p, header=f"### {aid}")
        for aid, p in other_proposals.items()
    )
    notes_block = render_prior_notes({
        "phase 1": your_phase_1_notes or [],
        "phase 4": your_phase_4_notes or [],
    })
    return (
        template
        .replace("{YOUR_AGENT_ID}", your_agent_id)
        .replace("{YOUR_PROPOSAL}", your_block)
        .replace("{OTHER_PROPOSALS}", others_block)
        .replace("{N_OTHERS}", str(len(other_proposals)))
        .replace("{YOUR_PRIOR_NOTES}", notes_block)
    )


def build_voice_commit_user_msg() -> str:
    """Phase 1 commit user message — produce the structured VoiceGenome.

    Continuation prompt; persona system + concept + bench context come
    from the prior tool-use turns' message history. Identifiers are
    attached by the orchestrator after parsing, so the agent fills body
    fields only — `pair_id` and `agent_id` aren't part of the schema
    the model sees (the structured-output target is `VoiceGenomeBody`).
    """
    return (
        "Pass 2 — commit. Produce the VoiceGenome body now.\n\n"
        "All three renderings must transform the bench's neutral drafts "
        "(same world, same characters, same physical events; different voice). "
        "Use the bench's scene_ids exactly. The schema fields and the verbal "
        "description must agree — if they say different things, the verbal "
        "description is the one that's wrong.\n\n"
        "Output the structured response now. No further tool calls."
    )


def render_bench_drafts(bench_drafts: list[dict]) -> str:
    """Render the full bench drafts inline — scene_id + demand + the actual
    neutral_draft prose. Used by Phase 1 + Phase 4 prompts.

    Each entry shape: `{"scene_id": ..., "synopsis": ..., "demand": ...,
    "why_distinct": ..., "neutral_draft": ...}` — match `AdjacentSceneBench.drafts`.
    Drafts are 150-300 words each; cached in the prefix on subsequent calls.
    """
    blocks: list[str] = []
    for i, d in enumerate(bench_drafts, 1):
        block_lines = [
            f"--- Scene {i}: `{d['scene_id']}` ---",
            f"demand: {d['demand']}",
            f"why_distinct: {d['why_distinct']}",
            "",
            "neutral draft (transform this in your voice):",
            d["neutral_draft"].strip(),
        ]
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


def render_prior_notes(notes_by_phase: dict[str, list[str]]) -> str:
    """Format an agent's own scratchpad notes from prior phases for inclusion
    in a later phase's prompt. Cross-phase ICL signal — the agent's curated
    working memory survives phase boundaries via this channel.

    `notes_by_phase` maps a label (e.g., "phase 1", "phase 4") to a list of
    note strings the agent wrote during that phase. Empty lists are skipped
    silently — when the agent didn't write notes, nothing surfaces.
    """
    sections: list[str] = []
    for label, notes in notes_by_phase.items():
        if not notes:
            continue
        sections.append(f"--- Your notes from {label} ---")
        for n in notes:
            sections.append(f"- {n.strip()}")
        sections.append("")
    if not sections:
        return "(no prior notes — you didn't use note_to_self earlier)"
    return "\n".join(sections).strip()


# Backwards-compat alias — earlier callers used render_bench_summary.
# The summary-only form is no longer used in v0.1 prompts, but kept
# importable for any external callers / scripts referencing it.
def render_bench_summary(bench_drafts: list[dict]) -> str:
    """DEPRECATED — kept for backwards compat. Use `render_bench_drafts`.

    The slim-summary form was the original Phase 1 prompt input; v0.1
    inlines the full drafts to avoid wasting `render_adjacent_scene`
    tool-use iterations on bench fetches.
    """
    lines: list[str] = []
    for d in bench_drafts:
        lines.append(f"- scene_id: {d['scene_id']}")
        lines.append(f"  demand: {d['demand']}")
        lines.append(f"  why_distinct: {d['why_distinct']}")
    return "\n".join(lines)


def build_adjacent_scene_drafter_prompt(
    concept: ConceptGenome,
    *,
    scene_id: str,
    synopsis: str,
) -> tuple[str, str]:
    """Assemble (system_msg, user_msg) for one neutral-voice draft call.

    Drafter renders one picked scene flat. One call per scene so each draft
    gets the model's full attention budget — the drafts are the baseline
    against which voice transformations are measured, so register-bleed
    between drafts is a real risk if batched.

    The picked scene's `demand` field is intentionally NOT passed to the
    drafter. The demand is voice-side context (what the bench is testing
    downstream); passing it to a prose-generating prompt risks literalization
    ("Render slow domestic time without filler" → prose saying "she moved
    slowly"). The demand stays attached to the scene record for downstream
    voice agents and judges.
    """
    system_msg = load_base_system()
    template = _load("adjacent_scene_drafter.txt")
    concept_context = _render_drafter_concept_context(concept)
    user_msg = (
        template
        .replace("{CONCEPT_CONTEXT}", concept_context)
        .replace("{SCENE_ID}", scene_id)
        .replace("{SCENE_SYNOPSIS}", synopsis)
    )
    return system_msg, user_msg
