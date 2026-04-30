"""Stage 3 voice-session composition.

Ties the upstream artifacts (cast list from `cast_voice_panel`, adjacent-
scene bench from `generate_adjacent_scenes`, concept + DAG from earlier
stages) into the orchestration framework: builds Agent objects with
exemplar-loaded persona system prompts, builds the ToolRegistry with the
phase allowlist, runs the 4 v0.1 phases, picks the winner by Borda, and
returns a VoiceSessionResult.

The persona system prompt is assembled here (not in `personas.py`) so the
loader stays focused on parsing while the session composer owns the
runtime persona-rendering: workshop frame, aesthetic body, exemplar
passages loaded from `data/voice-references/passages/`.
"""

from __future__ import annotations

import logging
from contextvars import Token
from pathlib import Path

from owtn.llm.call_logger import llm_log_dir as llm_log_dir_var
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_3 import (
    Critique,
    VoiceGenome,
    VoiceSessionConfig,
    VoiceSessionResult,
)
from owtn.orchestration import (
    Agent,
    SessionState,
    ToolRegistry,
    run_session,
    session_log_dir as session_log_dir_var,
)
from owtn.stage_3.adjacent_scenes import AdjacentSceneBench
from owtn.stage_3.personas import VoicePersona
from owtn.stage_3.phases import (
    BordaPhase,
    PrivateBriefPhase,
    RevealCritiquePhase,
    RevisePhase,
)
from owtn.stage_3.tools import ALL_VOICE_TOOLS, VOICE_PHASE_ALLOW

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_DIR = REPO_ROOT / "data" / "voice-references" / "passages"
"""Where exemplar prose lives — `<PASSAGES_DIR>/<category>/<id>.txt`.

The persona references exemplars by id; we resolve by walking the
category subdirs (`exemplars`, `baselines`, `defaults`) and reading the
first file whose stem matches.
"""


# ─── Persona system prompt assembly ──────────────────────────────────────


def _load_passage(passage_id: str) -> str | None:
    """Read a passage file by id; returns None if no matching .txt found."""
    for category in ("exemplars", "baselines", "defaults"):
        path = PASSAGES_DIR / category / f"{passage_id}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return None


def render_persona_system_prompt(persona: VoicePersona) -> str:
    """Compose a voice agent's system prompt from the persona YAML.

    Layout: identity → commitments → aversions → obsessions → epistemic
    skepticism → demonstrations → exemplar passages (each with a one-line
    note introducing what the passage exemplifies).

    Exemplars are required for voice work per `feedback_few_shot_exemplars
    _for_voice` — verbal description alone produces default-shaped prose.
    Missing passages are logged and skipped (the persona can still run on
    its verbal body, but flag pool drift).
    """
    parts: list[str] = [f"You are: {persona.name}."]
    parts.append("")
    parts.append(persona.identity.strip())
    parts.append("")

    parts.append("Aesthetic commitments (ranked; first is load-bearing):")
    for i, c in enumerate(persona.aesthetic_commitments, 1):
        parts.append(f"  {i}. {c}")
    parts.append("")

    if persona.aversions:
        parts.append("Aversions:")
        for a in persona.aversions:
            parts.append(f"  - {a}")
        parts.append("")

    if persona.obsessions:
        parts.append("Productive obsessions:")
        for o in persona.obsessions:
            parts.append(f"  - {o}")
        parts.append("")

    parts.append("Epistemic skepticism — distrust the first natural-feeling formulation:")
    parts.append(persona.epistemic_skepticism.strip())
    parts.append("")

    if persona.demonstrations:
        parts.append("Demonstrations of the voice in working mode:")
        for d in persona.demonstrations:
            parts.append(d.strip())
            parts.append("")

    if persona.exemplars:
        parts.append("Voice exemplars — passages that exemplify what your voice reaches for:")
        parts.append("")
        for ex in persona.exemplars:
            text = _load_passage(ex.id)
            if text is None:
                logger.warning("persona %s: exemplar %r not found in passages dir",
                               persona.id, ex.id)
                continue
            parts.append(f"### {ex.id}")
            parts.append(f"What this exemplifies: {ex.note}")
            parts.append("")
            parts.append(text)
            parts.append("")

    return "\n".join(parts).strip()


def build_voice_agent(persona: VoicePersona, *, generator_model: str | None = None) -> Agent:
    """Construct an orchestration Agent from a voice-pool persona.

    Sampler params come from the persona YAML (temperature, top_p, etc.).
    Tool allowlist is the union of all voice tools — gating happens at
    the per-phase level via `VOICE_PHASE_ALLOW`.

    `generator_model` overrides the persona's own `model` choice when set
    (Stage 3 config controls this; the persona YAML's model field is the
    persona-author's hint, not authoritative). Falls back to the persona's
    first model when generator_model is None, then to a sane default.
    """
    sampler: dict[str, object] = {"temperature": persona.temperature}
    if persona.top_p is not None:
        sampler["top_p"] = persona.top_p
    if persona.top_k is not None:
        sampler["top_k"] = persona.top_k
    if persona.min_p is not None:
        sampler["min_p"] = persona.min_p

    if generator_model is not None:
        model = generator_model
    elif persona.model:
        model = persona.model[0]
    else:
        model = "deepseek-v4-pro"

    return Agent(
        id=persona.id,
        system_prompt=render_persona_system_prompt(persona),
        model=model,
        sampler=sampler,
        tools=frozenset({spec.name for spec in ALL_VOICE_TOOLS}),
        metadata={"display_name": persona.name},
    )


# ─── Session entry ───────────────────────────────────────────────────────


async def run_voice_session(
    *,
    cast: list[VoicePersona],
    bench: AdjacentSceneBench,
    concept: ConceptGenome,
    dag_rendering: str,
    pair_id: str,
    session_log_dir: Path | None = None,
    config: VoiceSessionConfig | None = None,
) -> VoiceSessionResult:
    """Run one Stage 3 voice session end-to-end (v0.1).

    Phases run in order: PrivateBrief → RevealCritique → Revise → Borda.
    Phase 2 (judge consultation) deferred to v0.2. The winner is the
    agent with the highest Borda point total (writers' verdict); ties
    broken alphabetically.

    Returns a VoiceSessionResult; the orchestration log tree is written
    under `session_log_dir` if provided. Pass `config` to override phase
    iteration budgets, commit sampler, and analytical reasoning effort;
    defaults from the phase dataclasses apply when omitted.
    """
    if not cast:
        raise ValueError("run_voice_session requires a non-empty cast")

    generator_model = config.generator_model if config is not None else None
    agents = [build_voice_agent(p, generator_model=generator_model) for p in cast]
    state = SessionState.new(agents, pair_id=pair_id)
    state.payload["concept"] = concept
    state.payload["dag_rendering"] = dag_rendering
    state.payload["adjacent_scene_bench"] = bench.model_dump()
    state.payload["agent_models"] = {a.id: a.model for a in agents}

    registry = ToolRegistry(ALL_VOICE_TOOLS, per_phase_allow=VOICE_PHASE_ALLOW)
    if config is None:
        phases = [
            PrivateBriefPhase(),
            RevealCritiquePhase(),
            RevisePhase(),
            BordaPhase(),
        ]
    else:
        commit_t = config.commit_sampler.temperature
        commit_mt = config.commit_sampler.max_tokens
        phases = [
            PrivateBriefPhase(
                explore_max_iters=config.phase_1.explore_max_iters,
                commit_temperature=commit_t,
                commit_max_tokens=commit_mt,
            ),
            RevealCritiquePhase(
                reasoning_effort=config.analytical_reasoning_effort,
            ),
            RevisePhase(
                explore_max_iters=config.phase_4.explore_max_iters,
                commit_temperature=commit_t,
                commit_max_tokens=commit_mt,
            ),
            BordaPhase(
                reasoning_effort=config.analytical_reasoning_effort,
            ),
        ]

    session_token: Token | None = None
    llm_token: Token | None = None
    if session_log_dir is not None:
        session_log_dir.mkdir(parents=True, exist_ok=True)
        session_token = session_log_dir_var.set(str(session_log_dir))
        # Per-LLM-call YAMLs land at <session_log_dir>/llm/<model>/NNNN.yaml,
        # alongside the session/phases/agents trees. Mirrors the Stage 1/2
        # pattern of pointing both context vars at the same run root so the
        # LLM trace and the orchestration trace coexist.
        llm_token = llm_log_dir_var.set(str(session_log_dir))

    try:
        state = await run_session(state, phases, registry)
    finally:
        if session_token is not None:
            session_log_dir_var.reset(session_token)
        if llm_token is not None:
            llm_log_dir_var.reset(llm_token)

    return _assemble_result(state, session_log_dir)


def _assemble_result(
    state: SessionState,
    session_log_dir_path: Path | None,
) -> VoiceSessionResult:
    revised: dict[str, VoiceGenome] = state.payload.get("phase_4_revise", {})
    critiques_by_critic: dict[str, list[Critique]] = (
        state.payload.get("phase_3_reveal_critique", {})
    )
    borda = state.payload.get("phase_5_borda", {})
    points: dict[str, int] = borda.get("points", {})

    if not revised:
        raise RuntimeError("session ended without phase_4_revise output; cannot pick winner")

    # Tie-break: highest Borda points; on equal points, alphabetically earliest id.
    sorted_ids = sorted(revised.keys(), key=lambda aid: (-points.get(aid, 0), aid))
    winner_id = sorted_ids[0]

    flat_critiques: list[Critique] = []
    for lst in critiques_by_critic.values():
        flat_critiques.extend(lst)

    return VoiceSessionResult(
        pair_id=state.pair_id or "unknown",
        winner=revised[winner_id],
        proposals=list(revised.values()),
        critiques=flat_critiques,
        borda_points=points,
        cost_usd=state.cost_usd,
        session_log_dir=str(session_log_dir_path) if session_log_dir_path else "",
    )
