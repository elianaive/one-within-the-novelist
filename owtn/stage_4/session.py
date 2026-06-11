"""Stage 4 session composer.

`run_stage_4_session` wires the upstream artifacts (concept, DAG,
voice_genome) plus a Stage4Config into the orchestration framework:
runs the pre-stage filter (audience + experts), builds the writer
agent, scaffolds `story.md` and `pre_think.md`, sets up state.payload,
runs the three phases via `run_session`, returns a Stage4SessionResult.

Mirrors `owtn.stage_3.session.run_voice_session`; per-tuple shape is
single-agent (vs Stage 3's panel) and file-as-source-of-truth.
"""

from __future__ import annotations

import logging
from contextvars import Token
from pathlib import Path

from owtn.llm.call_logger import llm_log_dir as llm_log_dir_var
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG
from owtn.models.stage_3 import VoiceGenome
from owtn.models.stage_4 import (
    AudienceFraming,
    ExpertNeedsList,
    Stage4Config,
    Stage4SessionResult,
)
from owtn.orchestration import (
    Agent,
    SessionState,
    ToolRegistry,
    run_session,
    session_log_dir as session_log_dir_var,
)
from owtn.prompts.stage_4 import build_writer_system_prompt
from owtn.stage_2.rendering import render as render_dag
from owtn.stage_3.personas import load_persona_pool
from owtn.stage_4.critics import CriticRegistry
from owtn.stage_4.drafter import DownDraftPhase
from owtn.stage_4.filter import run_stage_4_filter
from owtn.stage_4.manuscript import scaffold_from_dag, write_text
from owtn.stage_4.personas import load_critic_pool
from owtn.stage_4.plateau import PlateauDetector
from owtn.stage_4.prethink import PreThinkPhase
from owtn.stage_4.revise import RevisePhase
from owtn.stage_4.tools import ALL_STAGE_4_TOOLS, STAGE_4_PHASE_ALLOW
from owtn.stage_4.domain_expert import instantiate_domain_experts
from owtn.stage_4.voice_fidelity import (
    VOICE_FIDELITY_ID,
    promote_voice_persona_to_critic,
)


logger = logging.getLogger(__name__)


WRITER_AGENT_ID = "stage_4_writer"


# ─── Scaffolding helpers ─────────────────────────────────────────────────


def _scaffold_pre_think(dag: DAG) -> str:
    """Empty `## scene_id` headings for each DAG node, one per scene, in
    topological order. The writer agent fills the bodies during PreThink."""
    topo = dag._check_acyclic_and_topo()
    ordered_ids = sorted(topo.keys(), key=lambda nid: topo[nid])
    parts: list[str] = []
    for nid in ordered_ids:
        parts.append(f"## {nid}\n\n")
    return "\n".join(parts).rstrip() + "\n"


def _build_writer_agent(
    *,
    voice_genome: VoiceGenome,
    concept: ConceptGenome,
    dag_rendering: str,
    audience_framing: AudienceFraming | None,
    model: str,
) -> Agent:
    """Compose the writer agent. No persona — the voice spec carries the
    aesthetic stance per the architecture's "no writer-persona" rule."""
    system_prompt = build_writer_system_prompt(
        voice_genome=voice_genome,
        concept=concept,
        dag_rendering=dag_rendering,
        audience_framing=audience_framing,
    )
    return Agent(
        id=WRITER_AGENT_ID,
        system_prompt=system_prompt,
        model=model,
        sampler={},  # phases set their own per-phase sampler
        tools=frozenset(spec.name for spec in ALL_STAGE_4_TOOLS),
        metadata={"display_name": "stage 4 writer"},
    )


# ─── Voice fidelity promotion ────────────────────────────────────────────


def _promote_voice_fidelity(
    registry: CriticRegistry,
    stage_3_persona_id: str,
    stage_3_pool_dir: Path | str | None,
    *,
    critic_model: str,
) -> CriticRegistry:
    """Replace the stub voice_fidelity in `registry` with a promoted
    version sourced from the Stage 3 winning persona.

    Returns a new `CriticRegistry` (registries are read-only after
    construction). When the Stage 3 persona can't be located, logs a
    warning and returns the registry unchanged — voice_fidelity falls
    back to the stub behaviour.
    """
    pool = load_persona_pool(stage_3_pool_dir)
    stage_3_persona = next((p for p in pool if p.id == stage_3_persona_id), None)
    if stage_3_persona is None:
        logger.warning(
            "Stage 3 persona %r not found in pool; voice_fidelity stub "
            "will be used instead", stage_3_persona_id,
        )
        return registry
    promoted = promote_voice_persona_to_critic(
        stage_3_persona,
        model=critic_model,
    )
    return registry.with_replaced(promoted)


# ─── Public entry ────────────────────────────────────────────────────────


async def run_stage_4_session(
    *,
    concept: ConceptGenome,
    dag: DAG,
    voice_genome: VoiceGenome,
    tuple_id: str,
    run_dir: Path | str,
    session_log_dir: Path | str | None = None,
    config: Stage4Config | None = None,
    audience_framing: AudienceFraming | None = None,
    expert_needs: ExpertNeedsList | None = None,
    critic_registry: CriticRegistry | None = None,
    stage_3_winner_persona_id: str | None = None,
    stage_3_persona_pool_dir: Path | str | None = None,
) -> Stage4SessionResult:
    """Run one Stage 4 session end-to-end on a (concept, struct, voice) tuple.

    `audience_framing` / `expert_needs` are normally produced by the
    pre-stage filter; pass them in to skip the filter call (useful for
    tests). `critic_registry` similarly defaults to loading from
    `configs/stage_4/critics/`; pass in to inject a test registry.

    `stage_3_winner_persona_id` (typically `voice_genome.agent_id`) and
    `stage_3_persona_pool_dir` enable the voice_fidelity promotion: the
    Stage 3 winning persona's identity is loaded and supersedes the
    stub voice_fidelity in the registry. Without these, the stub stays
    in place — useful for dev runs where the Stage 3 dossier isn't
    around.
    """
    cfg = config or Stage4Config()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    dag_rendering = render_dag(dag)

    if audience_framing is None or expert_needs is None:
        af, en = await run_stage_4_filter(
            concept=concept,
            voice_genome=voice_genome,
            dag_rendering=dag_rendering,
            config=cfg.filter,
        )
        if audience_framing is None:
            audience_framing = af
        if expert_needs is None:
            expert_needs = en

    if critic_registry is None:
        critic_registry = CriticRegistry(load_critic_pool())
    if len(critic_registry) == 0:
        raise RuntimeError("critic registry is empty; cannot run Stage 4")

    if stage_3_winner_persona_id is not None:
        critic_registry = _promote_voice_fidelity(
            critic_registry,
            stage_3_winner_persona_id,
            stage_3_persona_pool_dir,
            critic_model=critic_registry.get(VOICE_FIDELITY_ID).model,
        )

    if expert_needs is not None and expert_needs.experts:
        for spec in instantiate_domain_experts(expert_needs):
            critic_registry = critic_registry.with_replaced(spec)

    sandbox_dir = run_dir / "sandbox" / WRITER_AGENT_ID
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    story_path = sandbox_dir / "story.md"
    pre_think_path = sandbox_dir / "pre_think.md"
    write_text(story_path, scaffold_from_dag(dag))
    write_text(pre_think_path, _scaffold_pre_think(dag))

    agent = _build_writer_agent(
        voice_genome=voice_genome,
        concept=concept,
        dag_rendering=dag_rendering,
        audience_framing=audience_framing,
        model=cfg.generator_model,
    )
    state = SessionState.new([agent], pair_id=tuple_id)
    state.payload["concept"] = concept
    state.payload["voice_genome"] = voice_genome
    state.payload["dag_rendering"] = dag_rendering
    state.payload["audience_framing"] = audience_framing
    state.payload["expert_needs"] = expert_needs
    state.payload["critic_registry"] = critic_registry
    state.payload["run_dir"] = str(run_dir)
    state.payload["sandbox_dir"] = str(sandbox_dir)
    state.payload["story_path"] = str(story_path)
    state.payload["pre_think_path"] = str(pre_think_path)
    state.payload["agent_models"] = {agent.id: agent.model}
    state.payload["surgical_edit_config"] = cfg.surgical_edit

    registry = ToolRegistry(ALL_STAGE_4_TOOLS, per_phase_allow=STAGE_4_PHASE_ALLOW)
    phases = [
        PreThinkPhase(
            explore_max_iters=cfg.prethink.explore_max_iters,
            reasoning_effort=cfg.prethink.reasoning_effort,
        ),
        DownDraftPhase(
            explore_max_iters=cfg.downdraft.explore_max_iters,
            temperature=cfg.downdraft.temperature,
        ),
        RevisePhase(
            cycle_cap=cfg.revise.cycle_cap,
            call_ceiling=cfg.revise.call_ceiling,
            gather_max_iters=cfg.revise.gather_max_iters,
            revise_max_iters=cfg.revise.revise_max_iters,
            tier_b_in_cycle_zero=cfg.revise.tier_b_in_cycle_zero,
            reasoning_effort=cfg.revise.reasoning_effort,
            plateau_detector=PlateauDetector(
                window=cfg.plateau.window,
                require_total_decrease=cfg.plateau.require_total_decrease,
                require_severe_progress=cfg.plateau.require_severe_progress,
            ),
        ),
    ]

    session_token: Token | None = None
    llm_token: Token | None = None
    if session_log_dir is not None:
        log_dir = Path(session_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        session_token = session_log_dir_var.set(str(log_dir))
        llm_token = llm_log_dir_var.set(str(log_dir))

    try:
        state = await run_session(state, phases, registry)
    finally:
        if session_token is not None:
            session_log_dir_var.reset(session_token)
        if llm_token is not None:
            llm_log_dir_var.reset(llm_token)

    revise_state = state.payload.get("phase_3_revise", {})
    return Stage4SessionResult(
        tuple_id=tuple_id,
        manuscript_path=str(story_path),
        pre_think_path=str(pre_think_path),
        run_dir=str(run_dir),
        cost_usd=state.cost_usd,
        cycles_completed=revise_state.get("cycles_completed", 0),
        exit_reason=revise_state.get("exit_reason", "unknown"),
        session_log_dir=str(session_log_dir) if session_log_dir else "",
    )
