"""Stage 3 voice-session phase implementations.

Four phases compose the v0.1 session (Phase 2 deferred to v0.2):
- `PrivateBriefPhase` — explore-then-commit; agents draft voice specs
- `RevealCritiquePhase` — single-pass structured critique (no tools)
- `RevisePhase` — explore-then-commit revision under critique
- `BordaPhase` — rank others' final proposals (no tools)

Each phase fans out per-agent via `asyncio.gather` and wraps each
coroutine in `push_llm_context(agent_id=...)` so concurrent calls land in
the right per-agent log buckets. Phase exceptions abort the session per
the orchestrator's contract; per-agent retries (if any) live inside the
phase, not in the orchestrator.

Phase 1 / Phase 4 use the explore-then-commit shape: a tool-use loop
followed by a structured-output call that reuses the explore history. The
shape is repeated; once a third consumer arrives we promote it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from owtn.llm.query import query_async
from owtn.llm.tool_use import query_async_with_tools
from owtn.models.stage_3 import (
    BordaRanking,
    Critique,
    CritiqueSet,
    VoiceGenome,
    VoiceGenomeBody,
)
from owtn.orchestration import (
    Agent,
    SessionState,
    ToolContext,
    ToolRegistry,
    push_llm_context,
    session_log_path,
    write_yaml,
)
from owtn.prompts.stage_3 import (
    build_voice_borda_prompt,
    build_voice_brief_prompt,
    build_voice_commit_user_msg,
    build_voice_critique_prompt,
    build_voice_revise_prompt,
)
from owtn.stage_3.voting import borda_no_self_vote


logger = logging.getLogger(__name__)


# ─── Phase 1 — private brief ─────────────────────────────────────────────


COMMIT_TEMPERATURE = 0.6
"""Override temperature for structured-output commit calls. Voice agents
run their explore loops at the persona's natural temperature (typically
high for creative exploration); long structured outputs at that temp
destabilize and degenerate into word-soup near the end of generation.
The commit just packages the agent's already-decided voice work into
schema fields — exploration's done — so a lower temp suits."""


COMMIT_MAX_TOKENS = 16384
"""Output cap for structured commits. DeepSeek's default cap can truncate
dense concepts whose 3 renderings + verbal fields exceed the implicit
limit, producing whitespace-tail JSON parse failures. 16K is generous;
actual prose-bearing output is typically 4–8K tokens."""


def _extract_committed_body(state: SessionState, agent_id: str) -> "VoiceGenomeBody | None":
    """Pop the agent's tool-call commit from `state.payload["_pending_commits"]`.

    `finalize_voice_genome`'s handler stashes a validated `VoiceGenomeBody`
    here when the agent calls the tool. After the explore loop returns,
    the phase pops the body out and uses it directly — no separate
    structured-output commit call needed. Returns None when the agent
    didn't call the tool (orchestrator falls back to the legacy commit).
    """
    commits = state.payload.get("_pending_commits", {})
    return commits.pop(agent_id, None)


def _commit_sampler(
    persona_sampler: dict,
    *,
    temperature: float = COMMIT_TEMPERATURE,
    max_tokens: int = COMMIT_MAX_TOKENS,
) -> dict:
    """Override temperature + max_tokens for structured prose-bearing commits.

    Used by Phase 1 / Phase 4 commits — these produce VoiceGenome with
    prose renderings. Reasoning stays disabled because (a) memory
    `project_voice_api_techniques.md` notes thinking mode is "more
    detached" / unhelpful for prose; (b) per-token cost is bounded.
    """
    out = dict(persona_sampler)
    out["temperature"] = temperature
    out["max_tokens"] = max_tokens
    return out


ANALYTICAL_REASONING_EFFORT = "medium"
"""Reasoning effort for the analytical phases (Phase 3 critique, Phase 5
Borda). These are not prose-producing; they're judgment / ranking tasks
where reasoning materially helps. Stage 1's casting classifier uses the
same default; the consistency is intentional."""


def _analytical_sampler(
    persona_sampler: dict,
    *,
    temperature: float = COMMIT_TEMPERATURE,
    reasoning_effort: str = ANALYTICAL_REASONING_EFFORT,
) -> dict:
    """Sampler for analytical-phase commits (Phase 3 critique, Phase 5 Borda).

    Lower temp like the prose commit sampler, but with reasoning ENABLED —
    these phases are judgment tasks (specific-strength + concern; ranking)
    where extended thinking improves quality without the prose-detachment
    cost. DeepSeek's `build_call_kwargs` translates `reasoning_effort` to
    the right `extra_body.thinking` shape.

    Note: when `reasoning_effort` is non-disabled, deepseek's
    `build_call_kwargs` floors `max_tokens` to 32768 to cover reasoning +
    visible output. We don't need to set it explicitly here.
    """
    out = dict(persona_sampler)
    out["temperature"] = temperature
    out["reasoning_effort"] = reasoning_effort
    return out


def _summarize_tool_calls(msg_history: list[dict]) -> list[dict]:
    """Walk a tool-use loop's message history and surface what tools fired.

    Returns one entry per tool call in order: `{tool, args_keys}`. Args are
    not included verbatim (large prose passages would balloon the trace);
    only the parameter names so a reader can see at a glance what the
    agent invoked. Used by per-agent log records for inspection.
    """
    import json as _json

    out: list[dict] = []
    for msg in msg_history:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name", "?")
            args_raw = fn.get("arguments", "")
            try:
                args = _json.loads(args_raw) if args_raw else {}
                args_keys = sorted(args.keys()) if isinstance(args, dict) else []
            except Exception:
                args_keys = ["<unparsed>"]
            out.append({"tool": name, "args_keys": args_keys})
    return out


def _format_transcript(
    *,
    title: str,
    system_msg: str,
    msg_history: list[dict],
    final_output: str | None = None,
) -> str:
    """Render an agent's full LLM chain (system + user + assistant + tool
    results) as a Markdown chat transcript — readable like a chatbot history.

    Tool-use loops collapse to one composite QueryResult in the LLM call log,
    which hides the back-and-forth. This helper reads the full
    `new_msg_history` (which DOES contain every turn, including assistant
    `tool_calls` and `tool` results) and renders it as a flat markdown
    document under `<session_log_dir>/agents/<id>/<phase>.transcript.md`.

    `final_output`, if provided, is appended as the last assistant turn —
    useful when the structured-commit response isn't already in the
    msg_history (Phase 1/4 explore→commit shape).
    """
    import json as _json

    parts: list[str] = [f"# {title}", ""]

    # System prompt — collapsed by default since it's long; surface a
    # truncated preview plus the full text in a fenced block for readability.
    parts.append("## System")
    parts.append("")
    parts.append("```")
    parts.append(system_msg.strip())
    parts.append("```")
    parts.append("")

    turn = 0
    for msg in msg_history:
        role = msg.get("role")
        if role == "user":
            turn += 1
            parts.append(f"## Turn {turn} — User")
            parts.append("")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multi-block content (anthropic shape) — flatten
                content = "\n\n".join(
                    str(b.get("text", b)) if isinstance(b, dict) else str(b)
                    for b in content
                )
            parts.append(str(content).strip())
            parts.append("")

        elif role == "assistant":
            turn += 1
            parts.append(f"## Turn {turn} — Assistant")
            parts.append("")
            # Reasoning content (deepseek `reasoning_content`, anthropic
            # extended-thinking `thinking` blocks). Surface in a collapsed-
            # looking block so a reader can scan past it but it's there.
            reasoning = msg.get("reasoning_content") or ""
            if reasoning and str(reasoning).strip():
                parts.append("> **reasoning:**")
                parts.append("> ")
                for ln in str(reasoning).strip().split("\n"):
                    parts.append(f"> {ln}")
                parts.append("")
            text = msg.get("content") or ""
            if isinstance(text, list):
                text = "\n\n".join(
                    str(b.get("text", b)) if isinstance(b, dict) else str(b)
                    for b in text
                )
            if text and str(text).strip():
                parts.append(str(text).strip())
                parts.append("")
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "?")
                args_raw = fn.get("arguments", "")
                try:
                    args = _json.loads(args_raw) if args_raw else {}
                    args_pretty = _json.dumps(args, indent=2, ensure_ascii=False)
                except Exception:
                    args_pretty = args_raw
                parts.append(f"### → tool call: `{name}`")
                parts.append("")
                parts.append("```json")
                parts.append(args_pretty)
                parts.append("```")
                parts.append("")

        elif role == "tool":
            tcid = msg.get("tool_call_id", "?")
            parts.append(f"### ← tool result (`{tcid}`)")
            parts.append("")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            # Tool results are JSON-shaped; preserve as a fenced block.
            # Cap individual results at 32k chars to keep the transcript
            # readable when corpus lookups return very large passage dumps.
            parts.append("```")
            parts.append(content.strip()[:32000])
            if len(content) > 32000:
                parts.append(f"... [{len(content) - 32000} more chars truncated]")
            parts.append("```")
            parts.append("")

    if final_output is not None:
        # Display the final artifact for human readability, but as a
        # separate section — NOT a turn. For tool-call commits the
        # finalize_voice_genome call is already in the message history
        # above; appending it as "Turn N+1 (structured commit)" would
        # fabricate a turn that didn't happen.
        parts.append("---")
        parts.append("")
        parts.append("## Final committed artifact")
        parts.append("")
        if isinstance(final_output, str):
            text = final_output
        else:
            try:
                text = _json.dumps(final_output, indent=2, ensure_ascii=False, default=str)
            except Exception:
                text = str(final_output)
        parts.append("```")
        parts.append(text.strip())
        parts.append("```")
        parts.append("")

    return "\n".join(parts)


def _write_transcript(
    *,
    agent_id: str,
    phase_name: str,
    title: str,
    system_msg: str,
    msg_history: list[dict],
    final_output: object | None = None,
) -> None:
    """Write `agents/<agent_id>/<phase_name>.transcript.md` if logging is on."""
    path = session_log_path("agents", agent_id, f"{phase_name}.transcript.md")
    if path is None:
        return
    md = _format_transcript(
        title=title,
        system_msg=system_msg,
        msg_history=msg_history,
        final_output=final_output if isinstance(final_output, str) else (
            None if final_output is None else (
                final_output.model_dump_json(indent=2)
                if hasattr(final_output, "model_dump_json")
                else str(final_output)
            )
        ),
    )
    try:
        path.write_text(md, encoding="utf-8")
    except Exception as e:
        logger.warning("failed to write transcript %s: %s", path, e)


@dataclass
class PrivateBriefPhase:
    """Phase 1 — each agent independently drafts a VoiceGenome.

    Two-stage per agent:
        1. Explore via tool-use loop (render_adjacent_scene, note_to_self,
           lookup_reference, stylometry, slop_score, writing_style).
        2. Commit via structured-output call with `VoiceGenomeBody` and
           the explore history as `msg_history`. The orchestrator attaches
           `pair_id` and `agent_id` to construct the full VoiceGenome.

    Per-agent failure: a single agent's explore or commit failure aborts
    the whole phase per orchestration contract. Per-agent retries can be
    added when pilot data shows we need them.
    """

    name: str = "phase_1_private_brief"
    explore_max_iters: int = 30
    commit_temperature: float = COMMIT_TEMPERATURE
    commit_max_tokens: int = COMMIT_MAX_TOKENS

    async def run(
        self,
        agents: list[Agent],
        state: SessionState,
        registry: ToolRegistry,
    ) -> SessionState:
        bench = state.payload.get("adjacent_scene_bench")
        concept = state.payload.get("concept")
        dag_rendering = state.payload.get("dag_rendering", "")
        if bench is None or concept is None:
            raise ValueError(
                "PrivateBriefPhase requires 'adjacent_scene_bench' and "
                "'concept' in state.payload"
            )

        user_msg = build_voice_brief_prompt(concept, dag_rendering, bench["drafts"])

        proposals = await asyncio.gather(*[
            self._run_one_agent(agent, user_msg, state, registry)
            for agent in agents
        ])

        state.payload[self.name] = {agent.id: g for agent, g in zip(agents, proposals)}
        return state

    async def _run_one_agent(
        self,
        agent: Agent,
        user_msg: str,
        state: SessionState,
        registry: ToolRegistry,
    ) -> VoiceGenome:
        with push_llm_context(agent_id=agent.id):
            tool_schemas = registry.schemas_for(agent.tools, self.name)

            async def dispatch(tool_name: str, params: dict) -> str:
                ctx = ToolContext(
                    session_id=state.session_id,
                    phase_id=self.name,
                    agent_id=agent.id,
                    state_view=state.payload,
                )
                return await registry.dispatch(tool_name, params, ctx)

            explore_result = await query_async_with_tools(
                model_name=agent.model,
                msg=user_msg,
                system_msg=agent.system_prompt,
                tools=tool_schemas,
                dispatch=dispatch,
                max_iters=self.explore_max_iters,
                **dict(agent.sampler),
            )
            state.cost_usd += explore_result.cost

            body = _extract_committed_body(state, agent.id)
            commit_cost = 0.0
            commit_turns = 0
            if body is None:
                # Tool-use nudge fallback: the agent ran out of explore
                # iterations without calling finalize_voice_genome. Stay in
                # tool-use mode (same modality the agent has been using) and
                # restrict tools to finalize_voice_genome only — no thinking
                # or diagnostics, just commit. Switching to structured-output
                # mode here triggers DeepSeek failure under long thinking-mode
                # msg_history.
                finalize_only = [
                    s for s in tool_schemas if s["name"] == "finalize_voice_genome"
                ]
                nudge_result = await query_async_with_tools(
                    model_name=agent.model,
                    msg=(
                        "Your exploration budget is exhausted. Call "
                        "`finalize_voice_genome` now with the VoiceGenome "
                        "body based on the work above. Use the bench's "
                        "scene_ids exactly. Do not call any other tools."
                    ),
                    system_msg=agent.system_prompt,
                    msg_history=explore_result.new_msg_history,
                    tools=finalize_only,
                    dispatch=dispatch,
                    max_iters=2,
                    **_commit_sampler(
                        agent.sampler,
                        temperature=self.commit_temperature,
                        max_tokens=self.commit_max_tokens,
                    ),
                )
                state.cost_usd += nudge_result.cost
                commit_cost = nudge_result.cost
                commit_turns = len(nudge_result.new_msg_history) - len(explore_result.new_msg_history)
                body = _extract_committed_body(state, agent.id)
                if body is None:
                    raise RuntimeError(
                        f"agent {agent.id}: tool-use nudge fallback failed "
                        f"to elicit finalize_voice_genome call after explore "
                        f"loop hit max_iters"
                    )

            genome = VoiceGenome(
                **body.model_dump(),
                pair_id=state.pair_id or "unknown",
                agent_id=agent.id,
            )

            self._validate_renderings_match_bench(genome, state)
            self._write_agent_record(agent, genome, state, explore_result, commit_cost)
            _write_transcript(
                agent_id=agent.id,
                phase_name=self.name,
                title=f"Phase 1 (private brief) — {agent.id}",
                system_msg=agent.system_prompt,
                msg_history=explore_result.new_msg_history,
                final_output=genome,
            )

            logger.info(
                "phase_1: agent %s committed (cost $%.4f, %d explore turns)",
                agent.id,
                explore_result.cost + commit_cost,
                len(explore_result.new_msg_history),
            )
            return genome

    @staticmethod
    def _validate_renderings_match_bench(genome: VoiceGenome, state: SessionState) -> None:
        bench = state.payload["adjacent_scene_bench"]
        bench_ids = {d["scene_id"] for d in bench["drafts"]}
        genome_ids = {r.scene_id for r in genome.renderings}
        if genome_ids != bench_ids:
            raise RuntimeError(
                f"agent {genome.agent_id}: rendering scene_ids {sorted(genome_ids)} "
                f"do not match bench {sorted(bench_ids)}"
            )

    @staticmethod
    def _write_agent_record(
        agent: Agent,
        genome: VoiceGenome,
        state: SessionState,
        explore_result,
        commit_cost: float,
    ) -> None:
        path = session_log_path("agents", agent.id, "phase_1_private_brief.yaml")
        if path is None:
            return
        write_yaml(
            path,
            {
                "agent_id": agent.id,
                "model": agent.model,
                "session_id": state.session_id,
                "explore_cost_usd": round(explore_result.cost, 6),
                "commit_cost_usd": round(commit_cost, 6),
                "commit_via": "tool_call" if commit_cost == 0.0 else "structured_output_fallback",
                "explore_turns": len(explore_result.new_msg_history),
                "tool_calls": _summarize_tool_calls(explore_result.new_msg_history),
                "proposal": genome.model_dump(),
            },
        )


# ─── Phase 3 — reveal + critique ─────────────────────────────────────────


@dataclass
class RevealCritiquePhase:
    """Phase 3 — single-pass structured cross-critique (no tools).

    Each agent reads all final Phase 1 proposals and writes a Critique
    on each OTHER proposal. No dialogue, no follow-ups, no debate. The
    conformity literature is convergent on this — multi-turn debate
    produces opinion drift; mediated single-pass extracts cross-pollination
    benefit while limiting conformity risk.

    Reads `state.payload[phase_1_private_brief]` and produces
    `state.payload[phase_3_reveal_critique][critic_id] = list[Critique]`.
    """

    name: str = "phase_3_reveal_critique"
    source_phase: str = "phase_1_private_brief"
    reasoning_effort: str = ANALYTICAL_REASONING_EFFORT

    async def run(
        self,
        agents: list[Agent],
        state: SessionState,
        registry: ToolRegistry,
    ) -> SessionState:
        proposals = state.payload.get(self.source_phase)
        if not proposals:
            raise ValueError(
                f"{self.name} requires '{self.source_phase}' in state.payload"
            )

        proposals_dict = {aid: g.model_dump() for aid, g in proposals.items()}
        critique_lists = await asyncio.gather(*[
            self._run_one_agent(agent, proposals_dict, state)
            for agent in agents
        ])
        state.payload[self.name] = {
            agent.id: lst for agent, lst in zip(agents, critique_lists)
        }
        return state

    async def _run_one_agent(
        self,
        agent: Agent,
        proposals_dict: dict[str, dict],
        state: SessionState,
    ) -> list[Critique]:
        with push_llm_context(agent_id=agent.id):
            others = {aid: p for aid, p in proposals_dict.items() if aid != agent.id}
            user_msg = build_voice_critique_prompt(
                your_agent_id=agent.id,
                your_proposal=proposals_dict[agent.id],
                other_proposals=others,
            )
            result = await query_async(
                model_name=agent.model,
                msg=user_msg,
                system_msg=agent.system_prompt,
                output_model=CritiqueSet,
                **_analytical_sampler(agent.sampler, reasoning_effort=self.reasoning_effort),
            )
            state.cost_usd += result.cost

            cs = result.content
            if not isinstance(cs, CritiqueSet):
                raise RuntimeError(
                    f"{agent.id}: critique commit returned {type(cs).__name__}, "
                    f"expected CritiqueSet"
                )

            received_targets = {c.target_id for c in cs.critiques}
            expected_targets = set(others.keys())
            if received_targets != expected_targets:
                raise RuntimeError(
                    f"{agent.id}: critique target coverage mismatch; "
                    f"got {sorted(received_targets)}, expected {sorted(expected_targets)}"
                )
            if agent.id in received_targets:
                raise RuntimeError(f"{agent.id}: self-critique not allowed")

            critiques = [
                Critique(
                    critic_id=agent.id,
                    target_id=cb.target_id,
                    strengths=list(cb.strengths),
                    concern=cb.concern,
                )
                for cb in cs.critiques
            ]
            self._write_agent_record(agent, critiques, state, result)
            _write_transcript(
                agent_id=agent.id,
                phase_name=self.name,
                title=f"Phase 3 (reveal + critique) — {agent.id}",
                system_msg=agent.system_prompt,
                msg_history=result.new_msg_history,
                final_output=cs,
            )
            return critiques

    @staticmethod
    def _write_agent_record(agent, critiques, state, result) -> None:
        path = session_log_path("agents", agent.id, "phase_3_reveal_critique.yaml")
        if path is None:
            return
        write_yaml(
            path,
            {
                "agent_id": agent.id,
                "session_id": state.session_id,
                "cost_usd": round(result.cost, 6),
                "critiques": [c.model_dump() for c in critiques],
            },
        )


# ─── Phase 4 — revise ────────────────────────────────────────────────────


@dataclass
class RevisePhase:
    """Phase 4 — explore-then-commit revision under critiques + metrics.

    Mirrors Phase 1's two-stage shape; the explore loop has the metric
    ensemble (stylometry / slop_score / writing_style) for round-1
    critique-revise integration. Drops `lookup_reference` per the
    architecture decision: revision should not re-do Phase 1's reference
    search.

    Reads `state.payload[phase_1_private_brief]` and
    `state.payload[phase_3_reveal_critique]`; writes
    `state.payload[phase_4_revise][agent_id] = revised VoiceGenome`.
    """

    name: str = "phase_4_revise"
    explore_max_iters: int = 22
    proposal_phase: str = "phase_1_private_brief"
    critique_phase: str = "phase_3_reveal_critique"
    commit_temperature: float = COMMIT_TEMPERATURE
    commit_max_tokens: int = COMMIT_MAX_TOKENS

    async def run(
        self,
        agents: list[Agent],
        state: SessionState,
        registry: ToolRegistry,
    ) -> SessionState:
        proposals = state.payload.get(self.proposal_phase)
        all_critiques = state.payload.get(self.critique_phase)
        if not proposals or not all_critiques:
            raise ValueError(
                f"{self.name} requires '{self.proposal_phase}' and "
                f"'{self.critique_phase}' in state.payload"
            )

        # Index critiques by target — what each agent receives.
        critiques_by_target: dict[str, list[Critique]] = {a.id: [] for a in agents}
        for critic_id, lst in all_critiques.items():
            for c in lst:
                if c.target_id in critiques_by_target:
                    critiques_by_target[c.target_id].append(c)

        revised = await asyncio.gather(*[
            self._run_one_agent(
                agent, proposals[agent.id], critiques_by_target[agent.id],
                state, registry,
            )
            for agent in agents
        ])
        state.payload[self.name] = {agent.id: g for agent, g in zip(agents, revised)}
        return state

    async def _run_one_agent(
        self,
        agent: Agent,
        prior_proposal: VoiceGenome,
        critiques_received: list[Critique],
        state: SessionState,
        registry: ToolRegistry,
    ) -> VoiceGenome:
        with push_llm_context(agent_id=agent.id):
            bench = state.payload["adjacent_scene_bench"]
            phase_1_notes = (
                state.payload.get("scratchpads", {})
                .get(agent.id, {})
                .get("phase_1_private_brief", [])
            )
            user_msg = build_voice_revise_prompt(
                your_proposal=prior_proposal.model_dump(),
                critiques_received=[c.model_dump() for c in critiques_received],
                bench_drafts=bench["drafts"],
                your_phase_1_notes=phase_1_notes,
            )
            tool_schemas = registry.schemas_for(agent.tools, self.name)

            async def dispatch(tool_name: str, params: dict) -> str:
                ctx = ToolContext(
                    session_id=state.session_id,
                    phase_id=self.name,
                    agent_id=agent.id,
                    state_view=state.payload,
                )
                return await registry.dispatch(tool_name, params, ctx)

            explore_result = await query_async_with_tools(
                model_name=agent.model,
                msg=user_msg,
                system_msg=agent.system_prompt,
                tools=tool_schemas,
                dispatch=dispatch,
                max_iters=self.explore_max_iters,
                **dict(agent.sampler),
            )
            state.cost_usd += explore_result.cost

            # Commit-as-tool-call: prefer the agent's `finalize_voice_genome`
            # call from the explore loop. Falls back to a tool-use nudge
            # (same modality, restricted tool list) if the agent didn't call
            # finalize during explore.
            body = _extract_committed_body(state, agent.id)
            commit_cost = 0.0
            if body is None:
                finalize_only = [
                    s for s in tool_schemas if s["name"] == "finalize_voice_genome"
                ]
                nudge_result = await query_async_with_tools(
                    model_name=agent.model,
                    msg=(
                        "Your revision budget is exhausted. Call "
                        "`finalize_voice_genome` now with the revised "
                        "VoiceGenome body based on the work above. Keep "
                        "the bench's scene_ids. Do not call any other tools."
                    ),
                    system_msg=agent.system_prompt,
                    msg_history=explore_result.new_msg_history,
                    tools=finalize_only,
                    dispatch=dispatch,
                    max_iters=2,
                    **_commit_sampler(
                        agent.sampler,
                        temperature=self.commit_temperature,
                        max_tokens=self.commit_max_tokens,
                    ),
                )
                state.cost_usd += nudge_result.cost
                commit_cost = nudge_result.cost
                body = _extract_committed_body(state, agent.id)
                if body is None:
                    raise RuntimeError(
                        f"{agent.id}: revise tool-use nudge failed to "
                        f"elicit finalize_voice_genome call after explore "
                        f"loop hit max_iters"
                    )

            genome = VoiceGenome(
                **body.model_dump(),
                pair_id=state.pair_id or "unknown",
                agent_id=agent.id,
            )
            PrivateBriefPhase._validate_renderings_match_bench(genome, state)

            self._write_agent_record(agent, genome, state, explore_result, commit_cost)
            _write_transcript(
                agent_id=agent.id,
                phase_name=self.name,
                title=f"Phase 4 (revise) — {agent.id}",
                system_msg=agent.system_prompt,
                msg_history=explore_result.new_msg_history,
                final_output=genome,
            )
            return genome

    @staticmethod
    def _write_agent_record(agent, genome, state, explore_result, commit_cost: float) -> None:
        path = session_log_path("agents", agent.id, "phase_4_revise.yaml")
        if path is None:
            return
        write_yaml(
            path,
            {
                "agent_id": agent.id,
                "session_id": state.session_id,
                "explore_cost_usd": round(explore_result.cost, 6),
                "commit_cost_usd": round(commit_cost, 6),
                "commit_via": "tool_call" if commit_cost == 0.0 else "structured_output_fallback",
                "explore_turns": len(explore_result.new_msg_history),
                "tool_calls": _summarize_tool_calls(explore_result.new_msg_history),
                "revised_proposal": genome.model_dump(),
            },
        )


# ─── Phase 5 — Borda no-self-vote ────────────────────────────────────────


@dataclass
class BordaPhase:
    """Phase 5 — agents rank others' final proposals; Borda aggregation.

    Reads `state.payload[phase_4_revise]` (or `phase_1_private_brief` if
    revision was skipped). Writes:
        state.payload[phase_5_borda]["rankings"] = {agent_id: list[agent_id]}
        state.payload[phase_5_borda]["points"]   = {agent_id: int}
    """

    name: str = "phase_5_borda"
    proposal_phase: str = "phase_4_revise"
    reasoning_effort: str = ANALYTICAL_REASONING_EFFORT

    async def run(
        self,
        agents: list[Agent],
        state: SessionState,
        registry: ToolRegistry,
    ) -> SessionState:
        proposals = state.payload.get(self.proposal_phase)
        if not proposals:
            raise ValueError(
                f"{self.name} requires '{self.proposal_phase}' in state.payload"
            )

        proposals_dict = {aid: g.model_dump() for aid, g in proposals.items()}
        rankings_lists = await asyncio.gather(*[
            self._run_one_agent(agent, proposals_dict, state)
            for agent in agents
        ])
        rankings = {agent.id: ranking for agent, ranking in zip(agents, rankings_lists)}
        points = borda_no_self_vote(rankings)
        state.payload[self.name] = {"rankings": rankings, "points": points}
        return state

    async def _run_one_agent(
        self,
        agent: Agent,
        proposals_dict: dict[str, dict],
        state: SessionState,
    ) -> list[str]:
        with push_llm_context(agent_id=agent.id):
            others = {aid: p for aid, p in proposals_dict.items() if aid != agent.id}
            expected = set(others.keys())

            scratchpads = state.payload.get("scratchpads", {}).get(agent.id, {})
            user_msg = build_voice_borda_prompt(
                your_agent_id=agent.id,
                your_proposal=proposals_dict[agent.id],
                other_proposals=others,
                your_phase_1_notes=scratchpads.get("phase_1_private_brief", []),
                your_phase_4_notes=scratchpads.get("phase_4_revise", []),
            )
            result = await query_async(
                model_name=agent.model,
                msg=user_msg,
                system_msg=agent.system_prompt,
                output_model=BordaRanking,
                **_analytical_sampler(agent.sampler, reasoning_effort=self.reasoning_effort),
            )
            state.cost_usd += result.cost

            br = result.content
            if not isinstance(br, BordaRanking):
                raise RuntimeError(
                    f"{agent.id}: Borda commit returned {type(br).__name__}, "
                    f"expected BordaRanking"
                )

            ranking = list(br.ranking)
            if set(ranking) != expected:
                raise RuntimeError(
                    f"{agent.id}: Borda ranking mismatch; "
                    f"got {sorted(ranking)}, expected {sorted(expected)}"
                )
            if agent.id in ranking:
                raise RuntimeError(f"{agent.id}: self appears in own ranking")

            self._write_agent_record(agent, ranking, state, result)
            _write_transcript(
                agent_id=agent.id,
                phase_name=self.name,
                title=f"Phase 5 (Borda ranking) — {agent.id}",
                system_msg=agent.system_prompt,
                msg_history=result.new_msg_history,
                final_output=br,
            )
            return ranking

    @staticmethod
    def _write_agent_record(agent, ranking, state, result) -> None:
        path = session_log_path("agents", agent.id, "phase_5_borda.yaml")
        if path is None:
            return
        write_yaml(
            path,
            {
                "agent_id": agent.id,
                "session_id": state.session_id,
                "cost_usd": round(result.cost, 6),
                "ranking": ranking,
            },
        )
