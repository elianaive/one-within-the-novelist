"""Stage 3 voice-session tools.

ToolSpec instances wrapping the cross-stage `owtn.tools` implementations
plus stage-3-specific helpers. Tools read read-only context from
`ToolContext.state_view`, which is the orchestration `state.payload`
(the orchestrator and stage-3 session composer agree on which keys land
there at session start).

Convention for `state.payload` (set by `owtn.stage_3.session`):
    "adjacent_scene_bench": dict   # AdjacentSceneBench.model_dump()
    "agent_models":         dict   # {agent_id: model_name}

Tools serialize their reports as JSON text — the LLM tool-result channel
is a string, and the underlying tool report dataclasses already have
agent-facing field shapes.

Phase allowlist (`per_phase_allow`) is built in `session.py`; tools here
are stateless and reusable across phases. Tool gating per phase happens
at the registry level, not inside the handlers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Mapping

from pydantic import ValidationError

from owtn.models.stage_3 import SignatureRisk, VoiceGenomeBody
from owtn.orchestration import ToolContext, ToolSpec
from owtn.orchestration.session import session_log_path
from owtn.tools.lookup_exemplar import lookup_exemplar_async
from owtn.tools.slop_score import slop_score as compute_slop_score
from owtn.tools.stylometry import stylometry as compute_stylometry
from owtn.tools.thesaurus import MODE_TO_PARAM, thesaurus as compute_thesaurus
from owtn.tools.writing_style import writing_style as compute_writing_style


logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _to_jsonable(obj: Any) -> Any:
    """Coerce dataclass / Pydantic / dict / list to a JSON-serializable shape."""
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def _dump(report: Any) -> str:
    """Serialize a tool report to compact JSON for the LLM tool-result channel."""
    return json.dumps(_to_jsonable(report), ensure_ascii=False, indent=2)


def _scene_neutral_baseline(state_view: Mapping[str, Any], scene_id: str) -> str | None:
    """Look up the neutral draft for a scene_id from the cached bench."""
    bench = state_view.get("adjacent_scene_bench")
    if not bench:
        return None
    drafts = bench.get("drafts", []) if isinstance(bench, dict) else []
    for d in drafts:
        if d.get("scene_id") == scene_id:
            return d.get("neutral_draft")
    return None


# ─── Handlers ────────────────────────────────────────────────────────────


async def _render_adjacent_scene_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    scene_id = params.get("scene_id", "")
    bench = ctx.state_view.get("adjacent_scene_bench")
    if not bench:
        return "ERROR: adjacent_scene_bench not present in session state"
    drafts = bench.get("drafts", []) if isinstance(bench, dict) else []
    for d in drafts:
        if d.get("scene_id") == scene_id:
            return _dump({
                "scene_id": d["scene_id"],
                "synopsis": d.get("synopsis", ""),
                "demand": d.get("demand", ""),
                "neutral_draft": d.get("neutral_draft", ""),
            })
    available = [d.get("scene_id") for d in drafts]
    return f"ERROR: scene_id {scene_id!r} not found; available: {available}"


async def _think_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    thought = params.get("thought", "").strip()
    if not thought:
        return "ERROR: think called with empty thought"
    # Ephemeral reasoning — the call shows up in tool-use history (so the
    # agent can refer back within the phase) but is not persisted to
    # scratchpads and is not visible across phases. Pure thinking-place.
    return "Considered."


async def _note_to_self_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    text = params.get("text", "").strip()
    if not text:
        return "ERROR: note_to_self called with empty text"
    # Cross-phase memo — load-bearing decisions only. Stored under
    # state.payload["scratchpads"][agent_id][phase_id] = [notes].
    # Phase 4 + Phase 5 prompt builders inject prior-phase notes back in,
    # restoring ICL signal across the phase boundary. For ephemeral
    # in-phase reasoning the agent should use `think`; this tool is for
    # commitments worth carrying forward.
    #
    # The mutation goes through the underlying dict — `state_view` is typed
    # Mapping for tools that don't write, but the actual object is the live
    # `state.payload`. Restricted to the `scratchpads` key.
    if isinstance(ctx.state_view, dict):
        scratchpads = ctx.state_view.setdefault("scratchpads", {})
        agent_notes = scratchpads.setdefault(ctx.agent_id, {})
        phase_notes = agent_notes.setdefault(ctx.phase_id, [])
        phase_notes.append(text)
    return f"Noted ({len(text)} chars). Will be visible to you in later phases."


async def _lookup_reference_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    query = params.get("query", "")
    n = int(params.get("n", 2))
    if not query:
        return "ERROR: lookup_reference requires non-empty 'query'"
    try:
        result = await lookup_exemplar_async(query=query, n=n)
    except Exception as e:
        logger.warning("lookup_exemplar_async failed: %s", e)
        return f"ERROR: lookup failed ({type(e).__name__}: {e})"
    if result.get("n_returned", 0) == 0:
        _record_corpus_gap(ctx, result)
    return _dump(result)


def _record_corpus_gap(ctx: ToolContext, result: Mapping[str, Any]) -> None:
    """Append a JSONL entry to `<session_log_dir>/corpus_gaps.jsonl` when a
    lookup returns nothing. Each entry captures the NL query, the resolver's
    interpretation, and the resolver's note (which names closest-available
    alternatives) — high-quality intent signal for corpus curation.
    Append-only; safe under concurrent agent fan-out.
    """
    path = session_log_path("corpus_gaps.jsonl")
    if path is None:
        return
    entry = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "session_id": ctx.session_id,
        "phase_id": ctx.phase_id,
        "agent_id": ctx.agent_id,
        "query": result.get("query", ""),
        "match": result.get("match", "?"),
        "interpretation": result.get("interpretation", ""),
        "note": result.get("note", ""),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("failed to record corpus gap: %s", e)


async def _stylometry_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    passage = params.get("passage", "")
    if not passage.strip():
        return "ERROR: stylometry requires non-empty 'passage'"

    scene_id = params.get("scene_id")
    neutral_baseline = (
        _scene_neutral_baseline(ctx.state_view, scene_id) if scene_id else None
    )

    style_queries = params.get("style_queries") or None
    if style_queries is not None and not isinstance(style_queries, list):
        return "ERROR: style_queries must be a list of strings"

    agent_models = ctx.state_view.get("agent_models", {}) or {}
    caller_model = agent_models.get(ctx.agent_id)

    try:
        report = await compute_stylometry(
            passage=passage,
            caller_model=caller_model,
            neutral_baseline=neutral_baseline,
            style_queries=style_queries,
        )
    except Exception as e:
        logger.warning("stylometry failed: %s", e)
        return f"ERROR: stylometry failed ({type(e).__name__}: {e})"
    return _dump(report)


async def _slop_score_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    passage = params.get("passage", "")
    if not passage.strip():
        return "ERROR: slop_score requires non-empty 'passage'"
    compare_to = params.get("compare_to")
    try:
        report = compute_slop_score(passage=passage, compare_to=compare_to)
    except Exception as e:
        logger.warning("slop_score failed: %s", e)
        return f"ERROR: slop_score failed ({type(e).__name__}: {e})"
    return _dump(report)


async def _writing_style_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    passage = params.get("passage", "")
    if not passage.strip():
        return "ERROR: writing_style requires non-empty 'passage'"
    compare_to = params.get("compare_to")
    try:
        report = compute_writing_style(passage=passage, compare_to=compare_to)
    except Exception as e:
        logger.warning("writing_style failed: %s", e)
        return f"ERROR: writing_style failed ({type(e).__name__}: {e})"
    return _dump(report)


async def _thesaurus_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    word = params.get("word", "")
    mode = params.get("mode", "means_like")
    max_results = int(params.get("max_results", 20))
    if not word.strip():
        return "ERROR: thesaurus requires non-empty 'word'"
    try:
        report = compute_thesaurus(word=word, mode=mode, max_results=max_results)
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        logger.warning("thesaurus failed: %s", e)
        return f"ERROR: thesaurus failed ({type(e).__name__}: {e})"
    return _dump(report)


async def _ask_judge_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    """v0.1 stub. Phase 2 lands ask_judge as its own child issue."""
    return (
        "Phase 2 (judge consultation) is deferred to v0.2. "
        "Develop your voice spec without judge consultation in this session."
    )


async def _declare_signature_risk_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    """Required pre-commit step in Phase 1. The agent names the structural
    move it commits to that the model would not take by default for this
    concept, what the model's near-default would be, and why the concept
    demands the move. Stashed in state.payload so Phase 3 critique, Phase
    5 Borda, and Stage 4 critics can all read it.

    Forcing this articulation BEFORE finalize_voice_genome is the
    intervention against mid-shaped voices: the agent has to commit to a
    concrete risk before the description can be generated, so the
    description gets shaped around the risk rather than wandering into
    safe literary-fluent territory.
    """
    try:
        risk = SignatureRisk.model_validate(params)
    except ValidationError as e:
        return (
            f"ERROR: SignatureRisk validation failed. The fields must each "
            f"name a concrete commitment — vague register words ('lyrical', "
            f"'spare') do not pass the length threshold and would not be "
            f"useful even if they did. Fix the issues and call again.\n\n{e}"
        )

    if isinstance(ctx.state_view, dict):
        risks = ctx.state_view.setdefault("_pending_signature_risks", {})
        if ctx.agent_id in risks:
            return (
                "ERROR: signature_risk already declared for this phase. "
                "If you want to revise it, name what you'd change and why "
                "in `think`, then call `finalize_voice_genome` with the "
                "voice spec that follows from the declared risk."
            )
        risks[ctx.agent_id] = risk
    return (
        "Signature risk recorded. Now develop the voice and renderings "
        "around this commitment, then call `finalize_voice_genome`. The "
        "renderings are the proof — if they don't enact the move you "
        "named, the panel will see the gap and rank the voice as mid."
    )


async def _finalize_voice_genome_handler(
    params: Mapping[str, Any], ctx: ToolContext,
) -> str:
    """End-of-explore commit. The agent calls this when they've worked out
    a voice and are ready to submit. Args mirror `VoiceGenomeBody` exactly;
    we validate via Pydantic and stash the body in `state.payload` for the
    orchestrator to extract after the explore loop returns.

    Gates on `declare_signature_risk` having been called first — submission
    without an articulated risk is exactly the path to mid voices, and the
    gate exists to surface that early when the agent can still go back.

    Tool-call shape is more reliable than structured-output for the commit
    on long explore histories — the model is already operating in tool-call
    mode and just needs to fill the schema. The agent's commitment is also
    explicit at the agent level rather than inferred by the orchestrator
    from message-position heuristics.
    """
    declared_risk: SignatureRisk | None = None
    if isinstance(ctx.state_view, dict):
        risks = ctx.state_view.get("_pending_signature_risks", {})
        declared_risk = risks.get(ctx.agent_id)
        if declared_risk is None:
            return (
                "ERROR: you must call `declare_signature_risk` before "
                "`finalize_voice_genome`. Name the structural move you are "
                "committing to that the model would not take by default for "
                "this concept, what the model's near-default would be, and "
                "why the concept demands the move. Then call finalize."
            )

    # Merge the previously-declared risk into the body. The finalize schema
    # asks the agent for everything-except-signature_risk so the field
    # doesn't get re-typed; the handler inserts the prior commitment so the
    # final VoiceGenomeBody still satisfies its required-field invariant.
    merged = dict(params)
    if declared_risk is not None and "signature_risk" not in merged:
        merged["signature_risk"] = declared_risk.model_dump()

    try:
        body = VoiceGenomeBody.model_validate(merged)
    except ValidationError as e:
        # Surface the validation errors to the model so it can correct.
        return (
            f"ERROR: VoiceGenomeBody validation failed. Fix the issues and "
            f"call finalize_voice_genome again with corrected fields.\n\n{e}"
        )

    if isinstance(ctx.state_view, dict):
        commits = ctx.state_view.setdefault("_pending_commits", {})
        if ctx.agent_id in commits:
            return (
                "ERROR: voice already committed for this phase. "
                "Stop calling tools to end the explore loop."
            )
        commits[ctx.agent_id] = body
    return (
        "Voice committed. Stop calling tools — the explore loop will end "
        "when you produce a non-tool-call response. A short acknowledgment "
        "text response is fine."
    )


# ─── ToolSpec definitions ────────────────────────────────────────────────


RENDER_ADJACENT_SCENE = ToolSpec(
    name="render_adjacent_scene",
    description=(
        "Read the cached neutral-voice draft for one of the three adjacent "
        "scenes that anchor this session. Returns synopsis + the demand the "
        "scene places on voice + the neutral draft to transform."
    ),
    parameters={
        "type": "object",
        "properties": {
            "scene_id": {
                "type": "string",
                "description": "Scene id from the bench (e.g. 'morning-kitchen').",
            },
        },
        "required": ["scene_id"],
    },
    handler=_render_adjacent_scene_handler,
)


THINK = ToolSpec(
    name="think",
    description=(
        "Private thinking-place for in-the-moment reasoning. Use when you "
        "need to work through a problem before acting — what a critique is "
        "actually pointing at, which scene's voice work is hardest, why a "
        "draft isn't yet landing. Calling `think` does not change anything "
        "external; it gives you a deliberate pause to reason. Cheap, "
        "frequent, ephemeral — the thought stays in your tool-use history "
        "for this phase but is NOT carried forward. For decisions worth "
        "carrying to later phases, use `note_to_self` after thinking."
    ),
    parameters={
        "type": "object",
        "properties": {
            "thought": {"type": "string", "description": "The thought to work through."},
        },
        "required": ["thought"],
    },
    handler=_think_handler,
)


NOTE_TO_SELF = ToolSpec(
    name="note_to_self",
    description=(
        "Cross-phase memo for load-bearing decisions. Visible to you in "
        "later phases (Phase 4 sees Phase 1 notes; Phase 5 sees both). "
        "Use to record commitments worth carrying forward — the central "
        "voice axes you've chosen, the protected Saunders beats, the "
        "rejected approaches and why. Each note should be tight and "
        "decision-shaped. NOT a thinking-pad: for ephemeral in-the-moment "
        "reasoning use `think` instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The decision or commitment to memo forward."},
        },
        "required": ["text"],
    },
    handler=_note_to_self_handler,
)


LOOKUP_REFERENCE = ToolSpec(
    name="lookup_reference",
    description=(
        "Search the literary-reference corpus for prose passages to study. "
        "Ask in natural language — author names ('Toni Morrison'), styles "
        "('incantatory third-person', 'second-person direct address'), or "
        "both at once ('Morrison's incantatory mode', 'Saunders' tonal-"
        "disjunction maximalism'). Compound queries (author + style) are "
        "intersected: you'll get passages matching BOTH the author AND the "
        "style. If the intersection is empty, you'll get the closest "
        "available — passages by that author OR passages in that style — "
        "with a `note` explaining what was substituted. If nothing matches, "
        "you'll get an empty result whose `note` lists the closest existing "
        "options; that's a real corpus-gap signal worth recording with "
        "`note_to_self` if it constrains your voice work. You can also pass "
        "a specific entry id from a prior result (kebab-case ending in "
        "`-s<n>`, e.g. `morrison-beloved-s0`) to re-fetch that exact passage. "
        "This tool indexes voice/style/author for prose study — it is NOT "
        "for vocabulary definitions or topic research. Returns up to N "
        "passages with id, source citation, tags, and ~400 words each."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language description of the voice / style / "
                    "tradition / author you want to study, OR an exact "
                    "entry id from a prior result."
                ),
            },
            "n": {
                "type": "integer",
                "description": "Max passages to return (default 2).",
                "default": 2,
            },
        },
        "required": ["query"],
    },
    handler=_lookup_reference_handler,
)


STYLOMETRY = ToolSpec(
    name="stylometry",
    description=(
        "Compute function-word distribution distances, burstiness, and MATTR "
        "on a candidate passage. Returns the candidate's signals (burstiness, "
        "MATTR, sentence-length CV) plus distances to centroids you can "
        "position against — this model's default, the cross-LLM centroid, "
        "the human-literary centroid, the session's neutral baseline if "
        "you pass `scene_id`, and any natural-language style queries you "
        "pass in `style_queries` (e.g. \"Morrison's incantatory mode\", "
        "\"free indirect discourse\", \"minimalist Carver register\"). "
        "Burstiness is null when the passage is a single sentence — rhythm "
        "CV is undefined there; sample a longer window if you need a "
        "rhythm read. Use when you have an intentional position you want "
        "to verify a draft is moving toward — closer to a reference "
        "author, away from model default, at a target burstiness."
    ),
    parameters={
        "type": "object",
        "properties": {
            "passage": {
                "type": "string",
                "description": "The candidate prose to measure.",
            },
            "scene_id": {
                "type": "string",
                "description": (
                    "Optional. When set, the bench's neutral draft for this "
                    "scene becomes the baseline for fw-distance-from-baseline."
                ),
            },
            "style_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of natural-language style queries — same "
                    "shape as `lookup_reference`'s `query`. Examples: "
                    "\"Morrison's incantatory mode\", \"Saunders' tonal-"
                    "disjunction maximalism\", \"free indirect discourse\", "
                    "\"second-person direct address\". Each query is "
                    "resolved through the same haiku resolver; the report "
                    "returns the candidate's function-word distance from the "
                    "resolved cluster plus the resolver's interpretation/note."
                ),
            },
        },
        "required": ["passage"],
    },
    handler=_stylometry_handler,
)


SLOP_SCORE = ToolSpec(
    name="slop_score",
    description=(
        "EQ-Bench slop score on a passage — slop-list matches, AI-default "
        "register patterns, contrast-pattern frequency. 0-100 composite plus "
        "per-axis raw rates and concrete top-N hits. Use when you want to "
        "verify a draft is free of AI-tells — slop vocabulary, default "
        "register patterns, contrast-pair structures."
    ),
    parameters={
        "type": "object",
        "properties": {
            "passage": {
                "type": "string",
                "description": "The candidate prose to score.",
            },
            "compare_to": {
                "type": "string",
                "description": (
                    "Optional reference passage; report includes per-axis "
                    "deltas (candidate minus reference)."
                ),
            },
        },
        "required": ["passage"],
    },
    handler=_slop_score_handler,
)


WRITING_STYLE = ToolSpec(
    name="writing_style",
    description=(
        "Surface metrics on a passage — vocab grade (Flesch-Kincaid), "
        "sentence length, paragraph length, dialogue frequency — positioned "
        "against the project calibration corpus's per-bucket median + p90."
    ),
    parameters={
        "type": "object",
        "properties": {
            "passage": {
                "type": "string",
                "description": "The candidate prose to measure.",
            },
            "compare_to": {
                "type": "string",
                "description": "Optional reference passage for delta tracking.",
            },
        },
        "required": ["passage"],
    },
    handler=_writing_style_handler,
)


ASK_JUDGE = ToolSpec(
    name="ask_judge",
    description=(
        "Query a judge persona for taste / principle / method input. v0.1: "
        "this tool is a stub; Phase 2 (judge consultation) lands in v0.2."
    ),
    parameters={
        "type": "object",
        "properties": {
            "judge_id": {"type": "string"},
            "question": {"type": "string"},
        },
        "required": ["judge_id", "question"],
    },
    handler=_ask_judge_handler,
)


THESAURUS = ToolSpec(
    name="thesaurus",
    description=(
        "Datamuse lookup for a single word. Modes: means_like (synonyms), "
        "sounds_like (phonetic neighbours), related_to (co-occurrence "
        "triggers), adjective_for (adjectives modifying a noun), noun_for "
        "(nouns modified by an adjective), antonyms. Useful for diction "
        "specificity work; cached per session."
    ),
    parameters={
        "type": "object",
        "properties": {
            "word": {
                "type": "string",
                "description": "The query word.",
            },
            "mode": {
                "type": "string",
                "enum": sorted(MODE_TO_PARAM),
                "description": "Lookup mode (default means_like).",
                "default": "means_like",
            },
            "max_results": {
                "type": "integer",
                "description": "Max results to return (default 20).",
                "default": 20,
            },
        },
        "required": ["word"],
    },
    handler=_thesaurus_handler,
)


DECLARE_SIGNATURE_RISK = ToolSpec(
    name="declare_signature_risk",
    description=(
        "REQUIRED before finalize_voice_genome in Phase 1. Name the "
        "structural / syntactic / formal move you are committing to that "
        "the model would not take by default for this concept, what the "
        "model's near-default would be (the literary-fluent voice this "
        "concept would otherwise produce), and why the concept's load-"
        "bearing demands need this specific move. The risk you name will "
        "be visible to your peers in Phase 3 critique and to the writer "
        "and Stage 4 critics later, so be specific. A voice without a "
        "declared risk is mid by default; this gate exists to surface that "
        "before the renderings get drafted."
    ),
    parameters=SignatureRisk.model_json_schema(),
    handler=_declare_signature_risk_handler,
)


def _finalize_schema_without_signature_risk() -> dict:
    """VoiceGenomeBody's JSON schema with the signature_risk field stripped.

    The agent declares signature_risk via the dedicated tool earlier in the
    explore loop; the finalize handler merges that prior declaration into
    the body before validation. Asking the agent to repeat it on finalize
    would be redundant and risk drift between the two declarations."""
    schema = VoiceGenomeBody.model_json_schema()
    props = dict(schema.get("properties") or {})
    props.pop("signature_risk", None)
    schema["properties"] = props
    required = [r for r in (schema.get("required") or []) if r != "signature_risk"]
    schema["required"] = required
    return schema


FINALIZE_VOICE_GENOME = ToolSpec(
    name="finalize_voice_genome",
    description=(
        "Submit your final VoiceGenome — the deliberate end of exploration. "
        "Pass all schema fields filled, including the three renderings as "
        "complete transformations of the bench's neutral drafts (same world, "
        "same characters, same physical events; different voice). Use the "
        "bench's scene_ids exactly. The schema fields and the verbal "
        "description must agree — if they say different things, fix the "
        "verbal description. The signature_risk you declared earlier is "
        "merged in automatically; do not pass it again. Once you call this "
        "tool successfully, your explore loop terminates and the "
        "orchestrator treats the args as your committed voice spec — there "
        "is no separate commit step. If validation fails, you'll get an "
        "error message describing the issues; correct and call again."
    ),
    parameters=_finalize_schema_without_signature_risk(),
    handler=_finalize_voice_genome_handler,
)


ALL_VOICE_TOOLS: list[ToolSpec] = [
    RENDER_ADJACENT_SCENE,
    THINK,
    NOTE_TO_SELF,
    LOOKUP_REFERENCE,
    STYLOMETRY,
    SLOP_SCORE,
    WRITING_STYLE,
    THESAURUS,
    DECLARE_SIGNATURE_RISK,
    FINALIZE_VOICE_GENOME,
    ASK_JUDGE,
]


# ─── Per-phase allowlist ─────────────────────────────────────────────────


PHASE_1_TOOLS = frozenset({
    "think",
    "note_to_self",
    "lookup_reference",
    "thesaurus",
    "stylometry",
    "slop_score",
    "writing_style",
    "declare_signature_risk",
    "finalize_voice_genome",
})
"""Phase 1 (private brief) — drafting + reference + diction lookup + metric
ensemble. `render_adjacent_scene` removed: bench drafts are inlined in the
Phase 1 prompt directly, so tool-fetching them just burned explore-loop
iterations on data the agent should have started with."""


PHASE_4_TOOLS = frozenset({
    "think",
    "note_to_self",
    "stylometry",
    "slop_score",
    "writing_style",
    "thesaurus",
    "finalize_voice_genome",
})
"""Phase 4 (revise) — metric ensemble + diction lookup for revision work.
Drops `lookup_reference` because the agent has already committed to a
voice region; reopening reference search would re-do Phase 1 instead of
revising. Keeps `thesaurus` because word-level diction work is part of
revising (sharpening a specific word in a sentence is not the same as
re-discovering the voice). Drops `render_adjacent_scene` because the
bench drafts are inlined in the Phase 4 prompt."""


VOICE_PHASE_ALLOW: dict[str, frozenset[str]] = {
    "phase_1_private_brief": PHASE_1_TOOLS,
    "phase_4_revise": PHASE_4_TOOLS,
    # Phase 2 (judge consultation) deferred to v0.2:
    #     "phase_2_judge_consult": frozenset({"ask_judge", "render_adjacent_scene", "note_to_self"})
    # Phase 3 + Phase 5 are pure structured-output, no tools.
}
