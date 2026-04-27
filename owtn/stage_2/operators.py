"""Stage 2 operators.

Two layers:

1. **`seed_root`** — fires once per concept before MCTS begins. Wraps the
   Stage 1 anchor as a single-node root DAG and extracts motif_threads +
   concept_demands via one merged LLM call.

2. **Action appliers** (Phase 5) — `apply_action(dag, action)` returns a new
   DAG with the action applied. Pure functions, no LLM calls. Used by MCTS
   expansion: an `Action` (one of AddBeat / AddEdge / RewriteBeat) cached
   on a leaf gets applied to produce the child DAG. Validation happens via
   the DAG's own Pydantic constructors — invalid actions raise ValidationError
   and the MCTS expansion loop catches and skips.

Standalone CLI:
    uv run python -m owtn.stage_2.operators <stage1_champion.json>

Runs `seed_root` against a real Stage 1 champion, prints the resulting DAG
plus extracted motifs and demands. Useful for debugging extraction quality
on real concepts before wiring into MCTS.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Awaitable
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from owtn.llm.call_logger import llm_context
from owtn.llm.query import query_async
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.actions import (
    Action,
    AddBeatAction,
    AddEdgeAction,
    ExpansionProposals,
    RewriteBeatAction,
)
from owtn.models.stage_2.dag import DAG, Edge, MotifMention, Node
from owtn.models.stage_2.handoff import Stage1Winner
from owtn.prompts.stage_2.registry import (
    build_expansion_prompt,
    build_seed_motif_prompt,
)


logger = logging.getLogger(__name__)


DEFAULT_EXTRACTION_MODEL = "claude-sonnet-4-6"
"""Default model for seed_root's motif + demand extraction call.

Phase 3 default; Phase 9's run config will override via `Stage2Config.classifier_model`
once the YAML config ships. Same family as Stage 1's classifier — extraction
is a structured-output task, not generation, so we don't need the expansion
model's creativity here.
"""


_TARGET_BUCKET_NAMES = ("flash", "short_short", "standard_short", "long_short")
_TARGET_BUCKET_TO_WORDS = {
    "flash": 1000,
    "short_short": 3000,
    "standard_short": 5000,
    "long_short": 10000,
}


class SeedExtractionResult(BaseModel):
    """Structured output for the merged motif + concept_demand extraction call.

    The LLM is constrained to produce exactly this shape via `output_model`
    on `query_async`; downstream code reads these fields directly.

    `target_bucket` and `bucket_reasoning` were added when seed_root absorbed
    target-sizing classification — the LLM picks the natural prose-length
    bucket alongside motif/demand extraction. Both are optional so existing
    callers (and tests) that build SeedExtractionResult directly still work;
    when the LLM omits them, seed_root falls back to the caller's passed
    `target_node_count`. See `lab/scripts/target_sizing_experiment.py` for
    the calibration data behind this design.
    """
    motif_threads: list[str] = Field(min_length=2, max_length=3)
    concept_demands: list[str] = Field(default_factory=list)
    target_bucket: str | None = Field(
        default=None,
        description="Natural prose-length bucket. One of: flash, short_short, "
        "standard_short, long_short.",
    )
    bucket_reasoning: str | None = Field(
        default=None,
        description="2-4 sentences citing the concept signals (timeline span, "
        "character count, withholding load, thematic-engine complexity) that "
        "drove the bucket choice.",
    )


def _resolve_target_node_count(
    bucket: str | None,
    *,
    fallback: int,
    node_count_targets: dict[int, tuple[int, int]] | None,
) -> tuple[int, str]:
    """Map an LLM-chosen bucket → target_node_count midpoint.

    Returns (count, source) where source describes how the count was decided
    so callers can log it. Falls back to `fallback` when bucket is missing,
    invalid, or no targets dict is configured.
    """
    if bucket is None:
        return fallback, "fallback (LLM omitted bucket)"
    if bucket not in _TARGET_BUCKET_NAMES:
        return fallback, f"fallback (LLM returned unknown bucket {bucket!r})"
    if node_count_targets is None:
        return fallback, "fallback (no node_count_targets configured)"
    word_count = _TARGET_BUCKET_TO_WORDS[bucket]
    target_range = node_count_targets.get(word_count)
    if target_range is None:
        return fallback, f"fallback (no targets row for {word_count}-word bucket {bucket!r})"
    lo, hi = target_range
    return (lo + hi) // 2, f"bucket={bucket} (range [{lo},{hi}] → midpoint)"


async def seed_root(
    concept: ConceptGenome,
    *,
    concept_id: str,
    preset: str,
    target_node_count: int,
    node_count_targets: dict[int, tuple[int, int]] | None = None,
    model_name: str = DEFAULT_EXTRACTION_MODEL,
    **llm_kwargs,
) -> DAG:
    """Wrap the concept's anchor as a single-node root DAG; extract motifs +
    demands + (when `node_count_targets` is provided) target sizing.

    Makes one LLM call producing motifs, concept_demands, and an optional
    target_bucket. The bucket is mapped via `node_count_targets` to a
    midpoint node count; if the LLM omits the bucket or returns garbage,
    the caller's `target_node_count` is used as the fallback.

    On extraction failure, returns a DAG with empty motif_threads /
    concept_demands and the fallback target_node_count. Per
    `docs/stage-2/operators.md` §seed_root §"Failure handling", motifs and
    demands are bias-and-check, not load-bearing.

    The returned DAG has exactly one node (the anchor wrapped from
    `concept.anchor_scene`) and zero edges. Per-node motif attachments on the
    anchor are deferred to MCTS expansion — assigning motifs to a bare anchor
    without surrounding context risks arbitrary selection.

    Args:
        concept: Stage 1 concept genome (premise, anchor_scene, etc.).
        concept_id: Stable id used in the DAG and downstream artifacts.
        preset: Pacing preset name ("cassandra_ish" / "phoebe_ish" / etc.).
        target_node_count: Fallback target DAG size when the LLM doesn't
            provide a usable bucket (or no `node_count_targets` is given).
        node_count_targets: Map {target_word_count → [min_beats, max_beats]}.
            When provided, the LLM picks a named bucket and seed_root maps it
            to the midpoint of the corresponding range. When None, the LLM's
            bucket choice is ignored and `target_node_count` is used directly.
        model_name: Override extraction model (default: DEFAULT_EXTRACTION_MODEL).
        **llm_kwargs: Extra args forwarded to `query_async`.
    """
    # Step 1: deterministic anchor wrap. No LLM call.
    anchor = Node(
        id="anchor",
        sketch=concept.anchor_scene.sketch,
        role=[concept.anchor_scene.role],
        motifs=[],
    )

    # Step 2: motif + demand + bucket extraction. One LLM call.
    extraction = await _extract_seed_fields(
        concept, model_name=model_name, **llm_kwargs,
    )

    # Step 3: resolve target_node_count from the LLM's bucket choice. Falls
    # back to caller-passed default when LLM omits / returns invalid bucket.
    resolved_count, decision_source = _resolve_target_node_count(
        extraction.target_bucket,
        fallback=target_node_count,
        node_count_targets=node_count_targets,
    )
    if extraction.target_bucket is not None:
        logger.info(
            "seed_root[%s]: target_node_count=%d (%s); reasoning: %s",
            concept_id, resolved_count, decision_source,
            (extraction.bucket_reasoning or "(none)").replace("\n", " "),
        )

    return DAG(
        concept_id=concept_id,
        preset=preset,
        motif_threads=list(extraction.motif_threads),
        concept_demands=list(extraction.concept_demands),
        nodes=[anchor],
        edges=[],
        character_arcs=[],
        story_constraints=[],
        target_node_count=resolved_count,
    )


def _empty_extraction() -> SeedExtractionResult:
    """Returned when the extraction call fails entirely. Uses
    `model_construct` to bypass the schema's `min_length=2` floor on motifs —
    callers proceed with empty motifs/demands and the fallback target."""
    return SeedExtractionResult.model_construct(
        motif_threads=[], concept_demands=[],
        target_bucket=None, bucket_reasoning=None,
    )


async def _extract_seed_fields(
    concept: ConceptGenome,
    *,
    model_name: str,
    **llm_kwargs,
) -> SeedExtractionResult:
    """One LLM call producing motifs + demands + (optional) target_bucket.

    When `output_model` is passed to `query_async`, the provider runs the
    response through its native structured-output mechanism (Anthropic tool
    use, OpenAI Responses API text_format, Gemini response_schema, or
    DeepSeek json_object + recovery) and returns the *parsed Pydantic
    instance* on `result.content`. Schema-validation failures raise inside
    the call; we collapse all such failures into the broad except below.

    Returns a stub SeedExtractionResult on any failure (the caller proceeds
    with empty motifs/demands and the fallback target_node_count). Motifs
    bias expansion, demands gate Tier 3, and target_bucket only sizes the
    DAG — none are load-bearing for MCTS reachability.
    """
    system_msg, user_msg = build_seed_motif_prompt(concept)

    base_ctx = dict(llm_context.get({}))
    llm_context.set({**base_ctx, "role": "stage_2_seed_root"})
    try:
        result = await query_async(
            model_name=model_name,
            msg=user_msg,
            system_msg=system_msg,
            output_model=SeedExtractionResult,
            **llm_kwargs,
        )
    except Exception as e:  # broad: provider error, native parse failure, schema mismatch
        logger.warning("seed_root extraction failed for concept (%s): %s", type(e).__name__, e)
        return _empty_extraction()

    parsed = result.content
    if not isinstance(parsed, SeedExtractionResult):
        logger.warning(
            "seed_root extraction returned unexpected content type: %s "
            "(expected SeedExtractionResult)", type(parsed).__name__,
        )
        return _empty_extraction()

    return parsed


# ----- Real LLM expansion call (Phase 6 — the production expand_fn) -----


DEFAULT_EXPANSION_MODEL = "claude-sonnet-4-6"
"""Default model for MCTS expansion calls.

Phase 6 default; Phase 9 run config overrides via `Stage2Config.expansion_model`.
The expansion model is the *generative* one — its job is to propose creative
candidate actions. Cheap-judge model (different family per cross-family
discipline) lives in `owtn.evaluation.stage_2`.
"""


async def propose_actions_via_llm(
    dag: DAG,
    *,
    concept,  # ConceptGenome — typed loosely to avoid cyclic import surface
    phase: str,
    permitted_edge_types: list[str],
    pacing_hint: str = "",
    champion_brief: str = "(brief not yet available)",
    extra_context: str = "",
    k: int = 4,
    model_name: str = DEFAULT_EXPANSION_MODEL,
    **llm_kwargs,
) -> list[Action]:
    """One MCTS expansion call → up to K candidate actions.

    The production `expand_fn` for `MCTS`. Tests mock this; production wires
    it via the factory in `owtn.stage_2.operators.make_expand_factory`.

    On any LLM-side failure (provider error, native parse failure,
    schema mismatch), returns an empty list — MCTS treats empty as
    "no expansion possible" and marks the leaf fully_expanded. Same
    fallback discipline as `seed_root`'s motif extraction.
    """
    # imported here to avoid cyclic concerns with the rendering module
    from owtn.stage_2.rendering import render

    # include_progress=True so the LLM knows how many nodes remain in the
    # target budget and how the existing beats partition around the anchor.
    # Judges still call render() without this kwarg — they evaluate the DAG
    # on structural merit, not on its progress against the target.
    dag_rendering = render(dag, label="A", include_progress=True)
    system_msg, user_msg = build_expansion_prompt(
        concept,
        dag_rendering,
        phase=phase,
        permitted_edge_types=permitted_edge_types,
        pacing_hint=pacing_hint,
        champion_brief=champion_brief,
        extra_context=extra_context,
        k=k,
    )

    base_ctx = dict(llm_context.get({}))
    llm_context.set({**base_ctx, "role": "stage_2_expansion"})
    try:
        result = await query_async(
            model_name=model_name,
            msg=user_msg,
            system_msg=system_msg,
            output_model=ExpansionProposals,
            **llm_kwargs,
        )
    except Exception as e:
        logger.warning(
            "MCTS expansion LLM call failed (%s: %s); returning no actions.",
            type(e).__name__, e,
        )
        return []

    proposals = result.content
    if not isinstance(proposals, ExpansionProposals):
        logger.warning(
            "Expansion call returned unexpected content type: %s",
            type(proposals).__name__,
        )
        return []
    return list(proposals.actions)


def make_expand_factory(
    *,
    concept,
    brief_fetcher=None,
    extra_context_fn=None,
    model_name: str = DEFAULT_EXPANSION_MODEL,
    k: int = 4,
    **llm_kwargs,
):
    """Build an `ExpandFactory` for `run_bidirectional` / `run_refinement`.

    The bidirectional orchestrator passes a `PhaseContext` to the factory
    and gets back an `expand_fn(dag) -> list[Action]` for that phase. The
    factory closes over the concept + model + brief fetcher; phase-specific
    bits (permitted edge types, pacing hint) come from the PhaseContext.

    `brief_fetcher` is a `Callable[[], str]` — sync read of the current
    rendered champion brief (the cached render in `TreeBriefState`). Re-read
    on every expansion so a brief refresh between iterations propagates
    immediately. Pass `None` to suppress brief injection (cold-start mode).

    `extra_context_fn` is a `Callable[[DAG], str]` invoked per expansion
    to compute DAG-state-dependent prompt context (e.g., refinement passes
    the upstream/downstream node partition relative to the anchor). Pass
    `None` (default) to skip — forward/backward phases don't need it.
    """
    def factory(ctx) -> Awaitable[list[Action]]:  # type: ignore[return-value]
        async def expand(dag: DAG) -> list[Action]:
            brief = brief_fetcher() if brief_fetcher is not None else "(brief not yet available)"
            extra_context = extra_context_fn(dag) if extra_context_fn is not None else ""
            return await propose_actions_via_llm(
                dag,
                concept=concept,
                phase=ctx.phase,
                permitted_edge_types=sorted(ctx.permitted_edge_types),
                pacing_hint=ctx.pacing_hint,
                champion_brief=brief,
                extra_context=extra_context,
                k=k,
                model_name=model_name,
                **llm_kwargs,
            )
        return expand  # type: ignore[return-value]

    return factory


# ----- Action appliers (Phase 5) -----


def apply_action(dag: DAG, action: Action) -> DAG:
    """Apply an MCTS action to a DAG, returning a new DAG.

    Pure function: input DAG unchanged. Returned DAG passes Pydantic
    validation; if the action would produce an invalid DAG (cycle, missing
    payload, role conflict, etc.), `ValidationError` propagates and MCTS's
    expansion loop catches it.

    Dispatches on `action.action_type`. The discriminated-union shape from
    `owtn.models.stage_2.actions` makes this dispatch straightforward.
    """
    if isinstance(action, AddBeatAction):
        return _apply_add_beat(dag, action)
    if isinstance(action, AddEdgeAction):
        return _apply_add_edge(dag, action)
    if isinstance(action, RewriteBeatAction):
        return _apply_rewrite_beat(dag, action)
    raise TypeError(f"unknown action type: {type(action).__name__}")  # pragma: no cover


def _apply_add_beat(dag: DAG, action: AddBeatAction) -> DAG:
    """Insert a new beat + typed edge.

    `action.direction == "downstream"` (forward phase): edge is
    `anchor_id -> new_node_id`. `action.direction == "upstream"` (backward
    phase): edge is `new_node_id -> anchor_id`.

    Constructs a fresh DAG (rather than `model_copy`) so cross-field
    validators rerun — `model_copy(update=...)` skips validation by design,
    which would let invalid actions slip through.
    """
    new_node = Node(
        id=action.new_node_id,
        sketch=action.sketch,
        role=action.new_node_role,
        motifs=[],
    )
    if action.direction == "downstream":
        new_edge = _build_edge(
            src=action.anchor_id, dst=action.new_node_id,
            edge_type=action.edge_type, payload=action.edge_payload,
        )
    else:
        new_edge = _build_edge(
            src=action.new_node_id, dst=action.anchor_id,
            edge_type=action.edge_type, payload=action.edge_payload,
        )
    return _rebuild_dag(
        dag,
        nodes=list(dag.nodes) + [new_node],
        edges=list(dag.edges) + [new_edge],
    )


def _apply_add_edge(dag: DAG, action: AddEdgeAction) -> DAG:
    """Add a typed edge between two existing nodes."""
    new_edge = _build_edge(
        src=action.src_id, dst=action.dst_id,
        edge_type=action.edge_type, payload=action.edge_payload,
    )
    return _rebuild_dag(dag, edges=list(dag.edges) + [new_edge])


def _apply_rewrite_beat(dag: DAG, action: RewriteBeatAction) -> DAG:
    """Replace one node's sketch. Other fields preserved."""
    updated_nodes: list[Node] = []
    found = False
    for n in dag.nodes:
        if n.id == action.node_id:
            updated_nodes.append(n.model_copy(update={"sketch": action.new_sketch}))
            found = True
        else:
            updated_nodes.append(n)
    if not found:
        raise ValueError(
            f"rewrite_beat: node {action.node_id!r} not found in DAG"
        )
    return _rebuild_dag(dag, nodes=updated_nodes)


def _rebuild_dag(dag: DAG, **overrides) -> DAG:
    """Construct a new DAG with overrides, re-running cross-field validation.

    Pydantic's `model_copy(update=...)` skips validators — it shallow-copies
    and replaces the specified fields without re-checking invariants. For
    MCTS we need invalid actions to raise on application so they get skipped;
    constructing a fresh DAG forces validation."""
    fields = {
        "concept_id": dag.concept_id,
        "preset": dag.preset,
        "motif_threads": list(dag.motif_threads),
        "concept_demands": list(dag.concept_demands),
        "nodes": list(dag.nodes),
        "edges": list(dag.edges),
        "character_arcs": list(dag.character_arcs),
        "story_constraints": list(dag.story_constraints),
        "target_node_count": dag.target_node_count,
    }
    fields.update(overrides)
    return DAG(**fields)


def _build_edge(
    *, src: str, dst: str, edge_type: str,
    payload: dict[str, str | list[str]],
) -> Edge:
    """Construct a typed Edge, mapping the action's flat `payload` dict to
    the type-specific named fields on the Edge model.

    Edge's per-type validator enforces required fields are present and
    substantive; if `payload` lacks a required field for `edge_type`, Edge
    construction raises ValidationError and the caller (MCTS expansion) skips.

    `disclosed_to` is the one list-typed payload field. Pydantic now accepts
    it as a native list (the action schema is `dict[str, str | list[str]]`),
    but we still tolerate two legacy/error-mode encodings the LLM produced
    historically: a JSON-encoded string (`'["reader"]'`) and a comma-
    separated bare string (`"reader, mother"`). Both get coerced to a
    list here so audience validation downstream sees clean entries.
    """
    edge_kwargs: dict[str, object] = {"src": src, "dst": dst, "type": edge_type}
    type_field_map = {
        "causal": ("realizes",),
        "disclosure": ("reframes", "withheld", "disclosed_to"),
        "implication": ("entails",),
        "constraint": ("prohibits",),
        "motivates": ("agent", "goal", "stakes"),
    }
    fields = type_field_map.get(edge_type, ())
    for field_name in fields:
        if field_name in payload:
            value = payload[field_name]
            if field_name == "disclosed_to":
                value = _coerce_disclosed_to(value)
            edge_kwargs[field_name] = value
    return Edge(**edge_kwargs)


def _coerce_disclosed_to(value: str | list[str]) -> list[str]:
    """Normalize `disclosed_to` to a list[str], tolerating LLM-side encoding
    quirks: native lists pass through; JSON-encoded list strings are parsed;
    comma-separated bare strings are split."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = value.strip()
    # JSON-encoded list, e.g. '["reader", "mother"]'
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    # Bare or comma-separated string.
    return [part.strip() for part in s.split(",") if part.strip()]


# ----- Standalone CLI -----

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run seed_root against a Stage 1 champion JSON. Prints the "
            "resulting 1-node DAG plus extracted motifs and concept_demands. "
            "Makes one real LLM call — costs ~$0.005-0.02."
        ),
    )
    parser.add_argument("champion_path", type=Path, help="Path to champions/island_*.json")
    parser.add_argument(
        "--preset", default="cassandra_ish",
        help="Pacing preset to stamp on the seed DAG (default: cassandra_ish)",
    )
    parser.add_argument(
        "--target-node-count", type=int, default=5,
        help="Target full DAG size (default: 5; for ~1K-word prose targets)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_EXTRACTION_MODEL,
        help=f"Extraction model name (default: {DEFAULT_EXTRACTION_MODEL})",
    )
    args = parser.parse_args(argv)

    if not args.champion_path.exists():
        print(f"error: {args.champion_path} not found", file=sys.stderr)
        return 1

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        winner = Stage1Winner.from_champion_file(args.champion_path)
    except (ValidationError, KeyError, json.JSONDecodeError) as e:
        print(f"error loading champion: {e}", file=sys.stderr)
        return 1

    dag = asyncio.run(seed_root(
        winner.genome,
        concept_id=winner.program_id,
        preset=args.preset,
        target_node_count=args.target_node_count,
        model_name=args.model,
    ))

    print(f"seed_root produced 1-node DAG for {dag.concept_id}")
    print(f"  preset:           {dag.preset}")
    print(f"  anchor role:      {dag.nodes[0].role}")
    print(f"  motif_threads ({len(dag.motif_threads)}):")
    for m in dag.motif_threads:
        print(f"    - {m}")
    print(f"  concept_demands ({len(dag.concept_demands)}):")
    for d in dag.concept_demands:
        print(f"    - {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
