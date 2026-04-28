"""Artifact renderers: payload (concept genome / DAG) -> scoreable text.

The scorer is stage-agnostic; the renderer is the only stage-specific
dependency. Each call site picks the right renderer at composition time.
"""

from __future__ import annotations

from typing import Any

from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG
from owtn.stage_2.rendering import render as render_dag


def render_stage1_concept(genome: ConceptGenome | dict[str, Any]) -> str:
    """Concept genome -> formatted text (mirrors the existing pairwise format).

    Accepts either a ConceptGenome model or a raw dict (from JSONL test sets).
    """
    g = genome.model_dump() if hasattr(genome, "model_dump") else dict(genome)
    parts: list[str] = []

    if g.get("premise"):
        parts.append(f"PREMISE: {g['premise']}")
    if g.get("thematic_engine"):
        parts.append(f"THEMATIC ENGINE: {g['thematic_engine']}")
    if g.get("target_effect"):
        parts.append(f"TARGET EFFECT: {g['target_effect']}")

    anchor = g.get("anchor_scene")
    if isinstance(anchor, dict):
        parts.append(f"ANCHOR SCENE ({anchor.get('role', '?')}): {anchor.get('sketch', '')}")
    elif isinstance(anchor, str):
        parts.append(f"ANCHOR SCENE: {anchor}")

    seeds = g.get("character_seeds") or []
    if seeds:
        seed_blocks: list[str] = []
        for seed in seeds:
            block = [f"{seed.get('label', '(unnamed)')}: {seed.get('sketch', '')}"]
            for k in ("wound", "fear", "lie", "want", "need"):
                if seed.get(k):
                    block.append(f"  {k}: {seed[k]}")
            seed_blocks.append("\n".join(block))
        parts.append("CHARACTER SEEDS:\n" + "\n".join(seed_blocks))

    if g.get("setting_seeds"):
        parts.append(f"SETTING SEEDS: {g['setting_seeds']}")

    constraints = g.get("constraints") or []
    if constraints:
        parts.append("CONSTRAINTS:\n" + "\n".join(f"- {c}" for c in constraints))

    if g.get("style_hint"):
        parts.append(f"STYLE HINT: {g['style_hint']}")

    return "\n\n".join(parts)


def render_stage2_partial(partial: DAG | dict[str, Any]) -> str:
    """Stage-2 walked partial DAG -> outlined text.

    For DAG models, delegates to the existing rendering pipeline (the same
    format the pairwise judge sees). For pre-rendered dicts with a
    `dag_text` field, uses that verbatim.
    """
    if isinstance(partial, dict) and "dag_text" in partial:
        return partial["dag_text"]
    if hasattr(partial, "nodes"):
        return render_dag(partial)
    raise TypeError(
        f"render_stage2_partial expects DAG or dict-with-dag_text; got {type(partial).__name__}"
    )
