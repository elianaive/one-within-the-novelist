from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from owtn.evaluation.anti_cliche import gate_2_anti_cliche
from owtn.evaluation.models import (
    DIMENSION_NAMES,
    EvaluationResult,
    JudgeEvaluation,
    JudgeScores,
)
from owtn.evaluation.prompts import (
    build_classification_prompt,
    build_judge_system,
    build_judge_user,
)
from owtn.judging.scoring import aggregate_judge_scores, holder_mean
from owtn.llm.query import query_async
from owtn.models.judge import JudgePersona, load_panel
from owtn.models.stage_1.classification import ClassificationResult, Confidence
from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_1.config import StageConfig

logger = logging.getLogger(__name__)

# Detailed profiling at TRACE level (below DEBUG). Set logging to level 9
# to see per-judge and per-gate breakdowns. DEBUG shows per-concept summary.
TRACE = logging.DEBUG - 1

# Patterns that indicate trivial/placeholder content.
_TRIVIAL_PATTERNS = [
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\blorem ipsum\b", re.IGNORECASE),
    re.compile(r"\binsert .{0,40}here\b", re.IGNORECASE),
    re.compile(r"\bplaceholder\b", re.IGNORECASE),
    re.compile(r"^a (?:person|man|woman|character) (?:faces|deals with|encounters) a (?:challenge|problem|situation)\.?$", re.IGNORECASE),
    re.compile(r"^this is a story about\b", re.IGNORECASE),
    re.compile(r"^write a story\b", re.IGNORECASE),
    re.compile(r"^generate a\b", re.IGNORECASE),
]


def _is_trivial(genome: ConceptGenome) -> str | None:
    """Return error message if genome is trivial/placeholder, else None."""
    for pattern in _TRIVIAL_PATTERNS:
        if pattern.search(genome.premise):
            return f"Trivial premise (matched: {pattern.pattern})"
        if pattern.search(genome.target_effect):
            return f"Trivial target_effect (matched: {pattern.pattern})"
    return None


def _fail(error: str, results_dir: Path) -> EvaluationResult:
    """Write failure output and return result."""
    result = EvaluationResult(correct=False, error=error)
    _write_outputs(result, results_dir)
    return result


def _write_outputs(result: EvaluationResult, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "correct.json").write_text(
        json.dumps({"correct": result.correct})
    )
    (results_dir / "metrics.json").write_text(
        result.model_dump_json(indent=2)
    )


def _average_scores(evaluations: list[JudgeEvaluation]) -> dict[str, float]:
    """Average per-dimension scores across judges."""
    arrays = {name: [] for name in DIMENSION_NAMES}
    for ev in evaluations:
        for name in DIMENSION_NAMES:
            arrays[name].append(getattr(ev.scores, name))
    return {name: float(np.mean(vals)) for name, vals in arrays.items()}


async def _eval_one_judge(
    judge: JudgePersona,
    genome: ConceptGenome,
    config: StageConfig,
) -> JudgeEvaluation:
    """Run a single judge evaluation."""
    system_msg = build_judge_system(judge)
    user_msg = build_judge_user(genome)

    model_name = judge.model[0]

    t0 = time.perf_counter()
    result = await query_async(
        model_name=model_name,
        msg=user_msg,
        system_msg=system_msg,
        output_model=JudgeScores,
    )
    dt = time.perf_counter() - t0
    logger.log(TRACE, "[profile] Judge %s (%s): %.1fs", judge.id, result.model_name, dt)

    scores: JudgeScores = result.content
    h_score = holder_mean(scores.to_list(), p=config.evaluation.holder_p)

    return JudgeEvaluation(
        judge_id=judge.id,
        scores=scores,
        holder_score=h_score,
        model_used=result.model_name,
        cost=result.cost,
    )


async def _classify_concept(
    genome: ConceptGenome,
    avg_scores: dict[str, float],
    config: StageConfig,
) -> ClassificationResult:
    """Classify concept into MAP-Elites dimensions."""
    prompt = build_classification_prompt(genome, avg_scores)

    result = await query_async(
        model_name=config.llm.classifier_model,
        msg=prompt,
        system_msg="You are a precise classifier. Respond with JSON only.",
    )

    # Anthropic doesn't support structured output — parse JSON from raw response.
    raw = result.content
    json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group())
    else:
        data = json.loads(raw)

    # Merge LLM classification with rule-based constraint_density.
    data["constraint_density"] = genome.classify_constraint_density().value

    # Fill missing confidence fields with low if absent.
    for dim in ("concept_type", "arc_shape", "tonal_register", "thematic_domain"):
        conf_key = f"{dim}_confidence"
        if conf_key not in data:
            data[conf_key] = Confidence.LOW.value

    return ClassificationResult.model_validate(data)


async def evaluate(
    program_path: str,
    results_dir: str,
    config_path: str,
) -> EvaluationResult:
    """Run the full Stage 1 evaluation pipeline on a single concept genome.

    Called as a subprocess by ShinkaEvolve. Writes metrics.json and
    correct.json to results_dir.
    """
    results_path = Path(results_dir)
    config = StageConfig.from_yaml(config_path)
    t_total = time.perf_counter()

    # --- GATE 1: Validation ---
    t0 = time.perf_counter()
    program = Path(program_path)
    try:
        raw = json.loads(program.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return _fail(f"Invalid JSON: {e}", results_path)

    try:
        genome = ConceptGenome.model_validate(raw)
    except ValidationError as e:
        return _fail(f"Genome validation failed: {e}", results_path)

    trivial_error = _is_trivial(genome)
    if trivial_error:
        return _fail(trivial_error, results_path)
    logger.log(TRACE, "[profile] Gate 1 (validation): %.3fs", time.perf_counter() - t0)

    # --- GATE 2: Anti-Cliche (stub) ---
    t0 = time.perf_counter()
    anti_cliche = gate_2_anti_cliche(genome, config.evaluation.anti_cliche)
    logger.log(TRACE, "[profile] Gate 2 (anti-cliche): %.3fs", time.perf_counter() - t0)

    # --- GATE 3: Judge Panel ---
    panel = load_panel(config.judges.judges_dir, config.judges.panel)
    t0 = time.perf_counter()
    judge_results = await asyncio.gather(
        *[_eval_one_judge(judge, genome, config) for judge in panel]
    )
    logger.log(TRACE, "[profile] Gate 3 (judges parallel): %.1fs", time.perf_counter() - t0)

    # Aggregate scores.
    agg = aggregate_judge_scores(
        [j.holder_score for j in judge_results],
        diversity_weight=config.evaluation.diversity_weight,
        std_threshold=config.evaluation.std_threshold,
    )
    avg_scores = _average_scores(judge_results)

    # --- Classification ---
    t0 = time.perf_counter()
    classification = await _classify_concept(genome, avg_scores, config)
    logger.log(TRACE, "[profile] Classification (%s): %.1fs", config.llm.classifier_model, time.perf_counter() - t0)

    # --- Assemble output ---
    total_cost = sum(j.cost for j in judge_results)
    text_feedback = "\n\n---\n\n".join(
        f"[{j.judge_id}]\n{j.scores.reasoning}" for j in judge_results
    )

    result = EvaluationResult(
        correct=True,
        combined_score=agg.combined_score,
        holder_score=agg.holder_score,
        public_metrics={
            "dimensions": avg_scores,
            "cell_key": [v.value for v in classification.cell_key()],
            "classification": classification.model_dump(mode="json"),
        },
        private_metrics={
            "anti_cliche": anti_cliche,
            "judge_evaluations": [
                {
                    "judge_id": j.judge_id,
                    "holder_score": j.holder_score,
                    "scores": j.scores.to_dict(),
                    "model_used": j.model_used,
                    "cost": j.cost,
                }
                for j in judge_results
            ],
            "aggregate": {
                "judge_mean": agg.judge_mean,
                "judge_std": agg.judge_std,
                "diversity_bonus": agg.diversity_bonus,
            },
            "total_cost": total_cost,
        },
        text_feedback=text_feedback,
    )

    _write_outputs(result, results_path)
    dt_total = time.perf_counter() - t_total
    slowest = max(judge_results, key=lambda j: j.cost)
    logger.debug(
        "[profile] Concept evaluated: %.1fs | score=%.2f | cost=$%.4f | slowest=%s",
        dt_total, agg.combined_score, total_cost, slowest.model_used,
    )
    return result
