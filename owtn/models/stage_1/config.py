from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

from owtn.evaluation.models import DIMENSION_NAMES
from owtn.models.config import LLMConfig


class AnnealingSchedule(BaseModel):
    """Genesis-ratio schedule: high early (diversity), low late (refinement).

    Applied after ``warmup_fraction`` of total generations. Temperature is a
    fixed per-run setting now — see ``LLMConfig.generation_temperature``.
    """
    genesis_ratio_early: float = 0.6
    genesis_ratio_late: float = 0.1
    warmup_fraction: float = 0.4


class EvolutionConfig(BaseModel):
    num_generations: int
    language: str
    patch_types: list[str]
    patch_type_probs: list[float]
    max_patch_resamples: int
    llm_dynamic_selection: str
    use_text_feedback: bool
    evolve_prompts: bool
    meta_rec_interval: int
    code_embed_sim_threshold: float
    max_novelty_attempts: int
    genesis_ratio: float = 0.0
    tonal_inherit_rate: float = 0.5       # per-dimension probability of inheriting parent's register or mode on mutation
    tonal_crossover_new_rate: float = 0.33  # per-dimension probability of fresh roll on crossover (vs parent A or B)
    annealing: AnnealingSchedule = AnnealingSchedule()


class DatabaseConfig(BaseModel):
    num_islands: int
    archive_size: int
    archive_selection_strategy: str
    migration_interval: int
    migration_rate: float
    island_elitism: bool
    island_selection_strategy: str = "uniform"
    enable_dynamic_islands: bool
    stagnation_threshold: int
    parent_selection_strategy: str
    exploitation_alpha: float
    exploitation_ratio: float


class OperatorBanditConfig(BaseModel):
    enabled: bool
    warmup_generations: int
    exploration_constant: float
    min_probability_floor: float


class JudgesConfig(BaseModel):
    panel: list[str]
    judges_dir: str
    min_demanding_ratio: float


class AntiCliqueConfig(BaseModel):
    similarity_threshold: float
    elevated_novelty_threshold: float
    patterns_file: str


class PairwiseAggregationConfig(BaseModel):
    """Weights and tiebreaker rules for pairwise dim-vote aggregation.

    Encodes "mid is the primary failure mode" by tilting selection toward
    distinguishers (Indelibility, Grip, Novelty, Gen-Fert) and treating
    Coherence/Scope as table-stakes floors. Rationale:
    lab/issues/2026-04-21-rubric-reweighting.md.
    """
    dim_weights: dict[str, float]
    tiebreaker_threshold: float
    tiebreaker_dims: list[str]

    @model_validator(mode="after")
    def _validate(self) -> PairwiseAggregationConfig:
        expected = set(DIMENSION_NAMES)
        got = set(self.dim_weights)
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise ValueError(
                f"dim_weights must cover all 9 dimensions. "
                f"missing={missing} extra={extra}"
            )
        for d in self.tiebreaker_dims:
            if d not in expected:
                raise ValueError(f"tiebreaker_dims contains unknown dim: {d!r}")
        if any(w < 0 for w in self.dim_weights.values()):
            raise ValueError("dim_weights must be non-negative")
        if self.tiebreaker_threshold < 0:
            raise ValueError("tiebreaker_threshold must be non-negative")
        return self


class EvaluationConfig(BaseModel):
    holder_p: float
    diversity_weight: float
    std_threshold: float
    anti_cliche: AntiCliqueConfig
    pairwise: PairwiseAggregationConfig


class HandoffConfig(BaseModel):
    strategy: str
    max_concepts: int


class PathsConfig(BaseModel):
    seed_bank: str
    convergence_patterns: str
    # Where the seed-embedding cache lives. Sibling of the seed bank by default
    # so an updated seed bank keeps its embeddings nearby. Computed lazily on
    # first farthest-first selection; recomputed when a seed's content changes.
    seed_embeddings_cache: str = "data/seed-bank-embeddings.jsonl"


class StageConfig(BaseModel):
    stage: int
    prompt: str
    evolution: EvolutionConfig
    database: DatabaseConfig
    operator_bandit: OperatorBanditConfig
    llm: LLMConfig
    judges: JudgesConfig
    evaluation: EvaluationConfig
    handoff: HandoffConfig
    paths: PathsConfig

    # Seed sampling. "uniform" preserves prior behavior. "farthest_first" runs
    # the k-center heuristic on Qwen3-Embedding-0.6B vectors so each pick is
    # maximally unlike the run's already-used seeds. See
    # `lab/issues/2026-04-28-seed-embedding-diversity.md`.
    seed_sampling_strategy: Literal["uniform", "farthest_first"] = "uniform"

    @classmethod
    def from_yaml(cls, path: str) -> StageConfig:
        with open(Path(path)) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
