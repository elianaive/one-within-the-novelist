from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from owtn.models.config import LLMConfig


class AnnealingSchedule(BaseModel):
    """Annealing schedule for generation parameters: explore early, refine later.

    Temperature and genesis ratio both start high (diversity) and anneal down
    (refinement) after ``warmup_fraction`` of total generations.
    """
    temp_early: list[float] = [0.8, 0.9, 1.0]
    temp_late: list[float] = [0.5, 0.6, 0.7]
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


class EvaluationConfig(BaseModel):
    holder_p: float
    diversity_weight: float
    std_threshold: float
    anti_cliche: AntiCliqueConfig
    tier_a_enabled: bool
    pairwise_enabled: bool
    dynamic_rubrics_enabled: bool


class HandoffConfig(BaseModel):
    strategy: str
    max_concepts: int


class PathsConfig(BaseModel):
    seed_bank: str
    convergence_patterns: str


class StageConfig(BaseModel):
    stage: int
    steering: str
    evolution: EvolutionConfig
    database: DatabaseConfig
    operator_bandit: OperatorBanditConfig
    llm: LLMConfig
    judges: JudgesConfig
    evaluation: EvaluationConfig
    handoff: HandoffConfig
    paths: PathsConfig

    @classmethod
    def from_yaml(cls, path: str) -> StageConfig:
        with open(Path(path)) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
