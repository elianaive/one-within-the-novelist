"""Stage 2 run config: MCTS budgets, model routing, presets, evaluation knobs.

Mirrors `owtn.models.stage_1.config.StageConfig`. The actual YAML (
`configs/stage_2/light.yaml` and friends) ships in Phase 9 — this module
defines the schema now so other Phase 1+ code can typecheck against it.

Standalone CLI:
    uv run python -m owtn.models.stage_2.config <stage2_config.yaml>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from owtn.models.stage_2.pacing import PRESET_NAMES


PresetName = Literal["cassandra_ish", "phoebe_ish", "randy_ish", "winston_ish"]


class PresetParams(BaseModel):
    """Descriptive in v1; numeric only if v1.5 restores parametric priors."""
    min_rest_beats: int = Field(ge=0)
    max_flat_beats: int = Field(ge=1)
    intensity_variance: Literal["tight", "wide"]
    recovery_required: bool


class JudgesConfig(BaseModel):
    """Panel composition for the Stage 2 run.

    `full_panel_ids` is the panel that votes on full-panel commitment events
    (rollout confirmation, within-concept tournament, pilot head-to-head).
    `cheap_judge_id` must be one of `full_panel_ids` — the cheap-judge call
    samples one judge from the panel to deliver the rollout reward signal
    cheaply, and the same identity must also appear on the full panel so
    judge-level state stays consistent across tiers.
    """
    full_panel_ids: list[str] = Field(min_length=1)
    cheap_judge_id: str

    @model_validator(mode="after")
    def _cheap_in_panel(self) -> JudgesConfig:
        if self.cheap_judge_id not in self.full_panel_ids:
            raise ValueError(
                f"cheap_judge_id {self.cheap_judge_id!r} not in full_panel_ids "
                f"{self.full_panel_ids}"
            )
        return self


class PresetsByConfig(BaseModel):
    light: list[PresetName]
    medium: list[PresetName]
    heavy: list[PresetName]


class ArchiveBinBoundaries(BaseModel):
    disclosure_ratio: list[float]   # 4 internal cuts → 5 bins
    structural_density: list[float]


class NodeCountTargets(BaseModel):
    """target_prose_length (words) → [min_nodes, max_nodes]."""
    targets: dict[int, tuple[int, int]]


class Stage2Config(BaseModel):
    # MCTS budget
    iterations_per_phase: int = Field(ge=1)
    phase_3_iterations: int = Field(ge=0)
    k_candidates_per_expansion: int = Field(ge=1)

    # UCB / D-UCB
    exploration_constant: float = Field(ge=0.0)
    discount_gamma: float = Field(default=0.93, ge=0.0, le=1.0)
    rechallenge_interval: int = Field(ge=1)
    rechallenge_top_pct: float = Field(ge=0.0, le=1.0)

    # AlphaZero-style parallel rollouts. `mcts_parallel_workers > 1` runs that
    # many concurrent worker tasks against each MCTS tree, with virtual loss
    # at selection time spreading them across children. Each worker holds the
    # tree lock only during selection, expansion-cache claim, and backprop;
    # LLM calls (expand, rollout) execute outside the lock so multiple
    # requests are in flight simultaneously.
    mcts_parallel_workers: int = Field(default=1, ge=1)
    mcts_virtual_loss: float = Field(default=1.0, ge=0.0)

    # Models — cross-family discipline enforced by config conventions, not here
    expansion_model: str
    rollout_model: str
    cheap_judge_model: str
    classifier_model: str | None = None  # for champion-brief summarizer + Tier 3

    # Tiered judge (legacy pairwise-champion path)
    full_panel_on_promotion: bool = True
    full_panel_rejection_backprop: float = Field(default=0.5, ge=0.0, le=1.0)
    cheap_judge_agreement_alert: float = Field(default=0.70, ge=0.0, le=1.0)

    # Scoring mode. "pairwise_champion" = cheap-judge tiered against a running
    # champion (`docs/stage-2/mcts.md`). "scalar" = absolute scalar reward via
    # `owtn.evaluation.scalar`, fixing cold-champion saturation by removing
    # the reference point. Scalar mode reads composition names from the
    # `scoring_*_composition` fields below.
    scoring_mode: Literal["pairwise_champion", "scalar"] = "pairwise_champion"
    scoring_rollout_composition: str = "rollout_reward"
    scoring_handoff_composition: str = "handoff_rescore"
    scoring_handoff_top_k: int = Field(default=5, ge=1)

    # Bounded MCTS rollout simulation per `mcts.md` §Simulation. When
    # enabled, each rollout walks up to `simulation_max_steps` one-step
    # extensions via `simulation_model`; cheap-judge fires after each
    # accepted step, the walk halts on non-improvement, and the best partial
    # reached is what both cheap-judge and full panel evaluate.
    # `simulation_min_partial_size` skips the walk entirely when the
    # rollout's partial is too small for forward extensions to be
    # discriminating.
    simulate_rollouts: bool = False
    simulation_max_steps: int = Field(default=3, ge=1, le=8)
    simulation_min_partial_size: int = Field(default=0, ge=0)
    simulation_model: str

    # Panel composition (replaces hardcoded panel ids in the runner / pilot).
    judges: JudgesConfig

    # Presets
    presets: PresetsByConfig
    preset_params: dict[PresetName, PresetParams]

    # Handoff
    advance_from_stage_1: Literal["all", "top_k"]
    max_concepts_from_stage_1: int | None = None
    top_k_to_stage_3: int | None = None  # null = all
    near_tie_promoted: bool = True

    # Evaluation
    dimensions: list[str]   # 8 dimension names

    # Budget
    per_concept_time_budget_minutes: int = Field(ge=1)
    per_phase_time_budget_minutes: int = Field(ge=1)
    no_improvement_cutoff_iterations: int = Field(ge=1)

    # Archive
    archive_bin_boundaries: ArchiveBinBoundaries

    # Node-count targets
    node_count_targets: dict[int, tuple[int, int]]

    @model_validator(mode="after")
    def _validate(self) -> Stage2Config:
        # Each preset listed in any tier must be a known preset name.
        for tier in (self.presets.light, self.presets.medium, self.presets.heavy):
            for p in tier:
                if p not in PRESET_NAMES:
                    raise ValueError(
                        f"unknown preset {p!r}; known: {PRESET_NAMES}"
                    )
        # preset_params should cover every preset that appears in any tier.
        used = set(self.presets.light) | set(self.presets.medium) | set(self.presets.heavy)
        missing = used - set(self.preset_params)
        if missing:
            raise ValueError(
                f"preset_params missing entries for: {sorted(missing)}"
            )
        # Bin boundaries: 4 internal cuts → 5 bins. Sorted ascending.
        for axis_name, cuts in (
            ("disclosure_ratio", self.archive_bin_boundaries.disclosure_ratio),
            ("structural_density", self.archive_bin_boundaries.structural_density),
        ):
            if len(cuts) != 4:
                raise ValueError(
                    f"archive_bin_boundaries.{axis_name} must have 4 internal cuts "
                    f"(got {len(cuts)})"
                )
            if cuts != sorted(cuts):
                raise ValueError(
                    f"archive_bin_boundaries.{axis_name} must be ascending"
                )
        # 8 dimensions per the rev-7 rubric.
        if len(self.dimensions) != 8:
            raise ValueError(
                f"dimensions must have exactly 8 entries (got {len(self.dimensions)})"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> Stage2Config:
        with open(Path(path)) as f:
            data = yaml.safe_load(f)
        # Allow either a top-level "stage_2:" wrapper (matches the docs example)
        # or a flat document.
        if isinstance(data, dict) and "stage_2" in data:
            data = data["stage_2"]
        return cls.model_validate(data)


# ----- Standalone CLI -----

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Stage 2 config YAML file.",
    )
    parser.add_argument("config_path", type=Path, help="Path to stage_2 config YAML")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress success summary",
    )
    args = parser.parse_args(argv)

    if not args.config_path.exists():
        print(f"error: {args.config_path} not found", file=sys.stderr)
        return 1

    try:
        cfg = Stage2Config.from_yaml(args.config_path)
    except ValidationError as e:
        print(f"validation failed for {args.config_path}:\n{e}", file=sys.stderr)
        return 1
    except yaml.YAMLError as e:
        print(f"YAML parse error in {args.config_path}: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"valid: iterations_per_phase={cfg.iterations_per_phase} "
            f"k={cfg.k_candidates_per_expansion} "
            f"presets(light/med/heavy)="
            f"{len(cfg.presets.light)}/{len(cfg.presets.medium)}/{len(cfg.presets.heavy)} "
            f"dimensions={len(cfg.dimensions)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
