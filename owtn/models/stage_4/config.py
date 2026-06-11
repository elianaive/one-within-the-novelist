"""Stage 4 run config: filter, phases, revise, plateau.

Mirrors `owtn.models.stage_3.config.Stage3Config` shape — top-level
`Stage4Config` with sub-configs per concern, `from_yaml` that accepts
either a `stage_4:`-wrapped or flat document.

Standalone CLI:
    uv run python -m owtn.models.stage_4.config <stage_4_config.yaml>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


ReasoningEffort = Literal["disabled", "low", "medium", "high"]


class Stage4FilterConfig(BaseModel):
    """Pre-stage classification — two cheap haiku-class calls at session start."""
    audience_model: str = "claude-haiku-4-5-20251001"
    experts_model: str = "claude-haiku-4-5-20251001"
    reasoning_effort: ReasoningEffort = "medium"


class Stage4PreThinkConfig(BaseModel):
    """Phase 1 — free-form per-scene planning."""
    explore_max_iters: int = Field(default=20, ge=1)
    reasoning_effort: ReasoningEffort = "medium"


class Stage4DownDraftConfig(BaseModel):
    """Phase 2 — per-scene write with read-as-reader micro-loop. Reasoning
    is OFF here — thinking mode is "more detached" / unhelpful for prose
    generation per `project_voice_api_techniques.md`."""
    explore_max_iters: int = Field(default=80, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)


class Stage4ReviseConfig(BaseModel):
    """Phase 3 — cycled gather/revise sub-phases.

    `cycle_cap` is the empirical AutoNovel/CritiCS/Self-Refine ceiling
    above the convergence sweet spot. `call_ceiling` is the independent
    backstop that catches runaway loops the cycle counter misses.
    `tier_b_in_cycle_zero` controls whether cycle 0's pre-launch is the
    full sweep (Tier A + Tier B) or Tier A only.
    """
    cycle_cap: int = Field(default=6, ge=1)
    call_ceiling: int = Field(default=50, ge=1)
    gather_max_iters: int = Field(default=40, ge=1)
    revise_max_iters: int = Field(default=40, ge=1)
    tier_b_in_cycle_zero: bool = True
    reasoning_effort: ReasoningEffort = "medium"


class Stage4SurgicalEditConfig(BaseModel):
    """Surgical-edit dispatch — scope translator (haiku-class) and bounded
    subagent (sonnet-class by default). The translator turns natural-language
    scope into anchor strings; the subagent rewrites the bounded region with
    the full manuscript in mind."""
    translator_model: str = "claude-haiku-4-5-20251001"
    subagent_model: str = "claude-sonnet-4-6"


class Stage4PlateauConfig(BaseModel):
    """Plateau detection thresholds. Defaults match the architecture
    commitment; pilot-tunable via YAML override."""
    window: int = Field(default=2, ge=1)
    require_total_decrease: bool = True
    require_severe_progress: bool = True


class Stage4Config(BaseModel):
    """Top-level Stage 4 config.

    `generator_model` is the writer agent's model — the same agent runs
    PreThink, DownDraft, and Revise. Default `deepseek-v4-pro` for dev;
    prod uses `claude-opus-4-7`. Critic models are per-critic in the
    YAML files.
    """
    generator_model: str = "deepseek-v4-pro"
    filter: Stage4FilterConfig = Field(default_factory=Stage4FilterConfig)
    prethink: Stage4PreThinkConfig = Field(default_factory=Stage4PreThinkConfig)
    downdraft: Stage4DownDraftConfig = Field(default_factory=Stage4DownDraftConfig)
    revise: Stage4ReviseConfig = Field(default_factory=Stage4ReviseConfig)
    surgical_edit: Stage4SurgicalEditConfig = Field(default_factory=Stage4SurgicalEditConfig)
    plateau: Stage4PlateauConfig = Field(default_factory=Stage4PlateauConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> Stage4Config:
        with open(Path(path)) as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and "stage_4" in data:
            data = data["stage_4"]
        return cls.model_validate(data)


# ----- Standalone CLI -----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Stage 4 config YAML.")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.config_path.exists():
        print(f"error: {args.config_path} not found", file=sys.stderr)
        return 1
    try:
        cfg = Stage4Config.from_yaml(args.config_path)
    except (ValidationError, yaml.YAMLError) as e:
        print(f"validation failed for {args.config_path}: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"OK: {args.config_path}\n"
            f"  generator_model: {cfg.generator_model}\n"
            f"  cycle_cap:       {cfg.revise.cycle_cap}\n"
            f"  call_ceiling:    {cfg.revise.call_ceiling}\n"
            f"  plateau.window:  {cfg.plateau.window}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
