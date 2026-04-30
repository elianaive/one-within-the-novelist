"""Stage 3 run config: bench picker/drafter, casting, voice-session knobs.

Mirrors `owtn.models.stage_2.config.Stage2Config`. The fields that were
hardcoded across Stage 3 (picker/drafter/caster models, panel size,
phase explore-loop iteration caps, commit sampler) move here so a run is
fully describable by its YAML.

Standalone CLI:
    uv run python -m owtn.models.stage_3.config <stage3_config.yaml>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


ReasoningEffort = Literal["disabled", "low", "medium", "high"]


class AdjacentSceneConfig(BaseModel):
    """Bench picker + drafter parameters."""
    picker_model: str
    drafter_model: str
    picker_max_attempts: int = Field(default=3, ge=1)
    draft_word_cap: int = Field(default=400, ge=50)
    picker_reasoning_effort: ReasoningEffort = "medium"


class CastingConfig(BaseModel):
    """Voice-panel casting classifier parameters."""
    caster_model: str
    panel_size: int = Field(default=4, ge=1)
    reasoning_effort: ReasoningEffort = "medium"


class PhaseExploreConfig(BaseModel):
    """Tool-use loop budget for one phase."""
    explore_max_iters: int = Field(ge=1)


class CommitSamplerConfig(BaseModel):
    """Override sampler for the prose-bearing finalize_voice_genome call.

    Reasoning stays disabled by default (per `project_voice_api_techniques`
    memory: thinking mode is "more detached" / unhelpful for prose).
    """
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    max_tokens: int = Field(default=16384, ge=1024)


class VoiceSessionConfig(BaseModel):
    """Multi-agent voice-session orchestration parameters.

    `generator_model` is the LLM the voice agents run on for explore loops
    (Phase 1, Phase 4) and analytical commits (Phase 3, Phase 5). It
    overrides the persona-level `model` field in each agent's YAML so the
    Stage 3 run is fully describable from this config alone. Default
    `deepseek-v4-pro`: reasoning model with cheap-enough cost for the
    long tool-use loops voice work demands.
    """
    generator_model: str = "deepseek-v4-pro"
    phase_1: PhaseExploreConfig
    phase_4: PhaseExploreConfig
    commit_sampler: CommitSamplerConfig = Field(default_factory=CommitSamplerConfig)
    analytical_reasoning_effort: ReasoningEffort = "medium"


class Stage3Config(BaseModel):
    """Top-level Stage 3 config."""
    adjacent_scene: AdjacentSceneConfig
    casting: CastingConfig
    voice_session: VoiceSessionConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> Stage3Config:
        with open(Path(path)) as f:
            data = yaml.safe_load(f)
        # Allow either a top-level "stage_3:" wrapper (matches the docs example)
        # or a flat document.
        if isinstance(data, dict) and "stage_3" in data:
            data = data["stage_3"]
        return cls.model_validate(data)


# ----- Standalone CLI -----

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Stage 3 config YAML file.",
    )
    parser.add_argument("config_path", type=Path, help="Path to stage_3 config YAML")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress success summary",
    )
    args = parser.parse_args(argv)

    if not args.config_path.exists():
        print(f"error: {args.config_path} not found", file=sys.stderr)
        return 1

    try:
        cfg = Stage3Config.from_yaml(args.config_path)
    except ValidationError as e:
        print(f"validation failed for {args.config_path}:\n{e}", file=sys.stderr)
        return 1
    except yaml.YAMLError as e:
        print(f"YAML parse error in {args.config_path}: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"OK: {args.config_path}\n"
            f"  picker_model:  {cfg.adjacent_scene.picker_model}\n"
            f"  drafter_model: {cfg.adjacent_scene.drafter_model}\n"
            f"  caster_model:  {cfg.casting.caster_model}\n"
            f"  panel_size:    {cfg.casting.panel_size}\n"
            f"  phase_1 iters: {cfg.voice_session.phase_1.explore_max_iters}\n"
            f"  phase_4 iters: {cfg.voice_session.phase_4.explore_max_iters}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
