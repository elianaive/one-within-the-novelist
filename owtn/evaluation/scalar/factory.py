"""Build a Scorer from a YAML composition spec.

Specs live in `configs/scalar/scorer_compositions.yaml` keyed by name. Each
spec declares: rubric, judge_model, sampling_kwargs, mode (single/atomic),
optional persona panel for ensemble.

Example:
    rollout_reward:
      rubric: dag
      judge_model: deepseek-v4-flash
      mode: atomic
      reasoning_disabled: true
      persona: null   # S1 (no persona ensemble)

    handoff_rescore:
      rubric: dag
      judge_model: deepseek-v4-flash
      mode: atomic
      reasoning_disabled: true
      persona_panel: [alexander-wales, gwern, jamie-wahls, roon]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

from owtn.evaluation.scalar.rubrics import load_rubric
from owtn.evaluation.scalar.scorer import (
    AtomicPerDimScorer,
    EnsembleScorer,
    Scorer,
    SingleCallScorer,
)
from owtn.models.judge import JudgePersona

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "scalar"
_JUDGES_DIR = Path(__file__).resolve().parents[3] / "configs" / "judges"


def build_scorer_from_config(
    composition_name: str,
    artifact_renderer: Callable[[Any], str],
    *,
    config_dir: Path | None = None,
    judges_dir: Path | None = None,
) -> Scorer:
    """Build a `Scorer` for a named composition from `scorer_compositions.yaml`.

    `artifact_renderer` is provided by the call site (Stage-1 vs Stage-2).
    """
    config_dir = config_dir or _DEFAULT_CONFIG_DIR
    judges_dir = judges_dir or _JUDGES_DIR
    spec = _load_composition(composition_name, config_dir)

    rubric = load_rubric(spec["rubric"], config_dir=config_dir)
    sampling_kwargs = _build_sampling_kwargs(spec)
    base_cls = AtomicPerDimScorer if spec.get("mode", "atomic") == "atomic" else SingleCallScorer

    panel: list[str] | None = spec.get("persona_panel")
    if panel:
        base_scorers: list[Scorer] = [
            base_cls(
                rubric=rubric,
                judge_model=spec["judge_model"],
                artifact_renderer=artifact_renderer,
                persona_system_msg=_render_persona_system_msg(pid, judges_dir),
                persona_label=pid,
                sampling_kwargs=sampling_kwargs,
            )
            for pid in panel
        ]
        return EnsembleScorer(
            base_scorers=base_scorers,
            aggregation=spec.get("aggregation", "mean"),
            label=composition_name,
        )

    return base_cls(
        rubric=rubric,
        judge_model=spec["judge_model"],
        artifact_renderer=artifact_renderer,
        persona_system_msg=None,
        persona_label="neutral",
        sampling_kwargs=sampling_kwargs,
    )


def _load_composition(name: str, config_dir: Path) -> dict[str, Any]:
    path = config_dir / "scorer_compositions.yaml"
    if not path.exists():
        raise FileNotFoundError(f"composition YAML not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if name not in raw:
        raise KeyError(f"composition {name!r} not in {path} (have: {list(raw)})")
    return raw[name]


def _build_sampling_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    """Materialize provider-specific sampling overrides.

    DeepSeek's reasoning toggle goes through `extra_body.thinking.type`; the
    plain `reasoning_effort=disabled` is rejected by the API. For other
    providers, callers can pass the raw `sampling_kwargs` dict directly.
    """
    kwargs: dict[str, Any] = dict(spec.get("sampling_kwargs", {}))
    kwargs.setdefault("temperature", 0.0)

    if spec.get("reasoning_disabled"):
        provider_hint = spec.get("provider_family", "deepseek")
        if provider_hint == "deepseek":
            kwargs.setdefault("extra_body", {"thinking": {"type": "disabled"}})
        elif provider_hint == "openrouter":
            kwargs.setdefault("extra_body", {"reasoning": {"enabled": False}})

    return kwargs


def _render_persona_system_msg(persona_id: str, judges_dir: Path) -> str:
    """Load a JudgePersona YAML and render the identity-block system msg.

    Excludes the pairwise rubric / harshness blocks — the scalar scorer's
    system_msg owns the rubric. Persona contributes only identity, values,
    exemplars, and lean-ins.
    """
    path = judges_dir / f"{persona_id}.yaml"
    data = yaml.safe_load(path.read_text())
    persona = JudgePersona.model_validate(data)
    blocks = [
        f"You are {persona.name}.",
        "",
        persona.identity,
        "",
        "YOUR VALUES (in order of priority):",
        *[f"- {v}" for v in persona.values],
        "",
        "TASTE REFERENCES:",
        *[f"- {e}" for e in persona.exemplars],
        "",
        "WHAT CATCHES YOUR ATTENTION (lean-in signals):",
        *[f"- {s}" for s in persona.lean_in_signals],
        "",
        "VOICE: write your reasoning in your evaluative voice — your distinctive critical vocabulary visible where it applies. Generic professional-critic prose is itself a persona failure.",
    ]
    return "\n".join(blocks)
