"""Stage 2 config schema test. Skipped when the YAML doesn't exist yet.

`configs/stage_2/light.yaml` ships in Phase 9 (runner). Phase 1 just defines
the schema and skips the test until the file exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path("/home/user/one-within-the-novelist")
LIGHT_YAML = REPO_ROOT / "configs" / "stage_2" / "light.yaml"


@pytest.mark.skipif(
    not LIGHT_YAML.exists(),
    reason=f"Stage 2 config YAML not yet shipped (Phase 9 work). Expected: {LIGHT_YAML}",
)
def test_light_config_validates() -> None:
    from owtn.models.stage_2.config import Stage2Config

    cfg = Stage2Config.from_yaml(LIGHT_YAML)
    assert cfg.iterations_per_phase >= 1
    assert len(cfg.dimensions) == 8


def test_scalar_brief_re_summarize_every_validates() -> None:
    """`scalar_brief_re_summarize_every` is a real config knob threaded into
    `_make_rollout_fn_scalar` — must accept positive ints and reject 0."""
    import pydantic

    from owtn.models.stage_2.config import Stage2Config

    base = _minimal_config_dict()

    # Default applies when omitted.
    cfg = Stage2Config.model_validate(base)
    assert cfg.scalar_brief_re_summarize_every == 5

    # Custom value accepted.
    cfg2 = Stage2Config.model_validate({**base, "scalar_brief_re_summarize_every": 1})
    assert cfg2.scalar_brief_re_summarize_every == 1

    # Zero rejected (Field(ge=1)).
    with pytest.raises(pydantic.ValidationError):
        Stage2Config.model_validate({**base, "scalar_brief_re_summarize_every": 0})


def test_scoring_handoff_top_k_field_removed() -> None:
    """The redundant `scoring_handoff_top_k` field is gone from the schema.
    `top_k_to_stage_3` does the actual filtering in `build_handoff_for_concept`."""
    from owtn.models.stage_2.config import Stage2Config

    fields = Stage2Config.model_fields
    assert "scoring_handoff_top_k" not in fields
    assert "top_k_to_stage_3" in fields


def test_scalar_with_simulate_rollouts_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`simulate_rollouts: true` under `scoring_mode: scalar` is silently
    ignored at runtime (the simulator is pairwise-only). The validator
    surfaces a warning at config-load time so the YAML's stated intent
    matches what runs."""
    import logging as _logging

    from owtn.models.stage_2.config import Stage2Config

    base = _minimal_config_dict()
    base["scoring_mode"] = "scalar"
    base["simulate_rollouts"] = True
    # cheap_judge_id is not required under scalar; leave it unset.

    with caplog.at_level(_logging.WARNING, logger="owtn.models.stage_2.config"):
        Stage2Config.model_validate(base)

    assert any(
        "simulate_rollouts=true ignored under scoring_mode=scalar" in rec.message
        for rec in caplog.records
    ), [rec.message for rec in caplog.records]


def _minimal_config_dict() -> dict:
    """Smallest valid Stage2Config payload — used by tests that probe one
    field at a time. Mirrors the validated-fields surface in
    `Stage2Config.from_yaml`."""
    return {
        "iterations_per_phase": 5,
        "phase_3_iterations": 1,
        "k_candidates_per_expansion": 4,
        "exploration_constant": 0.5,
        "discount_gamma": 1.0,
        "rechallenge_interval": 25,
        "rechallenge_top_pct": 0.10,
        "expansion_model": "deepseek-v4-pro",
        "rollout_model": "deepseek-v4-pro",
        "cheap_judge_model": "gpt-5.4-mini",
        "classifier_model": "deepseek-v4-flash",
        "simulation_model": "deepseek-v4-flash",
        "scoring_mode": "scalar",
        "simulate_rollouts": False,
        "judges": {
            "full_panel_ids": ["gwern", "roon"],
        },
        "presets": {
            "light": ["cassandra_ish", "randy_ish"],
            "medium": ["cassandra_ish", "phoebe_ish", "randy_ish", "winston_ish"],
            "heavy": ["cassandra_ish", "phoebe_ish", "randy_ish", "winston_ish"],
        },
        "preset_params": {
            "cassandra_ish": {"min_rest_beats": 1, "max_flat_beats": 4,
                              "intensity_variance": "tight", "recovery_required": True},
            "phoebe_ish": {"min_rest_beats": 3, "max_flat_beats": 6,
                           "intensity_variance": "tight", "recovery_required": True},
            "randy_ish": {"min_rest_beats": 0, "max_flat_beats": 8,
                          "intensity_variance": "wide", "recovery_required": False},
            "winston_ish": {"min_rest_beats": 2, "max_flat_beats": 3,
                            "intensity_variance": "tight", "recovery_required": True},
        },
        "advance_from_stage_1": "all",
        "max_concepts_from_stage_1": None,
        "top_k_to_stage_3": 1,
        "near_tie_promoted": True,
        "dimensions": [
            "edge_logic", "motivational_coherence", "tension_information_arch",
            "post_dictability", "arc_integrity_ending", "structural_coherence",
            "beat_quality", "concept_fidelity_thematic",
        ],
        "per_concept_time_budget_minutes": 30,
        "per_phase_time_budget_minutes": 10,
        "no_improvement_cutoff_iterations": 15,
        "archive_bin_boundaries": {
            "disclosure_ratio": [0.10, 0.25, 0.40, 0.55],
            "structural_density": [1.2, 1.8, 2.5, 3.2],
        },
        "node_count_targets": {
            1000: [3, 5], 3000: [5, 8], 5000: [7, 12], 10000: [10, 18],
        },
    }
