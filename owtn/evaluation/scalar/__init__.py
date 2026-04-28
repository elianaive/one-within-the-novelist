"""Scalar (calibrated absolute) scoring.

Stage-2 MCTS rollout reward via pairwise-vs-incumbent saturates to 1.0 for
every walked partial when the incumbent is the cold-start seed. Scalar
scoring produces a per-artifact absolute score in [0, 1] that lets MCTS UCB
discriminate among siblings without a moving reference point.

Entry points:
- `Scorer` protocol with concrete implementations (`SingleCallScorer`,
  `AtomicPerDimScorer`, `EnsembleScorer`) — composition over inheritance.
- `Rubric`/`Dim` — score schema; loaded from `configs/scalar/rubric_*.yaml`.
- `AnchorCalibrator` — optional per-stage scale-mapping. Cross-stage anchor
  reuse is unsafe; each call site fits its own anchor set.
- `build_scorer_from_config` — factory consuming a named composition from
  `configs/scalar/scorer_compositions.yaml`.

Stage-2 wiring lives in `owtn.stage_2.tree_runtime` (scalar mode) and toggles
on `Stage2Config.scoring_mode == "scalar"`. The legacy pairwise-champion path
remains available behind `scoring_mode == "pairwise_champion"` for A/B comparison.
"""

from owtn.evaluation.scalar.calibration import AnchorCalibrator, bootstrap_ci
from owtn.evaluation.scalar.factory import build_scorer_from_config
from owtn.evaluation.scalar.rubrics import load_rubric
from owtn.evaluation.scalar.scorer import (
    AtomicPerDimScorer,
    EnsembleScorer,
    Scorer,
    SingleCallScorer,
)
from owtn.evaluation.scalar.types import (
    AggregatedScoreCard,
    Dim,
    Rubric,
    ScoreCard,
)

__all__ = [
    "AggregatedScoreCard",
    "AnchorCalibrator",
    "AtomicPerDimScorer",
    "Dim",
    "EnsembleScorer",
    "Rubric",
    "ScoreCard",
    "Scorer",
    "SingleCallScorer",
    "bootstrap_ci",
    "build_scorer_from_config",
    "load_rubric",
]
