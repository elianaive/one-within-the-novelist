"""Plateau detection for Phase 3 cycles.

The detector reads each completed cycle's accumulated `CriticReport`
collection (via `cycle["reports_seen"]`) and decides whether the
revision loop is converging or stuck. Two rules, both off issue counts +
severity tags so the project's judges-don't-score posture is preserved
(`feedback_judge_query_scope.md`):

- **No total decrease** — total issue count across all critics didn't
  decrease for the last `window` transitions
- **No severe progress** — severe-tagged issue count didn't decrease
  for the last `window` transitions

Either firing means plateau. Both are configurable on/off — the
default has both enabled. `window=2` means three completed cycles are
required before either rule can fire.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from owtn.models.stage_4 import CriticReport, Severity


@dataclass(frozen=True)
class PlateauVerdict:
    """One read of the cycles-so-far."""
    plateaued: bool
    reason: str
    total_per_cycle: list[int] = field(default_factory=list)
    severe_per_cycle: list[int] = field(default_factory=list)


@dataclass
class PlateauDetector:
    window: int = 2
    require_total_decrease: bool = True
    require_severe_progress: bool = True

    def check(self, cycles: list[dict]) -> PlateauVerdict:
        """Decide whether the cycles so far indicate plateau."""
        completed = [c for c in cycles if c.get("completed")]
        total = [_total_issue_count(c) for c in completed]
        severe = [_severe_issue_count(c) for c in completed]

        if len(completed) < self.window + 1:
            return PlateauVerdict(False, "insufficient-history", total, severe)

        recent_total = total[-(self.window + 1):]
        recent_severe = severe[-(self.window + 1):]

        if self.require_total_decrease and _no_decrease(recent_total):
            return PlateauVerdict(True, "total-issue-count-not-decreasing", total, severe)
        if self.require_severe_progress and _no_decrease(recent_severe):
            return PlateauVerdict(True, "no-severe-resolved", total, severe)

        return PlateauVerdict(False, "still-converging", total, severe)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _no_decrease(series: list[int]) -> bool:
    """True when each step in the series is >= the previous step AND
    there's something to decrease from.

    A series of all zeros means there were no issues to resolve — the
    rule has no signal, not a plateau. The rule fires only when the
    window contains some non-zero count and none of the transitions
    showed a strict decrease.
    """
    if len(series) < 2:
        return False
    if max(series) == 0:
        return False
    return all(series[i] >= series[i - 1] for i in range(1, len(series)))


def _total_issue_count(cycle: dict) -> int:
    """Sum issues across all critics in this cycle. Reports that flagged
    `not_load_bearing` count as zero issues — those are confirming
    observations, not corrections."""
    total = 0
    for report in cycle.get("reports_seen", {}).values():
        report = _coerce_report(report)
        if report is None or report.not_load_bearing:
            continue
        total += len(report.issues)
    return total


def _severe_issue_count(cycle: dict) -> int:
    severe = 0
    for report in cycle.get("reports_seen", {}).values():
        report = _coerce_report(report)
        if report is None or report.not_load_bearing:
            continue
        for issue in report.issues:
            if issue.severity == Severity.SEVERE:
                severe += 1
    return severe


def _coerce_report(value) -> CriticReport | None:
    """Accept either a CriticReport instance or a dict (from a model_dump
    or JSON round-trip) and normalize to the typed model. Returns None
    on parse failure."""
    if isinstance(value, CriticReport):
        return value
    if isinstance(value, dict):
        try:
            return CriticReport.model_validate(value)
        except Exception:
            return None
    return None
