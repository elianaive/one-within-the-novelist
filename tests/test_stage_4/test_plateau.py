"""Plateau detector tests — pure-function logic over synthetic cycles."""

from __future__ import annotations

from owtn.models.stage_4 import CriticReport, Issue, Severity
from owtn.stage_4.plateau import PlateauDetector, PlateauVerdict


def _report(critic_id: str, *, severe: int = 0, moderate: int = 0, minor: int = 0, not_load_bearing: bool = False) -> CriticReport:
    issues: list[Issue] = []
    for i in range(severe):
        issues.append(Issue(severity=Severity.SEVERE, observation=f"severe issue #{i} for testing"))
    for i in range(moderate):
        issues.append(Issue(severity=Severity.MODERATE, observation=f"moderate issue #{i} for testing"))
    for i in range(minor):
        issues.append(Issue(severity=Severity.MINOR, observation=f"minor issue #{i} for testing"))
    return CriticReport(
        critic_id=critic_id,
        cycle=0,
        not_load_bearing=not_load_bearing,
        issues=issues,
    )


def _cycle(reports: dict[str, CriticReport], *, completed: bool = True, cycle: int = 0) -> dict:
    return {
        "cycle": cycle,
        "critic_calls": list(reports.keys()),
        "plan": {} if completed else None,
        "completed": completed,
        "reports_seen": dict(reports),
    }


# ─── Helpers / edge cases ───────────────────────────────────────────────


def test_insufficient_history_below_window_plus_one():
    """Window=2 needs 3 completed cycles before either rule can fire."""
    detector = PlateauDetector(window=2)
    verdict = detector.check([_cycle({"continuity": _report("continuity", severe=2)})])
    assert verdict.plateaued is False
    assert verdict.reason == "insufficient-history"


def test_excludes_incomplete_cycles_from_history():
    """An in-flight cycle isn't counted toward window."""
    detector = PlateauDetector(window=2)
    verdict = detector.check([
        _cycle({"continuity": _report("continuity", severe=2)}, cycle=0),
        _cycle({"continuity": _report("continuity", severe=2)}, cycle=1),
        _cycle({"continuity": _report("continuity", severe=2)}, cycle=2, completed=False),
    ])
    # Only 2 completed; insufficient.
    assert verdict.reason == "insufficient-history"


# ─── Total-issue rule ──────────────────────────────────────────────────


def test_total_decrease_satisfied_returns_still_converging():
    """3 cycles where total strictly decreases across both transitions."""
    detector = PlateauDetector(window=2)
    cycles = [
        _cycle({"continuity": _report("continuity", severe=2, moderate=2)}, cycle=0),  # 4
        _cycle({"continuity": _report("continuity", severe=1, moderate=2)}, cycle=1),  # 3
        _cycle({"continuity": _report("continuity", severe=1, moderate=1)}, cycle=2),  # 2
    ]
    verdict = detector.check(cycles)
    assert verdict.plateaued is False
    assert verdict.reason == "still-converging"
    assert verdict.total_per_cycle == [4, 3, 2]


def test_total_count_flat_fires_plateau():
    detector = PlateauDetector(window=2)
    cycles = [
        _cycle({"continuity": _report("continuity", moderate=3)}, cycle=0),  # 3
        _cycle({"continuity": _report("continuity", moderate=3)}, cycle=1),  # 3
        _cycle({"continuity": _report("continuity", moderate=3)}, cycle=2),  # 3
    ]
    verdict = detector.check(cycles)
    assert verdict.plateaued is True
    assert verdict.reason == "total-issue-count-not-decreasing"


def test_total_count_increasing_fires_plateau():
    detector = PlateauDetector(window=2)
    cycles = [
        _cycle({"continuity": _report("continuity", moderate=2)}, cycle=0),  # 2
        _cycle({"continuity": _report("continuity", moderate=3)}, cycle=1),  # 3
        _cycle({"continuity": _report("continuity", moderate=4)}, cycle=2),  # 4
    ]
    verdict = detector.check(cycles)
    assert verdict.plateaued is True
    assert verdict.reason == "total-issue-count-not-decreasing"


def test_one_decrease_in_window_breaks_plateau():
    """If any step in the window decreased, no plateau."""
    detector = PlateauDetector(window=2)
    cycles = [
        _cycle({"continuity": _report("continuity", moderate=4)}, cycle=0),  # 4
        _cycle({"continuity": _report("continuity", moderate=4)}, cycle=1),  # 4 (no decrease)
        _cycle({"continuity": _report("continuity", moderate=3)}, cycle=2),  # 3 (decrease)
    ]
    verdict = detector.check(cycles)
    assert verdict.plateaued is False


# ─── Severe-progress rule ──────────────────────────────────────────────


def test_severe_count_decreasing_satisfies_rule():
    detector = PlateauDetector(window=2)
    cycles = [
        _cycle({"continuity": _report("continuity", severe=3)}, cycle=0),
        _cycle({"continuity": _report("continuity", severe=2)}, cycle=1),
        _cycle({"continuity": _report("continuity", severe=1)}, cycle=2),
    ]
    verdict = detector.check(cycles)
    assert verdict.plateaued is False


def test_severe_count_flat_with_total_changing_still_fires_severe_rule():
    """Severe count flat across 3 cycles; minors moving doesn't help."""
    detector = PlateauDetector(window=2)
    cycles = [
        _cycle({"continuity": _report("continuity", severe=2, minor=5)}, cycle=0),
        _cycle({"continuity": _report("continuity", severe=2, minor=3)}, cycle=1),
        _cycle({"continuity": _report("continuity", severe=2, minor=1)}, cycle=2),
    ]
    verdict = detector.check(cycles)
    # Total is decreasing (7→5→3), so total rule passes.
    # Severe is flat (2→2→2), so severe rule fires.
    assert verdict.plateaued is True
    assert verdict.reason == "no-severe-resolved"


def test_disabled_total_rule_only_severe_fires():
    detector = PlateauDetector(window=2, require_total_decrease=False)
    cycles = [
        _cycle({"continuity": _report("continuity", severe=2, moderate=2)}, cycle=0),  # 4 / 2
        _cycle({"continuity": _report("continuity", severe=2, moderate=2)}, cycle=1),  # 4 / 2
        _cycle({"continuity": _report("continuity", severe=2, moderate=2)}, cycle=2),  # 4 / 2
    ]
    verdict = detector.check(cycles)
    assert verdict.plateaued is True
    assert verdict.reason == "no-severe-resolved"


def test_both_rules_disabled_never_plateaus():
    detector = PlateauDetector(
        window=2, require_total_decrease=False, require_severe_progress=False,
    )
    cycles = [
        _cycle({"continuity": _report("continuity", severe=2, moderate=2)}, cycle=0),
        _cycle({"continuity": _report("continuity", severe=2, moderate=2)}, cycle=1),
        _cycle({"continuity": _report("continuity", severe=2, moderate=2)}, cycle=2),
    ]
    verdict = detector.check(cycles)
    assert verdict.plateaued is False
    assert verdict.reason == "still-converging"


# ─── not_load_bearing handling ─────────────────────────────────────────


def test_not_load_bearing_reports_count_zero_issues():
    """A critic that decided their dimension isn't load-bearing for this
    story contributes 0 issues regardless of any dummy issues set."""
    detector = PlateauDetector(window=2)
    nlb = _report("tension_curiosity", severe=2, not_load_bearing=True)
    cycles = [
        _cycle({"tension_curiosity": nlb, "continuity": _report("continuity", moderate=3)}, cycle=0),
        _cycle({"tension_curiosity": nlb, "continuity": _report("continuity", moderate=2)}, cycle=1),
        _cycle({"tension_curiosity": nlb, "continuity": _report("continuity", moderate=1)}, cycle=2),
    ]
    verdict = detector.check(cycles)
    assert verdict.plateaued is False
    assert verdict.total_per_cycle == [3, 2, 1]


# ─── Multi-critic aggregation ──────────────────────────────────────────


def test_total_aggregates_across_critics():
    detector = PlateauDetector(window=2)
    cycles = [
        _cycle({
            "continuity": _report("continuity", severe=1),
            "payload_enactment": _report("payload_enactment", moderate=2),
            "transportation": _report("transportation", minor=3),
        }, cycle=0),  # total 6
        _cycle({
            "continuity": _report("continuity", severe=1),
            "payload_enactment": _report("payload_enactment", moderate=1),
            "transportation": _report("transportation", minor=2),
        }, cycle=1),  # total 4
        _cycle({
            "continuity": _report("continuity", severe=1),
            "payload_enactment": _report("payload_enactment", moderate=1),
            "transportation": _report("transportation", minor=1),
        }, cycle=2),  # total 3
    ]
    verdict = detector.check(cycles)
    assert verdict.total_per_cycle == [6, 4, 3]


# ─── Dict-shaped reports (after JSON round-trip) ───────────────────────


def test_detector_handles_dict_shaped_reports():
    """Reports stored as dicts (e.g., from a YAML log round-trip) are
    coerced to typed CriticReport before counting."""
    detector = PlateauDetector(window=2)
    raw = _report("continuity", severe=2, moderate=2).model_dump()
    cycles = [
        _cycle({"continuity": raw}, cycle=0),
        _cycle({"continuity": raw}, cycle=1),
        _cycle({"continuity": raw}, cycle=2),
    ]
    verdict = detector.check(cycles)
    assert verdict.plateaued is True
    assert verdict.total_per_cycle == [4, 4, 4]


def test_window_one_fires_on_two_cycles():
    """Window=1 means a single non-decreasing transition fires plateau."""
    detector = PlateauDetector(window=1)
    cycles = [
        _cycle({"continuity": _report("continuity", moderate=3)}, cycle=0),
        _cycle({"continuity": _report("continuity", moderate=3)}, cycle=1),
    ]
    verdict = detector.check(cycles)
    assert verdict.plateaued is True
