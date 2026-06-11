"""CriticReport / CritiquePlan / Severity / Issue model tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from owtn.models.stage_4 import CriticReport, CritiquePlan, Issue, Severity


def test_severity_string_values():
    """StrEnum so JSON round-trips as plain strings."""
    assert Severity.SEVERE == "severe"
    assert Severity.MODERATE == "moderate"
    assert Severity.MINOR == "minor"


def test_issue_minimal_fields():
    issue = Issue(severity=Severity.MODERATE, observation="The third paragraph drifts to default register.")
    assert issue.severity == Severity.MODERATE
    assert issue.scene_id is None
    assert issue.quote is None


def test_issue_with_quote_and_scene():
    issue = Issue(
        severity=Severity.SEVERE,
        observation="Voice collapses to default Anthropic register here.",
        scene_id="weights-freeze",
        quote="The commissioner felt a deep sense of unease as she initiated",
        suggestion="Strip the affective gloss; render only the procedural surface.",
    )
    assert issue.scene_id == "weights-freeze"
    assert issue.quote.startswith("The commissioner")


def test_issue_rejects_short_observation():
    with pytest.raises(ValidationError):
        Issue(severity=Severity.MINOR, observation="too short")


def test_critic_report_with_issues():
    report = CriticReport(
        critic_id="voice_fidelity",
        cycle=0,
        issues=[
            Issue(severity=Severity.SEVERE, observation="Voice drift in scene 3."),
            Issue(severity=Severity.MINOR, observation="Two small Saunders-beat opportunities missed."),
        ],
    )
    assert report.critic_id == "voice_fidelity"
    assert len(report.issues) == 2
    assert report.not_load_bearing is False


def test_critic_report_not_load_bearing():
    """Valid output shape — confirming observation rather than correction."""
    report = CriticReport(
        critic_id="tension_curiosity",
        cycle=1,
        not_load_bearing=True,
        pursuit_observation=(
            "This piece works on permanent gap and refusal of naming, not "
            "Brewer & Lichtenstein suspense. The withholding IS the form."
        ),
        issues=[],
    )
    assert report.not_load_bearing is True
    assert report.pursuit_observation is not None
    assert report.issues == []


def test_critic_report_round_trips_json():
    report = CriticReport(
        critic_id="payload_enactment",
        cycle=2,
        focus="legislative-testimony-2028",
        issues=[Issue(severity=Severity.MODERATE, observation="Reveal is stated, not dramatized.")],
    )
    js = report.model_dump_json()
    restored = CriticReport.model_validate_json(js)
    assert restored == report


def test_critique_plan_minimal():
    plan = CritiquePlan(
        cycle=0,
        plan_summary="Tighten voice in scenes 2 and 3; sharpen the disclosure beat.",
    )
    assert plan.intended_revisions == []
    assert plan.unresolved_observations == []


def test_critique_plan_rejects_short_summary():
    with pytest.raises(ValidationError):
        CritiquePlan(cycle=0, plan_summary="short")
