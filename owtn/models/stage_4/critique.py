"""Stage 4 critic-report and revision-plan models.

Critics return structured *observations* — never rewrites. Each report
carries a list of issues with a severity tag (`severe`/`moderate`/`minor`)
that the orchestrator's plateau detector consumes; counts and tag
distribution carry the convergence signal without the Goodhart risk that
scalar scores invite.

A critic that judges its dimension isn't load-bearing for this story sets
`not_load_bearing=True` and writes its `pursuit_observation` describing
what the prose IS pursuing instead. This is an explicitly valid output
shape — irrelevant critiques become confirming observations rather than
corrections to non-problems.

The agent integrates reports across a cycle's gather sub-phase and
commits to a `CritiquePlan` via `finalize_critique_plan`. The plan is
the structured record the plateau detector and per-cycle log read.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Severity(StrEnum):
    """Tag a critic attaches to each flagged issue.

    The plateau detector watches `severe` resolution across cycles and
    total issue count regardless of severity — both as proxies for
    convergence without exposing scalar scores to the agent.
    """
    SEVERE = "severe"
    MODERATE = "moderate"
    MINOR = "minor"


class Issue(BaseModel):
    """One observation a critic flags.

    `quote` is a verbatim slice from the prose so the agent (and surgical-edit
    subagents) can locate the passage with string operations on the
    manuscript. `scene_id` is the scene the issue lives in when known;
    cross-scene observations leave it None.
    """
    severity: Severity
    observation: str = Field(min_length=10)
    scene_id: str | None = None
    quote: str | None = None
    suggestion: str | None = None


class CriticReportBody(BaseModel):
    """The fields a critic produces. Identifiers (`critic_id`, `cycle`)
    are attached by the dispatcher post-parse — the critic shouldn't
    invent them, and force-correcting after avoids the model's
    occasional hallucination of either field."""
    focus: str | None = None
    not_load_bearing: bool = False
    pursuit_observation: str | None = None
    issues: list[Issue] = Field(default_factory=list)


class CriticReport(CriticReportBody):
    """Structured output from one `call_critic` invocation.

    Critics produce reports; they never rewrite. Per
    `feedback_judge_query_scope.md`: judges/critics evaluate; the agent
    edits.

    `not_load_bearing` is the explicit "this dimension isn't load-bearing
    for this story; the prose is pursuing X" output shape — a valid
    confirming observation rather than a correction to a non-problem.
    When set, `issues` may be empty and `pursuit_observation` carries the
    critic's read of what the prose IS pursuing.
    """
    critic_id: str = Field(min_length=2)
    cycle: int = Field(ge=0)


class CritiquePlan(BaseModel):
    """Agent's structured commitment at the end of sub-phase A (gather).

    `intended_revisions` is the agent's planned response, packaged for
    logging and for the plateau detector's view of what the cycle is
    targeting. `unresolved_observations` lists issues the agent
    intentionally judges not-actionable this cycle (deferred or the agent
    stands behind the prose against the critic's flag); the plateau
    detector subtracts these from the unresolved count rather than
    treating them as failed-to-fix.
    """
    cycle: int = Field(ge=0)
    plan_summary: str = Field(min_length=20)
    intended_revisions: list[str] = Field(default_factory=list)
    unresolved_observations: list[str] = Field(default_factory=list)
