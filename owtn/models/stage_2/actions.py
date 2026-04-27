"""MCTS expansion action models.

Three actions extend a partial DAG during MCTS expansion:

- `AddBeatAction`   — insert a new beat attached via a typed edge
- `AddEdgeAction`   — add a typed edge between two existing beats
- `RewriteBeatAction` — revise an existing beat's sketch

The expansion LLM call returns up to K=4 candidate actions per leaf via the
`ExpansionProposals` wrapper. Application logic (validate + apply to a DAG)
lives in `owtn.stage_2.operators`. This module only defines the data shape.

Discriminated-union pattern: each action carries an `action_type` literal so
Pydantic v2's `Annotated[Union[...], Field(discriminator="action_type")]`
parses incoming structured outputs correctly.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from owtn.models.stage_2.dag import EdgeType, RoleName


class AddBeatAction(BaseModel):
    """Insert a new beat attached to an existing node via a typed edge.

    `direction` controls phase semantics:
      - "downstream" (forward phase): the new beat is the *target* of the
        new edge; `anchor_id` is the source. New beat sits temporally after.
      - "upstream" (backward phase): the new beat is the *source* of the
        new edge; `anchor_id` is the target. New beat sits temporally before.
    """
    action_type: Literal["add_beat"] = "add_beat"
    anchor_id: str = Field(min_length=1)
    direction: Literal["downstream", "upstream"]
    new_node_id: str = Field(min_length=1)
    sketch: str = Field(min_length=20)
    edge_type: EdgeType
    edge_payload: dict[str, str | list[str]] = Field(default_factory=dict)
    new_node_role: list[RoleName] | None = None
    reasoning: str = Field(default="")


class AddEdgeAction(BaseModel):
    """Add a typed edge between two existing nodes."""
    action_type: Literal["add_edge"] = "add_edge"
    src_id: str = Field(min_length=1)
    dst_id: str = Field(min_length=1)
    edge_type: EdgeType
    edge_payload: dict[str, str | list[str]] = Field(default_factory=dict)
    reasoning: str = Field(default="")


class RewriteBeatAction(BaseModel):
    """Revise an existing beat's sketch text. Used when downstream edge
    additions need a subtly different setup at an earlier beat."""
    action_type: Literal["rewrite_beat"] = "rewrite_beat"
    node_id: str = Field(min_length=1)
    new_sketch: str = Field(min_length=20)
    reasoning: str = Field(default="")


# Discriminated union — the LLM returns one of these per cached candidate.
Action = Annotated[
    Union[AddBeatAction, AddEdgeAction, RewriteBeatAction],
    Field(discriminator="action_type"),
]


class ExpansionProposals(BaseModel):
    """Structured output from one MCTS expansion LLM call.

    `actions` is the ranked candidate list Π(v) cached on the leaf. K is
    typically 4 (BiT-MCTS reported optimum); we accept 1..4 to gracefully
    handle LLMs that emit fewer than K when constrained options are scarce.
    Empty lists trigger the operator's retry-with-higher-temperature path.
    """
    actions: list[Action] = Field(min_length=0, max_length=4)
    reasoning: str = Field(default="")
