"""Per-validator unit tests on the Stage 2 DAG model.

Each test mutates a known-valid canonical (via `model_copy(deep=True, update=...)`
or by re-parsing JSON with a swapped field) and asserts the appropriate
validator fires with a recognizable error. Mutating a canonical rather than
hand-rolling a minimal DAG keeps tests anchored on real cases — when a future
schema change breaks a validator, the test surface mirrors the canonicals.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from owtn.models.stage_2.dag import DAG, Edge, Node


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_dict(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


class TestEdgePayloadValidation:
    def test_causal_edge_requires_substantive_realizes(self) -> None:
        with pytest.raises(ValidationError, match="realizes"):
            Edge(src="a", dst="b", type="causal", realizes="too short")

    def test_causal_edge_requires_realizes_at_all(self) -> None:
        with pytest.raises(ValidationError, match="realizes"):
            Edge(src="a", dst="b", type="causal")

    def test_disclosure_requires_reframes_and_withheld(self) -> None:
        with pytest.raises(ValidationError, match="reframes"):
            Edge(src="a", dst="b", type="disclosure",
                 withheld="something hidden between a and b")
        with pytest.raises(ValidationError, match="withheld"):
            Edge(src="a", dst="b", type="disclosure",
                 reframes="reframes the earlier beat in a real way")

    def test_disclosure_disclosed_to_defaults_to_reader(self) -> None:
        e = Edge(
            src="a", dst="b", type="disclosure",
            reframes="reframes earlier beat in a substantive way",
            withheld="withheld content for the reader to decode",
        )
        assert e.disclosed_to == ["reader"]

    def test_disclosure_empty_disclosed_to_rejected(self) -> None:
        with pytest.raises(ValidationError, match="disclosed_to"):
            Edge(
                src="a", dst="b", type="disclosure",
                reframes="reframes earlier beat in a substantive way",
                withheld="withheld content for the reader to decode",
                disclosed_to=[],
            )

    def test_motivates_requires_agent_and_goal(self) -> None:
        with pytest.raises(ValidationError, match="agent"):
            Edge(src="a", dst="b", type="motivates",
                 goal="to find the lost child before nightfall")
        with pytest.raises(ValidationError, match="goal"):
            Edge(src="a", dst="b", type="motivates", agent="the grandmother")

    def test_self_loop_rejected(self) -> None:
        with pytest.raises(ValidationError, match="self-loop"):
            Edge(src="a", dst="a", type="causal",
                 realizes="this would loop back on itself")


class TestNodeRoleValidation:
    def test_empty_role_list_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-empty list"):
            Node(id="x", sketch="A beat with enough words to count as substantive.",
                 role=[])

    def test_duplicate_roles_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicates"):
            Node(id="x", sketch="A beat with enough words to count as substantive.",
                 role=["climax", "climax"])

    def test_multi_role_allowed(self) -> None:
        n = Node(id="x", sketch="A beat with enough words to count as substantive.",
                 role=["climax", "pivot"])
        assert n.role == ["climax", "pivot"]


class TestDAGStructuralValidation:
    """These tests mutate the Lottery canonical to break specific invariants."""

    @pytest.fixture
    def lottery_dict(self) -> dict:
        return _load_dict("canonical_lottery.json")

    def test_baseline_canonical_validates(self, lottery_dict: dict) -> None:
        DAG.model_validate(lottery_dict)  # smoke

    def test_node_count_zero_rejected(self, lottery_dict: dict) -> None:
        lottery_dict["nodes"] = []
        lottery_dict["edges"] = []
        lottery_dict["character_arcs"] = []
        lottery_dict["story_constraints"] = []
        with pytest.raises(ValidationError, match="not in"):
            DAG.model_validate(lottery_dict)

    def test_node_count_too_high_rejected(self, lottery_dict: dict) -> None:
        # Replicate the gathering node 19 times to exceed the 18-node ceiling.
        proto = lottery_dict["nodes"][0]
        lottery_dict["nodes"] = [
            {**proto, "id": f"n_{i}", "role": (proto["role"] if i == 0 else None)}
            for i in range(19)
        ]
        lottery_dict["edges"] = []
        lottery_dict["story_constraints"] = []
        lottery_dict["character_arcs"] = []
        with pytest.raises(ValidationError, match="not in"):
            DAG.model_validate(lottery_dict)

    def test_single_node_seed_validates(self, lottery_dict: dict) -> None:
        """A fresh seed_root output: 1 node (the anchor), 0 edges. Must be
        legal so seed_root can return a fully-typed DAG instead of a
        weakly-typed seed object."""
        anchor = {**lottery_dict["nodes"][-1]}  # the role-bearing node
        anchor["id"] = "anchor"
        anchor["motifs"] = []
        lottery_dict["nodes"] = [anchor]
        lottery_dict["edges"] = []
        lottery_dict["character_arcs"] = []
        lottery_dict["story_constraints"] = []
        DAG.model_validate(lottery_dict)  # no exception

    def test_duplicate_node_ids_rejected(self, lottery_dict: dict) -> None:
        lottery_dict["nodes"][1]["id"] = lottery_dict["nodes"][0]["id"]
        with pytest.raises(ValidationError, match="duplicate node id"):
            DAG.model_validate(lottery_dict)

    def test_edge_to_unknown_node_rejected(self, lottery_dict: dict) -> None:
        lottery_dict["edges"][0]["dst"] = "nonexistent_node"
        with pytest.raises(ValidationError, match="unknown node"):
            DAG.model_validate(lottery_dict)

    def test_cycle_rejected(self, lottery_dict: dict) -> None:
        # Add a back-edge stoning → gathering, creating a cycle.
        lottery_dict["edges"].append({
            "src": "stoning",
            "dst": "gathering",
            "type": "causal",
            "realizes": "an artificial back-edge introducing a topological cycle",
        })
        with pytest.raises(ValidationError, match="cycle"):
            DAG.model_validate(lottery_dict)

    def test_zero_role_bearers_rejected(self, lottery_dict: dict) -> None:
        for n in lottery_dict["nodes"]:
            n["role"] = None
        with pytest.raises(ValidationError, match="exactly one node"):
            DAG.model_validate(lottery_dict)

    def test_two_role_bearers_rejected(self, lottery_dict: dict) -> None:
        # Add a second role to a non-anchor node.
        lottery_dict["nodes"][0]["role"] = ["reveal"]
        with pytest.raises(ValidationError, match="exactly one node"):
            DAG.model_validate(lottery_dict)


class TestMotifValidation:
    @pytest.fixture
    def lottery_dict(self) -> dict:
        return _load_dict("canonical_lottery.json")

    def test_motif_must_be_in_threads(self, lottery_dict: dict) -> None:
        lottery_dict["nodes"][0]["motifs"].append(
            {"motif": "not in motif_threads", "mode": "introduced"}
        )
        with pytest.raises(ValidationError, match="not in motif_threads"):
            DAG.model_validate(lottery_dict)

    def test_echoed_without_prior_appearance_rejected(self, lottery_dict: dict) -> None:
        # Replace gathering's "introduced" with "echoed" — there's no prior introducer.
        lottery_dict["nodes"][0]["motifs"] = [
            {"motif": "stones", "mode": "echoed"}
        ]
        # Also clear the stoning motif so no other "stones" remains as backup.
        lottery_dict["nodes"][-1]["motifs"] = [
            {"motif": "Old Man Warner's mutter", "mode": "echoed"}
        ]
        with pytest.raises(ValidationError, match="no earlier non-return-mode"):
            DAG.model_validate(lottery_dict)

    def test_inverted_without_prior_appearance_rejected(self, lottery_dict: dict) -> None:
        lottery_dict["nodes"][0]["motifs"] = [
            {"motif": "stones", "mode": "inverted"}
        ]
        lottery_dict["nodes"][-1]["motifs"] = [
            {"motif": "Old Man Warner's mutter", "mode": "echoed"}
        ]
        with pytest.raises(ValidationError, match="no earlier non-return-mode"):
            DAG.model_validate(lottery_dict)


class TestStoryConstraintValidation:
    @pytest.fixture
    def lottery_dict(self) -> dict:
        return _load_dict("canonical_lottery.json")

    def test_lifts_at_must_be_real_node(self, lottery_dict: dict) -> None:
        lottery_dict["story_constraints"][0]["lifts_at"] = "nonexistent"
        with pytest.raises(ValidationError, match="not a real node id"):
            DAG.model_validate(lottery_dict)

    def test_lifts_at_none_is_valid(self, lottery_dict: dict) -> None:
        lottery_dict["story_constraints"][0]["lifts_at"] = None
        DAG.model_validate(lottery_dict)


class TestDisclosureAudienceValidation:
    @pytest.fixture
    def oconnor_dict(self) -> dict:
        return _load_dict("canonical_oconnor.json")

    def test_audience_must_be_present_in_target_sketch(self, oconnor_dict: dict) -> None:
        # Find the multi-audience disclosure and inject a character not in the sketch.
        for e in oconnor_dict["edges"]:
            if e.get("type") == "disclosure" and "the grandmother" in (e.get("disclosed_to") or []):
                e["disclosed_to"] = ["reader", "Bobby Lee"]  # Bobby Lee not in arrival sketch
                break
        with pytest.raises(ValidationError, match="not present in target beat sketch"):
            DAG.model_validate(oconnor_dict)


class TestMotivatesLocality:
    @pytest.fixture
    def oconnor_dict(self) -> dict:
        return _load_dict("canonical_oconnor.json")

    def test_motivates_spanning_too_far_rejected(self, oconnor_dict: dict) -> None:
        # opening (topo 0) to grace (topo ~8) — definitely > 2 positions apart.
        oconnor_dict["edges"].append({
            "src": "opening",
            "dst": "grace",
            "type": "motivates",
            "agent": "the grandmother",
            "goal": "to be seen as a lady across the whole arc",
        })
        with pytest.raises(ValidationError, match="topological positions"):
            DAG.model_validate(oconnor_dict)


class TestCharacterArcPresence:
    @pytest.fixture
    def lottery_dict(self) -> dict:
        return _load_dict("canonical_lottery.json")

    def test_phantom_arc_agent_rejected(self, lottery_dict: dict) -> None:
        lottery_dict["character_arcs"].append({
            "agent": "Cthulhu",
            "goal": "to consume the village in unfathomable ways",
            "stakes": "the cosmic order",
        })
        with pytest.raises(ValidationError, match="does not appear"):
            DAG.model_validate(lottery_dict)
