"""Offline tests for `seed_root`. LLM call is mocked.

Live equivalent: `tests/test_stage_2/test_operators_live.py` (gated by
`@pytest.mark.live_api`). The live test exercises one real extraction call
and verifies the result; this file exercises the fallback paths and the
deterministic anchor wrap.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG
from owtn.prompts.stage_2.registry import build_seed_motif_prompt, load_base_system
from owtn.stage_2.operators import (
    SeedExtractionResult,
    seed_root,
)
from tests.conftest import HILLS_GENOME


@dataclass
class _FakeQueryResult:
    """Minimal stand-in for QueryResult.

    When `output_model` is passed to `query_async`, the provider runs the
    response through instructor and `result.content` is the *parsed Pydantic
    instance* — NOT a JSON string. Mocks here mirror that: pass a
    `SeedExtractionResult` instance as `content`, not a JSON string.
    """
    content: object  # SeedExtractionResult when output_model is set
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost: float = 0.0
    thought: str = ""
    model_name: str = "claude-sonnet-4-6"


@pytest.fixture
def hills_concept() -> ConceptGenome:
    return ConceptGenome.model_validate(HILLS_GENOME)


# ----- Prompt assembly (no LLM) -----

class TestPromptAssembly:
    def test_base_system_is_non_empty(self) -> None:
        base = load_base_system()
        assert "typed-edge" in base.lower()
        assert "directed acyclic graph" in base.lower()
        # Names all 5 edge types
        for et in ("CAUSAL", "DISCLOSURE", "IMPLICATION", "CONSTRAINT", "MOTIVATES"):
            assert et in base

    def test_seed_motif_prompt_substitutes_concept_fields(
        self, hills_concept: ConceptGenome,
    ) -> None:
        system_msg, user_msg = build_seed_motif_prompt(hills_concept)
        assert "{CONCEPT_JSON}" not in user_msg
        assert "{ANCHOR_SKETCH}" not in user_msg
        assert "{ANCHOR_ROLE}" not in user_msg
        # The concept's anchor sketch must appear verbatim in the user message.
        assert hills_concept.anchor_scene.sketch in user_msg
        # Anchor role appears (Hills' anchor is a "reveal")
        assert hills_concept.anchor_scene.role in user_msg

    def test_seed_motif_user_msg_contains_concept_json(
        self, hills_concept: ConceptGenome,
    ) -> None:
        _, user_msg = build_seed_motif_prompt(hills_concept)
        # The premise should be embedded in the JSON-formatted concept block
        assert hills_concept.premise[:30] in user_msg


# ----- seed_root behavior with mocked LLM -----

class TestSeedRootSuccess:
    """Happy path: extraction returns valid structured output."""

    def test_returns_one_node_dag_with_extracted_fields(
        self, hills_concept: ConceptGenome,
    ) -> None:
        fake_extraction = SeedExtractionResult(
            motif_threads=["the hills", "the beaded curtain", "the railway"],
            concept_demands=[],
        )

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=fake_extraction)

        with patch("owtn.stage_2.operators.query_async", side_effect=fake_query):
            dag = asyncio.run(seed_root(
                hills_concept,
                concept_id="test_hills",
                preset="phoebe_ish",
                target_node_count=4,
            ))

        assert isinstance(dag, DAG)
        assert dag.concept_id == "test_hills"
        assert dag.preset == "phoebe_ish"
        assert dag.target_node_count == 4
        assert len(dag.nodes) == 1
        assert dag.nodes[0].id == "anchor"
        assert dag.nodes[0].sketch == hills_concept.anchor_scene.sketch
        assert dag.nodes[0].role == [hills_concept.anchor_scene.role]
        assert dag.nodes[0].motifs == []  # per-node motifs deferred to MCTS expansion
        assert len(dag.edges) == 0
        assert dag.motif_threads == ["the hills", "the beaded curtain", "the railway"]
        assert dag.concept_demands == []

    def test_demand_extraction_populates_concept_demands(
        self, hills_concept: ConceptGenome,
    ) -> None:
        fake_extraction = SeedExtractionResult(
            motif_threads=["the hills", "the operation never named"],
            concept_demands=[
                "the DAG must include a beat where the not-naming becomes "
                "structurally untenable rather than continuing as authorial "
                "discipline",
            ],
        )

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=fake_extraction)

        with patch("owtn.stage_2.operators.query_async", side_effect=fake_query):
            dag = asyncio.run(seed_root(
                hills_concept,
                concept_id="test_hills",
                preset="phoebe_ish",
                target_node_count=4,
            ))

        assert len(dag.concept_demands) == 1
        assert "not-naming" in dag.concept_demands[0]

    def test_anchor_role_wrapped_as_list(self, hills_concept: ConceptGenome) -> None:
        """Stage 1 emits a single-string role; seed_root wraps it as a 1-element
        list so Phase 5 expansion can append secondary roles via rewrite_beat."""
        fake = SeedExtractionResult(motif_threads=["a", "b"], concept_demands=[])

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=fake)

        with patch("owtn.stage_2.operators.query_async", side_effect=fake_query):
            dag = asyncio.run(seed_root(
                hills_concept,
                concept_id="x",
                preset="cassandra_ish",
                target_node_count=5,
            ))
        assert isinstance(dag.nodes[0].role, list)
        assert len(dag.nodes[0].role) == 1


class TestSeedRootFailureFallbacks:
    """Per docs/stage-2/operators.md §seed_root §Failure handling: motif and
    demand extraction degrades to empty lists on failure rather than blocking
    the concept. The DAG still produces.

    With `output_model` set, instructor raises *inside* `query_async` on JSON
    parse failures or schema mismatches; both manifest as exceptions, which
    the operator's broad except clause catches. There's no separate
    post-call parsing step to test (cf. an earlier draft that did defensive
    re-parsing — removed once it was clear the provider returns a parsed
    instance directly).
    """

    def test_llm_exception_yields_empty_extracted_fields(
        self, hills_concept: ConceptGenome,
    ) -> None:
        async def failing_query(**kwargs):
            raise RuntimeError("simulated provider outage")

        with patch("owtn.stage_2.operators.query_async", side_effect=failing_query):
            dag = asyncio.run(seed_root(
                hills_concept,
                concept_id="x",
                preset="cassandra_ish",
                target_node_count=5,
            ))

        assert dag.motif_threads == []
        assert dag.concept_demands == []
        # Anchor wrap succeeded — it doesn't depend on the LLM call.
        assert len(dag.nodes) == 1
        assert dag.nodes[0].id == "anchor"

    def test_instructor_parse_failure_yields_empty_fields(
        self, hills_concept: ConceptGenome,
    ) -> None:
        """Instructor's parse-failure path manifests as an exception raised
        from query_async — same code path as a provider outage."""
        from pydantic import ValidationError

        async def schema_failing_query(**kwargs):
            # Simulate instructor surfacing a schema mismatch as a ValidationError
            # propagated out of the query call.
            raise ValidationError.from_exception_data("SeedExtractionResult", [])

        with patch("owtn.stage_2.operators.query_async", side_effect=schema_failing_query):
            dag = asyncio.run(seed_root(
                hills_concept,
                concept_id="x",
                preset="cassandra_ish",
                target_node_count=5,
            ))

        assert dag.motif_threads == []
        assert dag.concept_demands == []

    def test_unexpected_content_type_yields_empty_fields(
        self, hills_concept: ConceptGenome,
    ) -> None:
        """Defensive path: if the provider somehow returns content that
        isn't a SeedExtractionResult instance, fall back to empty rather
        than crash. This shouldn't happen under normal operation but
        guards against provider-wrapper drift."""
        async def fake_query(**kwargs):
            return _FakeQueryResult(content="raw string instead of model")

        with patch("owtn.stage_2.operators.query_async", side_effect=fake_query):
            dag = asyncio.run(seed_root(
                hills_concept,
                concept_id="x",
                preset="cassandra_ish",
                target_node_count=5,
            ))

        assert dag.motif_threads == []
        assert dag.concept_demands == []


class TestSeedExtractionResultValidation:
    def test_min_two_motif_threads_required(self) -> None:
        with pytest.raises(Exception):  # Pydantic ValidationError
            SeedExtractionResult(motif_threads=["only one"], concept_demands=[])

    def test_max_three_motif_threads(self) -> None:
        with pytest.raises(Exception):
            SeedExtractionResult(
                motif_threads=["a", "b", "c", "d"],
                concept_demands=[],
            )

    def test_concept_demands_default_empty(self) -> None:
        result = SeedExtractionResult(motif_threads=["a", "b"])
        assert result.concept_demands == []

    def test_target_bucket_defaults_to_none(self) -> None:
        result = SeedExtractionResult(motif_threads=["a", "b"])
        assert result.target_bucket is None
        assert result.bucket_reasoning is None


# ----- target_bucket → target_node_count resolution -----


_LIGHT_TARGETS = {1000: (3, 5), 3000: (5, 8), 5000: (7, 12), 10000: (10, 18)}


class TestSeedRootBucketResolution:
    """When the LLM picks a target_bucket and seed_root has node_count_targets
    available, the resulting DAG's target_node_count should be the midpoint
    of the corresponding range. Falls back to the caller-passed target when
    the LLM omits the bucket or returns garbage."""

    @pytest.mark.parametrize("bucket,expected_count", [
        ("flash", 4),            # midpoint([3, 5])
        ("short_short", 6),      # midpoint([5, 8])
        ("standard_short", 9),   # midpoint([7, 12])
        ("long_short", 14),      # midpoint([10, 18])
    ])
    def test_bucket_maps_to_midpoint(
        self, hills_concept: ConceptGenome, bucket: str, expected_count: int,
    ) -> None:
        fake = SeedExtractionResult(
            motif_threads=["a", "b"], concept_demands=[],
            target_bucket=bucket, bucket_reasoning="test",
        )

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=fake)

        with patch("owtn.stage_2.operators.query_async", side_effect=fake_query):
            dag = asyncio.run(seed_root(
                hills_concept,
                concept_id="x", preset="cassandra_ish",
                target_node_count=99,  # a value the LLM bucket should override
                node_count_targets=_LIGHT_TARGETS,
            ))
        assert dag.target_node_count == expected_count

    def test_unknown_bucket_falls_back_to_caller_target(
        self, hills_concept: ConceptGenome,
    ) -> None:
        fake = SeedExtractionResult(
            motif_threads=["a", "b"], concept_demands=[],
            target_bucket="novella",  # not in the canonical menu
        )

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=fake)

        with patch("owtn.stage_2.operators.query_async", side_effect=fake_query):
            dag = asyncio.run(seed_root(
                hills_concept,
                concept_id="x", preset="cassandra_ish",
                target_node_count=7,
                node_count_targets=_LIGHT_TARGETS,
            ))
        assert dag.target_node_count == 7  # caller's fallback used

    def test_omitted_bucket_falls_back_to_caller_target(
        self, hills_concept: ConceptGenome,
    ) -> None:
        fake = SeedExtractionResult(
            motif_threads=["a", "b"], concept_demands=[],
            target_bucket=None,
        )

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=fake)

        with patch("owtn.stage_2.operators.query_async", side_effect=fake_query):
            dag = asyncio.run(seed_root(
                hills_concept,
                concept_id="x", preset="cassandra_ish",
                target_node_count=7,
                node_count_targets=_LIGHT_TARGETS,
            ))
        assert dag.target_node_count == 7

    def test_no_targets_dict_ignores_bucket(
        self, hills_concept: ConceptGenome,
    ) -> None:
        """Backward compat: callers that don't pass `node_count_targets`
        keep the previous behavior — caller's target_node_count wins."""
        fake = SeedExtractionResult(
            motif_threads=["a", "b"], concept_demands=[],
            target_bucket="long_short",
            bucket_reasoning="big concept",
        )

        async def fake_query(**kwargs):
            return _FakeQueryResult(content=fake)

        with patch("owtn.stage_2.operators.query_async", side_effect=fake_query):
            dag = asyncio.run(seed_root(
                hills_concept,
                concept_id="x", preset="cassandra_ish",
                target_node_count=5,
                # node_count_targets omitted
            ))
        assert dag.target_node_count == 5
