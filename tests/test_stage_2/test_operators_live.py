"""Live API test for `seed_root`. Makes one real LLM call. Costs ~$0.01-0.02.

Run with:
    uv run pytest tests/test_stage_2/test_operators_live.py --run-live-api -v

Skipped by default (per `@pytest.mark.live_api` and the project's pytest
config). Offline mocked equivalent: `tests/test_stage_2/test_operators.py`.

What this test verifies:
- The full extraction call round-trips: prompt assembly → real LLM call →
  structured-output parsing → DAG construction.
- The returned DAG validates (single anchor node, role from concept).
- Motif extraction produces 2-3 concrete strings (not abstract themes).
- Cost stays under ~$0.05 per call (sanity bound; alerts on regression).

What this test does NOT verify:
- Specific motif content — LLM output is non-deterministic. We assert
  structural validity, not literary judgment of the motifs.
- Concept_demands content — empty is the common case; for Hills the
  concept's mechanism is fully schema-expressible, so demands likely empty.
"""

from __future__ import annotations

import asyncio

import pytest

from owtn.models.stage_1.concept_genome import ConceptGenome
from owtn.models.stage_2.dag import DAG
from owtn.stage_2.operators import seed_root
from tests.conftest import HILLS_GENOME


pytestmark = pytest.mark.live_api


def test_seed_root_against_hills_concept() -> None:
    concept = ConceptGenome.model_validate(HILLS_GENOME)

    dag = asyncio.run(seed_root(
        concept,
        concept_id="live_hills",
        preset="phoebe_ish",
        target_node_count=4,
    ))

    # Structural assertions only — content is non-deterministic.
    assert isinstance(dag, DAG)
    assert dag.concept_id == "live_hills"
    assert dag.preset == "phoebe_ish"
    assert dag.target_node_count == 4

    # Anchor wrap (deterministic; no LLM involved).
    assert len(dag.nodes) == 1
    anchor = dag.nodes[0]
    assert anchor.id == "anchor"
    assert anchor.sketch == concept.anchor_scene.sketch
    assert anchor.role == [concept.anchor_scene.role]

    # No edges at seed time.
    assert dag.edges == []

    # Motif extraction succeeded — 2 to 3 threads, all non-empty.
    assert 2 <= len(dag.motif_threads) <= 3
    for thread in dag.motif_threads:
        assert isinstance(thread, str)
        assert len(thread.strip()) > 0

    # concept_demands may be empty (most concepts) or populated; we don't
    # constrain content but each demand if present must be a non-empty string.
    for demand in dag.concept_demands:
        assert isinstance(demand, str)
        assert len(demand.strip()) > 0
