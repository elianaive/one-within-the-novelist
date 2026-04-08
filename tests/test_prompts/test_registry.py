"""Tests for the operator prompt registry."""

import numpy as np
import pytest

from owtn.models.stage_1.seed_bank import OPERATOR_SEED_TYPES, SeedBank
from owtn.prompts.stage_1.registry import (
    OPERATOR_DEFS,
    OperatorDef,
    build_mutation_feedback,
    build_operator_prompt,
    inject_seed,
    load_registry,
)
from shinka.defaults import default_patch_types, default_patch_type_probs
from shinka.edit.async_apply import _FULL_PATCH_TYPES, _DIFF_PATCH_TYPES


SEED_BANK_PATH = "data/seed-bank.yaml"

ALL_OPERATORS = [
    "collision", "noun_list", "thought_experiment", "compost", "crossover",
    "inversion", "discovery", "compression", "constraint_first",
    "anti_premise", "real_world_seed",
]

FULL_OPERATORS = {
    "collision", "noun_list", "thought_experiment", "compost", "crossover",
    "discovery", "compression", "constraint_first", "anti_premise",
    "real_world_seed",
}

DIFF_OPERATORS = {"inversion"}

CROSS_OPERATORS = {"collision", "compost", "crossover"}


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def seed_bank():
    return SeedBank.load(SEED_BANK_PATH)


class TestRegistryStructure:

    def test_all_operators_present(self, registry):
        for op in ALL_OPERATORS:
            assert op in registry, f"Missing operator: {op}"

    def test_no_extra_operators(self, registry):
        assert set(registry.keys()) == set(ALL_OPERATORS)

    def test_operator_def_fields(self, registry):
        for name, op in registry.items():
            assert isinstance(op, OperatorDef)
            assert op.name == name
            assert op.routing in ("full", "diff")
            assert isinstance(op.needs_inspiration, bool)
            assert isinstance(op.seed_types, list)
            assert isinstance(op.sys_format, str)
            assert isinstance(op.operator_instructions, str)

    def test_all_prompt_files_load(self, registry):
        for name, op in registry.items():
            assert len(op.operator_instructions) > 0, f"{name} has empty instructions"


class TestPatchRouting:

    def test_full_operators(self, registry):
        for op_name in FULL_OPERATORS:
            assert registry[op_name].routing == "full", f"{op_name} should be full"

    def test_diff_operators(self, registry):
        for op_name in DIFF_OPERATORS:
            assert registry[op_name].routing == "diff", f"{op_name} should be diff"


class TestCrossOperators:

    def test_cross_operators_flagged(self, registry):
        for op_name in CROSS_OPERATORS:
            assert registry[op_name].needs_inspiration, f"{op_name} should need inspiration"

    def test_non_cross_operators(self, registry):
        for name, op in registry.items():
            if name not in CROSS_OPERATORS:
                assert not op.needs_inspiration, f"{name} should not need inspiration"


class TestSeedTypes:

    def test_seed_types_match_operator_seed_map(self, registry):
        """Registry seed_types should match OPERATOR_SEED_TYPES from seed_bank.py."""
        for name, op in registry.items():
            expected = sorted(OPERATOR_SEED_TYPES.get(name, []))
            actual = sorted(op.seed_types)
            assert actual == expected, f"{name}: {actual} != {expected}"

    def test_operators_with_seeds(self, registry):
        has_seeds = {n for n, op in registry.items() if op.seed_types}
        assert "real_world_seed" in has_seeds
        assert "collision" in has_seeds
        assert "thought_experiment" in has_seeds

    def test_operators_without_seeds(self, registry):
        no_seeds = {n for n, op in registry.items() if not op.seed_types}
        assert "compost" in no_seeds
        assert "crossover" in no_seeds
        assert "inversion" in no_seeds


class TestInjectSeed:

    def test_returns_content_for_matching_type(self, seed_bank):
        result = inject_seed("real_world_seed", seed_bank)
        assert "Use this as your starting point" in result
        assert len(result) > 50

    def test_returns_empty_for_no_match(self, seed_bank):
        result = inject_seed("inversion", seed_bank)
        assert result == ""

    def test_returns_empty_for_unknown_operator(self, seed_bank):
        result = inject_seed("nonexistent_op", seed_bank)
        assert result == ""

    def test_exclusion_works(self, seed_bank):
        all_ids = {s.id for s in seed_bank.get_by_type("real_world")}
        result = inject_seed("real_world_seed", seed_bank, exclude_ids=all_ids)
        assert result == ""


class TestBuildOperatorPrompt:

    def test_initial_prompt_structure(self, registry):
        sys_msg, user_msg = build_operator_prompt(
            "thought_experiment",
            registry=registry,
            is_initial=True,
        )
        assert "story concepts" in sys_msg
        assert "never been spoken" in user_msg.lower()
        assert len(sys_msg) > 100
        assert len(user_msg) > 100

    def test_iteration_prompt_structure(self, registry):
        genome = '{"premise": "test", "target_effect": "test effect"}'
        sys_msg, user_msg = build_operator_prompt(
            "collision",
            registry=registry,
            parent_genome=genome,
            metrics="score: 3.2",
            feedback="Good originality, weak coherence.",
        )
        assert "story concepts" in sys_msg
        assert "The parent" in user_msg
        assert genome in user_msg
        assert "score: 3.2" in user_msg
        assert "Judge Feedback" in user_msg
        assert "Good originality" in user_msg

    def test_no_feedback_section_when_empty(self, registry):
        sys_msg, user_msg = build_operator_prompt(
            "noun_list",
            registry=registry,
            parent_genome="{}",
            metrics="",
        )
        assert "Judge Feedback" not in user_msg

    def test_seed_injection_in_prompt(self, registry, seed_bank):
        sys_msg, user_msg = build_operator_prompt(
            "real_world_seed",
            registry=registry,
            seed_bank=seed_bank,
            is_initial=True,
        )
        assert "starting point" in user_msg

    def test_steering_in_system(self, registry):
        sys_msg, user_msg = build_operator_prompt(
            "discovery",
            registry=registry,
            steering="Focus on maritime themes",
            is_initial=True,
        )
        assert "maritime themes" in sys_msg

    def test_no_steering_when_empty(self, registry):
        sys_msg, user_msg = build_operator_prompt(
            "discovery",
            registry=registry,
            steering="",
            is_initial=True,
        )
        assert "Creative direction" not in sys_msg

    def test_output_format_resolved(self, registry):
        """Operator instructions should have {output_format} resolved."""
        for name, op in registry.items():
            assert "{output_format}" not in op.operator_instructions, (
                f"{name} still has unresolved {{output_format}}"
            )

    def test_diff_operator_uses_diff_format(self, registry):
        sys_msg, user_msg = build_operator_prompt(
            "inversion",
            registry=registry,
            parent_genome='{"premise": "test"}',
            metrics="score: 2.0",
        )
        combined = sys_msg + user_msg
        assert "SEARCH/REPLACE" in combined or "DIFF" in combined

    def test_unknown_operator_raises(self, registry):
        with pytest.raises(KeyError):
            build_operator_prompt("nonexistent", registry=registry, is_initial=True)

    def test_all_operators_produce_output(self, registry):
        for name in ALL_OPERATORS:
            sys_msg, user_msg = build_operator_prompt(
                name,
                registry=registry,
                parent_genome='{"premise": "test", "target_effect": "effect"}',
                metrics="score: 3.0",
                is_initial=False,
            )
            assert len(sys_msg) > 50, f"{name} system msg too short"
            assert len(user_msg) > 50, f"{name} user msg too short"


class TestDefaults:
    """Tests absorbed from test_shinka/test_defaults.py."""

    def test_default_probs_sum_to_one(self):
        assert np.isclose(sum(default_patch_type_probs()), 1.0, atol=1e-6)

    def test_defaults_match_registry(self, registry):
        assert set(default_patch_types()) == set(registry.keys())

    def test_probs_match_types(self):
        assert len(default_patch_types()) == len(default_patch_type_probs())


class TestAsyncApplySync:
    """Verify async_apply routing sets stay in sync with the registry."""

    def test_all_operators_routed(self, registry):
        all_routed = _FULL_PATCH_TYPES | _DIFF_PATCH_TYPES
        for name in registry:
            assert name in all_routed, f"Operator {name} not routed in async_apply"

    def test_routing_matches_registry(self, registry):
        for name, op in registry.items():
            target = _FULL_PATCH_TYPES if op.routing == "full" else _DIFF_PATCH_TYPES
            assert name in target, f"{name} routing mismatch"

    def test_no_overlap(self):
        assert _FULL_PATCH_TYPES.isdisjoint(_DIFF_PATCH_TYPES)


class TestBuildMutationFeedback:
    """Tests for judge feedback compression."""

    # Realistic multi-format judge feedback matching actual run output.
    SAMPLE_FEEDBACK = (
        "[mira-okonkwo]\n"
        "NOVELTY: DOMAIN CROSSING: Instructions-as-horror crosses domestic manual with body horror. "
        "CONVERGENCE DISTANCE: Avoids standard patterns. GENERATIVE SURPRISE: Multiple executions possible.\n"
        "GRIP: THE THING YOU CAN'T LOOK AWAY FROM: Procedural narratives can be absorbing. "
        "EMOTIONAL STAKES: Sensory details strong. SENSORY SEED: Could be 5.\n"
        "TENSION ARCHITECTURE: SUSPENSE: Tension from procedural complicity. "
        "INFORMATION ARCHITECTURE: Built in. REFRAMING POTENTIAL: Near unputdownable.\n"
        "EMOTIONAL DEPTH: RECOGNITION: Love/control themes deeply felt. "
        "EMOTIONAL COMPLEXITY: Embedded contradictions. EMOTIONAL SOURCE: Strong voice.\n"
        "THEMATIC RESONANCE: QUESTION VS MESSAGE: Genuine question posed. "
        "EMBEDDEDNESS: Embedded. Strong.\n"
        "CONCEPT COHERENCE: LOAD-BEARING ELEMENTS: Tight. "
        "SURFACE/DEPTH ARCHITECTURE: Load-bearing. Execution dependent.\n"
        "GENERATIVE FERTILITY: EXECUTION DIVERSITY: Could feel like one path. "
        "GENERATIVE PRINCIPLE: Surprisingly fertile. Limited by form.\n"
        "SCOPE CALIBRATION: NATURAL SIZE: Fits short story. "
        "CONSTRAINT AS COMPRESSION: Naturally short. Repetitive risk.\n"
        "INDELIBILITY: INDELIBLE IMAGE: Instruction manuals resist forgetting. "
        "THE IRREDUCIBLE REMAINDER: Structurally impossible to forget. SILHOUETTE: Strong shape.\n"
        "\n\n---\n\n"
        "[tomas-varga]\n"
        "1) NOVELTY\n"
        "Variations of manual/instructions as horror exist. "
        "Distinct mechanics. High ceiling. Risk of collapse into sketch.\n"
        "2) GRIP\n"
        "Office-checkpoint deadpan absorbs. "
        "Vivid sensory anchors. 3.5-5 ceiling. Risk: static.\n"
        "3) TENSION ARCHITECTURE\n"
        "Institutional incompleteness is common. "
        "Multiple tension types. Exceptional ceiling. Risk: no resolution.\n"
        "4) EMOTIONAL DEPTH\n"
        "Trapped vs having somewhere to be. "
        "Articulated well. High ceiling. Risk: sloganized.\n"
        "5) THEMATIC RESONANCE\n"
        "Fits well thematically. "
        "1000-3500 words. Perfect. Risk: sprawl.\n"
        "6) CONCEPT COHERENCE\n"
        "Could smuggle in violations. "
        "Robust. High ceiling. Risk: constraint difficulty.\n"
        "7) GENERATIVE FERTILITY\n"
        "Can generate many settings. "
        "Switchable degrees of freedom. Multiple approaches. Risk: same shape.\n"
        "8) SCOPE CALIBRATION\n"
        "Fits short story form well. "
        "Good compression. Very high ceiling. Risk: hidden bloat.\n"
        "9) INDELIBILITY\n"
        "Removes narration and interiority. "
        "Bans usual exposition. Very high. Risk: forgettable mechanics.\n"
    )

    SAMPLE_DIMENSIONS = {
        "novelty": 4.17,
        "grip": 4.60,
        "tension_architecture": 4.73,
        "emotional_depth": 4.20,
        "thematic_resonance": 4.37,
        "concept_coherence": 4.80,
        "generative_fertility": 4.17,
        "scope_calibration": 4.83,
        "indelibility": 5.0,
    }

    def test_returns_empty_for_no_feedback(self):
        assert build_mutation_feedback("", {}) == ""
        assert build_mutation_feedback(None, {}) == ""
        assert build_mutation_feedback("  ", {}) == ""

    def test_identifies_weakest_and_strongest(self):
        result = build_mutation_feedback(
            self.SAMPLE_FEEDBACK,
            {"dimensions": self.SAMPLE_DIMENSIONS},
        )
        # Weakest: novelty (4.17), generative_fertility (4.17), emotional_depth (4.20)
        assert "novelty" in result
        assert "generative_fertility" in result
        # Strongest: indelibility (5.0), scope_calibration (4.83)
        assert "indelibility" in result
        assert "scope_calibration" in result
        assert "Weakest" in result
        assert "Strongest" in result

    def test_extracts_dimension_sections(self):
        result = build_mutation_feedback(
            self.SAMPLE_FEEDBACK,
            {"dimensions": self.SAMPLE_DIMENSIONS},
        )
        # Should include Mira's NOVELTY section (weakest dim)
        assert "Instructions-as-horror" in result
        # Should include Tomas's NOVELTY section
        assert "Variations of manual/instructions" in result
        # Should NOT include full GRIP (not weakest)
        assert "Office-checkpoint deadpan" not in result

    def test_output_is_concise(self):
        result = build_mutation_feedback(
            self.SAMPLE_FEEDBACK,
            {"dimensions": self.SAMPLE_DIMENSIONS},
        )
        assert len(result) < len(self.SAMPLE_FEEDBACK)

    def test_fallback_when_no_dimensions(self):
        result = build_mutation_feedback(
            "raw feedback text here",
            {"holder_score": 4.0},
        )
        assert "raw feedback text" in result
        assert len(result) <= 500
