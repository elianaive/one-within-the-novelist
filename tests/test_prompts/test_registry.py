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
        assert "first generation" in user_msg
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
        assert "Current concept" in user_msg
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
        "ORIGINALITY: RECOGNITION: I've seen stories told through instructions before. "
        "SPECIFICITY: The concrete details elevate it. CEILING: A 5 is possible. RISKS: Could slip.\n"
        "TRANSPORTATION POTENTIAL: RECOGNITION: Procedural narratives can be absorbing. "
        "SPECIFICITY: Sensory details strong. CEILING: Could be 5. RISKS: Too cold.\n"
        "NARRATIVE TENSION: RECOGNITION: Tension from procedural complicity. "
        "SPECIFICITY: Built in. CEILING: Near unputdownable. RISKS: Frustration.\n"
        "THEMATIC RESONANCE: RECOGNITION: Love/control themes. "
        "SPECIFICITY: Embedded. CEILING: Strong. RISKS: Too neat.\n"
        "SCOPE CALIBRATION: RECOGNITION: Fits short story. "
        "SPECIFICITY: Naturally short. CEILING: 5. RISKS: Repetitive.\n"
        "ANTI-CLICHE: RECOGNITION: Avoids patterns. "
        "SPECIFICITY: Domestic focus helps. CEILING: 4. RISKS: Could drift.\n"
        "CONCEPT COHERENCE: RECOGNITION: Tight. "
        "SPECIFICITY: Load-bearing. CEILING: 5. RISKS: Execution dependent.\n"
        "GENERATIVE FERTILITY: RECOGNITION: Could feel like one path. "
        "SPECIFICITY: Surprisingly fertile. CEILING: 4. RISKS: Limited by form.\n"
        "OVER-EXPLANATION RESISTANCE: RECOGNITION: Instruction manuals resist exposition. "
        "SPECIFICITY: Structurally impossible to explain. CEILING: 5. RISKS: Hidden exposition.\n"
        "\n\n---\n\n"
        "[tomas-varga]\n"
        "1) ORIGINALITY\n"
        "RECOGNITION: Variations of manual/instructions as horror exist. "
        "SPECIFICITY: Distinct mechanics. CEILING: High. RISKS: Collapse into sketch.\n"
        "2) TRANSPORTATION POTENTIAL\n"
        "RECOGNITION: Office-checkpoint deadpan absorbs. "
        "SPECIFICITY: Vivid sensory anchors. CEILING: 3.5-5. RISKS: Static.\n"
        "3) NARRATIVE TENSION\n"
        "RECOGNITION: Institutional incompleteness is common. "
        "SPECIFICITY: Multiple tension types. CEILING: Exceptional. RISKS: No resolution.\n"
        "4) THEMATIC RESONANCE\n"
        "RECOGNITION: Trapped vs having somewhere to be. "
        "SPECIFICITY: Articulated well. CEILING: High. RISKS: Sloganized.\n"
        "5) SCOPE CALIBRATION\n"
        "RECOGNITION: Fits well. "
        "SPECIFICITY: 1000-3500 words. CEILING: Perfect. RISKS: Sprawl.\n"
        "6) ANTI-CLICHE\n"
        "RECOGNITION: Document/handbook horror is known. "
        "SPECIFICITY: Genuinely followable. CEILING: Uncliche. RISKS: Remains.\n"
        "7) CONCEPT COHERENCE\n"
        "RECOGNITION: Could smuggle in violations. "
        "SPECIFICITY: Robust. CEILING: High. RISKS: Constraint difficulty.\n"
        "8) GENERATIVE FERTILITY\n"
        "RECOGNITION: Can generate many settings. "
        "SPECIFICITY: Switchable degrees of freedom. CEILING: Multiple approaches. RISKS: Same shape.\n"
        "9) OVER-EXPLANATION RESISTANCE\n"
        "RECOGNITION: Removes narration and interiority. "
        "SPECIFICITY: Bans usual exposition. CEILING: Very high. RISKS: Hidden exposition.\n"
    )

    SAMPLE_DIMENSIONS = {
        "originality": 4.17,
        "transportation_potential": 4.60,
        "narrative_tension": 4.73,
        "thematic_resonance": 4.37,
        "scope_calibration": 4.83,
        "anti_cliche": 4.20,
        "concept_coherence": 4.80,
        "generative_fertility": 4.17,
        "over_explanation_resistance": 5.0,
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
        # Weakest: originality (4.17), generative_fertility (4.17), anti_cliche (4.20)
        assert "originality" in result
        assert "generative_fertility" in result
        # Strongest: over_explanation_resistance (5.0), scope_calibration (4.83)
        assert "over_explanation_resistance" in result
        assert "scope_calibration" in result
        assert "Weakest" in result
        assert "Strongest" in result

    def test_extracts_dimension_sections(self):
        result = build_mutation_feedback(
            self.SAMPLE_FEEDBACK,
            {"dimensions": self.SAMPLE_DIMENSIONS},
        )
        # Should include Mira's ORIGINALITY section (weakest dim)
        assert "I've seen stories told through instructions" in result
        # Should include Tomas's ORIGINALITY section
        assert "Variations of manual/instructions" in result
        # Should NOT include full TRANSPORTATION POTENTIAL (not weakest)
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
