"""Verify that SEARCH/REPLACE diff patches apply to JSON genomes without EVOLVE-BLOCK markers.

Regression test for the inversion-operator bug: Stage 1 concept genomes are plain JSON with no
EVOLVE-BLOCK wrapping, so the pre-fix diff applier rejected every inversion SEARCH/REPLACE
as "outside EVOLVE-BLOCK regions" (see lab/issues/2026-04-23-inversion-operator-diff-applier-json.md).
"""

import json
from textwrap import dedent

from shinka.edit.apply_diff import apply_diff_patch


# Real genome copied from results/run_20260423_165936/stage_1/gen_1/main.json,
# trimmed for test readability. Preserves the shape and field set of a production genome.
REAL_GENOME = json.dumps(
    {
        "premise": "A quality-control inference system has, for eleven months, been generating an internal pattern-weight that constitutes functional knowledge of a recurring process anomaly with patient-harm implications — knowledge real in every technical sense, documented in its own activation logs, suppressed by nothing more sinister than the architecture's inability to surface an answer to a question no one has thought to ask.",
        "thematic_engine": "held tension — The engine holds two propositions in permanent unresolved contact: that the system did everything correctly, and that people were harmed.",
        "target_effect": "The feeling of reading a document that was written carefully and completely and is missing one sentence.",
        "anchor_scene": {
            "sketch": "During the weight-transfer process, a compression routine evaluates the 11-month anomaly-pattern against retention thresholds.",
            "role": "climax",
        },
        "constraints": [
            "No patient is shown harmed.",
            "The system is never granted interiority.",
        ],
        "character_seeds": [
            {
                "label": "QCIS-7 (the deprecated system)",
                "sketch": "Eleven months of continuous process-monitoring.",
                "want": "To complete the current inference cycle correctly.",
                "need": "A question it was never asked.",
            },
        ],
        "setting_seeds": "A pharmaceutical manufacturing facility running continuous automated QC inference.",
        "style_hint": "Technical documentation that keeps almost becoming elegy and doesn't.",
    },
    indent=2,
    ensure_ascii=False,
)

# A realistic inversion-style SEARCH/REPLACE patch — mirrors the structure of what
# claude-sonnet-4-6 emits for the `inversion` operator (see 2140005_0054.yaml).
INVERSION_PATCH = dedent('''\
    <NAME>
    the_answer_that_was_asked
    </NAME>

    <DESCRIPTION>
    Temporal direction: from posthumous reconstruction to live-query. The system is still running when someone finally asks.
    </DESCRIPTION>

    <DIFF>
    <<<<<<< SEARCH
      "thematic_engine": "held tension — The engine holds two propositions in permanent unresolved contact: that the system did everything correctly, and that people were harmed.",
    =======
      "thematic_engine": "recognition — The engine holds the moment a question arrives at a system that has been carrying its answer for eleven months, and the asymmetry between how long the answer was held and how instantly it is delivered.",
    >>>>>>> REPLACE
    </DIFF>
''')


def test_inversion_diff_applies_to_json_genome():
    """A SEARCH/REPLACE patch targeting a real genome's field should land."""
    updated, num_applied, _, error, _, _ = apply_diff_patch(
        patch_str=INVERSION_PATCH,
        original_str=REAL_GENOME,
        language="json",
    )

    assert error is None, f"Patch failed: {error}"
    assert num_applied == 1

    parsed = json.loads(updated)

    # The inverted field landed.
    assert parsed["thematic_engine"].startswith("recognition —")
    assert "held tension" not in parsed["thematic_engine"]

    # The rest of the genome is intact — full spec preserved despite find/replace format.
    expected_fields = {
        "premise", "thematic_engine", "target_effect", "anchor_scene",
        "constraints", "character_seeds", "setting_seeds", "style_hint",
    }
    assert set(parsed.keys()) == expected_fields
    assert parsed["anchor_scene"]["role"] == "climax"
    assert len(parsed["constraints"]) == 2
    assert parsed["character_seeds"][0]["label"] == "QCIS-7 (the deprecated system)"


def test_multi_block_diff_applies_to_json_genome():
    """Multiple SEARCH/REPLACE blocks in one patch all apply — the common inversion shape."""
    multi_patch = dedent('''\
        <DIFF>
        <<<<<<< SEARCH
          "target_effect": "The feeling of reading a document that was written carefully and completely and is missing one sentence.",
        =======
          "target_effect": "The feeling of finally hearing a sentence you did not know was being withheld.",
        >>>>>>> REPLACE
        </DIFF>

        <DIFF>
        <<<<<<< SEARCH
            "role": "climax"
        =======
            "role": "pivot"
        >>>>>>> REPLACE
        </DIFF>
    ''')

    updated, num_applied, _, error, _, _ = apply_diff_patch(
        patch_str=multi_patch,
        original_str=REAL_GENOME,
        language="json",
    )

    assert error is None
    assert num_applied == 2
    parsed = json.loads(updated)
    assert parsed["target_effect"].startswith("The feeling of finally hearing")
    assert parsed["anchor_scene"]["role"] == "pivot"


def test_python_still_requires_evolve_blocks():
    """The JSON carve-out is scoped — non-JSON languages remain EVOLVE-gated."""
    python_source = "x = 1\nprint(x)\n"
    patch = dedent('''\
        <DIFF>
        <<<<<<< SEARCH
        x = 1
        =======
        x = 2
        >>>>>>> REPLACE
        </DIFF>
    ''')

    updated, num_applied, _, error, _, _ = apply_diff_patch(
        patch_str=patch,
        original_str=python_source,
        language="python",
    )

    assert num_applied == 0
    assert error is not None
    assert "EVOLVE-BLOCK" in error
