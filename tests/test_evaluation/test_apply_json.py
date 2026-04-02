"""Test that ShinkaEvolve's apply_full_patch handles JSON genomes correctly."""

import json
import pytest
from pathlib import Path

from shinka.edit.apply_full import apply_full_patch


ORIGINAL_GENOME = json.dumps({
    "premise": "A lighthouse keeper discovers the light has been signaling someone.",
    "target_effect": "Creeping dread and the vertigo of complicity.",
}, indent=2)

MUTATED_GENOME = json.dumps({
    "premise": "A lighthouse keeper discovers the light has been answering something beneath the water.",
    "target_effect": "The slow horror of realizing you've been part of a conversation you didn't know was happening.",
    "constraints": ["No dialogue.", "The keeper never leaves the lighthouse."],
}, indent=2)


class TestApplyFullPatchJSON:
    def test_json_whole_file_replacement(self):
        """JSON patch replaces entire file content (no EVOLVE-BLOCK needed)."""
        patch_str = f"```json\n{MUTATED_GENOME}\n```"
        updated, num_applied, _, error, _, _ = apply_full_patch(
            patch_str=patch_str,
            original_str=ORIGINAL_GENOME,
            language="json",
        )
        assert error is None
        assert num_applied == 1
        parsed = json.loads(updated)
        assert "answering something beneath the water" in parsed["premise"]
        assert len(parsed["constraints"]) == 2

    def test_json_invalid_patch_rejected(self):
        """Invalid JSON in patch returns error, preserves original."""
        patch_str = "```json\n{not valid json\n```"
        updated, num_applied, _, error, _, _ = apply_full_patch(
            patch_str=patch_str,
            original_str=ORIGINAL_GENOME,
            language="json",
        )
        assert num_applied == 0
        assert error is not None
        assert "Invalid JSON" in error
        assert updated == ORIGINAL_GENOME

    def test_json_writes_output_files(self, tmp_path):
        """With patch_dir, JSON patch writes output files."""
        patch_str = f"```json\n{MUTATED_GENOME}\n```"
        updated, num_applied, output_path, error, patch_txt, diff_path = apply_full_patch(
            patch_str=patch_str,
            original_str=ORIGINAL_GENOME,
            patch_dir=tmp_path / "patches",
            language="json",
        )
        assert error is None
        assert num_applied == 1
        assert output_path is not None
        assert output_path.exists()
        assert json.loads(output_path.read_text()) == json.loads(MUTATED_GENOME)
        assert (tmp_path / "patches" / "original.json").exists()
        assert (tmp_path / "patches" / "rewrite.txt").exists()
        assert diff_path is not None

    def test_json_no_code_fence_returns_error(self):
        """If LLM output has no code fence, extraction fails gracefully."""
        patch_str = "Here's the updated genome: " + MUTATED_GENOME
        updated, num_applied, _, error, _, _ = apply_full_patch(
            patch_str=patch_str,
            original_str=ORIGINAL_GENOME,
            language="json",
        )
        assert num_applied == 0
        assert error is not None

    def test_python_still_requires_evolve_blocks(self):
        """Non-JSON languages still require EVOLVE-BLOCK markers."""
        patch_str = "```python\nprint('hello')\n```"
        original = "print('world')"
        updated, num_applied, _, error, _, _ = apply_full_patch(
            patch_str=patch_str,
            original_str=original,
            language="python",
        )
        assert num_applied == 0
        assert "EVOLVE-BLOCK" in error
