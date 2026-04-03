"""Tests for JSON/Python program loading."""

import json

from shinka.core.wrap_eval import load_program


def test_load_program_json(tmp_path):
    genome = {"premise": "What if memories were contagious?", "target_effect": "Dread"}
    json_file = tmp_path / "concept.json"
    json_file.write_text(json.dumps(genome))

    result = load_program(str(json_file))
    assert isinstance(result, dict)
    assert result["premise"] == genome["premise"]
    assert result["target_effect"] == genome["target_effect"]


def test_load_program_json_complex(tmp_path):
    genome = {
        "premise": "A translator discovers that the language she's working on has no word for 'self'",
        "target_effect": "Vertigo of lost identity",
        "character_seeds": [{"label": "The Translator", "sketch": "meticulous"}],
        "constraints": ["No flashbacks", "Single room"],
    }
    json_file = tmp_path / "complex.json"
    json_file.write_text(json.dumps(genome))

    result = load_program(str(json_file))
    assert len(result["character_seeds"]) == 1
    assert len(result["constraints"]) == 2


def test_load_program_python_still_works(tmp_path):
    py_file = tmp_path / "test_module.py"
    py_file.write_text("VALUE = 42\n")

    result = load_program(str(py_file))
    assert hasattr(result, "VALUE")
    assert result.VALUE == 42
