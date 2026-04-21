import pytest

from owtn.models.judge import JudgePersona, load_panel


JUDGES_DIR = "configs/judges"
PANEL_IDS = ["mira-okonkwo", "tomas-varga", "sable-ahn"]


class TestLoadPanel:
    def test_load_all_judges(self):
        panel = load_panel(JUDGES_DIR, PANEL_IDS)
        assert len(panel) == 3

    def test_judge_fields(self):
        panel = load_panel(JUDGES_DIR, PANEL_IDS)
        mira = panel[0]
        assert mira.id == "mira-okonkwo"
        assert mira.name == "Mira Okonkwo"
        assert mira.harshness == "advancing"
        assert mira.priority == "primary"
        assert isinstance(mira.model, list)
        assert len(mira.values) >= 3
        assert len(mira.exemplars) >= 1

    def test_demanding_judge(self):
        panel = load_panel(JUDGES_DIR, PANEL_IDS)
        tomas = panel[1]
        assert tomas.harshness == "demanding"

    def test_contrarian_judge(self):
        panel = load_panel(JUDGES_DIR, PANEL_IDS)
        sable = panel[2]
        assert sable.priority == "contrarian"

    def test_missing_judge_raises(self):
        with pytest.raises(FileNotFoundError):
            load_panel(JUDGES_DIR, ["nonexistent-judge"])

    def test_contrarian_harshness(self):
        panel = load_panel(JUDGES_DIR, PANEL_IDS)
        sable = panel[2]
        assert sable.harshness == "failing_unless_exceptional"

    def test_rejects_old_harshness_values(self):
        from pydantic import ValidationError

        for old in ("lenient", "moderate"):
            with pytest.raises(ValidationError):
                JudgePersona(
                    id="x", name="X", identity="x",
                    values=["x"], exemplars=["x"],
                    lean_in_signals=["x"],
                    harshness=old, priority="primary",
                    model=["gpt-4o"],
                )

    def test_accepts_all_four_bands(self):
        for band in ("advancing", "standard", "demanding", "failing_unless_exceptional"):
            j = JudgePersona(
                id="x", name="X", identity="x",
                values=["x"], exemplars=["x"],
                lean_in_signals=["x"],
                harshness=band, priority="primary",
                model=["gpt-4o"],
            )
            assert j.harshness == band

    def test_lean_in_signals_loaded(self):
        panel = load_panel(JUDGES_DIR, PANEL_IDS)
        for judge in panel:
            assert len(judge.lean_in_signals) >= 3, (
                f"{judge.id} has fewer than 3 lean_in_signals"
            )

    def test_lean_in_signals_required(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JudgePersona(
                id="x", name="X", identity="x",
                values=["x"], exemplars=["x"],
                harshness="standard", priority="primary",
                model=["gpt-4o"],
            )
