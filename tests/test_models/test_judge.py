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
        assert mira.harshness == "moderate"
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
