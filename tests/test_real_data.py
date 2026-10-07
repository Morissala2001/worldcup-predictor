"""Checks on the shipped dataset: the model must be accurate AND never absurdly certain."""

from pathlib import Path

import pytest

from worldcup import data as data_module
from worldcup import model as model_module
from worldcup import tournament

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def matches():
    return data_module.modern_era(data_module.load_results(ROOT / "data" / "results.csv"))


@pytest.fixture(scope="module")
def model(matches):
    return model_module.train(matches)


def test_model_clearly_beats_the_home_baseline_on_recent_matches(model):
    assert model.metrics["accuracy"] > model.metrics["baseline"] + 0.05


def test_probabilities_are_never_absurdly_certain(matches, model):
    """Regression test: fully grown trees used to announce 100% (France at home vs Algeria)."""
    teams = tournament.bracket_teams(tournament.load_bracket(ROOT / "data" / "bracket_2030.json"))
    forms = model_module.team_forms(matches, teams, model.window)
    for neutral in (True, False):
        probabilities = model_module.pairwise_win_probabilities(model, forms, neutral=neutral)
        assert 0.02 < min(probabilities.values()) and max(probabilities.values()) < 0.98


def test_france_at_home_against_algeria_is_not_a_certainty(matches, model):
    prediction = model_module.predict_match(model, matches, "France", "Algeria", neutral=False)
    assert prediction.winner == "France"
    assert prediction.win_probability < 0.95
