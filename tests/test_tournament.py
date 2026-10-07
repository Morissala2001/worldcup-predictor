from pathlib import Path

import numpy as np
import pytest

from worldcup import data as data_module
from worldcup import tournament

BRACKET = [("A", "B"), ("C", "D")]
RANK = {"A": 4, "B": 3, "C": 2, "D": 1}


def by_rank(a, b):
    """Deterministic odds: the better-ranked team wins 80% of the time."""
    return 0.8 if RANK[a] > RANK[b] else 0.2


def test_simulate_favourites_advance_to_the_final():
    rounds = tournament.simulate(BRACKET, by_rank)
    assert [len(r) for r in rounds] == [2, 1]
    assert [r.winner for r in rounds[0]] == ["A", "C"]
    assert tournament.champion(rounds) == "A"
    assert rounds[1][0].probability == 0.8


def test_simulate_with_rng_is_reproducible():
    first = tournament.simulate(BRACKET, by_rank, np.random.default_rng(1))
    second = tournament.simulate(BRACKET, by_rank, np.random.default_rng(1))
    assert first == second


def test_monte_carlo_title_odds_sum_to_one_and_follow_strength():
    odds = tournament.monte_carlo(BRACKET, by_rank, runs=3000, seed=0)
    titles = {team: o.get(tournament.CHAMPION, 0) for team, o in odds.items()}
    assert sum(titles.values()) == pytest.approx(1)
    assert titles["A"] > titles["B"] > titles["D"]
    assert odds["A"]["Semi-finals"] == 1  # everybody plays the first round


@pytest.mark.parametrize("bracket", [[], [("A", "B"), ("C", "D"), ("E", "F")], [("A", "B"), ("A", "C")]])
def test_invalid_brackets_are_rejected(bracket):
    with pytest.raises(ValueError):
        tournament.validate_bracket(bracket)


def test_shipped_bracket_is_valid_and_uses_known_teams():
    root = Path(__file__).resolve().parents[1]
    bracket = tournament.load_bracket(root / "data" / "bracket_2030.json")
    assert len(bracket) == 16
    matches = data_module.load_results(root / "data" / "results.csv")
    known = set(matches["home_team"]) | set(matches["away_team"])
    assert set(tournament.bracket_teams(bracket)) <= known
