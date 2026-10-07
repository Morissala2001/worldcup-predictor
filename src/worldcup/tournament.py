"""Knockout-bracket simulation: deterministic ("the favourite always wins") or Monte Carlo."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

WinProbability = Callable[[str, str], float]

CHAMPION = "Champion"

ROUND_NAMES = {
    64: "Round of 64",
    32: "Round of 32",
    16: "Round of 16",
    8: "Quarter-finals",
    4: "Semi-finals",
    2: "Final",
}


@dataclass(frozen=True)
class MatchResult:
    team_a: str
    team_b: str
    winner: str
    probability: float  # probability that `winner` wins this match


def load_bracket(path: str | Path) -> list[tuple[str, str]]:
    """Read a bracket from JSON: a list of [team_a, team_b] pairs, in bracket order."""
    pairs = [tuple(pair) for pair in json.loads(Path(path).read_text(encoding="utf-8"))]
    validate_bracket(pairs)
    return pairs


def validate_bracket(bracket: list[tuple[str, str]]) -> None:
    """Raise ValueError unless the bracket is a power-of-two number of unique, distinct teams."""
    n_matches = len(bracket)
    if n_matches < 1 or n_matches & (n_matches - 1):
        raise ValueError(f"A bracket needs a power-of-two number of matches, got {n_matches}.")
    teams = [team for pair in bracket for team in pair]
    duplicates = [team for team, count in Counter(teams).items() if count > 1]
    if duplicates:
        raise ValueError(f"Teams appear more than once in the bracket: {sorted(duplicates)}")


def bracket_teams(bracket: list[tuple[str, str]]) -> list[str]:
    return [team for pair in bracket for team in pair]


def round_name(n_matches: int) -> str:
    return ROUND_NAMES.get(n_matches * 2, f"Round of {n_matches * 2}")


def _play(pairs, win_prob: WinProbability, rng: np.random.Generator | None):
    """Play one round. With an `rng`, winners are drawn at random according to the odds."""
    results = []
    for a, b in pairs:
        p_a = win_prob(a, b)
        a_wins = rng.random() < p_a if rng is not None else p_a >= 0.5
        winner, p_winner = (a, p_a) if a_wins else (b, 1 - p_a)
        results.append(MatchResult(a, b, winner, p_winner))
    return results


def simulate(
    bracket: list[tuple[str, str]], win_prob: WinProbability, rng: np.random.Generator | None = None
) -> list[list[MatchResult]]:
    """Play the whole bracket and return the results round by round (the last round is the final).

    Winners of two neighbouring matches meet in the next round. Without `rng` the more likely
    team always wins; with one, each match is a weighted coin flip.
    """
    validate_bracket(bracket)
    rounds, pairs = [], list(bracket)
    while pairs:
        results = _play(pairs, win_prob, rng)
        rounds.append(results)
        winners = [r.winner for r in results]
        pairs = list(zip(winners[::2], winners[1::2]))
    return rounds


def champion(rounds: list[list[MatchResult]]) -> str:
    return rounds[-1][0].winner


def monte_carlo(
    bracket: list[tuple[str, str]], win_prob: WinProbability, runs: int = 2000, seed: int = 42
) -> dict[str, dict[str, float]]:
    """Share of simulated tournaments in which each team reaches each round, and wins it.

    Returns {team: {"<round name>": probability, ..., "Champion": probability}}.
    """
    validate_bracket(bracket)
    rng = np.random.default_rng(seed)
    reached: dict[str, Counter] = {team: Counter() for team in bracket_teams(bracket)}

    for _ in range(runs):
        rounds = simulate(bracket, win_prob, rng)
        for results in rounds:
            name = round_name(len(results))
            for result in results:
                reached[result.team_a][name] += 1
                reached[result.team_b][name] += 1
        reached[champion(rounds)][CHAMPION] += 1

    return {team: {name: count / runs for name, count in counts.items()} for team, counts in reached.items()}
