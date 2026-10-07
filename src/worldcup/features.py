"""Feature engineering: recent form of both teams, computed from past matches only."""

from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

FORM_COLUMNS = ["avg_points", "avg_goals_scored", "avg_goals_conceded"]
FEATURES = [
    "diff_avg_points",
    "diff_avg_goals_scored",
    "diff_avg_goals_conceded",
    "is_neutral",
    "is_friendly",
]


def points(goals_for: int, goals_against: int) -> int:
    """Points earned in one match: 3 for a win, 1 for a draw, 0 for a defeat."""
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def add_recent_form(df: pd.DataFrame, window: int = 10, min_matches: int = 5) -> pd.DataFrame:
    """Add each team's recent form, computed BEFORE the match, to every row.

    For both sides, the average points, goals scored and goals conceded over the team's last
    `window` matches become `home_*` / `away_*` columns. A team with fewer than `min_matches`
    past matches gets NaN. `df` must be sorted by date; the form of a match never includes
    the match itself nor any later one, so there is no leakage from the future.
    """
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    rows = []

    for match in df.itertuples():
        row = {}
        for side, team in (("home", match.home_team), ("away", match.away_team)):
            past = history[team]
            if len(past) >= min_matches:
                n = len(past)
                row[f"{side}_avg_points"] = sum(m[0] for m in past) / n
                row[f"{side}_avg_goals_scored"] = sum(m[1] for m in past) / n
                row[f"{side}_avg_goals_conceded"] = sum(m[2] for m in past) / n
        rows.append(row)

        for team, goals_for, goals_against in (
            (match.home_team, match.home_score, match.away_score),
            (match.away_team, match.away_score, match.home_score),
        ):
            history[team].append((points(goals_for, goals_against), goals_for, goals_against))

    form = pd.DataFrame(rows, index=df.index, columns=[f"{s}_{c}" for s in ("home", "away") for c in FORM_COLUMNS])
    return pd.concat([df, form], axis=1)


def current_form(df: pd.DataFrame, team: str, window: int = 10) -> dict[str, float]:
    """A team's form over its `window` most recent matches in `df` (draws included)."""
    played = df[(df["home_team"] == team) | (df["away_team"] == team)].tail(window)
    if played.empty:
        raise ValueError(f"No match found for team '{team}'. Team names are in English (e.g. 'Ivory Coast').")

    pts, scored, conceded = [], [], []
    for match in played.itertuples():
        if match.home_team == team:
            goals_for, goals_against = match.home_score, match.away_score
        else:
            goals_for, goals_against = match.away_score, match.home_score
        pts.append(points(goals_for, goals_against))
        scored.append(goals_for)
        conceded.append(goals_against)

    n = len(pts)
    return {"avg_points": sum(pts) / n, "avg_goals_scored": sum(scored) / n, "avg_goals_conceded": sum(conceded) / n}


def build_training_set(df: pd.DataFrame, window: int = 10, min_matches: int = 5) -> pd.DataFrame:
    """Turn raw matches into a supervised dataset: `FEATURES` -> `home_win` (1/0).

    Draws are dropped: a knockout match always has a winner, so the task is binary.
    """
    df = add_recent_form(df, window=window, min_matches=min_matches)
    df = df.dropna(subset=["home_avg_points", "away_avg_points"])
    data = df[df["home_score"] != df["away_score"]].copy()

    data["home_win"] = (data["home_score"] > data["away_score"]).astype(int)
    data["is_neutral"] = data["neutral"].astype(int)
    data["is_friendly"] = (data["tournament"] == "Friendly").astype(int)
    data["diff_avg_points"] = data["home_avg_points"] - data["away_avg_points"]
    data["diff_avg_goals_scored"] = data["home_avg_goals_scored"] - data["away_avg_goals_scored"]
    data["diff_avg_goals_conceded"] = data["home_avg_goals_conceded"] - data["away_avg_goals_conceded"]
    return data.reset_index(drop=True)


def match_features(form_home: dict[str, float], form_away: dict[str, float], neutral: bool, friendly: bool) -> list[float]:
    """The feature row (in `FEATURES` order) of a match between two teams of known form."""
    return [
        form_home["avg_points"] - form_away["avg_points"],
        form_home["avg_goals_scored"] - form_away["avg_goals_scored"],
        form_home["avg_goals_conceded"] - form_away["avg_goals_conceded"],
        int(neutral),
        int(friendly),
    ]
