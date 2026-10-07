import pandas as pd
import pytest

from worldcup.features import FEATURES, add_recent_form, build_training_set, current_form, points


def make_matches(results):
    """results: list of (home, away, home_score, away_score), played one per day."""
    return pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), "home_team": h, "away_team": a,
         "home_score": hs, "away_score": as_, "tournament": "Friendly", "neutral": False}
        for i, (h, a, hs, as_) in enumerate(results)
    ])


def test_points():
    assert (points(2, 0), points(1, 1), points(0, 3)) == (3, 1, 0)


def test_form_is_nan_until_enough_history():
    df = make_matches([("A", "B", 1, 0)] * 4 + [("A", "B", 0, 0)])
    out = add_recent_form(df, window=10, min_matches=3)
    assert out["home_avg_points"].isna().tolist() == [True, True, True, False, False]


def test_form_only_uses_past_matches():
    # A wins its first 3 matches 2-0, then the 4th match is a 5-0 thrashing.
    df = make_matches([("A", "B", 2, 0)] * 3 + [("A", "B", 5, 0)])
    out = add_recent_form(df, window=10, min_matches=3)
    last = out.iloc[3]
    # The form before match 4 reflects the three 2-0 wins, NOT the 5-0 of the match itself.
    assert last["home_avg_points"] == 3.0
    assert last["home_avg_goals_scored"] == 2.0
    assert last["away_avg_goals_conceded"] == 2.0


def test_form_window_forgets_old_matches():
    df = make_matches([("A", "B", 0, 4)] * 2 + [("A", "B", 3, 0)] * 2 + [("A", "B", 1, 1)])
    out = add_recent_form(df, window=2, min_matches=2)
    assert out.iloc[4]["home_avg_points"] == 3.0  # only the last two matches (both wins) count


def test_current_form_counts_draws_and_both_sides():
    df = make_matches([("A", "B", 1, 1), ("C", "A", 0, 2)])
    form = current_form(df, "A")
    assert form == {"avg_points": 2.0, "avg_goals_scored": 1.5, "avg_goals_conceded": 0.5}


def test_current_form_unknown_team():
    with pytest.raises(ValueError, match="No match found"):
        current_form(make_matches([("A", "B", 1, 0)]), "Nobody")


def test_training_set_drops_draws_and_builds_features(synthetic_matches):
    data = build_training_set(synthetic_matches, window=10, min_matches=5)
    assert (data["home_score"] != data["away_score"]).all()
    assert set(data["home_win"].unique()) == {0, 1}
    assert data[FEATURES].notna().all().all()
    row = data.iloc[0]
    assert row["diff_avg_points"] == pytest.approx(row["home_avg_points"] - row["away_avg_points"])
