import pandas as pd
import pytest

from worldcup.data import load_results, modern_era
from worldcup.display import percent_split


def write_csv(path, rows):
    pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
                 ).to_csv(path, index=False)


def test_load_results_sorts_parses_and_drops_unplayed_matches(tmp_path):
    path = tmp_path / "results.csv"
    write_csv(path, [
        ["2020-05-02", "B", "C", 1, 1, "Friendly", "TRUE"],
        ["2020-05-01", "A", "B", 2, 0, "Friendly", "FALSE"],
        ["2030-01-01", "A", "C", None, None, "Friendly", "FALSE"],
    ])
    df = load_results(path)
    assert df["date"].is_monotonic_increasing
    assert len(df) == 2  # the unplayed fixture is gone
    assert df["neutral"].tolist() == [False, True]
    assert df["year"].tolist() == [2020, 2020]


def test_load_results_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="worldcup download"):
        load_results(tmp_path / "nope.csv")


def test_load_results_rejects_unexpected_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_results(path)


def test_modern_era_bounds_are_inclusive(tmp_path):
    path = tmp_path / "results.csv"
    write_csv(path, [[d, "A", "B", 1, 0, "Friendly", "FALSE"] for d in ("1993-12-31", "1994-01-01", "2000-06-30", "2000-07-01")])
    df = load_results(path)
    assert len(modern_era(df, "1994-01-01", "2000-06-30")) == 2
    assert len(modern_era(df, "1994-01-01")) == 3


def test_percent_split_always_adds_up_to_100():
    assert percent_split(0.925) == (92, 8)  # rounding both sides independently would give 92 % and 7 %
    assert percent_split(0.5) == (50, 50)
    assert percent_split(1.0) == (100, 0)
    for p in (0.07, 0.333, 0.6595, 0.836774):
        a, b = percent_split(p)
        assert a + b == 100
