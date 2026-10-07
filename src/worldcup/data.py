"""Loading and filtering the international results dataset.

Data: https://github.com/martj42/international_results (CC0 / public domain).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DEFAULT_PATH = Path("data/results.csv")
REQUIRED_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]


def download_results(path: str | Path = DEFAULT_PATH, url: str = DATA_URL) -> Path:
    """Download the latest results.csv to `path` and return that path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
        path.write_bytes(response.read())
    return path


def load_results(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the results file: one row per match, sorted by date, with a `year` column.

    Rows without a final score (fixtures not played yet) are dropped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `worldcup download` or fetch results.csv from {DATA_URL}."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing the expected columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    df["year"] = df["date"].dt.year
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def modern_era(df: pd.DataFrame, start: str = "1994-01-01", end: str | None = None) -> pd.DataFrame:
    """Keep the matches played between `start` and `end` (both inclusive)."""
    mask = df["date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= df["date"] <= pd.Timestamp(end)
    return df[mask].reset_index(drop=True)
