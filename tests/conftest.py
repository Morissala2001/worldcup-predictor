"""Shared fixtures: a small synthetic league where team strength is known."""

import numpy as np
import pandas as pd
import pytest

STRENGTH = {"Alpha": 3.0, "Beta": 2.0, "Gamma": 1.0, "Delta": 0.3}


@pytest.fixture(scope="session")
def synthetic_matches() -> pd.DataFrame:
    """1,500 matches among four teams; goals ~ Poisson(strength), so Alpha > Beta > Gamma > Delta."""
    rng = np.random.default_rng(0)
    teams = list(STRENGTH)
    rows = []
    for i in range(1500):
        home, away = rng.choice(teams, size=2, replace=False)
        rows.append({
            "date": pd.Timestamp("2000-01-01") + pd.Timedelta(days=3 * i),
            "home_team": home,
            "away_team": away,
            "home_score": int(rng.poisson(STRENGTH[home])),
            "away_score": int(rng.poisson(STRENGTH[away])),
            "tournament": "Friendly" if i % 4 == 0 else "FIFA World Cup qualification",
            "neutral": bool(i % 5 == 0),
        })
    return pd.DataFrame(rows)
