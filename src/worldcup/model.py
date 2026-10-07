"""Train, evaluate and query the match-outcome model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, build_training_set, current_form, match_features


@dataclass
class MatchModel:
    """A fitted classifier plus everything needed to explain and use it."""

    pipeline: Pipeline
    window: int
    metrics: dict = field(default_factory=dict)
    importances: pd.Series | None = None
    confusion: pd.DataFrame | None = None

    def home_win_probability(self, rows: pd.DataFrame) -> np.ndarray:
        """P(home side wins) for each row of features (columns in `FEATURES` order)."""
        return self.pipeline.predict_proba(rows[FEATURES])[:, 1]


@dataclass(frozen=True)
class Prediction:
    team_a: str
    team_b: str
    prob_a: float

    @property
    def winner(self) -> str:
        return self.team_a if self.prob_a >= 0.5 else self.team_b

    @property
    def win_probability(self) -> float:
        return max(self.prob_a, 1 - self.prob_a)


# Every leaf of every tree must hold at least this many matches. Fully grown trees
# (min_samples_leaf=1) memorise the noise of the results: on five rolling test windows
# (2014-2026) they scored 66.1% accuracy and announced "100%" for matches that history says
# are won about 87% of the time. With 80 the forest reaches 71.0% (log loss 0.741 -> 0.559),
# the same level as a logistic regression, and its probabilities match what actually happens.
MIN_SAMPLES_LEAF = 80


def train(
    matches: pd.DataFrame,
    window: int = 10,
    test_size: float = 0.2,
    split: str = "temporal",
    n_estimators: int = 200,
    seed: int = 42,
    min_samples_leaf: int = MIN_SAMPLES_LEAF,
) -> MatchModel:
    """Fit a random forest on `matches` and evaluate it on held-out matches.

    `split="temporal"` (default) tests on the most recent matches, which mimics real use:
    predicting the future from the past. `split="random"` shuffles instead, which is easier
    (teams of the same era leak into both sets) and mostly useful for comparison.
    """
    data = build_training_set(matches, window=window)
    if split == "temporal":
        n_test = int(len(data) * test_size)
        train_data, test_data = data.iloc[:-n_test], data.iloc[-n_test:]
    elif split == "random":
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=seed)
    else:
        raise ValueError(f"split must be 'temporal' or 'random', got {split!r}")

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("forest", RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=-1
        )),
    ])
    pipeline.fit(train_data[FEATURES], train_data["home_win"])

    y_true = test_data["home_win"]
    y_pred = pipeline.predict(test_data[FEATURES])
    metrics = {
        "split": split,
        "n_train": len(train_data),
        "n_test": len(test_data),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # "Always bet on the home side": the bar a useful model has to clear.
        "baseline": float(y_true.mean()),
        "test_from": test_data["date"].min().date().isoformat(),
        "test_to": test_data["date"].max().date().isoformat(),
    }
    labels = ["no home win", "home win"]
    return MatchModel(
        pipeline=pipeline,
        window=window,
        metrics=metrics,
        importances=pd.Series(pipeline.named_steps["forest"].feature_importances_, index=FEATURES).sort_values(),
        confusion=pd.DataFrame(confusion_matrix(y_true, y_pred), index=[f"true: {l}" for l in labels],
                               columns=[f"pred: {l}" for l in labels]),
    )


def team_forms(matches: pd.DataFrame, teams: list[str], window: int = 10) -> dict[str, dict[str, float]]:
    """Current form of each team, computed once and reused for many predictions."""
    return {team: current_form(matches, team, window) for team in teams}


def pairwise_win_probabilities(
    model: MatchModel,
    forms: dict[str, dict[str, float]],
    neutral: bool = True,
    friendly: bool = False,
) -> dict[tuple[str, str], float]:
    """P(a beats b) for every ordered pair of teams in `forms`, in one batched call.

    On neutral ground nobody is really "at home", yet the model was trained with a home side.
    To remove that arbitrary asymmetry, both orderings are queried and averaged, so that
    P(a beats b) + P(b beats a) = 1 exactly. When `neutral` is False, the first team of each
    pair is the home side and the raw model output is used.
    """
    teams = list(forms)
    pairs = [(a, b) for a in teams for b in teams if a != b]
    rows = pd.DataFrame(
        [match_features(forms[a], forms[b], neutral, friendly) for a, b in pairs], columns=FEATURES
    )
    raw = dict(zip(pairs, model.home_win_probability(rows)))
    if not neutral:
        return {pair: float(p) for pair, p in raw.items()}
    return {(a, b): float((raw[(a, b)] + 1 - raw[(b, a)]) / 2) for a, b in pairs}


def predict_match(
    model: MatchModel,
    matches: pd.DataFrame,
    team_a: str,
    team_b: str,
    neutral: bool = True,
    friendly: bool = False,
) -> Prediction:
    """Predict a single match. `team_a` is the home side unless `neutral` is True."""
    if team_a == team_b:
        raise ValueError("A team cannot play against itself.")
    forms = team_forms(matches, [team_a, team_b], model.window)
    probs = pairwise_win_probabilities(model, forms, neutral=neutral, friendly=friendly)
    return Prediction(team_a, team_b, probs[(team_a, team_b)])
