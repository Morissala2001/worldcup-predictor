"""Streamlit app: predict a match, simulate a bracket, inspect the model.

    uv run --extra app streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from worldcup import data as data_module
from worldcup import model as model_module
from worldcup import tournament
from worldcup.display import label, percent_split

ROOT = Path(__file__).parent
RESULTS = ROOT / "data" / "results.csv"
DEFAULT_BRACKET = ROOT / "data" / "bracket_2030.json"

NEUTRAL, AT_HOME = "Neutral ground", "Team A at home"

st.set_page_config(page_title="World Cup Predictor", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner="Loading matches…")
def load_matches(start_year: int) -> pd.DataFrame:
    return data_module.modern_era(data_module.load_results(RESULTS), start=f"{start_year}-01-01")


@st.cache_resource(show_spinner="Training the model…")
def fit(start_year: int, window: int) -> model_module.MatchModel:
    return model_module.train(load_matches(start_year), window=window)


with st.sidebar:
    st.header("Settings")
    start_year = st.slider("Use matches since", 1980, 2015, 1994)
    window = st.slider("Form = last N matches", 3, 20, 10)
    st.caption("Data: martj42/international_results (CC0).")

matches = load_matches(start_year)
model = fit(start_year, window)

recent = matches[matches["date"] >= matches["date"].max() - pd.DateOffset(years=3)]
counts = pd.concat([recent["home_team"], recent["away_team"]]).value_counts()
teams = sorted(counts[counts >= 5].index)

st.title("⚽ World Cup Predictor")
st.caption(
    f"{len(matches):,} international matches, from {matches['date'].min():%d %b %Y} "
    f"to {matches['date'].max():%d %b %Y}."
)

tab_match, tab_bracket, tab_model = st.tabs(["Single match", "Tournament", "Model"])

# ---------------------------------------------------------------- one match
with tab_match:
    col_a, col_b = st.columns(2)
    team_a = col_a.selectbox("Team A", teams, index=teams.index("France") if "France" in teams else 0, format_func=label)
    team_b = col_b.selectbox("Team B", teams, index=teams.index("Brazil") if "Brazil" in teams else 1, format_func=label)
    venue = st.radio("Venue", [NEUTRAL, AT_HOME], horizontal=True)
    friendly = st.checkbox("Friendly match")

    if team_a == team_b:
        st.warning("Pick two different teams.")
    else:
        prediction = model_module.predict_match(
            model, matches, team_a, team_b, neutral=venue == NEUTRAL, friendly=friendly
        )
        st.subheader(f"Predicted winner: {label(prediction.winner)}")
        percent_a, percent_b = percent_split(prediction.prob_a)
        metric_a, metric_b = st.columns(2)
        metric_a.metric(label(team_a), f"{percent_a}%")
        metric_b.metric(label(team_b), f"{percent_b}%")
        st.progress(prediction.prob_a)

        forms = model_module.team_forms(matches, [team_a, team_b], window)
        st.caption(f'Form over the last {window} matches (what the model "sees"):')
        st.dataframe(
            pd.DataFrame(forms).T.rename(columns={
                "avg_points": "Points per match", "avg_goals_scored": "Goals scored", "avg_goals_conceded": "Goals conceded",
            }).round(2),
            width="stretch",
            alt="Recent form of both teams: points per match, goals scored and goals conceded",
        )

# --------------------------------------------------------------- bracket
with tab_bracket:
    uploaded = st.file_uploader("Custom bracket (JSON: a list of [team A, team B] pairs)", type="json")
    try:
        bracket = (
            [tuple(pair) for pair in json.load(uploaded)] if uploaded else tournament.load_bracket(DEFAULT_BRACKET)
        )
        tournament.validate_bracket(bracket)
        unknown = sorted(set(tournament.bracket_teams(bracket)) - set(matches["home_team"]) - set(matches["away_team"]))
        if unknown:
            raise ValueError(f"Teams not found in the data: {unknown}")
    except (ValueError, json.JSONDecodeError) as error:
        st.error(f"Invalid bracket: {error}")
        st.stop()

    forms = model_module.team_forms(matches, tournament.bracket_teams(bracket), window)
    probabilities = model_module.pairwise_win_probabilities(model, forms)
    win_prob = lambda a, b: probabilities[(a, b)]  # noqa: E731

    rounds = tournament.simulate(bracket, win_prob)
    st.subheader(f"If the favourite always wins: 🏆 {label(tournament.champion(rounds))}")
    for results in rounds:
        name = tournament.round_name(len(results))
        with st.expander(name, expanded=len(results) <= 2):
            st.table(
                pd.DataFrame(
                    {"Match": f"{label(r.team_a)}  vs  {label(r.team_b)}", "Winner": label(r.winner),
                     "Probability": f"{r.probability:.0%}"} for r in results
                ).set_index("Match"),
                alt=f"Predicted results of the {name.lower()}",
            )

    st.divider()
    runs = st.slider("Monte Carlo simulations", 200, 5000, 2000, step=200)
    st.caption("Each match is drawn at random according to the model's probabilities, so upsets can happen.")
    odds = tournament.monte_carlo(bracket, win_prob, runs=runs)
    title_odds = pd.Series({team: o.get(tournament.CHAMPION, 0.0) for team, o in odds.items()}).sort_values(ascending=False)
    st.bar_chart(
        title_odds.head(12).rename(index=label),
        horizontal=True,
        alt="Share of simulated tournaments won by the 12 most likely champions",
    )

# ----------------------------------------------------------------- model
with tab_model:
    m = model.metrics
    st.write(
        f"Evaluated on the **{m['n_test']:,} most recent matches** ({m['test_from']} → {m['test_to']}), "
        f"never seen during training ({m['n_train']:,} matches)."
    )
    col_acc, col_base = st.columns(2)
    col_acc.metric("Model accuracy", f"{m['accuracy']:.1%}")
    col_base.metric("Baseline: always predict a home win", f"{m['baseline']:.1%}")
    st.subheader("What drives the predictions")
    st.bar_chart(model.importances, horizontal=True, alt="Importance of each feature in the random forest")
    st.subheader("Confusion matrix")
    st.dataframe(model.confusion, width="stretch", alt="Confusion matrix of the model on the test matches")
    st.info(
        "Football is a low-scoring sport, so even a good model is right only about seven times out of ten. "
        "Read the probabilities, not just the verdict."
    )
