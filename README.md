# ⚽ World Cup Predictor

*Predict international football matches from recent form, and simulate a World Cup knockout bracket.*

A machine-learning model (random forest) that describes each match by the **recent form** of both teams, estimates each side's probability of winning, then replays a knockout bracket: once with "the favourite always wins", and thousands of times with Monte Carlo.

The project can be used in three ways: a **web app** (Streamlit), a **CLI** (`worldcup`) and a Python **library** (`src/worldcup`), covered by tests.

## Results

Evaluated on the **4,414 most recent matches** (March 2021 → July 2026), never seen during training (17,658 matches, since 1994):

| | Accuracy |
|---|---|
| Baseline: always bet on the home side | 62.1% |
| **Random forest (200 trees)** | **71.7%** |

Football is a low-scoring sport where a goalpost can change a result, so the **probabilities** matter more than the verdict (France 80% against Greece, but 59% against Brazil). The three form differences (points, goals scored, goals conceded) carry almost all the information; neutral ground and friendlies matter little.

## Run it

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
git clone https://github.com/Morissala2001/worldcup-predictor.git
cd worldcup-predictor
uv sync --all-extras
```

**Web app**

```bash
uv run streamlit run app.py
```

Three tabs: *Single match* (probabilities and form of both teams), *Tournament* (bracket, simulation, title odds from Monte Carlo, custom bracket as JSON) and *Model* (accuracy, feature importance, confusion matrix).

**Command line** (team names are in English, as in the data)

```bash
uv run worldcup train                          # train and evaluate the model
uv run worldcup predict France Brazil          # one match, neutral ground
uv run worldcup predict Spain Argentina --home # Spain plays at home
uv run worldcup simulate --runs 2000           # bracket + title odds
uv run worldcup download                       # refresh data/results.csv
```

**Tests**

```bash
uv run pytest
```

## How it works

1. **Data**: every international match since 1872 ([martj42/international_results](https://github.com/martj42/international_results), CC0 licence). The model uses the matches since 1994.
2. **Recent form** (`features.py`): for each match, the average points, goals scored and goals conceded of each team over its last 10 matches, computed **only from earlier matches** (a test checks that nothing leaks from the future).
3. **Features**: the three form differences (home − away), plus *neutral ground* and *friendly match*. Draws are dropped: a knockout match always has a winner.
4. **Model** (`model.py`): `StandardScaler` + `RandomForestClassifier`, in a scikit-learn `Pipeline`.
5. **A smoothed forest**: fully grown trees memorise the noise of football results. Requiring at least 80 matches per leaf (`min_samples_leaf=80`) raised accuracy from 66.1% to 71.0% and cut the log loss from 0.741 to 0.559 on five rolling test windows (8,829 matches, 2014–2026), and the probabilities became trustworthy: among the predictions announced at 90% or more (93% on average), 94% were right. A logistic regression (70.9%) and a gradient-boosting model (70.8%) land at the same level, so the limit comes from the features, not from the algorithm.
6. **Temporal evaluation**: the model is tested on the most recent matches rather than on a random draw, which mimics real use (predicting the future from the past). With a random split (`worldcup train --split random`) accuracy is 72.3%, against a 64.1% baseline.
7. **Symmetric predictions**: on neutral ground nobody is "at home", yet the model was trained with a home side. Both orderings are therefore queried and averaged, so that P(A beats B) + P(B beats A) = 1.
8. **Tournament** (`tournament.py`): the winners of two neighbouring matches meet in the next round. The "deterministic" simulation lets the favourite win every match; the Monte Carlo simulation draws each match at random according to the model's probabilities.

## Limits

- The model only knows **recent form**: no players, no injuries, and no strength of the opponents faced (a winning streak against weak teams inflates form). It can therefore surprise: a result such as *Brazil beaten by Morocco* reflects the measured form, not an expert's opinion.
- The accuracy gain over the baseline is real but football stays unpredictable: about three matches out of ten are still called wrong.
- `data/bracket_2030.json` is a fictional example of 32 teams, freely editable.

## Structure

```
├── app.py                  # Streamlit app
├── data/
│   ├── results.csv         # match results (CC0)
│   └── bracket_2030.json   # example bracket
├── src/worldcup/
│   ├── data.py             # loading and filtering
│   ├── features.py         # recent form, features
│   ├── model.py            # training, evaluation, prediction
│   ├── tournament.py       # simulation and Monte Carlo
│   ├── display.py          # flags and labels
│   └── cli.py              # the `worldcup` command
└── tests/
```

## Ideas for improvement

- Weight the form by **opponent strength** (for example with an Elo rating).
- Add richer signals (squad value, rest days, tournament stage): simpler and more complex models plateau at the same ~71%.
- Also predict the **draw** (3 classes) and the scores.
- Give host nations a home advantage in the simulation.

## License

[MIT](LICENSE). The match data is in the public domain (CC0).
