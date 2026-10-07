import pytest

from worldcup.model import pairwise_win_probabilities, predict_match, team_forms, train


@pytest.fixture(scope="module")
def model(synthetic_matches):
    return train(synthetic_matches, n_estimators=100)


def test_model_beats_the_home_baseline(model):
    assert model.metrics["accuracy"] > model.metrics["baseline"]
    assert model.metrics["n_train"] > model.metrics["n_test"]


def test_temporal_split_tests_on_the_most_recent_matches(synthetic_matches):
    model = train(synthetic_matches, n_estimators=10, split="temporal")
    assert model.metrics["test_to"] == synthetic_matches["date"].max().date().isoformat()


def test_unknown_split_is_rejected(synthetic_matches):
    with pytest.raises(ValueError, match="split"):
        train(synthetic_matches, split="nope")


def test_stronger_team_is_favourite(model, synthetic_matches):
    prediction = predict_match(model, synthetic_matches, "Alpha", "Delta")
    assert prediction.winner == "Alpha"
    assert prediction.win_probability > 0.5


def test_neutral_predictions_are_symmetric(model, synthetic_matches):
    ab = predict_match(model, synthetic_matches, "Beta", "Gamma", neutral=True)
    ba = predict_match(model, synthetic_matches, "Gamma", "Beta", neutral=True)
    assert ab.prob_a == pytest.approx(1 - ba.prob_a)
    assert ab.winner == ba.winner


def test_pairwise_probabilities_cover_every_ordered_pair(model, synthetic_matches):
    forms = team_forms(synthetic_matches, ["Alpha", "Beta", "Gamma"], window=10)
    probs = pairwise_win_probabilities(model, forms)
    assert len(probs) == 6
    for (a, b), p in probs.items():
        assert 0 <= p <= 1
        assert p + probs[(b, a)] == pytest.approx(1)


def test_a_team_cannot_play_itself(model, synthetic_matches):
    with pytest.raises(ValueError):
        predict_match(model, synthetic_matches, "Alpha", "Alpha")
