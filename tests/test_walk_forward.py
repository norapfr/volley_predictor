import pandas as pd
import pytest

from src.evaluation.walk_forward import evaluate_walk_forward, make_walk_forward_folds
from src.models.bradley_terry import BradleyTerryModel
from src.models.logistic_baseline import EloLogisticModel
from src.ratings.elo import EloConfig, compute_elo_history


def test_folds_are_strictly_chronological(synthetic_matches):
    folds = make_walk_forward_folds(synthetic_matches, n_folds=4, min_train_fraction=0.3)
    for train_idx, test_idx in folds:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        max_train_date = pd.to_datetime(synthetic_matches.loc[train_idx, "date"]).max()
        min_test_date = pd.to_datetime(synthetic_matches.loc[test_idx, "date"]).min()
        assert max_train_date <= min_test_date


def test_folds_expanding_window_train_grows(synthetic_matches):
    folds = make_walk_forward_folds(synthetic_matches, n_folds=4, min_train_fraction=0.3)
    train_sizes = [len(train_idx) for train_idx, _ in folds]
    assert train_sizes == sorted(train_sizes)  # nunca decrece


def test_too_few_matches_raises():
    tiny_df = pd.DataFrame(
        [{"date": "2020-01-01", "team_a": "A", "team_b": "B", "winner": "team_a"}] * 3
    )
    with pytest.raises(ValueError):
        make_walk_forward_folds(tiny_df, n_folds=5, min_train_fraction=0.3)


def test_evaluate_walk_forward_runs_all_three_models(synthetic_matches):
    history, _ = compute_elo_history(synthetic_matches, EloConfig())

    def elo_factory(train_df):
        return lambda test_df: test_df["elo_pred_a"].to_numpy()

    def bt_factory(train_df):
        model = BradleyTerryModel().fit(train_df)
        return lambda test_df: model.predict_proba_batch(test_df)

    def logistic_factory(train_df):
        model = EloLogisticModel().fit(train_df)
        return lambda test_df: model.predict_proba_batch(test_df)

    results = evaluate_walk_forward(
        history,
        {"elo": elo_factory, "bradley_terry": bt_factory, "logistic_elo": logistic_factory},
        n_folds=3,
        min_train_fraction=0.3,
    )

    assert set(results["model"].unique()) == {"elo", "bradley_terry", "logistic_elo"}
    assert (results["log_loss"] > 0).all()
    assert (results["accuracy"] >= 0).all() and (results["accuracy"] <= 1).all()
    # El baseline de 4 equipos con jerarquía estable debería predecir mejor que el azar.
    assert (results.groupby("model")["accuracy"].mean() > 0.5).all()
