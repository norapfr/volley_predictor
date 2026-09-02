import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import mae, rmse, summarize_regression
from src.models.point_diff import LinearEloBaseline, MeanBaseline, PointDiffModel, compute_point_diff


def test_compute_point_diff():
    df = pd.DataFrame({"team_a_score": [100, 80], "team_b_score": [90, 95]})
    result = compute_point_diff(df)
    assert list(result) == [10, -15]


def test_mae_and_rmse_known_values():
    y_true = np.array([10.0, -5.0, 0.0])
    y_pred = np.array([8.0, -5.0, 3.0])
    assert mae(y_true, y_pred) == pytest.approx((2 + 0 + 3) / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((4 + 0 + 9) / 3))


def test_summarize_regression_has_expected_keys():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 1.5])
    result = summarize_regression(y_true, y_pred)
    assert set(result.keys()) == {"mae", "rmse", "n"}


def test_mean_baseline_predicts_train_mean():
    train_df = pd.DataFrame({"team_a_score": [100, 80, 90], "team_b_score": [90, 90, 90]})
    # point_diff = [10, -10, 0] -> media 0
    model = MeanBaseline().fit(train_df, [])
    test_df = pd.DataFrame({"team_a_score": [1, 1], "team_b_score": [1, 1]})
    preds = model.predict(test_df, [])
    assert np.allclose(preds, 0.0)


def test_linear_elo_baseline_captures_positive_relationship():
    rng = np.random.default_rng(0)
    n = 200
    elo_diff = rng.uniform(-300, 300, n)
    noise = rng.normal(0, 5, n)
    point_diff = elo_diff * 0.05 + noise  # relación lineal conocida
    train_df = pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "team_a_score": 100 + point_diff / 2,
            "team_b_score": 100 - point_diff / 2,
        }
    )
    model = LinearEloBaseline().fit(train_df, ["elo_diff"])
    preds = model.predict(pd.DataFrame({"elo_diff": [300.0, -300.0]}), ["elo_diff"])
    assert preds[0] > 0  # más Elo -> se espera ganar por más puntos
    assert preds[1] < 0


def test_point_diff_model_fits_and_predicts(synthetic_matches, competition_catalog):
    pytest.importorskip("catboost")
    from src.features.build_features import FEATURE_COLUMNS, build_features

    df = synthetic_matches.copy()
    df["competition_id"] = "VNL_MEN"
    df["stage"] = "pool"
    df["team_a_score"] = 75 + (df["sets_a"] - df["sets_b"]) * 5
    df["team_b_score"] = 75 - (df["sets_a"] - df["sets_b"]) * 5
    features_df = build_features(df, competition_catalog)

    model = PointDiffModel().fit(features_df, FEATURE_COLUMNS)
    preds = model.predict(features_df, FEATURE_COLUMNS)

    assert len(preds) == len(features_df)
    assert np.isfinite(preds).all()


def test_point_diff_model_save_load_roundtrip(synthetic_matches, competition_catalog, tmp_path):
    pytest.importorskip("catboost")
    from src.features.build_features import FEATURE_COLUMNS, build_features

    df = synthetic_matches.copy()
    df["competition_id"] = "VNL_MEN"
    df["stage"] = "pool"
    df["team_a_score"] = 75 + (df["sets_a"] - df["sets_b"]) * 5
    df["team_b_score"] = 75 - (df["sets_a"] - df["sets_b"]) * 5
    features_df = build_features(df, competition_catalog)

    model = PointDiffModel().fit(features_df, FEATURE_COLUMNS)
    original_preds = model.predict(features_df, FEATURE_COLUMNS)

    save_path = tmp_path / "point_diff_model.cbm"
    model.save(save_path)
    loaded = PointDiffModel.load(save_path)
    loaded_preds = loaded.predict(features_df, FEATURE_COLUMNS)

    assert np.allclose(original_preds, loaded_preds)
