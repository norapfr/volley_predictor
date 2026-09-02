import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FEATURE_COLUMNS, build_features


@pytest.fixture
def features_df(synthetic_matches, competition_catalog):
    df = synthetic_matches.copy()
    df["competition_id"] = "VNL_MEN"
    df["stage"] = "pool"
    return build_features(df, competition_catalog)


def test_neural_model_fits_and_predicts(features_df):
    pytest.importorskip("torch")
    from src.models.neural import NeuralWinModel

    model = NeuralWinModel(max_epochs=20, patience=5).fit(features_df, FEATURE_COLUMNS)
    preds = model.predict_proba_batch(features_df, FEATURE_COLUMNS)

    assert len(preds) == len(features_df)
    assert ((preds >= 0) & (preds <= 1)).all()
    assert np.isfinite(preds).all()


def test_neural_model_handles_missing_values(features_df):
    """h2h_win_rate_a viene con NaN reales — la red debe imputarlos, no crashear."""
    pytest.importorskip("torch")
    from src.models.neural import NeuralWinModel

    assert features_df["h2h_win_rate_a"].isna().any(), "el fixture debería tener NaN para probar esto"

    model = NeuralWinModel(max_epochs=20, patience=5).fit(features_df, FEATURE_COLUMNS)
    preds = model.predict_proba_batch(features_df, FEATURE_COLUMNS)
    assert not pd.isna(preds).any()


def test_neural_model_imputation_uses_train_median_not_test():
    """La imputación de test debe usar la mediana del TRAIN, nunca recalcularla con datos de test."""
    pytest.importorskip("torch")
    from src.models.neural import NeuralWinModel

    train_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=20).strftime("%Y-%m-%d"),
            "winner": ["team_a", "team_b"] * 10,
            "elo_diff": [10.0] * 20,
        }
    )
    model = NeuralWinModel(max_epochs=5, patience=3)
    imputed_fit = model._impute(train_df, ["elo_diff"], fitting=True)
    assert imputed_fit["elo_diff"].iloc[0] == 10.0
    assert model._medians["elo_diff"] == 10.0

    # Con un valor de test muy distinto, la mediana guardada no debe cambiar.
    test_df = pd.DataFrame({"elo_diff": [999.0]})
    model._impute(test_df, ["elo_diff"], fitting=False)
    assert model._medians["elo_diff"] == 10.0  # no se recalculó con el valor de test


def test_neural_model_missing_flag_columns_detected_from_fit_part(features_df):
    pytest.importorskip("torch")
    from src.models.neural import NeuralWinModel

    model = NeuralWinModel(max_epochs=5, patience=3).fit(features_df, FEATURE_COLUMNS)
    assert "h2h_win_rate_a" in model._missing_flag_columns
