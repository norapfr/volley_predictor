import pandas as pd
import pytest

from src.features.build_features import FEATURE_COLUMNS, build_features
from src.models.tabular import available_tabular_models


@pytest.fixture
def features_with_competition(synthetic_matches, competition_catalog):
    df = synthetic_matches.copy()
    df["competition_id"] = "VNL_MEN"
    df["stage"] = "pool"
    return build_features(df, competition_catalog)


def test_available_tabular_models_never_crashes():
    """Debe devolver lo que esté instalado sin reventar, aunque falte todo."""
    models = available_tabular_models()
    assert isinstance(models, dict)


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_tabular_model_fits_and_predicts(model_name, features_with_competition):
    available = available_tabular_models()
    if model_name not in available:
        pytest.skip(f"{model_name} no está instalado en este entorno")

    model_cls = available[model_name]
    n = len(features_with_competition)
    train_df = features_with_competition.iloc[: int(n * 0.7)]
    test_df = features_with_competition.iloc[int(n * 0.7):]

    model = model_cls().fit(train_df, FEATURE_COLUMNS)
    preds = model.predict_proba_batch(test_df, FEATURE_COLUMNS)

    assert len(preds) == len(test_df)
    assert ((preds >= 0) & (preds <= 1)).all()


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_tabular_model_feature_importance(model_name, features_with_competition):
    available = available_tabular_models()
    if model_name not in available:
        pytest.skip(f"{model_name} no está instalado en este entorno")

    model_cls = available[model_name]
    model = model_cls().fit(features_with_competition, FEATURE_COLUMNS)
    importance = model.feature_importance()

    assert importance is not None
    assert set(importance.index) == set(FEATURE_COLUMNS)
    assert (importance >= 0).all()


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_tabular_model_handles_missing_values(model_name, features_with_competition):
    """h2h_win_rate_a y varias otras vienen con NaN reales para partidos sin historial — no debe crashear."""
    available = available_tabular_models()
    if model_name not in available:
        pytest.skip(f"{model_name} no está instalado en este entorno")

    df = features_with_competition.copy()
    assert df["h2h_win_rate_a"].isna().any(), "el fixture debería tener NaN reales para probar esto"

    model_cls = available[model_name]
    model = model_cls().fit(df, FEATURE_COLUMNS)
    preds = model.predict_proba_batch(df, FEATURE_COLUMNS)
    assert not pd.isna(preds).any()
