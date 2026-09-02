import pytest

from src.features.build_features import FEATURE_COLUMNS, build_features
from src.models.set_score import SetMarginModel, build_margin_training_frame
from src.models.tabular import CatBoostModel


@pytest.fixture
def features_df(synthetic_matches, competition_catalog):
    df = synthetic_matches.copy()
    df["competition_id"] = "VNL_MEN"
    df["stage"] = "pool"
    return build_features(df, competition_catalog)


def test_catboost_model_save_load_roundtrip(features_df, tmp_path):
    pytest.importorskip("catboost")

    model = CatBoostModel().fit(features_df, FEATURE_COLUMNS)
    original_preds = model.predict_proba_batch(features_df, FEATURE_COLUMNS)

    save_path = tmp_path / "win_model.cbm"
    model.save(save_path)
    assert save_path.exists()
    assert save_path.with_suffix(".cbm.features.json").exists()

    loaded = CatBoostModel.load(save_path)
    loaded_preds = loaded.predict_proba_batch(features_df, FEATURE_COLUMNS)

    assert list(original_preds) == list(loaded_preds)
    assert loaded._feature_columns == FEATURE_COLUMNS


def test_set_margin_model_save_load_roundtrip(features_df, tmp_path):
    pytest.importorskip("catboost")

    margin_train = build_margin_training_frame(features_df, FEATURE_COLUMNS)
    model = SetMarginModel().fit(margin_train, FEATURE_COLUMNS)

    from src.models.set_score import orient_features

    oriented = orient_features(features_df, FEATURE_COLUMNS, from_a_perspective=True)
    original_preds = model.predict_proba(oriented)

    save_path = tmp_path / "margin_model.cbm"
    model.save(save_path)

    loaded = SetMarginModel.load(save_path)
    loaded_preds = loaded.predict_proba(oriented)

    assert (original_preds.to_numpy() == loaded_preds.to_numpy()).all()
