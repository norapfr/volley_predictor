import numpy as np
import pandas as pd
import pytest

from src.models.set_score import (
    MARGIN_CLASSES,
    SetMarginModel,
    build_margin_training_frame,
    combine_set_score_probabilities,
    margin_class_from_sets,
    orient_features,
)


def test_margin_class_from_sets():
    assert margin_class_from_sets(3, 0) == "3-0"
    assert margin_class_from_sets(3, 1) == "3-1"
    assert margin_class_from_sets(3, 2) == "3-2"
    assert margin_class_from_sets(0, 3) == "3-0"  # el ganador (3) siempre define la clase
    assert margin_class_from_sets(1, 3) == "3-1"


def test_orient_features_a_perspective_is_identity():
    df = pd.DataFrame({"elo_diff": [100.0], "h2h_win_rate_a": [0.7], "h2h_matches_played": [3]})
    cols = ["elo_diff", "h2h_win_rate_a", "h2h_matches_played"]
    result = orient_features(df, cols, from_a_perspective=True)
    assert result["elo_diff"].iloc[0] == 100.0
    assert result["h2h_win_rate_a"].iloc[0] == 0.7


def test_orient_features_b_perspective_negates_antisymmetric():
    df = pd.DataFrame({"elo_diff": [100.0], "matches_played_diff": [5.0]})
    cols = ["elo_diff", "matches_played_diff"]
    result = orient_features(df, cols, from_a_perspective=False)
    assert result["elo_diff"].iloc[0] == -100.0
    assert result["matches_played_diff"].iloc[0] == -5.0


def test_orient_features_b_perspective_complements_rate_features():
    df = pd.DataFrame({"h2h_win_rate_a": [0.7]})
    cols = ["h2h_win_rate_a"]
    result = orient_features(df, cols, from_a_perspective=False)
    assert result["h2h_win_rate_a"].iloc[0] == pytest.approx(0.3)


def test_orient_features_b_perspective_leaves_symmetric_untouched():
    df = pd.DataFrame({"h2h_matches_played": [4], "competition_importance": [3.0]})
    cols = ["h2h_matches_played", "competition_importance"]
    result = orient_features(df, cols, from_a_perspective=False)
    assert result["h2h_matches_played"].iloc[0] == 4
    assert result["competition_importance"].iloc[0] == 3.0


def test_build_margin_training_frame_uses_all_matches():
    df = pd.DataFrame(
        {
            "winner": ["team_a", "team_b", "team_a"],
            "sets_a": [3, 2, 3],
            "sets_b": [1, 3, 0],
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "elo_diff": [50.0, -30.0, 10.0],
        }
    )
    result = build_margin_training_frame(df, ["elo_diff"])
    assert len(result) == 3  # una fila por partido, no se descarta ninguno
    assert set(result["margin_class"]) == {"3-1", "3-2", "3-0"}


def test_build_margin_training_frame_orients_toward_actual_winner():
    df = pd.DataFrame(
        {
            "winner": ["team_b"],
            "sets_a": [1],
            "sets_b": [3],
            "date": ["2020-01-01"],
            "elo_diff": [50.0],  # A tenía más Elo pero perdió
        }
    )
    result = build_margin_training_frame(df, ["elo_diff"])
    # Ganó B, así que la feature orientada al ganador debe estar invertida (negativa).
    assert result["elo_diff"].iloc[0] == -50.0
    assert result["margin_class"].iloc[0] == "3-1"


def test_combine_set_score_probabilities_sums_to_one():
    p_a_wins = np.array([0.7, 0.3])
    margin_a = pd.DataFrame({"3-0": [0.5, 0.5], "3-1": [0.3, 0.3], "3-2": [0.2, 0.2]})
    margin_b = pd.DataFrame({"3-0": [0.4, 0.4], "3-1": [0.4, 0.4], "3-2": [0.2, 0.2]})

    result = combine_set_score_probabilities(p_a_wins, margin_a, margin_b)
    assert set(result.columns) == {"3-0", "3-1", "3-2", "2-3", "1-3", "0-3"}
    row_sums = result.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_combine_set_score_probabilities_favorite_gets_more_mass_on_win_columns():
    p_a_wins = np.array([0.9])
    margin_a = pd.DataFrame({"3-0": [0.5], "3-1": [0.3], "3-2": [0.2]})
    margin_b = pd.DataFrame({"3-0": [0.5], "3-1": [0.3], "3-2": [0.2]})

    result = combine_set_score_probabilities(p_a_wins, margin_a, margin_b)
    a_win_mass = result[["3-0", "3-1", "3-2"]].sum(axis=1).iloc[0]
    b_win_mass = result[["0-3", "1-3", "2-3"]].sum(axis=1).iloc[0]
    assert a_win_mass == pytest.approx(0.9)
    assert b_win_mass == pytest.approx(0.1)


def test_set_margin_model_fits_and_predicts(synthetic_matches, competition_catalog):
    pytest.importorskip("catboost")
    from src.features.build_features import build_features

    df = synthetic_matches.copy()
    df["competition_id"] = "VNL_MEN"
    df["stage"] = "pool"
    features_df = build_features(df, competition_catalog)

    from src.features.build_features import FEATURE_COLUMNS

    margin_train = build_margin_training_frame(features_df, FEATURE_COLUMNS)
    model = SetMarginModel().fit(margin_train, FEATURE_COLUMNS)

    oriented = orient_features(features_df, FEATURE_COLUMNS, from_a_perspective=True)
    proba = model.predict_proba(oriented)

    assert list(proba.columns) == MARGIN_CLASSES
    assert np.allclose(proba.sum(axis=1), 1.0)
