import pandas as pd

from src.features.build_features import FEATURE_COLUMNS, build_features


def test_build_features_end_to_end(synthetic_matches, competition_catalog):
    df = synthetic_matches.copy()
    df["competition_id"] = "VNL_MEN"
    df["stage"] = "pool"

    result = build_features(df, competition_catalog)

    for col in FEATURE_COLUMNS:
        assert col in result.columns, f"falta la columna {col}"

    assert len(result) == len(df)
    # elo_diff coherente con elo_a_pre/elo_b_pre ya presentes.
    assert (result["elo_diff"] == result["elo_a_pre"] - result["elo_b_pre"]).all()


def test_build_features_first_match_has_neutral_h2h(synthetic_matches, competition_catalog):
    df = synthetic_matches.copy()
    df["competition_id"] = "VNL_MEN"
    df["stage"] = "pool"
    result = build_features(df, competition_catalog)
    first = result.iloc[0]
    assert first["h2h_matches_played"] == 0
    assert pd.isna(first["h2h_win_rate_a"])
