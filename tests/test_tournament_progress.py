import pandas as pd

from src.features.tournament_progress import add_tournament_progress


def test_progress_scales_from_zero_to_one_within_tournament():
    df = pd.DataFrame(
        [
            {"tournament_no": "T1", "no_in_tournament": 1},
            {"tournament_no": "T1", "no_in_tournament": 5},
            {"tournament_no": "T1", "no_in_tournament": 10},
        ]
    )
    result = add_tournament_progress(df)
    assert result.iloc[0]["tournament_progress"] == 0.1
    assert result.iloc[1]["tournament_progress"] == 0.5
    assert result.iloc[2]["tournament_progress"] == 1.0


def test_progress_is_independent_per_tournament():
    df = pd.DataFrame(
        [
            {"tournament_no": "T1", "no_in_tournament": 5},
            {"tournament_no": "T1", "no_in_tournament": 10},
            {"tournament_no": "T2", "no_in_tournament": 3},
            {"tournament_no": "T2", "no_in_tournament": 6},
        ]
    )
    result = add_tournament_progress(df)
    # El último partido de T2 (no_in_tournament=6) debe llegar a 1.0 igual
    # que el de T1, aunque T2 tenga números absolutos más bajos.
    assert result.iloc[3]["tournament_progress"] == 1.0
    assert result.iloc[2]["tournament_progress"] == 0.5


def test_missing_columns_returns_nan_not_crash():
    df = pd.DataFrame([{"date": "2020-01-01"}])
    result = add_tournament_progress(df)
    assert result["tournament_progress"].isna().all()


def test_single_match_tournament_is_nan_not_zero_division():
    df = pd.DataFrame([{"tournament_no": "T1", "no_in_tournament": 0}])
    result = add_tournament_progress(df)
    assert result.iloc[0]["tournament_progress"] != result.iloc[0]["tournament_progress"] or pd.isna(
        result.iloc[0]["tournament_progress"]
    )
