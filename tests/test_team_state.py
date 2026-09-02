import pandas as pd
import pytest

from src.features.team_state import compute_current_state, load_current_state, save_current_state


def _row(date, team_a, team_b, sets_a, sets_b, winner):
    return {
        "gender": "men",
        "date": date,
        "team_a": team_a,
        "team_b": team_b,
        "sets_a": sets_a,
        "sets_b": sets_b,
        "winner": winner,
    }


def test_current_state_reflects_last_match_of_each_team():
    df = pd.DataFrame(
        [
            _row("2020-01-01", "A", "B", 3, 1, "team_a"),
            _row("2020-01-10", "A", "C", 3, 0, "team_a"),
        ]
    )
    team_states, _ = compute_current_state(df)
    state_a = team_states["men"]["A"]
    assert state_a["matches_played"] == 2
    assert state_a["current_streak"] == 2  # ganó los dos
    assert state_a["last_match_date"] == "2020-01-10"


def test_h2h_state_tracks_specific_pair():
    df = pd.DataFrame(
        [
            _row("2020-01-01", "A", "B", 3, 0, "team_a"),
            _row("2020-01-05", "A", "C", 3, 0, "team_a"),
        ]
    )
    _, h2h_states = compute_current_state(df)
    assert "A|B" in h2h_states["men"]
    assert "A|C" in h2h_states["men"]
    assert len(h2h_states["men"]["A|B"]) == 1


def test_save_and_load_current_state_roundtrip(tmp_path):
    df = pd.DataFrame([_row("2020-01-01", "A", "B", 3, 1, "team_a")])
    team_states, h2h_states = compute_current_state(df)

    save_current_state(team_states["men"], h2h_states["men"], tmp_path)
    loaded_team, loaded_h2h = load_current_state(tmp_path)

    assert loaded_team["A"]["matches_played"] == 1
    assert loaded_h2h["A|B"][0]["winner_team"] == "A"


def test_build_features_for_match_matches_historical_pipeline(competition_catalog):
    """
    El feature-builder de inferencia (predict_features.py) debe producir
    EXACTAMENTE los mismos valores que el pipeline histórico
    (build_features.py) para el mismo partido — si no coinciden, el modelo
    entrenado con uno vería datos distintos al servir con el otro.
    """
    from src.features.build_features import build_features
    from src.features.predict_features import build_features_for_match
    from src.ratings.elo import EloConfig, compute_elo_history

    df = pd.DataFrame(
        [
            _row("2020-01-01", "A", "B", 3, 1, "team_a"),
            _row("2020-01-05", "A", "C", 3, 0, "team_a"),
            _row("2020-01-08", "B", "C", 3, 2, "team_a"),
            _row("2020-01-12", "A", "B", 3, 2, "team_a"),  # partido "objetivo"
        ]
    )
    df["competition_id"] = "VNL_MEN"
    df["stage"] = "pool"

    historical = build_features(df, competition_catalog)
    target_row = historical.iloc[3]  # el partido A vs B del 12 de enero

    # Estado "tal y como estaba" justo antes del partido objetivo: todo lo anterior.
    df_before = df.iloc[:3]
    _, final_elo = compute_elo_history(df_before, EloConfig())
    team_states, h2h_states = compute_current_state(df_before)

    inferred = build_features_for_match(
        team_a="A",
        team_b="B",
        match_date="2020-01-12",
        competition_id="VNL_MEN",
        elo_ratings=final_elo["men"],
        team_states=team_states["men"],
        h2h_states=h2h_states["men"],
        competition_catalog=competition_catalog,
    )

    assert inferred["elo_diff"] == pytest.approx(target_row["elo_diff"])
    assert inferred["matches_played_diff"] == target_row["matches_played_diff"]
    assert inferred["current_streak_diff"] == target_row["current_streak_diff"]
    assert inferred["win_rate_last5_diff"] == pytest.approx(target_row["win_rate_last5_diff"])
    assert inferred["h2h_matches_played"] == target_row["h2h_matches_played"]
    if pd.notna(target_row["h2h_win_rate_a"]):
        assert inferred["h2h_win_rate_a"] == pytest.approx(target_row["h2h_win_rate_a"])
    else:
        assert pd.isna(inferred["h2h_win_rate_a"])


def test_build_features_for_match_handles_unknown_team(competition_catalog):
    """Un equipo nunca visto (debut) no debe crashear — se le trata como neutral (Elo 1500, sin historial)."""
    from src.features.predict_features import build_features_for_match

    result = build_features_for_match(
        team_a="BRAND_NEW_TEAM",
        team_b="ANOTHER_NEW_TEAM",
        match_date="2026-01-01",
        competition_id="VNL_MEN",
        elo_ratings={},
        team_states={},
        h2h_states={},
        competition_catalog=competition_catalog,
    )
    assert result["elo_diff"] == 0.0
    assert result["matches_played_diff"] == 0
    assert result["h2h_matches_played"] == 0
    assert pd.isna(result["h2h_win_rate_a"])
