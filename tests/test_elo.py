import pandas as pd

from src.ratings.elo import EloConfig, compute_elo_history


def test_elo_starts_at_initial_rating():
    df = pd.DataFrame(
        [{"date": "2020-01-01", "gender": "men", "team_a": "A", "team_b": "B", "winner": "team_a"}]
    )
    history, _ = compute_elo_history(df, EloConfig(initial_rating=1500))
    assert history.iloc[0]["elo_a_pre"] == 1500
    assert history.iloc[0]["elo_b_pre"] == 1500


def test_elo_winner_gains_rating_loser_loses_it():
    df = pd.DataFrame(
        [{"date": "2020-01-01", "gender": "men", "team_a": "A", "team_b": "B", "winner": "team_a"}]
    )
    history, ratings = compute_elo_history(df, EloConfig())
    assert history.iloc[0]["elo_a_post"] > 1500
    assert history.iloc[0]["elo_b_post"] < 1500
    # Elo es de suma cero entre los dos equipos del partido.
    gained = history.iloc[0]["elo_a_post"] - 1500
    lost = 1500 - history.iloc[0]["elo_b_post"]
    assert abs(gained - lost) < 1e-9


def test_elo_pre_rating_never_uses_future_matches():
    """El rating 'pre' del segundo partido de un equipo debe ser el 'post' del primero, no otra cosa."""
    df = pd.DataFrame(
        [
            {"date": "2020-01-01", "gender": "men", "team_a": "A", "team_b": "B", "winner": "team_a"},
            {"date": "2020-01-02", "gender": "men", "team_a": "A", "team_b": "C", "winner": "team_a"},
        ]
    )
    history, _ = compute_elo_history(df, EloConfig())
    first_match_a_post = history.iloc[0]["elo_a_post"]
    second_match_a_pre = history.iloc[1]["elo_a_pre"]
    assert first_match_a_post == second_match_a_pre


def test_elo_ignores_row_order_only_respects_date():
    """Si las filas llegan desordenadas, compute_elo_history debe ordenar por fecha igualmente."""
    df_ordered = pd.DataFrame(
        [
            {"date": "2020-01-01", "gender": "men", "team_a": "A", "team_b": "B", "winner": "team_a"},
            {"date": "2020-01-02", "gender": "men", "team_a": "A", "team_b": "C", "winner": "team_a"},
        ]
    )
    df_shuffled = df_ordered.iloc[::-1].reset_index(drop=True)

    history_ordered, ratings_ordered = compute_elo_history(df_ordered, EloConfig())
    history_shuffled, ratings_shuffled = compute_elo_history(df_shuffled, EloConfig())

    assert ratings_ordered["men"]["A"] == ratings_shuffled["men"]["A"]


def test_elo_keeps_genders_completely_separate():
    df = pd.DataFrame(
        [
            {"date": "2020-01-01", "gender": "men", "team_a": "USA", "team_b": "BRA", "winner": "team_a"},
            {"date": "2020-01-01", "gender": "women", "team_a": "USA", "team_b": "BRA", "winner": "team_b"},
        ]
    )
    _, ratings = compute_elo_history(df, EloConfig())
    assert ratings["men"]["USA"] > 1500
    assert ratings["women"]["USA"] < 1500


def test_elo_higher_rated_team_has_higher_win_probability():
    df = pd.DataFrame(
        [{"date": "2020-01-01", "gender": "men", "team_a": "A", "team_b": "B", "winner": "team_a"}] * 10
        + [{"date": "2020-02-01", "gender": "men", "team_a": "A", "team_b": "B", "winner": "team_a"}]
    )
    history, _ = compute_elo_history(df, EloConfig())
    assert history.iloc[-1]["elo_pred_a"] > 0.5


def test_elo_learns_strength_hierarchy(synthetic_matches):
    _, ratings = compute_elo_history(synthetic_matches, EloConfig())
    pool = ratings["men"]
    assert pool["STRONG"] > pool["MID"] > pool["WEAK"] > pool["BOTTOM"]
