import pandas as pd

from src.features.rolling_form import compute_form_and_h2h_features


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


def test_first_match_has_no_history():
    df = pd.DataFrame([_row("2020-01-01", "A", "B", 3, 1, "team_a")])
    result = compute_form_and_h2h_features(df)
    row = result.iloc[0]
    assert row["a_matches_played"] == 0
    assert row["b_matches_played"] == 0
    assert row["a_current_streak"] == 0
    assert row["a_days_since_last"] is None
    assert row["h2h_matches_played"] == 0
    assert row["h2h_win_rate_a"] is None


def test_second_match_reflects_first_but_not_itself():
    df = pd.DataFrame(
        [
            _row("2020-01-01", "A", "B", 3, 1, "team_a"),  # A gana
            _row("2020-01-10", "A", "C", 3, 0, "team_a"),  # A gana otra vez
        ]
    )
    result = compute_form_and_h2h_features(df)
    second = result.iloc[1]
    assert second["a_matches_played"] == 1  # solo el partido anterior, no el actual
    assert second["a_win_rate_last5"] == 1.0
    assert second["a_current_streak"] == 1
    assert second["a_days_since_last"] == 9  # 10 enero - 1 enero
    assert second["a_set_margin_avg_last5"] == 2  # 3-1 = +2


def test_win_rate_reflects_mixed_results():
    df = pd.DataFrame(
        [
            _row("2020-01-01", "A", "X", 3, 0, "team_a"),  # A gana
            _row("2020-01-02", "A", "Y", 0, 3, "team_b"),  # A pierde
            _row("2020-01-03", "A", "B", 3, 1, "team_a"),  # partido a evaluar
        ]
    )
    result = compute_form_and_h2h_features(df)
    third = result.iloc[2]
    assert third["a_matches_played"] == 2
    assert third["a_win_rate_last5"] == 0.5  # 1 de 2
    assert third["a_current_streak"] == -1  # la derrota más reciente


def test_streak_breaks_correctly():
    df = pd.DataFrame(
        [
            _row("2020-01-01", "A", "X", 3, 0, "team_a"),
            _row("2020-01-02", "A", "Y", 3, 0, "team_a"),
            _row("2020-01-03", "A", "Z", 3, 0, "team_a"),
            _row("2020-01-04", "A", "B", 3, 1, "team_a"),  # a evaluar: racha de 3 victorias antes
        ]
    )
    result = compute_form_and_h2h_features(df)
    last = result.iloc[3]
    assert last["a_current_streak"] == 3


def test_head_to_head_tracks_specific_pair_only():
    df = pd.DataFrame(
        [
            _row("2020-01-01", "A", "B", 3, 0, "team_a"),  # A vence a B
            _row("2020-01-02", "A", "C", 3, 0, "team_a"),  # A vence a C (no cuenta para h2h A-B)
            _row("2020-01-03", "B", "A", 3, 1, "team_a"),  # revancha: B es team_a ahora, gana B
        ]
    )
    result = compute_form_and_h2h_features(df)
    third = result.iloc[2]  # B vs A, con B como team_a
    assert third["h2h_matches_played"] == 1  # solo cuenta el primer A-B, no el A-C
    # h2h_win_rate_a es la tasa de VICTORIAS DE TEAM_A (aquí "B") en los enfrentamientos previos.
    # En el único precedente, ganó A (el otro equipo), así que team_a="B" tiene 0% de victorias previas.
    assert third["h2h_win_rate_a"] == 0.0


def test_head_to_head_swapped_positions_still_recognized_as_same_pair():
    df = pd.DataFrame(
        [
            _row("2020-01-01", "A", "B", 3, 0, "team_a"),
            _row("2020-01-05", "B", "A", 3, 0, "team_a"),  # posiciones invertidas, mismo par
            _row("2020-01-10", "A", "B", 3, 0, "team_a"),  # a evaluar
        ]
    )
    result = compute_form_and_h2h_features(df)
    third = result.iloc[2]
    assert third["h2h_matches_played"] == 2  # cuenta ambos, pese al cambio de posición


def test_no_future_data_leaks_into_earlier_rows_even_if_input_unsorted():
    df_sorted = pd.DataFrame(
        [
            _row("2020-01-01", "A", "B", 3, 0, "team_a"),
            _row("2020-01-10", "A", "C", 3, 0, "team_a"),
        ]
    )
    df_shuffled = df_sorted.iloc[::-1].reset_index(drop=True)

    result_sorted = compute_form_and_h2h_features(df_sorted)
    result_shuffled = compute_form_and_h2h_features(df_shuffled)

    # El partido del 10 de enero debe ver 1 partido previo de A, esté como esté el input.
    row_sorted = result_sorted[result_sorted["date"] == "2020-01-10"].iloc[0]
    row_shuffled = result_shuffled[result_shuffled["date"] == "2020-01-10"].iloc[0]
    assert row_sorted["a_matches_played"] == row_shuffled["a_matches_played"] == 1
