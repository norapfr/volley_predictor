"""
Orquestador de feature engineering — Fase 3.

Combina Elo (Fase 2), forma reciente + head-to-head (`rolling_form.py`) e
importancia de competición (`competition_importance.py`) en un único
DataFrame, y añade columnas "diff" (team_a - team_b) simétricas: son las
que consumirá directamente el modelo tabular de la Fase 4, porque no le
importa "cuánto vale el equipo A" en absoluto, sino la diferencia relativa
frente al equipo B.

`FEATURE_COLUMNS` es la lista de columnas numéricas listas para entrenar —
mantenerla centralizada aquí evita que Fase 4 tenga que saber los nombres
internos de cada módulo de features.
"""

from __future__ import annotations

import pandas as pd

from src.features.competition_importance import add_importance_features
from src.features.rolling_form import DEFAULT_WINDOWS, compute_form_and_h2h_features
from src.features.tournament_progress import add_tournament_progress
from src.normalization.competitions import CompetitionCatalog
from src.ratings.elo import EloConfig, compute_elo_history

FEATURE_COLUMNS = [
    "elo_diff",
    "matches_played_diff",
    "current_streak_diff",
    "days_since_last_diff",
    "h2h_matches_played",
    "h2h_win_rate_a",
    "competition_importance",
    "tournament_progress",
] + [f"win_rate_last{w}_diff" for w in DEFAULT_WINDOWS] + [
    f"set_margin_avg_last{w}_diff" for w in DEFAULT_WINDOWS
]


def build_features(
    df: pd.DataFrame,
    competition_catalog: CompetitionCatalog,
    elo_config: EloConfig = EloConfig(),
    windows=DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """
    Punto de entrada único de la Fase 3. Recibe el DataFrame de partidos
    limpio (el `matches.csv` de la Fase 1) y devuelve ese mismo DataFrame
    con todas las columnas de features añadidas, incluidas las "diff"
    listas para `FEATURE_COLUMNS`.
    """
    result, _final_ratings = compute_elo_history(df, elo_config)
    result = compute_form_and_h2h_features(result, windows=windows)
    result = add_importance_features(result, competition_catalog)
    result = add_tournament_progress(result)

    result["elo_diff"] = result["elo_a_pre"] - result["elo_b_pre"]
    result["matches_played_diff"] = result["a_matches_played"] - result["b_matches_played"]
    result["current_streak_diff"] = result["a_current_streak"] - result["b_current_streak"]
    result["days_since_last_diff"] = result["a_days_since_last"] - result["b_days_since_last"]

    for w in windows:
        result[f"win_rate_last{w}_diff"] = result[f"a_win_rate_last{w}"] - result[f"b_win_rate_last{w}"]
        result[f"set_margin_avg_last{w}_diff"] = (
            result[f"a_set_margin_avg_last{w}"] - result[f"b_set_margin_avg_last{w}"]
        )

    # h2h_win_rate_a en None significa "nunca se han enfrentado" — se deja
    # como NaN a propósito (no se rellena con 0.5) para que un modelo de
    # árboles (Fase 4) pueda tratarlo como su propia categoría ("sin
    # historial") en vez de fingir que "empatados" es lo mismo que
    # "desconocido". Los modelos lineales de la Fase 2 no usan esta columna,
    # así que no hace falta imputar aquí.

    return result
