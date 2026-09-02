"""
Sistema de rating Elo — Fase 2 (baselines).

`compute_elo_history` procesa los partidos en orden cronológico estricto y,
para cada uno, guarda el rating de CADA equipo tal y como estaba justo ANTES
de ese partido (`elo_a_pre`, `elo_b_pre`). Esto es la base anti-leakage de
toda la Fase 2: cualquier modelo que use estas columnas como feature nunca
puede estar viendo el futuro, porque el valor en la fila del partido T solo
se calculó con partidos estrictamente anteriores a T (sección 6 de la spec).

Hombres y mujeres llevan ratings completamente separados — un mismo
`team_id` (p.ej. "USA") tiene un Elo distinto en cada rating pool, porque
son selecciones distintas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass
class EloConfig:
    k_factor: float = 32.0
    initial_rating: float = 1500.0
    scale: float = 400.0  # divisor clásico de la fórmula de Elo


def _expected_score(rating_a: float, rating_b: float, scale: float) -> float:
    """P(A gana) según la fórmula logística clásica de Elo."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / scale))


def compute_elo_history(
    df: pd.DataFrame,
    config: EloConfig = EloConfig(),
    gender_col: str = "gender",
    date_col: str = "date",
    team_a_col: str = "team_a",
    team_b_col: str = "team_b",
    winner_col: str = "winner",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Devuelve (history_df, final_ratings).

    history_df: copia de `df` + columnas nuevas:
        elo_a_pre, elo_b_pre  — rating de cada equipo ANTES del partido
        elo_pred_a            — P(gana A) según Elo puro en ese momento
        elo_a_post, elo_b_post — rating después de aplicar el resultado

    final_ratings: {gender: {team_id: rating}} — el rating final de cada
    equipo tras procesar todo el histórico (útil para predecir partidos
    futuros que no están en el dataset).

    Requiere que `df` tenga una columna de fecha ordenable (string ISO
    'YYYY-MM-DD' o datetime); la función ordena internamente, nunca asume
    que el caller ya lo hizo — depender de eso sería una fuente fácil de
    leakage silencioso.
    """
    if len(df) == 0:
        return df.copy(), {}

    working = df.copy()
    working["_orig_order"] = range(len(working))
    working["_date_sort"] = pd.to_datetime(working[date_col])
    working = working.sort_values(["_date_sort", "_orig_order"], kind="stable")

    ratings: Dict[str, Dict[str, float]] = {}
    a_pre, b_pre, pred_a, a_post, b_post = [], [], [], [], []

    for row in working.itertuples(index=False):
        gender = getattr(row, gender_col)
        team_a = getattr(row, team_a_col)
        team_b = getattr(row, team_b_col)
        winner = getattr(row, winner_col)

        pool = ratings.setdefault(gender, {})
        rating_a = pool.get(team_a, config.initial_rating)
        rating_b = pool.get(team_b, config.initial_rating)

        expected_a = _expected_score(rating_a, rating_b, config.scale)
        actual_a = 1.0 if winner == "team_a" else 0.0

        new_a = rating_a + config.k_factor * (actual_a - expected_a)
        new_b = rating_b + config.k_factor * ((1.0 - actual_a) - (1.0 - expected_a))

        pool[team_a] = new_a
        pool[team_b] = new_b

        a_pre.append(rating_a)
        b_pre.append(rating_b)
        pred_a.append(expected_a)
        a_post.append(new_a)
        b_post.append(new_b)

    working["elo_a_pre"] = a_pre
    working["elo_b_pre"] = b_pre
    working["elo_pred_a"] = pred_a
    working["elo_a_post"] = a_post
    working["elo_b_post"] = b_post

    # Devolver en el orden original del df de entrada, no en el orden cronológico interno.
    result = working.sort_values("_orig_order", kind="stable").drop(columns=["_orig_order", "_date_sort"])
    result.index = df.index
    return result, ratings
