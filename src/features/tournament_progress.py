"""
Progreso dentro del torneo — Fase 3 (feature engineering).

FIVB VIS no devuelve la fase del partido (`Phase`) para ningún partido de
este dataset (comprobado con datos reales: 7060/7060 partidos sin ese
campo) — así que no podemos usar directamente "final"/"semifinal"/"pool"
como pensábamos originalmente (ver `competition_importance.py`, que sigue
existiendo para fuentes que sí tengan esa información).

En su lugar, usamos una señal real y disponible: la posición del partido
dentro de su torneo (`no_in_tournament`, tal y como lo numera la propia
fuente). En la inmensa mayoría de formatos de torneo de vóley (fase de
grupos primero, cruces eliminatorios después), los partidos con número más
alto dentro del torneo tienden a ser los de mayor peso. No es una etiqueta
exacta de fase, es una aproximación — se documenta como tal.

Esto NO es leakage: el número total de partidos de un torneo es información
estructural conocida de antemano (el formato del torneo se define antes de
jugarse), no depende del resultado de ningún partido.
"""

from __future__ import annotations

import pandas as pd


def add_tournament_progress(
    df: pd.DataFrame,
    tournament_no_col: str = "tournament_no",
    no_in_tournament_col: str = "no_in_tournament",
) -> pd.DataFrame:
    """
    Añade `tournament_progress` en [0, 1]: 0 = primer partido del torneo,
    1 = último partido conocido de ese torneo en el dataset. NaN cuando no
    hay `tournament_no`/`no_in_tournament` (p.ej. filas de Kaggle, o partidos
    de VIS sin ese campo).
    """
    result = df.copy()

    if tournament_no_col not in result.columns or no_in_tournament_col not in result.columns:
        result["tournament_progress"] = float("nan")
        return result

    max_per_tournament = result.groupby(tournament_no_col)[no_in_tournament_col].transform("max")
    progress = result[no_in_tournament_col] / max_per_tournament
    # Torneos con un único partido conocido (max=0 o max=NaN) -> progreso indefinido, no 0/0.
    progress = progress.where(max_per_tournament > 0)
    result["tournament_progress"] = progress
    return result
