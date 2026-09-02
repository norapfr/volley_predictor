"""
Importancia de competición y fase — Fase 3 (feature engineering).

No es lo mismo un partido de fase de grupos de la VNL que una final
olímpica: los equipos suelen tomarse más en serio (mejor alineación, más
presión, menos rotación de jugadores) los partidos de mayor peso. Estas
features dan al modelo esa noción de "cuánto importa este partido", que no
está en ningún otro dato crudo del dataset maestro.

Las escalas (1-5 y 1-4) son ordinales, no absolutas — lo que importa es el
orden relativo, no el valor exacto. Un modelo de árboles (Fase 4) no
necesita más que eso para partir por el punto de corte correcto.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.normalization.competitions import CompetitionCatalog

COMPETITION_LEVEL_IMPORTANCE: Dict[str, float] = {
    "olympic_games": 5.0,
    "world_championship": 5.0,
    "world_cup": 4.0,
    "nations_league": 3.0,
    "continental": 2.0,
}
DEFAULT_COMPETITION_IMPORTANCE = 1.0

STAGE_IMPORTANCE: Dict[str, float] = {
    "final": 4.0,
    "bronze": 3.0,
    "semifinal": 2.5,
    "quarterfinal": 2.0,
    "preliminary": 1.5,
    "pool": 1.0,
    "other": 1.0,
}
DEFAULT_STAGE_IMPORTANCE = 1.0


def add_importance_features(
    df: pd.DataFrame,
    competition_catalog: CompetitionCatalog,
    competition_id_col: str = "competition_id",
    stage_col: str = "stage",
) -> pd.DataFrame:
    result = df.copy()

    def competition_importance(competition_id: str) -> float:
        comp = competition_catalog.get(competition_id)
        if comp is None:
            return DEFAULT_COMPETITION_IMPORTANCE
        return COMPETITION_LEVEL_IMPORTANCE.get(comp.level, DEFAULT_COMPETITION_IMPORTANCE)

    def stage_importance(stage: Optional[str]) -> float:
        if not isinstance(stage, str):
            return DEFAULT_STAGE_IMPORTANCE
        return STAGE_IMPORTANCE.get(stage, DEFAULT_STAGE_IMPORTANCE)

    result["competition_importance"] = result[competition_id_col].map(competition_importance)
    result["stage_importance"] = result[stage_col].map(stage_importance)
    result["match_importance"] = result["competition_importance"] * result["stage_importance"]
    return result
