"""
Loader para datasets de Kaggle de la VNL (sección 3.2) en formato CSV.

Este loader NO descarga datos: los datasets de Kaggle requieren
autenticación y no forman parte de los dominios accesibles desde este
entorno. Su trabajo es leer un CSV ya descargado por el usuario (con las
columnas típicas de los datasets VNL de Kaggle referenciados en la spec) y
producir `RawMatchRow`, dejando la normalización de nombres de equipo y la
validación estricta para las fases siguientes del pipeline.

Columnas esperadas (ajustar `column_map` si un dataset concreto difiere):
    date, gender, competition, stage, team_a, team_b,
    team_a_score, team_b_score, sets_a, sets_b, venue, country_host
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, Optional

from .base import RawMatchRow, RawMatchSource

DEFAULT_COLUMN_MAP: Dict[str, str] = {
    "date": "date",
    "gender": "gender",
    "competition": "competition",
    "stage": "stage",
    "team_a": "team_a",
    "team_b": "team_b",
    "team_a_score": "team_a_score",
    "team_b_score": "team_b_score",
    "sets_a": "sets_a",
    "sets_b": "sets_b",
    "venue": "venue",
    "country_host": "country_host",
}


class KaggleVNLSource(RawMatchSource):
    """Lee un CSV de VNL estilo Kaggle y produce RawMatchRow. source_match_id = número de fila."""

    source_name = "kaggle_vnl"

    def __init__(self, csv_path: str | Path, column_map: Optional[Dict[str, str]] = None) -> None:
        self.csv_path = Path(csv_path)
        self.column_map = column_map or DEFAULT_COLUMN_MAP

    def iter_matches(self) -> Iterator[RawMatchRow]:
        with open(self.csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                cm = self.column_map
                yield RawMatchRow(
                    source=self.source_name,
                    source_match_id=f"{self.csv_path.stem}_{i}",
                    date_str=row[cm["date"]],
                    gender=row[cm["gender"]].strip().lower(),
                    competition_name=row[cm["competition"]],
                    stage_str=row.get(cm["stage"]) or None,
                    team_a_name=row[cm["team_a"]],
                    team_b_name=row[cm["team_b"]],
                    team_a_score=_to_int(row.get(cm["team_a_score"])),
                    team_b_score=_to_int(row.get(cm["team_b_score"])),
                    sets_a=int(row[cm["sets_a"]]),
                    sets_b=int(row[cm["sets_b"]]),
                    venue=row.get(cm["venue"]) or None,
                    country_host=row.get(cm["country_host"]) or None,
                    neutral=None,
                )


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(float(value))
