"""
Utilidad de detección de leakage temporal — regla crítica de la spec (sección 6):

    features(match_T) no puede depender de result(match_T)

Este módulo no depende de pandas para mantenerse ligero en Fase 1; trabaja
sobre estructuras simples (listas de dicts). Las fases de feature engineering
(Fase 3) deben usar `assert_no_future_leakage` sobre cada tabla de features
generada, pasando para cada fila el `match_id`, la fecha del partido y las
fechas de todos los partidos que contribuyeron a esa fila (p.ej. los
partidos usados para calcular una media móvil o un Elo).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Sequence


@dataclass
class LeakageViolation:
    match_id: str
    match_date: date
    offending_source_date: date
    detail: str = ""


def assert_no_future_leakage(
    feature_rows: Iterable["FeatureProvenance"],
) -> List[LeakageViolation]:
    """
    Verifica, para cada fila de features, que ningún partido usado para
    construirla ocurrió en o después de la fecha del partido objetivo.

    Devuelve la lista de violaciones encontradas (vacía si todo está limpio).
    No lanza excepción directamente para que el caller decida si es un error
    fatal (pytest) o un reporte de calidad de datos.
    """
    violations: List[LeakageViolation] = []
    for row in feature_rows:
        for source_date in row.source_match_dates:
            if source_date >= row.match_date:
                violations.append(
                    LeakageViolation(
                        match_id=row.match_id,
                        match_date=row.match_date,
                        offending_source_date=source_date,
                        detail=(
                            f"feature de match_id={row.match_id} (fecha {row.match_date}) "
                            f"usa un partido con fecha {source_date}, que no es estrictamente anterior."
                        ),
                    )
                )
    return violations


@dataclass
class FeatureProvenance:
    """Para una fila de features de un partido: qué partidos previos la alimentaron."""

    match_id: str
    match_date: date
    source_match_dates: Sequence[date]
