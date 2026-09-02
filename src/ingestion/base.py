"""
Interfaz base de ingestión — Fase 1.

Cada fuente (FIVB VIS, un dataset de Kaggle, etc) implementa `RawMatchSource`
y produce filas "crudas" en un formato común (RawMatchRow) que luego pasa por
limpieza y normalización antes de convertirse en registros `schema.Match`.

Mantener esta capa separada de `schema.py` es intencional: los datos crudos
de una fuente casi nunca cumplen las validaciones estrictas del dataset
maestro (nombres de equipo sin normalizar, tipos como texto, campos
ausentes), así que RawMatchRow es deliberadamente permisivo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class RawMatchRow:
    """Fila cruda de un partido, tal y como llega de la fuente, sin normalizar."""

    source: str
    source_match_id: str
    date_str: str
    gender: str
    competition_name: str
    stage_str: Optional[str]
    team_a_name: str
    team_b_name: str
    team_a_score: Optional[int]
    team_b_score: Optional[int]
    sets_a: int
    sets_b: int
    venue: Optional[str] = None
    country_host: Optional[str] = None
    neutral: Optional[bool] = None
    # Identificadores de posición dentro del torneo — no todas las fuentes los
    # tienen (Kaggle no los trae), de ahí que sean opcionales. Sirven para
    # derivar una señal real de "cuán avanzado está el torneo en este partido"
    # cuando la fuente no expone la fase explícitamente (ver Fase 3, caso VIS).
    tournament_no: Optional[str] = None
    no_in_tournament: Optional[int] = None


class RawMatchSource(ABC):
    """Cada fuente de datos (VIS, Kaggle, ...) implementa esta interfaz."""

    #: Nombre corto y estable de la fuente, usado como `source` en el dataset maestro.
    source_name: str

    @abstractmethod
    def iter_matches(self) -> Iterator[RawMatchRow]:
        """Produce las filas crudas de partidos disponibles en esta fuente."""
        raise NotImplementedError
