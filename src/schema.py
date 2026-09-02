"""
Schema del dataset maestro — Fase 1.

Define las tablas descritas en la especificación (sección 5) más las
tablas auxiliares mencionadas en el prompt maestro (sección 32):
competitions y team_ratings.

Reglas de diseño:
- gender es siempre "men" o "women" (nunca mezclar en un mismo registro).
- Todo lo que en la spec se marca como estadística técnica opcional
  ("cuando estén disponibles") se modela como Optional y NO se rellena
  con 0 en la limpieza — un 0 real y un dato ausente son cosas distintas.
- source / source_match_id se conservan siempre para trazabilidad y para
  poder desduplicar entre fuentes (VIS, Kaggle, etc).
- neutral es explícito en vez de inferido, porque en torneos
  internacionales la localía casi siempre es neutral.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Gender(str, Enum):
    men = "men"
    women = "women"


class Stage(str, Enum):
    pool = "pool"
    preliminary = "preliminary"
    quarterfinal = "quarterfinal"
    semifinal = "semifinal"
    bronze = "bronze"
    final = "final"
    other = "other"


class Winner(str, Enum):
    team_a = "team_a"
    team_b = "team_b"


# ---------------------------------------------------------------------------
# competitions
# ---------------------------------------------------------------------------
class Competition(BaseModel):
    """Catálogo de competiciones oficiales de selecciones senior (sección 2)."""

    competition_id: str
    name: str
    gender: Gender
    level: str = Field(
        description=(
            "Categoría orientativa para competition_level / competition_importance "
            "(sección 7.6), p.ej. 'nations_league', 'world_championship', "
            "'olympic_games', 'world_cup', 'continental'. No debe usarse como "
            "proxy directo de fuerza del equipo."
        )
    )
    is_club_competition: bool = Field(
        default=False,
        description="Debe ser siempre False en V1 — clubes y ligas están excluidos (sección 2).",
    )

    @model_validator(mode="after")
    def _no_clubs(self) -> "Competition":
        if self.is_club_competition:
            raise ValueError(
                "Competición de clubes detectada — fuera de alcance de V1 (sección 2)."
            )
        return self


# ---------------------------------------------------------------------------
# teams / team_aliases
# ---------------------------------------------------------------------------
class Team(BaseModel):
    team_id: str
    canonical_name: str
    gender: Gender
    country_code: str = Field(description="Código ISO del país/selección, p.ej. 'USA', 'BRA'.")
    continent: str
    active: bool = True


class TeamAlias(BaseModel):
    """Mapea nombres de distintas fuentes a un team_id canónico (sección 5, ejemplo USA)."""

    canonical_team_id: str
    source: str
    source_name: str


# ---------------------------------------------------------------------------
# matches
# ---------------------------------------------------------------------------
class Match(BaseModel):
    match_id: str
    date: date
    gender: Gender
    competition: str
    competition_id: str
    stage: Stage
    team_a: str = Field(description="team_id de la selección A.")
    team_b: str = Field(description="team_id de la selección B.")
    team_a_score: int = Field(ge=0, description="Puntos totales del partido para A, si están disponibles.")
    team_b_score: int = Field(ge=0)
    sets_a: int = Field(ge=0, le=3)
    sets_b: int = Field(ge=0, le=3)
    winner: Winner
    venue: Optional[str] = None
    country_host: Optional[str] = None
    neutral: bool = Field(
        default=True,
        description="En torneos internacionales se asume neutral salvo evidencia contraria (sección 7.7).",
    )
    source: str
    source_match_id: str
    tournament_no: Optional[str] = Field(
        default=None,
        description=(
            "ID interno del torneo en la fuente (p.ej. 'No' de FIVB VIS). "
            "None para fuentes que no lo exponen (p.ej. Kaggle)."
        ),
    )
    no_in_tournament: Optional[int] = Field(
        default=None,
        description=(
            "Posición del partido dentro de su torneo (orden de la fuente). "
            "Se usa en Fase 3 para aproximar 'cuán avanzado está el torneo' "
            "cuando la fase (final/semifinal/...) no está disponible."
        ),
    )

    @model_validator(mode="after")
    def _consistent_result(self) -> "Match":
        if self.team_a == self.team_b:
            raise ValueError(f"match_id={self.match_id}: team_a y team_b no pueden ser el mismo equipo.")
        if {self.sets_a, self.sets_b} not in ({3, 0}, {3, 1}, {3, 2}):
            raise ValueError(
                f"match_id={self.match_id}: marcador de sets inválido "
                f"({self.sets_a}-{self.sets_b}); el vóley se juega al mejor de 5."
            )
        expected_winner = Winner.team_a if self.sets_a > self.sets_b else Winner.team_b
        if self.winner != expected_winner:
            raise ValueError(
                f"match_id={self.match_id}: winner={self.winner} no coincide con el "
                f"marcador de sets {self.sets_a}-{self.sets_b}."
            )
        return self


# ---------------------------------------------------------------------------
# sets
# ---------------------------------------------------------------------------
class SetScore(BaseModel):
    match_id: str
    set_number: int = Field(ge=1, le=5)
    team_a_points: int = Field(ge=0)
    team_b_points: int = Field(ge=0)


# ---------------------------------------------------------------------------
# team_match_stats
# ---------------------------------------------------------------------------
class TeamMatchStats(BaseModel):
    """
    Estadísticas técnicas por equipo y partido (sección 5 / 7.4).

    Todos los campos estadísticos son Optional[float] a propósito: la spec
    prohíbe explícitamente convertir `missing` en `0` (sección 5), ya que no
    todas las fuentes/torneos exponen todas las estadísticas.
    """

    match_id: str
    team: str
    attack_attempts: Optional[float] = None
    attack_points: Optional[float] = None
    attack_errors: Optional[float] = None
    attack_efficiency: Optional[float] = None
    blocks: Optional[float] = None
    aces: Optional[float] = None
    serve_errors: Optional[float] = None
    reception_attempts: Optional[float] = None
    positive_reception: Optional[float] = None
    perfect_reception: Optional[float] = None
    digs: Optional[float] = None
    sideout: Optional[float] = None
    break_points: Optional[float] = None
    points: Optional[float] = None
    sets: Optional[float] = None


# ---------------------------------------------------------------------------
# team_ratings (mencionada en el prompt maestro, sección 32)
# ---------------------------------------------------------------------------
class TeamRating(BaseModel):
    """
    Snapshot temporal del rating de un equipo — la base de las features de
    Elo (sección 7.1). Se guarda un registro por equipo tras cada partido
    para poder reconstruir "el Elo tal y como era antes de T" sin leakage.
    """

    team_id: str
    gender: Gender
    as_of_date: date
    as_of_match_id: Optional[str] = Field(
        default=None,
        description="Partido después del cual se calculó este rating; None para el rating inicial.",
    )
    elo_global: float
    elo_recent: Optional[float] = None
    elo_time_decay: Optional[float] = None
    elo_margin_adjusted: Optional[float] = None


ALL_TABLES = {
    "competitions": Competition,
    "teams": Team,
    "team_aliases": TeamAlias,
    "matches": Match,
    "sets": SetScore,
    "team_match_stats": TeamMatchStats,
    "team_ratings": TeamRating,
}
