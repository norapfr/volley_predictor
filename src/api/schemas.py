"""Esquemas Pydantic de la API — Fase 9. Validan la entrada y documentan la salida (OpenAPI/Swagger automático de FastAPI)."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    gender: Literal["men", "women"]
    team_a: str = Field(description="Código de equipo (p.ej. 'USA', 'BRA') — mismo team_id que en el dataset maestro.")
    team_b: str
    date: str = Field(description="Fecha del partido, 'YYYY-MM-DD'.")
    competition_id: str = Field(description="Id de competición del catálogo (p.ej. 'VNL_MEN', 'OG_WOMEN').")
    tournament_progress: Optional[float] = Field(
        default=None,
        description=(
            "Opcional, 0-1. Posición del partido dentro de su torneo, si se conoce "
            "(p.ej. 0.9 para una final). Se deja en blanco si no se sabe — no se inventa."
        ),
    )


class PredictionResponse(BaseModel):
    gender: str
    team_a: str
    team_b: str
    date: str
    competition_id: str
    p_team_a_wins: float
    p_team_b_wins: float
    set_score_probabilities: Dict[str, float]
    most_likely_score: str
    expected_point_diff: float
    elo_team_a: float
    elo_team_b: float
    confidence: str
    explanatory_factors: List[str]


class TeamListResponse(BaseModel):
    gender: str
    n_teams: int
    teams: List[str]


class ErrorResponse(BaseModel):
    detail: str
