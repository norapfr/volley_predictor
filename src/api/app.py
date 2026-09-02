"""
API — Fase 9.

Carga los modelos UNA VEZ al arrancar (no en cada petición — cargar un
modelo CatBoost desde disco no es instantáneo, y con tráfico real esto
importaría mucho). Los dos géneros se cargan de forma perezosa la primera
vez que se piden, así arrancar la API no falla si por ejemplo solo se han
entrenado los modelos de un género todavía.

Uso:
    uvicorn src.api.app:app --reload
    # documentación interactiva en http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.api.predictor import GenderModelBundle, predict_match
from src.api.schemas import PredictionRequest, PredictionResponse, TeamListResponse
from src.normalization.competitions import CompetitionCatalog

MODELS_DIR = ROOT / "models"

app = FastAPI(
    title="Volley Predictor API",
    description="Predicción probabilística de partidos de vóley entre selecciones nacionales.",
    version="0.1.0",
)

_bundles: Dict[str, GenderModelBundle] = {}
_competition_catalog: CompetitionCatalog | None = None


def _get_bundle(gender: str) -> GenderModelBundle:
    if gender not in _bundles:
        try:
            _bundles[gender] = GenderModelBundle.load(MODELS_DIR, gender)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _bundles[gender]


def _get_competition_catalog() -> CompetitionCatalog:
    global _competition_catalog
    if _competition_catalog is None:
        _competition_catalog = CompetitionCatalog.from_csv(ROOT / "configs" / "competitions_seed.csv")
    return _competition_catalog


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/teams/{gender}", response_model=TeamListResponse)
def list_teams(gender: str) -> TeamListResponse:
    if gender not in ("men", "women"):
        raise HTTPException(status_code=422, detail="gender debe ser 'men' o 'women'.")
    bundle = _get_bundle(gender)
    teams = bundle.known_teams()
    return TeamListResponse(gender=gender, n_teams=len(teams), teams=teams)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if request.team_a == request.team_b:
        raise HTTPException(status_code=422, detail="team_a y team_b no pueden ser el mismo equipo.")

    bundle = _get_bundle(request.gender)
    competition_catalog = _get_competition_catalog()

    if competition_catalog.get(request.competition_id) is None:
        raise HTTPException(
            status_code=422,
            detail=f"competition_id={request.competition_id!r} no está en el catálogo de competiciones.",
        )

    result = predict_match(
        bundle=bundle,
        team_a=request.team_a,
        team_b=request.team_b,
        match_date=request.date,
        competition_id=request.competition_id,
        competition_catalog=competition_catalog,
        tournament_progress=request.tournament_progress,
    )
    return PredictionResponse(**result)
