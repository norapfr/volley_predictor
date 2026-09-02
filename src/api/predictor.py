"""
Lógica central de predicción — Fase 9 (API).

Separada del framework web (FastAPI) a propósito: `predict_match` no sabe
nada de HTTP, solo de modelos y features — así se puede probar con pytest
normal y corriente, sin levantar un servidor, y reutilizar desde un script
de línea de comandos si algún día hace falta.

Junta TODO lo construido en las fases anteriores:
  - Fase 3: features (vía predict_features.py, la versión "para partidos futuros").
  - Fase 5: modelo de ganador (CatBoost, sin calibrar — la recomendación final).
  - Fase 6: modelo de marcador de sets (descomposición ganador × margen).
  - Fase 7: modelo de diferencia de puntos.
  - Fase 8.5: estado persistido (Elo actual, forma reciente, head-to-head).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS
from src.features.predict_features import build_features_for_match
from src.models.point_diff import PointDiffModel
from src.models.set_score import SetMarginModel, combine_set_score_probabilities, orient_features
from src.models.tabular import CatBoostModel
from src.normalization.competitions import CompetitionCatalog

SCORE_ORDER = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]


class TeamNotFoundError(Exception):
    """Equipo no reconocido en el alcance del proyecto (no confundir con 'sin historial')."""


@dataclass
class GenderModelBundle:
    """Todo lo que hace falta para predecir partidos de un género, cargado una sola vez."""

    gender: str
    win_model: CatBoostModel
    margin_model: SetMarginModel
    point_diff_model: PointDiffModel
    elo_ratings: Dict[str, float]
    team_states: Dict[str, dict]
    h2h_states: Dict[str, List[dict]]
    metadata: dict

    @classmethod
    def load(cls, models_dir: Path, gender: str) -> "GenderModelBundle":
        gender_dir = Path(models_dir) / gender
        if not gender_dir.exists():
            raise FileNotFoundError(
                f"No hay modelos guardados para gender={gender!r} en {gender_dir}. "
                f"Corre scripts/train_final_models.py primero."
            )
        return cls(
            gender=gender,
            win_model=CatBoostModel.load(gender_dir / "win_model.cbm"),
            margin_model=SetMarginModel.load(gender_dir / "margin_model.cbm"),
            point_diff_model=PointDiffModel.load(gender_dir / "point_diff_model.cbm"),
            elo_ratings=json.loads((gender_dir / "elo_ratings.json").read_text()),
            team_states=json.loads((gender_dir / "team_state.json").read_text()),
            h2h_states=json.loads((gender_dir / "h2h_state.json").read_text()),
            metadata=json.loads((gender_dir / "metadata.json").read_text()),
        )

    def known_teams(self) -> List[str]:
        return sorted(self.elo_ratings.keys())


def _confidence_level(matches_played_a: int, matches_played_b: int) -> str:
    """
    Heurística simple: la confianza depende de cuánto historial hay de AMBOS
    equipos, no de lo extrema que sea la probabilidad predicha — un modelo
    puede estar muy "seguro" con poquísimos datos, y eso es precisamente lo
    que esta señal quiere advertir, no ocultar.
    """
    min_matches = min(matches_played_a, matches_played_b)
    if min_matches == 0:
        return "baja (al menos un equipo sin historial en el dataset)"
    if min_matches < 5:
        return "baja"
    if min_matches < 15:
        return "media"
    return "alta"


def _explanatory_factors(features: dict, team_a: str, team_b: str) -> List[str]:
    """Traduce las features numéricas a frases legibles, ordenadas por relevancia intuitiva."""
    factors: List[str] = []

    elo_diff = features["elo_diff"]
    if abs(elo_diff) >= 30:
        favored = team_a if elo_diff > 0 else team_b
        factors.append(f"Diferencia de Elo: {abs(elo_diff):.0f} puntos a favor de {favored}")

    streak_diff = features["current_streak_diff"]
    if streak_diff and abs(streak_diff) >= 2:
        team, streak = (team_a, features["current_streak_diff"]) if streak_diff > 0 else (
            team_b, -features["current_streak_diff"],
        )
        word = "victorias" if streak > 0 else "derrotas"
        factors.append(f"{team} llega con racha de {abs(streak):.0f} {word} seguidas")

    h2h_n = features["h2h_matches_played"]
    h2h_rate = features["h2h_win_rate_a"]
    if h2h_n > 0 and h2h_rate is not None:
        pct = h2h_rate * 100 if h2h_rate >= 0.5 else (1 - h2h_rate) * 100
        favored = team_a if h2h_rate >= 0.5 else team_b
        factors.append(f"Historial directo: {favored} ganó {pct:.0f}% de los {h2h_n} enfrentamientos previos")
    elif h2h_n == 0:
        factors.append("Sin enfrentamientos previos registrados entre estos dos equipos")

    for w in (5, 10):
        wr_diff = features.get(f"win_rate_last{w}_diff")
        if wr_diff is not None and abs(wr_diff) >= 0.3:
            favored = team_a if wr_diff > 0 else team_b
            factors.append(f"{favored} con mejor forma reciente (últimos {w} partidos)")
            break  # no repetir el mismo mensaje para las dos ventanas

    importance = features["competition_importance"]
    if importance >= 4:
        factors.append("Partido de alta relevancia (Mundial/Juegos Olímpicos/Copa del Mundo)")

    return factors


def predict_match(
    bundle: GenderModelBundle,
    team_a: str,
    team_b: str,
    match_date: str,
    competition_id: str,
    competition_catalog: CompetitionCatalog,
    tournament_progress: Optional[float] = None,
) -> dict:
    """Devuelve la predicción completa de un partido — el punto de entrada de toda la API."""
    features = build_features_for_match(
        team_a=team_a,
        team_b=team_b,
        match_date=match_date,
        competition_id=competition_id,
        elo_ratings=bundle.elo_ratings,
        team_states=bundle.team_states,
        h2h_states=bundle.h2h_states,
        competition_catalog=competition_catalog,
        tournament_progress=tournament_progress,
    )
    features_df = pd.DataFrame([features])

    p_a = float(bundle.win_model.predict_proba_batch(features_df, FEATURE_COLUMNS)[0])
    p_b = 1.0 - p_a

    oriented_a = orient_features(features_df, FEATURE_COLUMNS, from_a_perspective=True)
    oriented_b = orient_features(features_df, FEATURE_COLUMNS, from_a_perspective=False)
    margin_if_a = bundle.margin_model.predict_proba(oriented_a)
    margin_if_b = bundle.margin_model.predict_proba(oriented_b)
    score_probs = combine_set_score_probabilities(np.array([p_a]), margin_if_a, margin_if_b)
    score_probs = score_probs[SCORE_ORDER].iloc[0]
    most_likely_score = score_probs.idxmax()

    expected_point_diff = float(bundle.point_diff_model.predict(features_df, FEATURE_COLUMNS)[0])

    state_a = bundle.team_states.get(team_a, {})
    state_b = bundle.team_states.get(team_b, {})
    matches_a = state_a.get("matches_played", 0)
    matches_b = state_b.get("matches_played", 0)

    return {
        "gender": bundle.gender,
        "team_a": team_a,
        "team_b": team_b,
        "date": match_date,
        "competition_id": competition_id,
        "p_team_a_wins": round(p_a, 4),
        "p_team_b_wins": round(p_b, 4),
        "set_score_probabilities": {k: round(float(v), 4) for k, v in score_probs.items()},
        "most_likely_score": most_likely_score,
        "expected_point_diff": round(expected_point_diff, 1),
        "elo_team_a": round(bundle.elo_ratings.get(team_a, 1500.0), 1),
        "elo_team_b": round(bundle.elo_ratings.get(team_b, 1500.0), 1),
        "confidence": _confidence_level(matches_a, matches_b),
        "explanatory_factors": _explanatory_factors(features, team_a, team_b),
    }
