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
        return "low (at least one team has no history in the dataset)"
    if min_matches < 5:
        return "low"
    if min_matches < 15:
        return "medium"
    return "high"


def _score_confidence(score_probs: pd.Series) -> dict:
    """
    Resume how decisive the exact-score prediction is.

    Exact scores are naturally noisy: a top score can be the argmax while
    still being very close to the next alternatives. This keeps the API
    from presenting idxmax as stronger evidence than it really is.
    """
    ordered = score_probs.sort_values(ascending=False)
    top_score = str(ordered.index[0])
    top_probability = float(ordered.iloc[0])
    runner_up_probability = float(ordered.iloc[1]) if len(ordered) > 1 else 0.0
    gap = top_probability - runner_up_probability

    if top_probability >= 0.40 and gap >= 0.12:
        level = "high"
    elif top_probability >= 0.30 and gap >= 0.07:
        level = "medium"
    else:
        level = "low"

    close = ordered.iloc[1:]
    close = close[close >= max(runner_up_probability, top_probability - 0.08)]
    alternatives = [
        {"score": str(score), "probability": round(float(prob), 4)}
        for score, prob in close.head(2).items()
    ]

    return {
        "most_likely_score": top_score,
        "most_likely_score_probability": round(top_probability, 4),
        "score_confidence": level,
        "score_confidence_gap": round(gap, 4),
        "close_score_alternatives": alternatives,
    }


def _explanatory_factors(
    features: dict,
    team_a: str,
    team_b: str,
    streak_a: int,
    streak_b: int,
) -> List[str]:
    """
    Traduce las features numéricas a frases legibles, ordenadas por
    relevancia intuitiva.

    IMPORTANTE — qué son y qué NO son estos factores: son estadísticas
    descriptivas sobre el partido (Elo, racha, head-to-head...), no una
    extracción del razonamiento interno del modelo. CatBoost no expone un
    "por qué" causal de una predicción concreta sin herramientas dedicadas
    (p.ej. valores SHAP) — esto no las usa, así que un factor puede
    apuntar en una dirección y el modelo, combinando TODAS las features a
    la vez de forma no lineal, decidir lo contrario. Son contexto útil
    sobre el partido, no una justificación exacta de la predicción.

    Recibe `streak_a`/`streak_b` por separado (no la diferencia) a
    propósito: `current_streak_diff` puede ser positivo aunque los DOS
    equipos vengan de perder (p.ej. A con 1 derrota, B con 5 derrotas —
    la diferencia favorece a A, pero A sigue perdiendo, no ganando). Usar
    la diferencia para decidir "victorias" o "derrotas" describía mal el
    partido en ese caso — bug real, corregido usando el dato de cada
    equipo tal cual.
    """
    factors: List[str] = []

    elo_diff = features["elo_diff"]
    if abs(elo_diff) >= 30:
        favored = team_a if elo_diff > 0 else team_b
        factors.append(f"Elo rating: {abs(elo_diff):.0f} points in favor of {favored}")

    if streak_a >= 2:
        factors.append(f"{team_a} arrives on a {streak_a}-match winning streak")
    elif streak_a <= -2:
        factors.append(f"{team_a} arrives on a {abs(streak_a)}-match losing streak")
    if streak_b >= 2:
        factors.append(f"{team_b} arrives on a {streak_b}-match winning streak")
    elif streak_b <= -2:
        factors.append(f"{team_b} arrives on a {abs(streak_b)}-match losing streak")

    h2h_n = features["h2h_matches_played"]
    h2h_rate = features["h2h_win_rate_a"]
    if h2h_n > 0 and h2h_rate is not None:
        pct = h2h_rate * 100 if h2h_rate >= 0.5 else (1 - h2h_rate) * 100
        favored = team_a if h2h_rate >= 0.5 else team_b
        factors.append(f"Head-to-head: {favored} won {pct:.0f}% of the last {h2h_n} meetings")
    elif h2h_n == 0:
        factors.append("No previous meetings on record between these two teams")

    for w in (5, 10):
        wr_diff = features.get(f"win_rate_last{w}_diff")
        if wr_diff is not None and abs(wr_diff) >= 0.3:
            favored = team_a if wr_diff > 0 else team_b
            factors.append(f"{favored} in better recent form (last {w} matches)")
            break  # no repetir el mismo mensaje para las dos ventanas

    importance = features["competition_importance"]
    if importance >= 4:
        factors.append("High-stakes match (World Championship / Olympics / World Cup)")

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
    score_summary = _score_confidence(score_probs)

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
        "most_likely_score": score_summary["most_likely_score"],
        "most_likely_score_probability": score_summary["most_likely_score_probability"],
        "score_confidence": score_summary["score_confidence"],
        "score_confidence_gap": score_summary["score_confidence_gap"],
        "close_score_alternatives": score_summary["close_score_alternatives"],
        "expected_point_diff": round(expected_point_diff, 1),
        "elo_team_a": round(bundle.elo_ratings.get(team_a, 1500.0), 1),
        "elo_team_b": round(bundle.elo_ratings.get(team_b, 1500.0), 1),
        "confidence": _confidence_level(matches_a, matches_b),
        "explanatory_factors": _explanatory_factors(
            features, team_a, team_b,
            streak_a=state_a.get("current_streak", 0),
            streak_b=state_b.get("current_streak", 0),
        ),
    }
