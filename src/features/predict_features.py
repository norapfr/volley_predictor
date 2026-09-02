"""
Features para un partido FUTURO (no en el dataset) — la pieza que faltaba
para poder predecir de verdad, no solo validar sobre el histórico.

Usa exactamente las mismas fórmulas que `build_features.py` (Fase 3), pero
en vez de leerlas del propio DataFrame histórico, las reconstruye a partir
del estado persistido (`team_state.py`) — el rating Elo actual, la forma
reciente y el head-to-head tal y como quedaron tras el último partido
conocido de cada equipo.

`tournament_progress` es la única feature que NO se puede reconstruir con
garantía para un partido futuro suelto: depende de saber cuántos partidos
tendrá en total el torneo, algo que solo se conoce con certeza una vez el
torneo está definido. Se deja en `None` (NaN) salvo que el caller la
proporcione explícitamente — más honesto que inventar un valor.
"""

from __future__ import annotations

from datetime import date as date_cls
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.features.competition_importance import (
    COMPETITION_LEVEL_IMPORTANCE,
    DEFAULT_COMPETITION_IMPORTANCE,
)
from src.normalization.competitions import CompetitionCatalog

DEFAULT_WINDOWS = (5, 10)


def _parse_date(value) -> date_cls:
    if isinstance(value, date_cls):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _team_stats_from_state(state: Optional[dict], windows: Tuple[int, ...]) -> dict:
    if state is None:
        return {
            "matches_played": 0,
            "current_streak": 0,
            "last_match_date": None,
            **{f"win_rate_last{w}": None for w in windows},
            **{f"set_margin_avg_last{w}": None for w in windows},
        }

    recent = state.get("recent_results", [])
    out = {
        "matches_played": state["matches_played"],
        "current_streak": state["current_streak"],
        "last_match_date": state["last_match_date"],
    }
    for w in windows:
        window_results = recent[-w:]
        if window_results:
            out[f"win_rate_last{w}"] = sum(1 for r in window_results if r["won"]) / len(window_results)
            out[f"set_margin_avg_last{w}"] = sum(
                r["sets_for"] - r["sets_against"] for r in window_results
            ) / len(window_results)
        else:
            out[f"win_rate_last{w}"] = None
            out[f"set_margin_avg_last{w}"] = None
    return out


def _days_since(last_match_date: Optional[str], match_date: date_cls) -> Optional[float]:
    if last_match_date is None:
        return None
    return float((match_date - _parse_date(last_match_date)).days)


def build_features_for_match(
    team_a: str,
    team_b: str,
    match_date,
    competition_id: str,
    elo_ratings: Dict[str, float],
    team_states: Dict[str, dict],
    h2h_states: Dict[str, List[dict]],
    competition_catalog: CompetitionCatalog,
    windows: Tuple[int, ...] = DEFAULT_WINDOWS,
    tournament_progress: Optional[float] = None,
) -> dict:
    """
    Devuelve un dict con las mismas claves que `FEATURE_COLUMNS`
    (`build_features.py`), para un partido que aún no se ha jugado.

    `elo_ratings`, `team_states`, `h2h_states` son los diccionarios ya
    cargados de `models/<gender>/elo_ratings.json` / `team_state.json` /
    `h2h_state.json` (ver `src/features/team_state.py` y
    `scripts/train_final_models.py`).
    """
    match_date = _parse_date(match_date)

    elo_a = elo_ratings.get(team_a, 1500.0)
    elo_b = elo_ratings.get(team_b, 1500.0)

    stats_a = _team_stats_from_state(team_states.get(team_a), windows)
    stats_b = _team_stats_from_state(team_states.get(team_b), windows)

    days_a = _days_since(stats_a["last_match_date"], match_date)
    days_b = _days_since(stats_b["last_match_date"], match_date)

    pair_key = "|".join(sorted([team_a, team_b]))
    meetings = [m for m in h2h_states.get(pair_key, []) if _parse_date(m["date"]) < match_date]
    h2h_matches_played = len(meetings)
    h2h_win_rate_a = (
        sum(1 for m in meetings if m["winner_team"] == team_a) / h2h_matches_played
        if h2h_matches_played > 0
        else float("nan")
    )

    comp = competition_catalog.get(competition_id)
    competition_importance = (
        COMPETITION_LEVEL_IMPORTANCE.get(comp.level, DEFAULT_COMPETITION_IMPORTANCE)
        if comp is not None
        else DEFAULT_COMPETITION_IMPORTANCE
    )

    features = {
        "elo_diff": elo_a - elo_b,
        "matches_played_diff": stats_a["matches_played"] - stats_b["matches_played"],
        "current_streak_diff": stats_a["current_streak"] - stats_b["current_streak"],
        "days_since_last_diff": (
            (days_a - days_b) if (days_a is not None and days_b is not None) else float("nan")
        ),
        "h2h_matches_played": h2h_matches_played,
        "h2h_win_rate_a": h2h_win_rate_a,
        "competition_importance": competition_importance,
        "tournament_progress": tournament_progress if tournament_progress is not None else float("nan"),
    }
    for w in windows:
        wr_a, wr_b = stats_a[f"win_rate_last{w}"], stats_b[f"win_rate_last{w}"]
        sm_a, sm_b = stats_a[f"set_margin_avg_last{w}"], stats_b[f"set_margin_avg_last{w}"]
        features[f"win_rate_last{w}_diff"] = (
            (wr_a - wr_b) if (wr_a is not None and wr_b is not None) else float("nan")
        )
        features[f"set_margin_avg_last{w}_diff"] = (
            (sm_a - sm_b) if (sm_a is not None and sm_b is not None) else float("nan")
        )

    return features
