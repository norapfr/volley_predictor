"""
Estado "actual" de cada equipo — puente entre el histórico (Fase 3) y la
predicción de partidos futuros (preparación para la Fase 9, API).

`compute_form_and_h2h_features` (rolling_form.py) calcula, por cada FILA
del histórico, la forma que tenía cada equipo justo antes de ese partido —
pero descarta el estado al terminar. Para predecir un partido que no está
en el dataset (el caso real de uso de la API) hace falta el estado tal y
como queda DESPUÉS del último partido conocido de cada equipo — este
módulo lo calcula y lo deja en un formato serializable (JSON), reusando la
misma lógica interna (`_TeamHistory`/`_TeamRecord`) que ya está probada en
`rolling_form.py`, para no duplicar ni desincronizar el criterio.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.features.rolling_form import DEFAULT_WINDOWS, _TeamHistory, _TeamRecord


@dataclass
class TeamCurrentState:
    """Snapshot de un equipo tal y como queda tras su último partido conocido."""

    team: str
    matches_played: int
    current_streak: int
    last_match_date: Optional[str]  # ISO 'YYYY-MM-DD', None si nunca jugó
    # Resultados recientes (hasta el mayor de los windows) — permiten recalcular
    # win_rate/set_margin exactamente igual que en el histórico, y también
    # encadenar predicciones futuras (predecir un partido, actualizar el estado,
    # predecir el siguiente) sin tener que reprocesar todo el CSV.
    recent_results: List[dict]  # [{date, won, sets_for, sets_against}, ...]


def _history_to_state(team: str, hist: _TeamHistory, max_window: int) -> TeamCurrentState:
    records = list(hist.records)
    last_date = records[-1].date.strftime("%Y-%m-%d") if records else None
    recent = records[-max_window:]
    return TeamCurrentState(
        team=team,
        matches_played=len(records),
        current_streak=hist.current_streak(),
        last_match_date=last_date,
        recent_results=[
            {
                "date": r.date.strftime("%Y-%m-%d"),
                "won": r.won,
                "sets_for": r.sets_for,
                "sets_against": r.sets_against,
            }
            for r in recent
        ],
    )


def compute_current_state(
    df: pd.DataFrame,
    windows: Tuple[int, ...] = DEFAULT_WINDOWS,
    gender_col: str = "gender",
    date_col: str = "date",
    team_a_col: str = "team_a",
    team_b_col: str = "team_b",
    sets_a_col: str = "sets_a",
    sets_b_col: str = "sets_b",
    winner_col: str = "winner",
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, Dict[str, List[dict]]]]:
    """
    Recorre `df` en orden cronológico (igual que `compute_form_and_h2h_features`)
    y devuelve el estado FINAL, listo para serializar:

        team_states: {gender: {team: TeamCurrentState-como-dict}}
        h2h_states:  {gender: {"TEAM_X|TEAM_Y" (orden alfabético): [{date, winner_team}, ...]}}
    """
    max_window = max(windows) if windows else 10

    working = df.copy()
    working["_date_sort"] = pd.to_datetime(working[date_col])
    working = working.sort_values(["_date_sort"], kind="stable")
    dates_sorted = working["_date_sort"].tolist()

    histories: Dict[Tuple[str, str], _TeamHistory] = {}
    h2h: Dict[Tuple[str, Tuple[str, str]], List[Tuple[pd.Timestamp, str]]] = {}

    for i, row in enumerate(working.itertuples(index=False)):
        gender = getattr(row, gender_col)
        team_a = getattr(row, team_a_col)
        team_b = getattr(row, team_b_col)
        sets_a = int(getattr(row, sets_a_col))
        sets_b = int(getattr(row, sets_b_col))
        winner = getattr(row, winner_col)
        date = dates_sorted[i]

        hist_a = histories.setdefault((gender, team_a), _TeamHistory())
        hist_b = histories.setdefault((gender, team_b), _TeamHistory())
        hist_a.records.append(_TeamRecord(date, winner == "team_a", sets_a, sets_b))
        hist_b.records.append(_TeamRecord(date, winner == "team_b", sets_b, sets_a))

        pair_key = tuple(sorted([team_a, team_b]))
        winner_team = team_a if winner == "team_a" else team_b
        h2h.setdefault((gender, pair_key), []).append((date, winner_team))

    team_states: Dict[str, Dict[str, dict]] = {}
    for (gender, team), hist in histories.items():
        team_states.setdefault(gender, {})[team] = asdict(_history_to_state(team, hist, max_window))

    h2h_states: Dict[str, Dict[str, List[dict]]] = {}
    for (gender, pair_key), meetings in h2h.items():
        key_str = f"{pair_key[0]}|{pair_key[1]}"
        h2h_states.setdefault(gender, {})[key_str] = [
            {"date": d.strftime("%Y-%m-%d"), "winner_team": w} for d, w in meetings
        ]

    return team_states, h2h_states


def save_current_state(team_states: dict, h2h_states: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "team_state.json").write_text(json.dumps(team_states, indent=2))
    (out_dir / "h2h_state.json").write_text(json.dumps(h2h_states, indent=2))


def load_current_state(out_dir: Path) -> Tuple[dict, dict]:
    out_dir = Path(out_dir)
    team_states = json.loads((out_dir / "team_state.json").read_text())
    h2h_states = json.loads((out_dir / "h2h_state.json").read_text())
    return team_states, h2h_states
