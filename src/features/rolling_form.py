"""
Forma reciente y head-to-head — Fase 3 (feature engineering).

Mismo patrón anti-leakage que `src/ratings/elo.py`: se procesan los
partidos en orden cronológico estricto y, para cada uno, se calculan las
features de cada equipo usando SOLO su historial hasta ANTES de ese
partido. El partido en curso se añade al historial después de calcular
sus features, nunca antes.

`head_to_head` es específico: no es "cómo de bien juega el equipo en
general" sino "qué tal le ha ido a este equipo concretamente contra este
rival concreto" — dos equipos pueden tener Elo parecido pero un historial
head-to-head muy desequilibrado (estilos de juego que chocan mal/bien).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_WINDOWS = (5, 10)


@dataclass
class _TeamRecord:
    date: pd.Timestamp
    won: bool
    sets_for: int
    sets_against: int


@dataclass
class _TeamHistory:
    records: Deque[_TeamRecord] = field(default_factory=deque)

    def stats(self, window: int) -> Tuple[Optional[float], Optional[float], int]:
        """Devuelve (win_rate, set_margin_avg, n_disponibles) sobre los últimos `window` partidos."""
        recent = list(self.records)[-window:]
        if not recent:
            return None, None, 0
        win_rate = sum(1 for r in recent if r.won) / len(recent)
        set_margin = sum(r.sets_for - r.sets_against for r in recent) / len(recent)
        return win_rate, set_margin, len(recent)

    def current_streak(self) -> int:
        """Positivo = racha de victorias, negativo = racha de derrotas, 0 si no hay historial."""
        if not self.records:
            return 0
        last_result = self.records[-1].won
        streak = 0
        for r in reversed(self.records):
            if r.won != last_result:
                break
            streak += 1
        return streak if last_result else -streak

    def days_since_last(self, current_date: pd.Timestamp) -> Optional[float]:
        if not self.records:
            return None
        return (current_date - self.records[-1].date).days

    def matches_played(self) -> int:
        return len(self.records)


def _pair_key(team_x: str, team_y: str) -> Tuple[str, str]:
    return tuple(sorted([team_x, team_y]))  # type: ignore[return-value]


def compute_form_and_h2h_features(
    df: pd.DataFrame,
    windows: Tuple[int, ...] = DEFAULT_WINDOWS,
    gender_col: str = "gender",
    date_col: str = "date",
    team_a_col: str = "team_a",
    team_b_col: str = "team_b",
    sets_a_col: str = "sets_a",
    sets_b_col: str = "sets_b",
    winner_col: str = "winner",
) -> pd.DataFrame:
    """
    Devuelve una copia de `df` con columnas nuevas por equipo (`a_`/`b_`) y de
    head-to-head (`h2h_`). Todas calculadas SOLO con partidos estrictamente
    anteriores a cada fila — nunca con el propio partido ni con partidos futuros.
    """
    if len(df) == 0:
        return df.copy()

    working = df.copy()
    working["_orig_order"] = range(len(working))
    working["_date_sort"] = pd.to_datetime(working[date_col])
    working = working.sort_values(["_date_sort", "_orig_order"], kind="stable")

    histories: Dict[Tuple[str, str], _TeamHistory] = defaultdict(_TeamHistory)
    h2h: Dict[Tuple[str, Tuple[str, str]], List[Tuple[pd.Timestamp, str]]] = defaultdict(list)

    rows: List[dict] = []
    dates_sorted = working["_date_sort"].tolist()

    for i, row in enumerate(working.itertuples(index=False)):
        gender = getattr(row, gender_col)
        team_a = getattr(row, team_a_col)
        team_b = getattr(row, team_b_col)
        sets_a = int(getattr(row, sets_a_col))
        sets_b = int(getattr(row, sets_b_col))
        winner = getattr(row, winner_col)
        date = dates_sorted[i]

        hist_a = histories[(gender, team_a)]
        hist_b = histories[(gender, team_b)]

        out: dict = {}
        for prefix, hist in (("a", hist_a), ("b", hist_b)):
            out[f"{prefix}_matches_played"] = hist.matches_played()
            out[f"{prefix}_current_streak"] = hist.current_streak()
            out[f"{prefix}_days_since_last"] = hist.days_since_last(date)
            for w in windows:
                win_rate, set_margin, n = hist.stats(w)
                out[f"{prefix}_win_rate_last{w}"] = win_rate
                out[f"{prefix}_set_margin_avg_last{w}"] = set_margin

        # Head-to-head: historial específico entre estos dos equipos, antes de este partido.
        pair = _pair_key(team_a, team_b)
        prior_meetings = h2h[(gender, pair)]
        out["h2h_matches_played"] = len(prior_meetings)
        if prior_meetings:
            a_wins = sum(1 for _, winner_team in prior_meetings if winner_team == team_a)
            out["h2h_win_rate_a"] = a_wins / len(prior_meetings)
        else:
            out["h2h_win_rate_a"] = None

        rows.append(out)

        # Actualizar historiales DESPUÉS de calcular las features de esta fila.
        hist_a.records.append(_TeamRecord(date, winner == "team_a", sets_a, sets_b))
        hist_b.records.append(_TeamRecord(date, winner == "team_b", sets_b, sets_a))
        winner_team = team_a if winner == "team_a" else team_b
        h2h[(gender, pair)].append((date, winner_team))

    features_df = pd.DataFrame(rows, index=working.index)
    result = pd.concat([working, features_df], axis=1)
    result = result.sort_values("_orig_order", kind="stable").drop(columns=["_orig_order", "_date_sort"])
    result.index = df.index
    return result
