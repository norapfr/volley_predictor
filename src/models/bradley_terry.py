"""
Modelo Bradley-Terry — Fase 2 (baselines).

Implementado como regresión logística sin intercepto sobre variables
indicadoras por equipo (+1 en la posición de team_a, -1 en la de team_b):
es la forma exacta de ajustar Bradley-Terry por máxima verosimilitud, y
así se reutiliza `sklearn` en vez de reimplementar el algoritmo iterativo
clásico (Zermelo/MM). La regularización L2 de `LogisticRegression` actúa
como un prior gaussiano suave sobre la fuerza de cada equipo — necesario
porque, sin ella, un equipo con pocos partidos y 100% de victorias tendría
una fuerza estimada infinita.

Se entrena una vez por fold de walk-forward (ventana de entrenamiento que
crece con el tiempo), nunca sobre partidos futuros respecto al fold.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class BradleyTerryModel:
    def __init__(self, C: float = 1.0):
        self.C = C
        self.strengths_: Dict[str, float] = {}
        self._team_index: Dict[str, int] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        team_a_col: str = "team_a",
        team_b_col: str = "team_b",
        winner_col: str = "winner",
    ) -> "BradleyTerryModel":
        teams: List[str] = sorted(set(train_df[team_a_col]) | set(train_df[team_b_col]))
        self._team_index = {team: i for i, team in enumerate(teams)}
        n_teams = len(teams)
        n_matches = len(train_df)

        X = np.zeros((n_matches, n_teams), dtype=float)
        y = np.zeros(n_matches, dtype=float)

        for i, row in enumerate(train_df.itertuples(index=False)):
            team_a = getattr(row, team_a_col)
            team_b = getattr(row, team_b_col)
            winner = getattr(row, winner_col)
            X[i, self._team_index[team_a]] += 1.0
            X[i, self._team_index[team_b]] -= 1.0
            y[i] = 1.0 if winner == "team_a" else 0.0

        if n_teams == 0 or len(set(y)) < 2:
            # Sin variedad de resultados (o sin equipos) no hay nada que ajustar;
            # todas las fuerzas quedan en 0 (equivalente a "todos iguales").
            self.strengths_ = {team: 0.0 for team in teams}
            return self

        model = LogisticRegression(fit_intercept=False, C=self.C, solver="lbfgs", max_iter=1000)
        model.fit(X, y)
        self.strengths_ = {team: float(model.coef_[0][idx]) for team, idx in self._team_index.items()}
        return self

    def predict_proba(self, team_a: str, team_b: str) -> float:
        """P(team_a gana). Equipos no vistos en el entrenamiento reciben fuerza 0 (promedio)."""
        strength_a = self.strengths_.get(team_a, 0.0)
        strength_b = self.strengths_.get(team_b, 0.0)
        diff = strength_a - strength_b
        return 1.0 / (1.0 + np.exp(-diff))

    def predict_proba_batch(self, test_df: pd.DataFrame, team_a_col: str = "team_a", team_b_col: str = "team_b") -> np.ndarray:
        return np.array(
            [self.predict_proba(a, b) for a, b in zip(test_df[team_a_col], test_df[team_b_col])]
        )
