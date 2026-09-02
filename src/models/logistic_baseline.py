"""
Regresión logística sobre diferencia de Elo — Fase 2 (baselines).

El Elo puro asume una escala fija (400) para convertir diferencia de rating
en probabilidad. Este modelo deja que la regresión logística aprenda esa
escala (y el intercepto) a partir de los datos de entrenamiento, en vez de
asumirla — es habitual que la escala óptima real no sea exactamente 400.

Requiere que el DataFrame ya tenga `elo_a_pre`/`elo_b_pre` calculados por
`src.ratings.elo.compute_elo_history` — ese cálculo es el que garantiza que
no hay leakage (cada valor solo depende de partidos anteriores). Este
modelo en sí mismo se entrena por fold de walk-forward, solo con partidos
de la ventana de entrenamiento de ese fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class EloLogisticModel:
    def __init__(self):
        self._model: LogisticRegression | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        winner_col: str = "winner",
    ) -> "EloLogisticModel":
        diff = (train_df["elo_a_pre"] - train_df["elo_b_pre"]).to_numpy().reshape(-1, 1)
        y = (train_df[winner_col] == "team_a").astype(float).to_numpy()

        if len(set(y)) < 2:
            self._model = None  # sin variedad de resultados, no hay nada que ajustar
            return self

        self._model = LogisticRegression(solver="lbfgs")
        self._model.fit(diff, y)
        return self

    def predict_proba_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        diff = (test_df["elo_a_pre"] - test_df["elo_b_pre"]).to_numpy().reshape(-1, 1)
        if self._model is None:
            return np.full(len(test_df), 0.5)
        return self._model.predict_proba(diff)[:, 1]
