"""
Modelo de diferencia de puntos — Fase 7.

Regresión: predice `team_a_score - team_b_score` (puntos totales del
partido, sumando los 3-5 sets), a partir de las mismas features de la Fase
3. Complementa al modelo de "quién gana" (Fase 5) y al de marcador de sets
(Fase 6) con la magnitud esperada de la diferencia — parte de lo que pide
la spec ("diferencia de puntos esperada").

Se comparan dos referencias sencillas para saber si el modelo tabular
aporta de verdad, con la misma disciplina de toda la Fase 4/5:
  - `mean_baseline`: predice siempre la media de point_diff del train de
    cada fold (referencia mínima — "no sé nada, uso el promedio").
  - `linear_elo`: regresión lineal de point_diff sobre elo_diff — Elo ya
    predice bastante bien el margen de forma aproximadamente lineal; si el
    modelo tabular no le gana con margen claro, no compensa la complejidad.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_point_diff(df: pd.DataFrame) -> pd.Series:
    return df["team_a_score"] - df["team_b_score"]


class MeanBaseline:
    def __init__(self):
        self._mean = 0.0

    def fit(self, train_df: pd.DataFrame, feature_columns: List[str]) -> "MeanBaseline":
        self._mean = float(compute_point_diff(train_df).mean())
        return self

    def predict(self, test_df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        return np.full(len(test_df), self._mean)


class LinearEloBaseline:
    def __init__(self):
        self._model = LinearRegression()

    def fit(self, train_df: pd.DataFrame, feature_columns: List[str]) -> "LinearEloBaseline":
        X = train_df[["elo_diff"]]
        y = compute_point_diff(train_df)
        self._model.fit(X, y)
        return self

    def predict(self, test_df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        return self._model.predict(test_df[["elo_diff"]])


class PointDiffModel:
    """CatBoost en modo regresión sobre las mismas features de la Fase 3."""

    def __init__(self, **params):
        from catboost import CatBoostRegressor

        default_params = dict(depth=4, learning_rate=0.05, iterations=200, l2_leaf_reg=5, verbose=False)
        default_params.update(params)
        self._model = CatBoostRegressor(**default_params)
        self._feature_columns: List[str] = []

    def fit(self, train_df: pd.DataFrame, feature_columns: List[str]) -> "PointDiffModel":
        self._feature_columns = feature_columns
        X = train_df[feature_columns]
        y = compute_point_diff(train_df)
        self._model.fit(X, y)
        return self

    def predict(self, test_df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        return self._model.predict(test_df[feature_columns])

    def save(self, path) -> None:
        import json
        from pathlib import Path

        path = Path(path)
        self._model.save_model(str(path))
        path.with_suffix(path.suffix + ".features.json").write_text(json.dumps(self._feature_columns))

    @classmethod
    def load(cls, path) -> "PointDiffModel":
        import json
        from pathlib import Path

        from catboost import CatBoostRegressor

        path = Path(path)
        instance = cls.__new__(cls)
        instance._model = CatBoostRegressor()
        instance._model.load_model(str(path))
        instance._feature_columns = json.loads(path.with_suffix(path.suffix + ".features.json").read_text())
        return instance
