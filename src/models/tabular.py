"""
Modelos tabulares — Fase 4 (LightGBM/XGBoost/CatBoost).

Los tres comparten la misma interfaz que ya usa Fase 2
(`fit(train_df)` -> objeto -> `predict_proba_batch(test_df)`), así que
enchufan directo en `src.evaluation.walk_forward.evaluate_walk_forward`
sin tocar ese módulo.

Cada wrapper importa su librería de forma perezosa (dentro de `__init__`,
no a nivel de módulo) para que el archivo se pueda importar aunque falte
alguna librería — `available_tabular_models()` es lo que decide en tiempo
de ejecución cuáles están realmente disponibles, y el script de Fase 4 usa
solo esas, avisando de las que faltan en vez de reventar.

Hiperparámetros deliberadamente modestos (profundidad/hojas limitadas):
con ~1000-3000 filas de entrenamiento por fold, un árbol demasiado
expresivo memoriza en vez de generalizar. Ajuste fino de hiperparámetros
es tarea de Fase 5 (calibración/ensemble), no de aquí.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd


class _BaseTabularModel:
    """Interfaz común. No instanciar directamente."""

    def fit(self, train_df: pd.DataFrame, feature_columns: List[str], winner_col: str = "winner") -> "_BaseTabularModel":
        raise NotImplementedError

    def predict_proba_batch(self, test_df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        raise NotImplementedError

    def feature_importance(self) -> Optional[pd.Series]:
        """Serie ordenada de mayor a menor importancia, o None si el modelo no está entrenado."""
        raise NotImplementedError


class LightGBMModel(_BaseTabularModel):
    def __init__(self, **params):
        import lightgbm as lgb  # import perezoso: falla aquí, no al importar el módulo

        default_params = dict(
            num_leaves=15,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=200,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1,
        )
        default_params.update(params)
        self._model = lgb.LGBMClassifier(**default_params)
        self._feature_columns: List[str] = []

    def fit(self, train_df, feature_columns, winner_col="winner"):
        self._feature_columns = feature_columns
        X = train_df[feature_columns]
        y = (train_df[winner_col] == "team_a").astype(int)
        self._model.fit(X, y)
        return self

    def predict_proba_batch(self, test_df, feature_columns):
        X = test_df[feature_columns]
        return self._model.predict_proba(X)[:, 1]

    def feature_importance(self):
        if not self._feature_columns:
            return None
        return pd.Series(self._model.feature_importances_, index=self._feature_columns).sort_values(ascending=False)


class XGBoostModel(_BaseTabularModel):
    def __init__(self, **params):
        import xgboost as xgb

        default_params = dict(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            eval_metric="logloss",
        )
        default_params.update(params)
        self._model = xgb.XGBClassifier(**default_params)
        self._feature_columns: List[str] = []

    def fit(self, train_df, feature_columns, winner_col="winner"):
        self._feature_columns = feature_columns
        X = train_df[feature_columns]
        y = (train_df[winner_col] == "team_a").astype(int)
        self._model.fit(X, y)
        return self

    def predict_proba_batch(self, test_df, feature_columns):
        X = test_df[feature_columns]
        return self._model.predict_proba(X)[:, 1]

    def feature_importance(self):
        if not self._feature_columns:
            return None
        return pd.Series(self._model.feature_importances_, index=self._feature_columns).sort_values(ascending=False)


class CatBoostModel(_BaseTabularModel):
    def __init__(self, **params):
        from catboost import CatBoostClassifier

        default_params = dict(
            depth=4,
            learning_rate=0.05,
            iterations=200,
            l2_leaf_reg=5,
            verbose=False,
        )
        default_params.update(params)
        self._model = CatBoostClassifier(**default_params)
        self._feature_columns: List[str] = []

    def fit(self, train_df, feature_columns, winner_col="winner"):
        self._feature_columns = feature_columns
        X = train_df[feature_columns]
        y = (train_df[winner_col] == "team_a").astype(int)
        self._model.fit(X, y)
        return self

    def predict_proba_batch(self, test_df, feature_columns):
        X = test_df[feature_columns]
        return self._model.predict_proba(X)[:, 1]

    def feature_importance(self):
        if not self._feature_columns:
            return None
        return pd.Series(self._model.feature_importances_, index=self._feature_columns).sort_values(ascending=False)

    def save(self, path) -> None:
        """Guarda el modelo (formato nativo CatBoost) + las columnas de features usadas, como sidecar JSON."""
        import json
        from pathlib import Path

        path = Path(path)
        self._model.save_model(str(path))
        path.with_suffix(path.suffix + ".features.json").write_text(json.dumps(self._feature_columns))

    @classmethod
    def load(cls, path) -> "CatBoostModel":
        import json
        from pathlib import Path

        from catboost import CatBoostClassifier

        path = Path(path)
        instance = cls.__new__(cls)
        instance._model = CatBoostClassifier()
        instance._model.load_model(str(path))
        instance._feature_columns = json.loads(path.with_suffix(path.suffix + ".features.json").read_text())
        return instance


_ALL_MODELS: Dict[str, Type[_BaseTabularModel]] = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
}


def available_tabular_models() -> Dict[str, Type[_BaseTabularModel]]:
    """
    Devuelve solo las clases cuya librería está realmente instalada,
    intentando instanciar cada una con parámetros por defecto.
    """
    available = {}
    for name, cls in _ALL_MODELS.items():
        try:
            cls()  # el import perezoso vive en __init__; si falla, no está instalada
            available[name] = cls
        except ImportError:
            continue
    return available
