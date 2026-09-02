"""
Ensemble y calibración envueltos como "predictor factory" — Fase 5.

Ambas piezas siguen el mismo contrato que ya usa
`src.evaluation.walk_forward.evaluate_walk_forward`:
    factory(train_df) -> predict_fn(test_df) -> np.ndarray de P(gana A)

Así se pueden comparar directamente contra los modelos de las Fases 2 y 4
en los mismos folds, sin tocar el motor de validación.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

PredictorFactory = Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]]


def make_average_ensemble_factory(
    component_factories: Dict[str, PredictorFactory],
    weights: Optional[Dict[str, float]] = None,
) -> PredictorFactory:
    """
    Combina varios modelos ya existentes por media (ponderada o simple).
    No necesita entrenar nada propio — solo promedia lo que cada componente
    ya predice, así que no hay riesgo de leakage adicional más allá del que
    ya gestiona cada componente por separado.
    """
    names = list(component_factories.keys())
    w = weights or {name: 1.0 / len(names) for name in names}

    def factory(train_df: pd.DataFrame):
        fitted_predictors = {name: component_factories[name](train_df) for name in names}

        def predict(test_df: pd.DataFrame) -> np.ndarray:
            preds = np.array([fitted_predictors[name](test_df) * w[name] for name in names])
            return preds.sum(axis=0)

        return predict

    return factory


def make_calibrated_factory(
    base_factory: PredictorFactory,
    calibrator_cls,
    date_col: str = "date",
    calib_fraction: float = 0.2,
) -> PredictorFactory:
    """
    Envuelve `base_factory` con un calibrador (Platt o Isotonic).

    Dentro de cada fold walk-forward, el propio `train_df` se subdivide
    cronológicamente en dos tramos:
      - el primer (1 - calib_fraction), usado para entrenar el modelo base
        tal cual (igual que sin calibrar);
      - el último `calib_fraction`, reservado SOLO para ajustar el
        calibrador — el modelo base nunca lo ve al entrenar, así que el
        calibrador corrige sesgos reales, no memoriza el propio ajuste del
        modelo sobre esas filas.

    Sigue siendo walk-forward de principio a fin: el tramo de calibración
    es cronológicamente posterior al de ajuste del modelo, y ambos son
    anteriores al fold de test.
    """

    def factory(train_df: pd.DataFrame):
        ordered = train_df.sort_values(date_col, kind="stable")
        split_point = int(len(ordered) * (1 - calib_fraction))
        fit_part = ordered.iloc[:split_point]
        calib_part = ordered.iloc[split_point:]

        base_predict = base_factory(fit_part)

        calibrator = calibrator_cls()
        if len(calib_part) > 0:
            raw_calib_preds = base_predict(calib_part)
            y_calib = (calib_part["winner"] == "team_a").astype(float).to_numpy()
            calibrator.fit(raw_calib_preds, y_calib)

        def predict(test_df: pd.DataFrame) -> np.ndarray:
            raw = base_predict(test_df)
            return calibrator.transform(raw)

        return predict

    return factory
