"""
Calibración de probabilidades — Fase 5.

Un modelo puede acertar el ganador bien y aun así estar mal calibrado: decir
"80% de confianza" en partidos que en realidad gana el favorito solo el 65%
de las veces. Estos calibradores reajustan la probabilidad de salida de
CUALQUIER modelo ya entrenado, sin tocar el modelo en sí.

Se entrenan SIEMPRE sobre un tramo de datos que el modelo base no vio al
entrenar — nunca sobre las mismas filas con las que se ajustó el modelo, o
el calibrador aprendería a confiar en un sobreajuste en vez de corregirlo
(ver `make_calibrated_factory` en `ensemble.py`, que separa un tramo de
"calibración" dentro del propio train de cada fold walk-forward).
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Ajusta una regresión logística 1-D sobre el logit de la probabilidad cruda."""

    def __init__(self):
        self._model: LogisticRegression | None = None

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        if len(set(y_true)) < 2:
            self._model = None
            return self
        X = _logit(np.asarray(raw_probs, dtype=float)).reshape(-1, 1)
        self._model = LogisticRegression(solver="lbfgs")
        self._model.fit(X, y_true)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.asarray(raw_probs, dtype=float)
        X = _logit(np.asarray(raw_probs, dtype=float)).reshape(-1, 1)
        return self._model.predict_proba(X)[:, 1]


class IsotonicCalibrator:
    """
    Ajusta una función monótona no paramétrica probabilidad_cruda -> probabilidad_calibrada.

    `y_min`/`y_max` se acotan lejos de 0/1 a propósito (no en [0, 1] tal
    cual): con pocos puntos de calibración — como los folds tempranos de un
    walk-forward, que pueden tener solo unas pocas decenas de filas — la
    isotónica puede devolver 0.0 o 1.0 exactos para tramos enteros. Si el
    resultado real contradice aunque sea una sola de esas predicciones
    "imposibles", el log loss se dispara (log(0) es -infinito). Acotar a
    [0.02, 0.98] evita ese colapso sin cambiar apenas nada cuando sí hay
    datos suficientes para calibrar con confianza real.
    """

    def __init__(self):
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
        self._fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        if len(set(y_true)) < 2:
            self._fitted = False
            return self
        self._model.fit(np.asarray(raw_probs, dtype=float), y_true)
        self._fitted = True
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.asarray(raw_probs, dtype=float)
        return self._model.predict(np.asarray(raw_probs, dtype=float))
