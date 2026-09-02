"""
Métricas de evaluación — Fase 2.

Orden de prioridad según la spec (sección "validación"): Log Loss, Brier
Score, calibración, estabilidad temporal, y solo después accuracy — un
modelo con buena accuracy pero mal calibrado (p.ej. siempre dice 90% de
confianza y acierta el 60% de las veces) es peor para este proyecto que uno
con accuracy algo menor pero probabilidades fiables, porque el producto
final expone probabilidades al usuario, no solo un ganador.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-15


def log_loss(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    p = np.clip(y_pred_proba, _EPS, 1 - _EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    return float(np.mean((y_pred_proba - y_true) ** 2))


def accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> float:
    preds = (y_pred_proba >= threshold).astype(float)
    return float(np.mean(preds == y_true))


def calibration_table(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Para cada bin de probabilidad predicha, compara la media predicha con la
    tasa real de aciertos — un modelo bien calibrado tiene ambas columnas
    prácticamente iguales en cada fila.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_proba})
    df["bin"] = pd.cut(df["y_pred"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        predicted_mean=("y_pred", "mean"),
        actual_rate=("y_true", "mean"),
        count=("y_true", "size"),
    )
    return grouped.reset_index()


def summarize(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    return {
        "log_loss": log_loss(y_true, y_pred_proba),
        "brier_score": brier_score(y_true, y_pred_proba),
        "accuracy": accuracy(y_true, y_pred_proba),
        "n": len(y_true),
    }


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Error absoluto medio — en las mismas unidades que la variable (puntos), fácil de interpretar."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raíz del error cuadrático medio — penaliza más los fallos grandes que el MAE."""
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def summarize_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "n": len(y_true)}
