"""
Validación walk-forward (expanding window) — Fase 2.

La spec prohíbe explícitamente split aleatorio (sección "validación"): cada
fold entrena solo con partidos estrictamente anteriores a los que evalúa,
igual que pasaría en producción (nunca se tiene el futuro para predecir el
pasado). `make_walk_forward_folds` construye los folds por FECHA, no por
posición arbitraria, y `evaluate_walk_forward` verifica en tiempo de
ejecución que ningún fold viola esa regla — si algún día un cambio de
código rompe esto, falla ruidosamente aquí en vez de colar un número de
validación optimista y engañoso.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import summarize

# Firma de un "predictor": recibe el DataFrame de entrenamiento del fold y
# devuelve una función que, dado el DataFrame de test del fold, produce un
# array de P(gana team_a) por fila. Elo no necesita entrenar nada (ya viene
# precalculado en el propio DataFrame de forma segura, ver src/ratings/elo.py)
# así que su factory simplemente ignora train_df.
PredictorFactory = Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]]


def make_walk_forward_folds(
    df: pd.DataFrame,
    date_col: str = "date",
    n_folds: int = 5,
    min_train_fraction: float = 0.3,
) -> List[Tuple[pd.Index, pd.Index]]:
    """
    Devuelve una lista de (train_index, test_index), ordenados por fecha,
    con ventana de entrenamiento creciente (expanding window): el fold i
    entrena con todo lo anterior al bloque de test i.

    `min_train_fraction`: fracción del histórico total reservada como
    "calentamiento" antes del primer fold — con muy poco entrenamiento
    inicial, el primer fold sería puro ruido (p.ej. Bradley-Terry con 3
    equipos vistos no significa nada).
    """
    ordered = df.sort_values(date_col, kind="stable")
    n = len(ordered)
    train_start = max(1, int(n * min_train_fraction))
    remaining = n - train_start
    if remaining < n_folds:
        raise ValueError(
            f"No hay suficientes partidos ({n}) para {n_folds} folds con "
            f"min_train_fraction={min_train_fraction}. Reduce n_folds o min_train_fraction."
        )
    fold_size = remaining // n_folds

    folds = []
    for i in range(n_folds):
        test_start = train_start + i * fold_size
        test_end = train_start + (i + 1) * fold_size if i < n_folds - 1 else n
        train_idx = ordered.index[:test_start]
        test_idx = ordered.index[test_start:test_end]
        folds.append((train_idx, test_idx))
    return folds


def _assert_no_temporal_leakage(df: pd.DataFrame, train_idx: pd.Index, test_idx: pd.Index, date_col: str) -> None:
    if len(train_idx) == 0 or len(test_idx) == 0:
        return
    max_train_date = pd.to_datetime(df.loc[train_idx, date_col]).max()
    min_test_date = pd.to_datetime(df.loc[test_idx, date_col]).min()
    if max_train_date > min_test_date:
        raise AssertionError(
            f"Leakage temporal detectado: fecha máxima de train ({max_train_date}) "
            f"es posterior a la fecha mínima de test ({min_test_date})."
        )


def evaluate_walk_forward(
    df: pd.DataFrame,
    predictors: Dict[str, PredictorFactory],
    n_folds: int = 5,
    min_train_fraction: float = 0.3,
    date_col: str = "date",
    winner_col: str = "winner",
) -> pd.DataFrame:
    """
    Evalúa cada predictor en `predictors` sobre los mismos folds walk-forward.
    Devuelve un DataFrame con una fila por (modelo, fold) y las métricas de
    la sección "validación" de la spec: log_loss, brier_score, accuracy, n.
    """
    folds = make_walk_forward_folds(df, date_col=date_col, n_folds=n_folds, min_train_fraction=min_train_fraction)
    rows = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        _assert_no_temporal_leakage(df, train_idx, test_idx, date_col)
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]
        y_true = (test_df[winner_col] == "team_a").astype(float).to_numpy()

        for name, factory in predictors.items():
            predict_fn = factory(train_df)
            y_pred = predict_fn(test_df)
            metrics = summarize(y_true, np.asarray(y_pred, dtype=float))
            metrics["model"] = name
            metrics["fold"] = fold_i
            metrics["train_size"] = len(train_idx)
            rows.append(metrics)

    return pd.DataFrame(rows)[["model", "fold", "train_size", "n", "log_loss", "brier_score", "accuracy"]]


def summarize_across_folds(results: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación estándar por modelo a través de los folds — mide estabilidad temporal (spec)."""
    return (
        results.groupby("model")[["log_loss", "brier_score", "accuracy"]]
        .agg(["mean", "std"])
        .round(4)
    )
