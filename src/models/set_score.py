"""
Modelo de marcador de sets — Fase 6.

Estrategia: no se entrena un clasificador de 6 clases directo (3-0, 3-1,
3-2, 2-3, 1-3, 0-3) desde cero. En vez de eso se descompone en dos piezas
que ya conocemos bien:

1. P(gana A) — el modelo de la Fase 5 (CatBoost), ya validado y calibrado.
2. P(margen | quién gana) — un modelo NUEVO, más simple (3 clases: 3-0,
   3-1, 3-2), entrenado sobre TODOS los partidos reorientados desde la
   perspectiva del ganador ("features del ganador menos las del
   perdedor"), en vez de "equipo A menos equipo B".

La combinación da las 6 probabilidades finales:
    P(A gana 3-0) = P(gana A) * P(margen=3-0 | A es el ganador)
    P(B gana 3-0) = P(gana B) * P(margen=3-0 | B es el ganador)
    ... y así con 3-1 y 3-2.

Por qué reorientar en vez de entrenar dos modelos (uno por cada mitad de
los datos, "cuando gana A" / "cuando gana B"): reorientar deja usar TODOS
los partidos para un único modelo de margen, en vez de tirar la mitad —
con ~3300 partidos por género, cada partido cuenta.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# Features cuyo signo depende de si se mira desde la perspectiva de A o de B
# (p.ej. elo_diff = elo_a - elo_b; desde la perspectiva del ganador, si ganó
# B, el "elo_diff del ganador" es -elo_diff).
ANTISYMMETRIC_FEATURES = [
    "elo_diff",
    "matches_played_diff",
    "current_streak_diff",
    "days_since_last_diff",
    "win_rate_last5_diff",
    "win_rate_last10_diff",
    "set_margin_avg_last5_diff",
    "set_margin_avg_last10_diff",
]

# Features en escala [0,1] tipo tasa — su "versión B" es el complementario
# (1 - x), no el negativo, porque son probabilidades/proporciones, no diferencias.
COMPLEMENT_FEATURES = ["h2h_win_rate_a"]

# Features que no dependen de quién es "A" o "B" — se dejan tal cual.
SYMMETRIC_FEATURES = ["h2h_matches_played", "competition_importance", "tournament_progress"]

MARGIN_CLASSES = ["3-0", "3-1", "3-2"]


def margin_class_from_sets(sets_a: int, sets_b: int) -> str:
    loser_sets = min(sets_a, sets_b)
    return f"3-{loser_sets}"


def orient_features(features_df: pd.DataFrame, feature_columns: List[str], from_a_perspective: bool) -> pd.DataFrame:
    """
    Devuelve una copia de las columnas de features reorientadas: si
    `from_a_perspective` es True, se dejan tal cual (A es la referencia);
    si es False, se invierten como si B fuera la referencia (el "ganador
    hipotético" para el que se quiere el margen).
    """
    out = features_df[feature_columns].copy()
    if from_a_perspective:
        return out
    for col in feature_columns:
        if col in ANTISYMMETRIC_FEATURES:
            out[col] = -out[col]
        elif col in COMPLEMENT_FEATURES:
            out[col] = 1.0 - out[col]
        # SYMMETRIC_FEATURES y cualquier otra no listada: se dejan igual.
    return out


def build_margin_training_frame(features_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Una fila por partido, con las features orientadas desde la perspectiva
    de quien REALMENTE ganó ese partido, y la etiqueta de margen real. Esto
    es lo que entrena `SetMarginModel` — usa el 100% de los partidos, no
    solo "los que ganó A".
    """
    is_a_winner = (features_df["winner"] == "team_a").to_numpy()
    oriented_rows = []
    for from_a in (True, False):
        mask = is_a_winner if from_a else ~is_a_winner
        subset = features_df.loc[mask]
        oriented = orient_features(subset, feature_columns, from_a_perspective=from_a)
        oriented["margin_class"] = [
            margin_class_from_sets(a, b) for a, b in zip(subset["sets_a"], subset["sets_b"])
        ]
        oriented["date"] = subset["date"].to_numpy()
        oriented_rows.append(oriented)
    result = pd.concat(oriented_rows, axis=0)
    return result.sort_index()


class SetMarginModel:
    """Clasificador de 3 clases (3-0/3-1/3-2) sobre features orientadas al ganador."""

    def __init__(self, **params):
        from catboost import CatBoostClassifier

        default_params = dict(depth=4, learning_rate=0.05, iterations=200, l2_leaf_reg=5, verbose=False)
        default_params.update(params)
        self._model = CatBoostClassifier(**default_params)
        self._feature_columns: List[str] = []

    def fit(self, margin_train_df: pd.DataFrame, feature_columns: List[str]) -> "SetMarginModel":
        self._feature_columns = feature_columns
        X = margin_train_df[feature_columns]
        y = margin_train_df["margin_class"]
        self._model.fit(X, y)
        return self

    def predict_proba(self, oriented_features: pd.DataFrame) -> pd.DataFrame:
        """Devuelve un DataFrame con una columna por clase (3-0, 3-1, 3-2), en ese orden."""
        raw = self._model.predict_proba(oriented_features[self._feature_columns])
        classes = list(self._model.classes_)
        proba_df = pd.DataFrame(raw, columns=classes, index=oriented_features.index)
        return proba_df[MARGIN_CLASSES]  # orden fijo, por si CatBoost las devuelve en otro orden

    def save(self, path) -> None:
        import json
        from pathlib import Path

        path = Path(path)
        self._model.save_model(str(path))
        path.with_suffix(path.suffix + ".features.json").write_text(json.dumps(self._feature_columns))

    @classmethod
    def load(cls, path) -> "SetMarginModel":
        import json
        from pathlib import Path

        from catboost import CatBoostClassifier

        path = Path(path)
        instance = cls.__new__(cls)
        instance._model = CatBoostClassifier()
        instance._model.load_model(str(path))
        instance._feature_columns = json.loads(path.with_suffix(path.suffix + ".features.json").read_text())
        return instance


def combine_set_score_probabilities(
    p_a_wins: np.ndarray,
    margin_probs_if_a_wins: pd.DataFrame,
    margin_probs_if_b_wins: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combina P(gana A) + P(margen | A gana) + P(margen | B gana) en las 6
    probabilidades finales de marcador exacto, nombradas como
    "sets_a-sets_b" (p.ej. "3-1" = A gana 3 a 1; "1-3" = B gana 3 a 1).
    Suman 1 por fila por construcción.
    """
    p_a_wins = np.asarray(p_a_wins, dtype=float)
    p_b_wins = 1.0 - p_a_wins
    result = pd.DataFrame(index=margin_probs_if_a_wins.index)

    for cls in MARGIN_CLASSES:  # "3-0", "3-1", "3-2" -> A gana con ese marcador
        result[cls] = p_a_wins * margin_probs_if_a_wins[cls].to_numpy()

    for cls in MARGIN_CLASSES:  # B gana con ese marcador -> marcador global es el espejo, p.ej "3-1" de B es "1-3"
        loser_sets = cls.split("-")[1]
        mirrored_label = f"{loser_sets}-3"
        result[mirrored_label] = p_b_wins * margin_probs_if_b_wins[cls].to_numpy()

    return result
