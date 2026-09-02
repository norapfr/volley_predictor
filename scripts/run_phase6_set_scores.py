"""
Fase 6 — modelo de marcador de sets.

Combina el modelo de "quién gana" (CatBoost, Fase 5, sin calibrar — el
ganador de esa fase) con un modelo nuevo de "margen" (3-0/3-1/3-2) para
dar las 6 probabilidades de marcador exacto de cada partido.

Uso:
    python scripts/run_phase6_set_scores.py
    python scripts/run_phase6_set_scores.py --n-folds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.walk_forward import make_walk_forward_folds
from src.features.build_features import FEATURE_COLUMNS, build_features
from src.models.set_score import (
    MARGIN_CLASSES,
    SetMarginModel,
    build_margin_training_frame,
    combine_set_score_probabilities,
    margin_class_from_sets,
    orient_features,
)
from src.models.tabular import available_tabular_models
from src.normalization.competitions import CompetitionCatalog
from src.ratings.elo import EloConfig

SCORE_CLASSES = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]


def run_for_gender(df_gender: pd.DataFrame, gender_label: str, n_folds: int, competition_catalog: CompetitionCatalog) -> None:
    print(f"\n{'=' * 70}\n{gender_label.upper()} — {len(df_gender)} partidos\n{'=' * 70}")

    tabular_models = available_tabular_models()
    if "catboost" not in tabular_models:
        print("⚠ catboost no está instalado. `pip install catboost`.")
        return

    features_df = build_features(df_gender, competition_catalog, EloConfig())
    folds = make_walk_forward_folds(features_df, n_folds=n_folds)

    fold_rows = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_df = features_df.loc[train_idx]
        test_df = features_df.loc[test_idx]

        # Modelo de ganador (igual que Fase 5: CatBoost solo, sin calibrar).
        win_model = tabular_models["catboost"]().fit(train_df, FEATURE_COLUMNS)
        p_a_wins = win_model.predict_proba_batch(test_df, FEATURE_COLUMNS)

        # Modelo de margen, entrenado sobre TODO train_df reorientado al ganador real.
        margin_train = build_margin_training_frame(train_df, FEATURE_COLUMNS)
        margin_model = SetMarginModel().fit(margin_train, FEATURE_COLUMNS)

        oriented_a = orient_features(test_df, FEATURE_COLUMNS, from_a_perspective=True)
        oriented_b = orient_features(test_df, FEATURE_COLUMNS, from_a_perspective=False)
        margin_if_a = margin_model.predict_proba(oriented_a)
        margin_if_b = margin_model.predict_proba(oriented_b)

        combined = combine_set_score_probabilities(p_a_wins, margin_if_a, margin_if_b)

        y_true_labels = [
            f"{a}-{b}" for a, b in zip(test_df["sets_a"], test_df["sets_b"])
        ]

        # sklearn.metrics.log_loss asume orden ALFABÉTICO de columnas en y_pred,
        # sin importar el orden de `labels=` que se le pase — es un detalle poco
        # documentado que, si se ignora, compara cada probabilidad con la clase
        # equivocada (produce un log loss peor que el azar, sin avisar de forma
        # clara más que con un UserWarning fácil de pasar por alto).
        sorted_classes = sorted(SCORE_CLASSES)
        y_pred_matrix = combined[sorted_classes].to_numpy()
        y_pred_matrix = y_pred_matrix / y_pred_matrix.sum(axis=1, keepdims=True)  # por seguridad numérica

        fold_log_loss = log_loss(y_true_labels, y_pred_matrix, labels=sorted_classes)
        predicted_labels = combined[SCORE_CLASSES].idxmax(axis=1).to_numpy()
        fold_accuracy = accuracy_score(y_true_labels, predicted_labels)

        fold_rows.append(
            {"fold": fold_i, "train_size": len(train_idx), "n": len(test_idx), "log_loss": fold_log_loss, "accuracy": fold_accuracy}
        )

    results = pd.DataFrame(fold_rows)
    print("\nPor fold (predicción de marcador exacto, 6 clases):")
    print(results.to_string(index=False))
    print(f"\nMedia log_loss: {results['log_loss'].mean():.4f}  (azar uniforme entre 6 = {np.log(6):.4f})")
    print(f"Media accuracy: {results['accuracy'].mean():.4f}  (azar uniforme entre 6 = {1/6:.4f})")

    # Distribución real vs media predicha de cada marcador, sobre el último fold (más entrenamiento).
    print("\nMarcador exacto — distribución real vs. media predicha (último fold):")
    last_test_idx = folds[-1][1]
    last_test = features_df.loc[last_test_idx]
    real_dist = pd.Series(
        [f"{a}-{b}" for a, b in zip(last_test["sets_a"], last_test["sets_b"])]
    ).value_counts(normalize=True).reindex(SCORE_CLASSES).fillna(0)
    print(pd.DataFrame({"frecuencia_real": real_dist, "media_predicha": combined[SCORE_CLASSES].mean()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=ROOT / "data" / "processed" / "matches.csv")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.matches)
    competition_catalog = CompetitionCatalog.from_csv(ROOT / "configs" / "competitions_seed.csv")

    for gender in sorted(df["gender"].unique()):
        subset = df[df["gender"] == gender].reset_index(drop=True)
        run_for_gender(subset, gender, args.n_folds, competition_catalog)


if __name__ == "__main__":
    main()
