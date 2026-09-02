"""
Fase 7 — modelo de diferencia de puntos esperada.

Compara, sobre los mismos folds walk-forward de siempre:
  - mean_baseline: predice la media de point_diff del train de cada fold.
  - linear_elo: regresión lineal simple de point_diff sobre elo_diff.
  - catboost: CatBoost en modo regresión sobre todas las features de Fase 3.

Uso:
    python scripts/run_phase7_point_diff.py
    python scripts/run_phase7_point_diff.py --n-folds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import summarize_regression
from src.evaluation.walk_forward import make_walk_forward_folds
from src.features.build_features import FEATURE_COLUMNS, build_features
from src.models.point_diff import LinearEloBaseline, MeanBaseline, PointDiffModel, compute_point_diff
from src.normalization.competitions import CompetitionCatalog
from src.ratings.elo import EloConfig


def run_for_gender(df_gender: pd.DataFrame, gender_label: str, n_folds: int, competition_catalog: CompetitionCatalog) -> None:
    print(f"\n{'=' * 70}\n{gender_label.upper()} — {len(df_gender)} partidos\n{'=' * 70}")

    features_df = build_features(df_gender, competition_catalog, EloConfig())
    folds = make_walk_forward_folds(features_df, n_folds=n_folds)

    model_classes = {"mean_baseline": MeanBaseline, "linear_elo": LinearEloBaseline, "catboost": PointDiffModel}

    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_df = features_df.loc[train_idx]
        test_df = features_df.loc[test_idx]
        y_true = compute_point_diff(test_df).to_numpy()

        for name, cls in model_classes.items():
            model = cls().fit(train_df, FEATURE_COLUMNS)
            y_pred = model.predict(test_df, FEATURE_COLUMNS)
            metrics = summarize_regression(y_true, y_pred)
            metrics.update({"model": name, "fold": fold_i, "train_size": len(train_idx)})
            rows.append(metrics)

    results = pd.DataFrame(rows)[["model", "fold", "train_size", "n", "mae", "rmse"]]
    print("\nPor fold:")
    print(results.to_string(index=False))
    print("\nResumen (media entre folds):")
    print(results.groupby("model")[["mae", "rmse"]].mean().round(3))

    # Puntos totales típicos de un partido, para dar contexto a los MAE/RMSE.
    typical_total_points = (df_gender["team_a_score"] + df_gender["team_b_score"]).median()
    print(f"\n(Referencia: mediana de puntos totales por partido = {typical_total_points:.0f})")


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
