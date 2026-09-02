"""
Fase 5 — calibración y ensemble.

Dos preguntas, sobre los MISMOS folds walk-forward que ya usan las Fases
2/4, para que la comparación sea justa:

1. ¿Las probabilidades del mejor modelo de la Fase 4 ya están bien
   calibradas, o conviene recalibrarlas (Platt scaling / isotonic
   regression)?
2. ¿Combinar los dos mejores modelos tabulares (media) mejora sobre usar
   solo el mejor?

Uso:
    python scripts/run_phase5_calibration_ensemble.py
    python scripts/run_phase5_calibration_ensemble.py --n-folds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import calibration_table, summarize
from src.evaluation.walk_forward import evaluate_walk_forward, make_walk_forward_folds, summarize_across_folds
from src.features.build_features import FEATURE_COLUMNS, build_features
from src.models.calibration import IsotonicCalibrator, PlattCalibrator
from src.models.ensemble import make_average_ensemble_factory, make_calibrated_factory
from src.models.tabular import available_tabular_models
from src.normalization.competitions import CompetitionCatalog
from src.ratings.elo import EloConfig


def _tabular_factory(model_cls, feature_columns):
    def factory(train_df: pd.DataFrame):
        model = model_cls().fit(train_df, feature_columns)

        def predict(test_df: pd.DataFrame):
            return model.predict_proba_batch(test_df, feature_columns)

        return predict

    return factory


def _pooled_test_predictions(df: pd.DataFrame, factory, n_folds: int) -> tuple[np.ndarray, np.ndarray]:
    """Predicciones crudas de TODOS los folds de test concatenadas, para ver un reliability diagram global."""
    folds = make_walk_forward_folds(df, n_folds=n_folds)
    all_preds, all_true = [], []
    for train_idx, test_idx in folds:
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]
        predict_fn = factory(train_df)
        all_preds.append(predict_fn(test_df))
        all_true.append((test_df["winner"] == "team_a").astype(float).to_numpy())
    return np.concatenate(all_preds), np.concatenate(all_true)


def run_for_gender(df_gender: pd.DataFrame, gender_label: str, n_folds: int, competition_catalog: CompetitionCatalog) -> pd.DataFrame:
    print(f"\n{'=' * 70}\n{gender_label.upper()} — {len(df_gender)} partidos\n{'=' * 70}")

    features_df = build_features(df_gender, competition_catalog, EloConfig())

    tabular_models = available_tabular_models()
    if "catboost" not in tabular_models:
        print("⚠ catboost no está instalado — la Fase 5 lo usa como modelo base principal. `pip install catboost`.")
        return pd.DataFrame()

    catboost_factory = _tabular_factory(tabular_models["catboost"], FEATURE_COLUMNS)

    predictors = {"catboost": catboost_factory}

    if "xgboost" in tabular_models:
        xgboost_factory = _tabular_factory(tabular_models["xgboost"], FEATURE_COLUMNS)
        predictors["xgboost"] = xgboost_factory
        predictors["ensemble_avg"] = make_average_ensemble_factory(
            {"catboost": catboost_factory, "xgboost": xgboost_factory}
        )

    predictors["catboost_platt"] = make_calibrated_factory(catboost_factory, PlattCalibrator)
    predictors["catboost_isotonic"] = make_calibrated_factory(catboost_factory, IsotonicCalibrator)

    results = evaluate_walk_forward(features_df, predictors, n_folds=n_folds)
    print("\nPor fold:")
    print(results.to_string(index=False))
    print("\nResumen (media ± desviación estándar entre folds):")
    print(summarize_across_folds(results))

    # Reliability diagram (tabla de calibración) de catboost crudo vs calibrado, sobre todos los folds de test.
    raw_preds, y_true = _pooled_test_predictions(features_df, catboost_factory, n_folds)
    print("\nCalibración de catboost SIN ajustar (probabilidad predicha vs tasa real de victoria):")
    print(calibration_table(y_true, raw_preds).to_string(index=False))
    print(f"  Log loss agregado: {summarize(y_true, raw_preds)['log_loss']:.4f}")

    platt_factory = make_calibrated_factory(catboost_factory, PlattCalibrator)
    calib_preds, y_true_2 = _pooled_test_predictions(features_df, platt_factory, n_folds)
    print("\nCalibración de catboost CON Platt scaling:")
    print(calibration_table(y_true_2, calib_preds).to_string(index=False))
    print(f"  Log loss agregado: {summarize(y_true_2, calib_preds)['log_loss']:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=ROOT / "data" / "processed" / "matches.csv")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.matches)
    competition_catalog = CompetitionCatalog.from_csv(ROOT / "configs" / "competitions_seed.csv")

    all_results = []
    for gender in sorted(df["gender"].unique()):
        subset = df[df["gender"] == gender].reset_index(drop=True)
        results = run_for_gender(subset, gender, args.n_folds, competition_catalog)
        if len(results) > 0:
            results["gender"] = gender
            all_results.append(results)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = ROOT / "reports" / "phase5_calibration_ensemble_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        print(f"\nResultados completos guardados en: {out_path}")


if __name__ == "__main__":
    main()
