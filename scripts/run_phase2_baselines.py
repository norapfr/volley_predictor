"""
Fase 2 — baselines: Elo, Bradley-Terry, regresión logística sobre Elo.
Validación walk-forward, modelos separados por género (sección de la spec).

Uso:
    python scripts/run_phase2_baselines.py
    python scripts/run_phase2_baselines.py --matches data/processed/matches.csv --n-folds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.walk_forward import evaluate_walk_forward, summarize_across_folds
from src.models.bradley_terry import BradleyTerryModel
from src.models.logistic_baseline import EloLogisticModel
from src.ratings.elo import EloConfig, compute_elo_history


def _elo_predictor_factory(train_df: pd.DataFrame):
    # Elo ya viene precalculado de forma segura en el propio DataFrame (elo_pred_a);
    # no hace falta "entrenar" nada por fold, así que se ignora train_df.
    def predict(test_df: pd.DataFrame):
        return test_df["elo_pred_a"].to_numpy()

    return predict


def _bradley_terry_factory(train_df: pd.DataFrame):
    model = BradleyTerryModel(C=1.0)
    model.fit(train_df)

    def predict(test_df: pd.DataFrame):
        return model.predict_proba_batch(test_df)

    return predict


def _logistic_elo_factory(train_df: pd.DataFrame):
    model = EloLogisticModel()
    model.fit(train_df)

    def predict(test_df: pd.DataFrame):
        return model.predict_proba_batch(test_df)

    return predict


def run_for_gender(df_gender: pd.DataFrame, gender_label: str, n_folds: int) -> pd.DataFrame:
    print(f"\n{'=' * 60}\n{gender_label.upper()} — {len(df_gender)} partidos\n{'=' * 60}")

    history_df, _final_ratings = compute_elo_history(df_gender, EloConfig())

    predictors = {
        "elo": _elo_predictor_factory,
        "bradley_terry": _bradley_terry_factory,
        "logistic_elo": _logistic_elo_factory,
    }

    results = evaluate_walk_forward(history_df, predictors, n_folds=n_folds)
    print("\nPor fold:")
    print(results.to_string(index=False))

    print("\nResumen (media ± desviación estándar entre folds — estabilidad temporal):")
    print(summarize_across_folds(results))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=ROOT / "data" / "processed" / "matches.csv")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.matches)
    required = {"date", "gender", "team_a", "team_b", "winner"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Faltan columnas requeridas en {args.matches}: {missing}")

    all_results = []
    for gender in sorted(df["gender"].unique()):
        subset = df[df["gender"] == gender].reset_index(drop=True)
        results = run_for_gender(subset, gender, args.n_folds)
        results["gender"] = gender
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    out_path = ROOT / "reports" / "phase2_baseline_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nResultados completos guardados en: {out_path}")


if __name__ == "__main__":
    main()
