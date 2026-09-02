"""
Fase 4 — ML tabular: LightGBM/XGBoost/CatBoost, comparados contra los
baselines de la Fase 2 (Elo, Bradley-Terry, logistic-sobre-Elo) sobre los
MISMOS folds walk-forward — es la única forma honesta de saber si la
complejidad extra realmente aporta o no.

Uso:
    python scripts/run_phase4_tabular.py
    python scripts/run_phase4_tabular.py --matches data/processed/matches.csv --n-folds 5

Instala solo lo que vayas a usar (evita instalar todo `[ml]` de golpe,
xgboost se queda resolviendo versiones mucho tiempo en Windows):
    pip install lightgbm      # candidato principal según la spec
    pip install xgboost       # opcional
    pip install catboost      # opcional
El script detecta automáticamente cuáles tienes instaladas y usa solo esas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.walk_forward import evaluate_walk_forward, summarize_across_folds
from src.features.build_features import FEATURE_COLUMNS, build_features
from src.models.bradley_terry import BradleyTerryModel
from src.models.logistic_baseline import EloLogisticModel
from src.models.tabular import available_tabular_models
from src.normalization.competitions import CompetitionCatalog
from src.ratings.elo import EloConfig


def _elo_predictor_factory(train_df: pd.DataFrame):
    def predict(test_df: pd.DataFrame):
        return test_df["elo_pred_a"].to_numpy()

    return predict


def _bradley_terry_factory(train_df: pd.DataFrame):
    model = BradleyTerryModel(C=1.0).fit(train_df)

    def predict(test_df: pd.DataFrame):
        return model.predict_proba_batch(test_df)

    return predict


def _logistic_elo_factory(train_df: pd.DataFrame):
    model = EloLogisticModel().fit(train_df)

    def predict(test_df: pd.DataFrame):
        return model.predict_proba_batch(test_df)

    return predict


def _tabular_factory(model_cls, feature_columns):
    def factory(train_df: pd.DataFrame):
        model = model_cls().fit(train_df, feature_columns)

        def predict(test_df: pd.DataFrame):
            return model.predict_proba_batch(test_df, feature_columns)

        return predict

    return factory


def run_for_gender(df_gender: pd.DataFrame, gender_label: str, n_folds: int, competition_catalog: CompetitionCatalog) -> pd.DataFrame:
    print(f"\n{'=' * 70}\n{gender_label.upper()} — {len(df_gender)} partidos\n{'=' * 70}")

    features_df = build_features(df_gender, competition_catalog, EloConfig())

    predictors = {
        "elo": _elo_predictor_factory,
        "bradley_terry": _bradley_terry_factory,
        "logistic_elo": _logistic_elo_factory,
    }

    tabular_models = available_tabular_models()
    if not tabular_models:
        print(
            "\n⚠ No hay ninguna librería tabular instalada (lightgbm/xgboost/catboost).\n"
            "  Solo se compararán los baselines de la Fase 2. Instala al menos\n"
            "  `pip install lightgbm` para ver el punto real de la Fase 4."
        )
    for name, cls in tabular_models.items():
        predictors[name] = _tabular_factory(cls, FEATURE_COLUMNS)

    results = evaluate_walk_forward(features_df, predictors, n_folds=n_folds)
    print("\nPor fold:")
    print(results.to_string(index=False))

    print("\nResumen (media ± desviación estándar entre folds):")
    print(summarize_across_folds(results))

    # Importancia de features del último fold (el que más entrenamiento acumula) — solo si hay modelo tabular.
    if tabular_models:
        primary_name = "lightgbm" if "lightgbm" in tabular_models else next(iter(tabular_models))
        primary_cls = tabular_models[primary_name]
        model = primary_cls().fit(features_df, FEATURE_COLUMNS)
        importance = model.feature_importance()
        if importance is not None:
            print(f"\nImportancia de features ({primary_name}, entrenado con todo el histórico):")
            print(importance.to_string())

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

    competition_catalog = CompetitionCatalog.from_csv(ROOT / "configs" / "competitions_seed.csv")

    all_results = []
    for gender in sorted(df["gender"].unique()):
        subset = df[df["gender"] == gender].reset_index(drop=True)
        results = run_for_gender(subset, gender, args.n_folds, competition_catalog)
        results["gender"] = gender
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    out_path = ROOT / "reports" / "phase4_tabular_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nResultados completos guardados en: {out_path}")


if __name__ == "__main__":
    main()
