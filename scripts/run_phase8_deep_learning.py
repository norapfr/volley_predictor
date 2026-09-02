"""
Fase 8 — deep learning, solo si mejora sobre CatBoost (spec explícita).

Compara, sobre los mismos folds walk-forward:
  - elo, catboost — las dos referencias ya establecidas (Fases 2 y 5).
  - neural_net — MLP compacto con early stopping (src/models/neural.py).

Uso:
    python scripts/run_phase8_deep_learning.py
    python scripts/run_phase8_deep_learning.py --n-folds 5
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
from src.models.neural import NeuralWinModel
from src.models.tabular import available_tabular_models
from src.normalization.competitions import CompetitionCatalog
from src.ratings.elo import EloConfig


def _elo_predictor_factory(train_df: pd.DataFrame):
    def predict(test_df: pd.DataFrame):
        return test_df["elo_pred_a"].to_numpy()

    return predict


def _tabular_factory(model_cls, feature_columns):
    def factory(train_df: pd.DataFrame):
        model = model_cls().fit(train_df, feature_columns)

        def predict(test_df: pd.DataFrame):
            return model.predict_proba_batch(test_df, feature_columns)

        return predict

    return factory


def _neural_factory(feature_columns):
    def factory(train_df: pd.DataFrame):
        model = NeuralWinModel().fit(train_df, feature_columns)

        def predict(test_df: pd.DataFrame):
            return model.predict_proba_batch(test_df, feature_columns)

        return predict

    return factory


def run_for_gender(df_gender: pd.DataFrame, gender_label: str, n_folds: int, competition_catalog: CompetitionCatalog) -> pd.DataFrame:
    print(f"\n{'=' * 70}\n{gender_label.upper()} — {len(df_gender)} partidos\n{'=' * 70}")

    try:
        import torch  # noqa: F401
    except ImportError:
        print("⚠ PyTorch no está instalado. `pip install torch`.")
        return pd.DataFrame()

    tabular_models = available_tabular_models()
    if "catboost" not in tabular_models:
        print("⚠ catboost no está instalado. `pip install catboost`.")
        return pd.DataFrame()

    features_df = build_features(df_gender, competition_catalog, EloConfig())

    predictors = {
        "elo": _elo_predictor_factory,
        "catboost": _tabular_factory(tabular_models["catboost"], FEATURE_COLUMNS),
        "neural_net": _neural_factory(FEATURE_COLUMNS),
    }

    results = evaluate_walk_forward(features_df, predictors, n_folds=n_folds)
    print("\nPor fold:")
    print(results.to_string(index=False))
    print("\nResumen (media ± desviación estándar entre folds):")
    print(summarize_across_folds(results))

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
        out_path = ROOT / "reports" / "phase8_deep_learning_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        print(f"\nResultados completos guardados en: {out_path}")

        print(f"\n{'=' * 70}\nVEREDICTO\n{'=' * 70}")
        for gender in combined["gender"].unique():
            gender_results = combined[combined["gender"] == gender]
            means = gender_results.groupby("model")["log_loss"].mean()
            if "neural_net" in means and "catboost" in means:
                if means["neural_net"] < means["catboost"]:
                    print(f"{gender}: la red neuronal MEJORA sobre CatBoost ({means['neural_net']:.4f} vs {means['catboost']:.4f}) — mantener.")
                else:
                    print(f"{gender}: la red neuronal NO mejora sobre CatBoost ({means['neural_net']:.4f} vs {means['catboost']:.4f}) — no se adopta, según el criterio de la spec.")


if __name__ == "__main__":
    main()
