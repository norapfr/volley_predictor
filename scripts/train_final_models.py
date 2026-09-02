"""
Entrena los modelos finales de producción — uno por género — con TODO el
histórico disponible (a diferencia de los scripts de las Fases 2/4/5/6,
que reservan folds para validar, aquí no se reserva nada: para el modelo
que de verdad se va a usar, más datos de entrenamiento es estrictamente
mejor).

Guarda en models/<gender>/:
    win_model.cbm              — CatBoost, P(gana A). Recomendación de Fase 5: sin calibrar, sin ensemble.
    win_model.cbm.features.json
    margin_model.cbm           — CatBoost, P(margen 3-0/3-1/3-2 | quién gana). Fase 6.
    margin_model.cbm.features.json
    point_diff_model.cbm       — CatBoost regresor, diferencia de puntos esperada. Fase 7.
    point_diff_model.cbm.features.json
    elo_ratings.json           — rating Elo final de cada equipo, necesario para puntuar partidos futuros.
    team_state.json            — forma reciente de cada equipo (racha, % victorias/margen últimos N, última fecha).
    h2h_state.json             — historial head-to-head entre cada par de equipos que se ha enfrentado.
    metadata.json              — fecha de entrenamiento, nº de partidos, rango de fechas, columnas de features.

Con `elo_ratings.json` + `team_state.json` + `h2h_state.json` ya se puede
predecir un partido que NO está en el dataset histórico — ver
`src/features/predict_features.py::build_features_for_match`.

Uso:
    python scripts/train_final_models.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build_features import FEATURE_COLUMNS, build_features
from src.features.team_state import compute_current_state, save_current_state
from src.models.point_diff import PointDiffModel
from src.models.set_score import SetMarginModel, build_margin_training_frame
from src.models.tabular import available_tabular_models
from src.normalization.competitions import CompetitionCatalog
from src.ratings.elo import EloConfig, compute_elo_history


def train_for_gender(df_gender: pd.DataFrame, gender: str, competition_catalog: CompetitionCatalog, out_dir: Path) -> None:
    print(f"\n{'=' * 60}\n{gender.upper()} — entrenando con {len(df_gender)} partidos (todo el histórico)\n{'=' * 60}")

    tabular_models = available_tabular_models()
    if "catboost" not in tabular_models:
        print("⚠ catboost no está instalado. `pip install catboost`. Nada que guardar para este género.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Features + Elo, sobre el 100% del histórico.
    features_df = build_features(df_gender, competition_catalog, EloConfig())
    _, final_ratings = compute_elo_history(df_gender, EloConfig())

    # Modelo de ganador — recomendación de Fase 5: CatBoost solo, sin calibrar ni ensemble.
    win_model = tabular_models["catboost"]()
    win_model.fit(features_df, FEATURE_COLUMNS)
    win_model.save(out_dir / "win_model.cbm")
    print(f"  Guardado: {out_dir / 'win_model.cbm'}")

    # Modelo de margen — Fase 6.
    margin_train = build_margin_training_frame(features_df, FEATURE_COLUMNS)
    margin_model = SetMarginModel()
    margin_model.fit(margin_train, FEATURE_COLUMNS)
    margin_model.save(out_dir / "margin_model.cbm")
    print(f"  Guardado: {out_dir / 'margin_model.cbm'}")

    # Modelo de diferencia de puntos — Fase 7.
    point_diff_model = PointDiffModel()
    point_diff_model.fit(features_df, FEATURE_COLUMNS)
    point_diff_model.save(out_dir / "point_diff_model.cbm")
    print(f"  Guardado: {out_dir / 'point_diff_model.cbm'}")

    # Ratings Elo finales — imprescindibles para puntuar partidos futuros.
    elo_path = out_dir / "elo_ratings.json"
    elo_path.write_text(json.dumps(final_ratings.get(gender, {}), indent=2))
    print(f"  Guardado: {elo_path} ({len(final_ratings.get(gender, {}))} equipos)")

    # Estado de forma reciente + head-to-head — la pieza que faltaba para
    # poder predecir un partido que no está en el histórico (ver
    # src/features/team_state.py y src/features/predict_features.py).
    team_states, h2h_states = compute_current_state(df_gender)
    save_current_state(team_states.get(gender, {}), h2h_states.get(gender, {}), out_dir)
    print(f"  Guardado: {out_dir / 'team_state.json'} / {out_dir / 'h2h_state.json'}")

    # Metadata para trazabilidad — qué datos y qué features se usaron.
    metadata = {
        "gender": gender,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_matches": len(df_gender),
        "date_range": [str(df_gender["date"].min()), str(df_gender["date"].max())],
        "feature_columns": FEATURE_COLUMNS,
        "n_teams": len(final_ratings.get(gender, {})),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"  Guardado: {metadata_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=ROOT / "data" / "processed" / "matches.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.matches)
    competition_catalog = CompetitionCatalog.from_csv(ROOT / "configs" / "competitions_seed.csv")

    for gender in sorted(df["gender"].unique()):
        subset = df[df["gender"] == gender].reset_index(drop=True)
        out_dir = ROOT / "models" / gender
        train_for_gender(subset, gender, competition_catalog, out_dir)

    print("\nListo. Los modelos de producción están en models/<gender>/.")


if __name__ == "__main__":
    main()
