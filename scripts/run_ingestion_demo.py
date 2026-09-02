"""
Demo end-to-end de Fase 1: ingestión -> limpieza -> normalización -> validación.

Uso:
    python scripts/run_ingestion_demo.py [--csv path/al/dataset.csv]

Sin --csv usa el fixture sintético de pruebas (tests/fixtures/sample_vnl_men.csv)
para que el pipeline se pueda ejecutar y revisar sin depender de descargar
datos reales de Kaggle/FIVB todavía.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaning.matches import build_matches
from src.ingestion.kaggle_vnl import KaggleVNLSource
from src.normalization.competitions import CompetitionCatalog
from src.normalization.teams import TeamNormalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "sample_vnl_men.csv",
        help="Ruta a un CSV estilo Kaggle VNL. Por defecto usa el fixture de prueba.",
    )
    args = parser.parse_args()

    team_normalizer = TeamNormalizer.from_csv(ROOT / "configs" / "team_aliases_seed.csv")
    competition_catalog = CompetitionCatalog.from_csv(ROOT / "configs" / "competitions_seed.csv")

    source = KaggleVNLSource(args.csv)
    rows = list(source.iter_matches())
    result = build_matches(rows, team_normalizer, competition_catalog)

    print(f"Archivo:              {args.csv}")
    print(f"Filas crudas leídas:  {len(rows)}")
    print(f"Partidos válidos:     {len(result.matches)}")
    print(f"Duplicados omitidos:  {result.duplicates_skipped}")
    print(f"Filas rechazadas:     {len(result.rejected)}")

    if result.rejected:
        print("\nMotivos de rechazo:")
        for r in result.rejected:
            print(f"  - [{r.source}:{r.source_match_id}] {r.reason}")

    unresolved = team_normalizer.unresolved_names()
    if unresolved:
        print("\nEquipos no resueltos (revisar/añadir a team_aliases_seed.csv):")
        for source_name, name in unresolved:
            print(f"  - source={source_name!r} name={name!r}")

    if result.matches:
        print("\nEjemplo de partido limpio y validado:")
        print(result.matches[0].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
