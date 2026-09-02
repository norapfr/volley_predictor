"""
Orquesta el crawler histórico de FIVB VIS y produce el dataset maestro
limpio (matches + sets), reusando la misma capa de limpieza/normalización
que Kaggle (sección 5-6 de la spec).

Ejecutar DESDE TU MÁQUINA — este entorno no tiene acceso de red a fivb.org.

Uso:
    python scripts/run_fivb_vis_crawl.py
    python scripts/run_fivb_vis_crawl.py --resume   # continúa un crawl anterior
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaning.matches import build_matches
from src.ingestion.fivb_vis_client import FivbVisClient, VisClientConfig
from src.ingestion.fivb_vis_crawler import FivbVisCrawler
from src.ingestion.fivb_vis_source import iter_fivb_vis_raw_matches
from src.normalization.competitions import CompetitionCatalog
from src.normalization.teams import TeamNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

RAW_DIR = ROOT / "data" / "raw" / "fivb_vis"
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds-between-requests",
        type=float,
        default=1.0,
        help="Espera mínima entre peticiones a VIS (ser respetuoso con un servicio público).",
    )
    args = parser.parse_args()

    # --- Paso 1: crawl (con checkpoint/resume automático vía state.json) ---
    client = FivbVisClient(VisClientConfig(min_seconds_between_requests=args.seconds_between_requests))
    crawler = FivbVisCrawler(output_dir=RAW_DIR, client=client)
    print(f"Crawleando FIVB VIS -> {RAW_DIR} (resume automático si ya hay progreso)...")
    crawler.crawl()

    # --- Paso 2: convertir crudo -> RawMatchRow + sets ---
    team_normalizer = TeamNormalizer.from_csv(ROOT / "configs" / "team_aliases_seed.csv")
    competition_catalog = CompetitionCatalog.from_csv(ROOT / "configs" / "competitions_seed.csv")

    raw_rows = []
    all_sets = []
    discard_reasons = []
    for raw, sets, reason in iter_fivb_vis_raw_matches(RAW_DIR):
        if raw is None:
            discard_reasons.append(reason)
            continue
        raw_rows.append(raw)
        all_sets.append((raw.source_match_id, sets))

    # --- Paso 3: limpieza/normalización/validación (mismo pipeline que Kaggle) ---
    result = build_matches(raw_rows, team_normalizer, competition_catalog)

    print(f"\nPartidos crudos descargados:        {len(raw_rows) + len(discard_reasons)}")
    print(f"Descartados antes de limpieza:       {len(discard_reasons)}")
    print(f"Partidos válidos tras limpieza:       {len(result.matches)}")
    print(f"Rechazados en limpieza (schema/team): {len(result.rejected)}")
    print(f"Duplicados omitidos:                  {result.duplicates_skipped}")

    unresolved = team_normalizer.unresolved_names()
    auto_resolved = team_normalizer.auto_resolved_names()
    unique_auto = sorted(set((s, n, t) for s, n, t in auto_resolved))
    print(f"\n{len(unique_auto)} nombres resueltos automáticamente vía ISO (pycountry) — revisar de pasada:")
    for source, name, team_id in unique_auto[:15]:
        print(f"  - {name!r} -> {team_id}")
    if len(unique_auto) > 15:
        print(f"  ... y {len(unique_auto) - 15} más.")

    unique_unresolved = sorted(set(unresolved))
    if unique_unresolved:
        print(f"\n{len(unique_unresolved)} nombres de equipo SIN resolver — añade alias a configs/team_aliases_seed.csv:")
        for source, name in unique_unresolved[:30]:
            print(f"  - source={source!r} name={name!r}")
        if len(unique_unresolved) > 30:
            print(f"  ... y {len(unique_unresolved) - 30} más.")
    else:
        print("\nTodos los nombres de equipo se resolvieron.")

    # --- Paso 4: escribir dataset maestro ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches_path = PROCESSED_DIR / "matches.csv"
    with open(matches_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "match_id", "date", "gender", "competition", "competition_id", "stage",
                "team_a", "team_b", "team_a_score", "team_b_score", "sets_a", "sets_b",
                "winner", "venue", "country_host", "neutral", "source", "source_match_id",
                "tournament_no", "no_in_tournament",
            ]
        )
        for m in result.matches:
            writer.writerow(
                [
                    m.match_id, m.date, m.gender.value, m.competition, m.competition_id, m.stage.value,
                    m.team_a, m.team_b, m.team_a_score, m.team_b_score, m.sets_a, m.sets_b,
                    m.winner.value, m.venue or "", m.country_host or "", m.neutral, m.source, m.source_match_id,
                    m.tournament_no or "", m.no_in_tournament if m.no_in_tournament is not None else "",
                ]
            )
    print(f"\nEscrito: {matches_path} ({len(result.matches)} partidos)")

    valid_match_ids = {m.source_match_id for m in result.matches if m.source == "fivb_vis"}
    sets_path = PROCESSED_DIR / "sets.csv"
    with open(sets_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["match_id", "set_number", "team_a_points", "team_b_points"])
        for source_match_id, sets in all_sets:
            if source_match_id not in valid_match_ids:
                continue
            match_id = f"fivb_vis:{source_match_id}"
            for s in sets:
                writer.writerow([match_id, s.set_number, s.team_a_points, s.team_b_points])
    print(f"Escrito: {sets_path}")


if __name__ == "__main__":
    main()
