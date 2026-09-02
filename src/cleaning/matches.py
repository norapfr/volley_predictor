"""
Limpieza y normalización de partidos crudos -> `schema.Match` — Fase 1.

`build_matches` es deliberadamente tolerante a fallos por fila: un partido
mal formado, con un equipo desconocido o con marcador inconsistente no debe
tirar abajo la ingestión completa. En vez de eso se acumula en `rejected`
con un motivo legible, para que el reporte de validación (sección 28) pueda
mostrarlo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_cls
from datetime import datetime
from typing import Dict, Iterable, List, Set, Tuple

from pydantic import ValidationError

from src.ingestion.base import RawMatchRow
from src.normalization.competitions import CompetitionCatalog
from src.normalization.teams import TeamNormalizer
from src.schema import Gender, Match, Stage, Winner

_STAGE_ALIASES: Dict[str, Stage] = {
    "pool": Stage.pool,
    "preliminary": Stage.preliminary,
    "preliminary round": Stage.preliminary,
    "quarterfinal": Stage.quarterfinal,
    "quarter-final": Stage.quarterfinal,
    "qf": Stage.quarterfinal,
    "semifinal": Stage.semifinal,
    "semi-final": Stage.semifinal,
    "sf": Stage.semifinal,
    "bronze": Stage.bronze,
    "bronze medal match": Stage.bronze,
    "final": Stage.final,
    "gold medal match": Stage.final,
}

# Rango de fechas plausible para el dataset (voleibol internacional moderno,
# con margen hacia el futuro para torneos ya calendarizados). Cualquier
# fecha fuera de este rango se rechaza en vez de dejarla pasar — protege
# contra corrupciones de datos de la fuente (p.ej. fechas en calendario
# budista tailandés como "2568-11-08", que sin este chequeo llegan hasta
# pandas y rompen `pd.to_datetime` más adelante en el análisis).
MIN_PLAUSIBLE_DATE = date_cls(1990, 1, 1)
MAX_PLAUSIBLE_DATE = date_cls(2030, 12, 31)


def _parse_stage(raw: str | None) -> Stage:
    if not raw:
        return Stage.other
    return _STAGE_ALIASES.get(raw.strip().lower(), Stage.other)


def _parse_date(raw: str) -> date_cls:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(raw.strip(), fmt).date()
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Formato de fecha no reconocido: {raw!r}")

    if not (MIN_PLAUSIBLE_DATE <= parsed <= MAX_PLAUSIBLE_DATE):
        raise ValueError(
            f"Fecha fuera de rango plausible ({MIN_PLAUSIBLE_DATE}..{MAX_PLAUSIBLE_DATE}): {parsed}"
        )
    return parsed


@dataclass
class RejectedRow:
    source: str
    source_match_id: str
    reason: str


@dataclass
class CleaningResult:
    matches: List[Match] = field(default_factory=list)
    rejected: List[RejectedRow] = field(default_factory=list)
    duplicates_skipped: int = 0


def build_matches(
    rows: Iterable[RawMatchRow],
    team_normalizer: TeamNormalizer,
    competition_catalog: CompetitionCatalog,
) -> CleaningResult:
    result = CleaningResult()
    seen_source_ids: Set[Tuple[str, str]] = set()
    seen_natural_keys: Set[Tuple[str, str, str, str]] = set()  # (gender, date, team_a, team_b) tras normalizar

    for row in rows:
        source_key = (row.source, row.source_match_id)
        if source_key in seen_source_ids:
            result.duplicates_skipped += 1
            continue
        seen_source_ids.add(source_key)

        try:
            match = _clean_row(row, team_normalizer, competition_catalog)
        except (ValueError, ValidationError) as exc:
            result.rejected.append(
                RejectedRow(source=row.source, source_match_id=row.source_match_id, reason=str(exc))
            )
            continue

        natural_key = (
            match.gender.value,
            match.date.isoformat(),
            *sorted([match.team_a, match.team_b]),
        )
        if natural_key in seen_natural_keys:
            result.duplicates_skipped += 1
            continue
        seen_natural_keys.add(natural_key)

        result.matches.append(match)

    return result


def _clean_row(
    row: RawMatchRow,
    team_normalizer: TeamNormalizer,
    competition_catalog: CompetitionCatalog,
) -> Match:
    gender = Gender(row.gender.strip().lower())

    team_a_id = team_normalizer.resolve(row.source, row.team_a_name)
    team_b_id = team_normalizer.resolve(row.source, row.team_b_name)
    if team_a_id is None:
        raise ValueError(f"Equipo desconocido (team_a): {row.team_a_name!r} en source={row.source}")
    if team_b_id is None:
        raise ValueError(f"Equipo desconocido (team_b): {row.team_b_name!r} en source={row.source}")

    competition = competition_catalog.resolve(row.competition_name, gender.value)
    if competition is None:
        raise ValueError(
            f"Competición no encontrada en el catálogo: {row.competition_name!r} (gender={gender.value})"
        )

    parsed_date = _parse_date(row.date_str)
    stage = _parse_stage(row.stage_str)
    winner = Winner.team_a if row.sets_a > row.sets_b else Winner.team_b

    match_id = f"{row.source}:{row.source_match_id}"

    return Match(
        match_id=match_id,
        date=parsed_date,
        gender=gender,
        competition=competition.name,
        competition_id=competition.competition_id,
        stage=stage,
        team_a=team_a_id,
        team_b=team_b_id,
        team_a_score=row.team_a_score if row.team_a_score is not None else 0,
        team_b_score=row.team_b_score if row.team_b_score is not None else 0,
        sets_a=row.sets_a,
        sets_b=row.sets_b,
        winner=winner,
        venue=row.venue,
        country_host=row.country_host,
        neutral=row.neutral if row.neutral is not None else True,
        source=row.source,
        source_match_id=row.source_match_id,
        tournament_no=row.tournament_no,
        no_in_tournament=row.no_in_tournament,
    )
