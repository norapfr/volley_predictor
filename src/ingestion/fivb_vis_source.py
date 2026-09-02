"""
Traduce lo que dejó `FivbVisCrawler` en `data/raw/fivb_vis/*.jsonl` al mismo
formato `RawMatchRow` que usa la fuente de Kaggle, para que ambas pasen por
el mismo `build_matches` (sección 5-6 de la spec: un único pipeline de
limpieza/validación/anti-leakage para todas las fuentes).

También produce, por separado, las filas de `sets` (puntos por set) — el
dataset maestro las guarda en una tabla aparte (sección 5).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from src.ingestion.base import RawMatchRow
from src.ingestion.fivb_vis_crawler import canonical_competition_name, iter_raw_tournament_files

# Valores observados/documentados para el campo Gender de VIS son inciertos
# sin poder verificarlos en vivo; se cubren las variantes más probables
# (texto y código numérico habitual 0=men/1=women en varios feeds FIVB).
GENDER_MAP: Dict[str, str] = {
    "men": "men", "man": "men", "male": "men", "m": "men", "0": "men",
    "women": "women", "woman": "women", "female": "women", "w": "women", "1": "women",
}


_WOMEN_RE = re.compile(r"\bwomen'?s?\b", re.IGNORECASE)
_MEN_RE = re.compile(r"\bmen'?s?\b", re.IGNORECASE)


def resolve_gender(tournament: Dict, tournament_name: str) -> Optional[str]:
    raw = str(tournament.get("Gender", "")).strip().lower()
    if raw in GENDER_MAP:
        return GENDER_MAP[raw]
    # Fallback: muchos nombres de torneo incluyen "Men"/"Women" explícitamente.
    # Se usa \b (límite de palabra) para no confundir con substrings como
    # "tournament" (contiene "men") o "women" dentro de otra palabra.
    if _WOMEN_RE.search(tournament_name):
        return "women"
    if _MEN_RE.search(tournament_name):
        return "men"
    return None


@dataclass
class VisSetRow:
    match_source_match_id: str
    set_number: int
    team_a_points: int
    team_b_points: int


def _parse_set_points(match: Dict) -> List[VisSetRow]:
    sets: List[VisSetRow] = []
    for set_number in range(1, 6):
        a_key, b_key = f"PointsTeamASet{set_number}", f"PointsTeamBSet{set_number}"
        a_val, b_val = match.get(a_key), match.get(b_key)
        if a_val in (None, "", "0") and b_val in (None, "", "0"):
            continue  # set no jugado (partido terminó antes)
        try:
            sets.append(
                VisSetRow(
                    match_source_match_id=str(match.get("No", "")),
                    set_number=set_number,
                    team_a_points=int(a_val or 0),
                    team_b_points=int(b_val or 0),
                )
            )
        except (TypeError, ValueError):
            continue
    return sets


_THAI_BUDDHIST_ERA_OFFSET = 543  # año budista tailandés = año gregoriano + 543


def _extract_date_only(datetime_local: str) -> Optional[str]:
    """
    'DateTimeLocal' llega como fecha+hora (p.ej. '2026-06-10T18:00:00' o
    '2026-06-10 18:00:00'); nos quedamos solo con la parte 'YYYY-MM-DD' para
    que encaje con el parseo de fecha ya existente en el pipeline de limpieza
    (src/cleaning/matches.py::_parse_date), que es agnóstico a la fuente.

    Caso real observado: algún torneo alojado en Tailandia devuelve la fecha
    en calendario budista tailandés en vez de gregoriano (p.ej. '2568-11-08'
    en vez de '2025-11-08' — la era budista tailandesa va 543 años por
    delante). Se detecta y convierte aquí en vez de en el pipeline de
    limpieza genérico porque es un defecto específico de esta fuente, no
    algo que deba conocer `_parse_date`.
    """
    if not datetime_local:
        return None
    date_part = datetime_local.strip()[:10]
    if not (len(date_part) == 10 and date_part[4] == "-" and date_part[7] == "-"):
        return None

    year_str = date_part[:4]
    try:
        year = int(year_str)
    except ValueError:
        return date_part  # no es un año numérico; deja que _parse_date decida

    # Un año claramente fuera de rango gregoriano plausible, pero que sí
    # cae en rango plausible al restarle la era budista, es casi con toda
    # seguridad este defecto — se corrige.
    if year > 2100 and 1990 <= (year - _THAI_BUDDHIST_ERA_OFFSET) <= 2100:
        corrected_year = year - _THAI_BUDDHIST_ERA_OFFSET
        return f"{corrected_year:04d}{date_part[4:]}"

    return date_part


def iter_fivb_vis_raw_matches(
    output_dir: Path,
) -> Iterator[Tuple[Optional[RawMatchRow], List[VisSetRow], Optional[str]]]:
    """
    Recorre los .jsonl crudos del crawler y produce, por cada partido,
    (RawMatchRow | None, sets, motivo_de_descarte | None).

    Un partido se descarta (RawMatchRow=None) cuando falta un dato
    imprescindible (fecha, nombres de equipo, marcador de sets, o no se pudo
    determinar género/competición) — el motivo se registra para el reporte
    de validación en vez de fallar silenciosamente.
    """
    for path in iter_raw_tournament_files(output_dir):
        tournament: Optional[Dict] = None
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("_type") == "tournament":
                    tournament = row
                    continue
                if tournament is None:
                    yield None, [], f"{path.name}: partido antes de cabecera de torneo, se omite."
                    continue

                match = row
                tournament_name = tournament.get("Name", "")
                gender = resolve_gender(tournament, tournament_name)
                competition_name = canonical_competition_name(tournament_name)

                if gender is None:
                    yield None, [], f"No se pudo determinar género para torneo {tournament_name!r}"
                    continue
                if competition_name is None:
                    yield None, [], f"Torneo fuera del catálogo canónico: {tournament_name!r}"
                    continue
                if not match.get("TeamAName") or not match.get("TeamBName"):
                    yield None, [], f"Partido {match.get('No')}: falta nombre de equipo A o B"
                    continue
                date_only = _extract_date_only(match.get("DateTimeLocal", ""))
                if date_only is None:
                    yield None, [], f"Partido {match.get('No')}: falta o no se pudo parsear DateTimeLocal"
                    continue

                try:
                    sets_a = int(match["MatchPointsA"])
                    sets_b = int(match["MatchPointsB"])
                except (KeyError, TypeError, ValueError):
                    yield None, [], f"Partido {match.get('No')}: MatchPointsA/B ausente o no numérico"
                    continue

                set_rows = _parse_set_points(match)
                total_a = sum(s.team_a_points for s in set_rows) or None
                total_b = sum(s.team_b_points for s in set_rows) or None

                raw = RawMatchRow(
                    source="fivb_vis",
                    source_match_id=str(match.get("No", "")),
                    date_str=date_only,
                    gender=gender,
                    competition_name=competition_name,
                    stage_str=match.get("Phase"),
                    team_a_name=match["TeamAName"],
                    team_b_name=match["TeamBName"],
                    team_a_score=total_a,
                    team_b_score=total_b,
                    sets_a=sets_a,
                    sets_b=sets_b,
                    venue=match.get("Hall") or match.get("City"),
                    country_host=None,
                    neutral=None,
                )
                yield raw, set_rows, None
