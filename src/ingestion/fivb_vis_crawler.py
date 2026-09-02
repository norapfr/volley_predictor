"""
Crawler histórico de FIVB VIS — descubre torneos de selecciones senior y
descarga todos los partidos disponibles (sección 3.1 de la spec).

Estrategia (deliberadamente conservadora, dado que no se pudo verificar en
vivo contra fivb.org desde este entorno):

1. Descubrir TODOS los torneos vía `GetVolleyTournamentList` (sin filtro de
   servidor, porque no se pudo confirmar la sintaxis exacta de
   VolleyTournamentFilter contra el servicio real). El filtrado a las
   competiciones de selecciones senior del alcance (sección 2) se hace en
   cliente, por nombre — ver `COMPETITION_KEYWORDS`.
2. Para cada torneo que matchea el alcance, pedir sus partidos vía
   `GetVolleyMatchList` con `filter="NoTournament='<id>'"` (patrón
   documentado para Beach Volleyball en el README de `fivbvis`; se asume el
   mismo patrón para indoor — a confirmar con `scripts/probe_fivb_vis_fields.py`).
3. Guardar cada torneo crudo como JSONL en `data/raw/fivb_vis/` y llevar un
   `state.json` con los torneos ya completados, para poder cortar y
   reanudar un crawl largo sin perder trabajo ni re-pedir de más (VIS es un
   servicio público compartido).
4. NO convierte a `schema.Match` aquí — eso lo hace
   `raw_to_master(...)` reusando `TeamNormalizer` / `CompetitionCatalog` /
   `build_matches`, igual que con Kaggle, para que ambas fuentes compartan
   la misma capa de limpieza y así los tests de leakage/duplicados apliquen
   por igual.

Campos solicitados — deliberadamente explícitos y ajustables en
`TOURNAMENT_FIELDS` / `MATCH_FIELDS` para poder recortarlos si VIS rechaza
alguno (ver `FivbVisClient.get_list_with_field_probing`).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.ingestion.fivb_vis_client import FivbVisClient, VisClientConfig

logger = logging.getLogger(__name__)

# Cada competición del alcance (sección 2) se identifica por un conjunto de
# palabras que deben aparecer TODAS en el nombre del torneo (no
# necesariamente pegadas ni en ese orden) — FIVB a veces inserta "Men's"/
# "Women's" en medio del nombre (p.ej. "Asian Men's Championship"), lo que
# rompía una comparación de subcadena contigua. Se usa `\b` (límite de
# palabra) para no confundir "norceca" con una subcadena de otra palabra.
#
# El nombre canónico es el que aparece en configs/competitions_seed.csv —
# única fuente de verdad para is_in_scope() y para mapear al catálogo.
COMPETITION_PATTERNS: List[Tuple[Tuple[str, ...], str]] = [
    (("nations", "league"), "Volleyball Nations League"),
    (("world", "championship"), "World Championship"),
    (("olympic",), "Olympic Games"),
    (("world", "cup"), "World Cup"),
    (("eurovolley",), "EuroVolley"),
    (("european", "championship"), "EuroVolley"),
    (("norceca",), "NORCECA Championship"),
    (("south", "american", "championship"), "South American Championship"),
    (("asian", "championship"), "Asian Championship"),
    (("african", "championship"), "African Championship"),
]

# Excluir explícitamente lo que la spec marca fuera de alcance (sección 2),
# más falsos positivos reales detectados por el matching de palabras sueltas
# (p.ej. "World Grand Champions Cup" contiene "world" y "cup" pero NO es la
# Copa del Mundo oficial; los campeonatos CISM son entre equipos militares,
# no las selecciones absolutas oficiales).
EXCLUDE_KEYWORDS: List[str] = [
    "club",
    "u23",
    "u21",
    "u20",
    "u19",
    "u18",
    "u17",
    "u16",
    "u15",
    "u14",
    "youth",
    "junior",
    "beach",
    "military",
    "cism",
    "grand champions cup",
    "zonal",  # p.ej. "Asian Eastern Zonal Championship" — sub-región de una confederación
    "central asian",  # sub-región de Asia, no el campeonato continental completo
]


def _tokens_present(name_lower: str, tokens: Tuple[str, ...]) -> bool:
    return all(re.search(rf"\b{re.escape(t)}\b", name_lower) for t in tokens)


def canonical_competition_name(tournament_name: str) -> Optional[str]:
    """Nombre canónico (tal y como está en competitions_seed.csv), o None si no matchea ningún patrón."""
    name_lower = tournament_name.lower()
    for tokens, canonical in COMPETITION_PATTERNS:
        if _tokens_present(name_lower, tokens):
            return canonical
    return None

TOURNAMENT_FIELDS = [
    "No",
    "Name",
    "Code",
    "Season",
    "Gender",
    "StartDate",
    "EndDate",
]

MATCH_FIELDS = [
    "No",
    "NoInTournament",
    "DateTimeLocal",
    "TeamAName",
    "TeamBName",
    "MatchPointsA",
    "MatchPointsB",
    "PointsTeamASet1",
    "PointsTeamBSet1",
    "PointsTeamASet2",
    "PointsTeamBSet2",
    "PointsTeamASet3",
    "PointsTeamBSet3",
    "PointsTeamASet4",
    "PointsTeamBSet4",
    "PointsTeamASet5",
    "PointsTeamBSet5",
    "City",
    "Hall",
    "Pool",
    "Phase",
]


@dataclass
class CrawlState:
    completed_tournament_ids: Set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "CrawlState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(completed_tournament_ids=set(data.get("completed_tournament_ids", [])))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"completed_tournament_ids": sorted(self.completed_tournament_ids)}, indent=2),
            encoding="utf-8",
        )


def is_in_scope(tournament_name: str) -> bool:
    name = tournament_name.lower()
    if any(bad in name for bad in EXCLUDE_KEYWORDS):
        return False
    return canonical_competition_name(tournament_name) is not None


class FivbVisCrawler:
    def __init__(
        self,
        output_dir: Path,
        client: Optional[FivbVisClient] = None,
        state_path: Optional[Path] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.client = client or FivbVisClient(VisClientConfig())
        self.state_path = state_path or (self.output_dir / "state.json")
        self.state = CrawlState.load(self.state_path)

    # -- paso 1: descubrimiento -------------------------------------------------
    def discover_tournaments(self) -> List[Dict]:
        rows, dropped = self.client.get_list_with_field_probing("GetVolleyTournamentList", TOURNAMENT_FIELDS)
        if dropped:
            logger.warning("GetVolleyTournamentList: VIS rechazó estos campos, se omitieron: %s", dropped)
        in_scope = [r for r in rows if is_in_scope(r.get("Name", ""))]
        logger.info("Torneos descubiertos: %d totales, %d dentro de alcance.", len(rows), len(in_scope))
        return in_scope

    # -- paso 2: partidos por torneo ---------------------------------------
    def fetch_tournament_matches(self, tournament_no: str) -> List[Dict]:
        rows, dropped = self.client.get_list_with_field_probing(
            "GetVolleyMatchList", MATCH_FIELDS, filter=f"NoTournament='{tournament_no}'"
        )
        if dropped:
            logger.warning(
                "GetVolleyMatchList (torneo %s): VIS rechazó estos campos, se omitieron: %s",
                tournament_no,
                dropped,
            )
        return rows

    # -- orquestación con checkpoint ----------------------------------------
    def crawl(self, tournaments: Optional[List[Dict]] = None) -> None:
        tournaments = tournaments if tournaments is not None else self.discover_tournaments()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for tournament in tournaments:
            tno = tournament.get("No")
            if tno is None:
                logger.warning("Torneo sin 'No', se omite: %s", tournament)
                continue
            if tno in self.state.completed_tournament_ids:
                logger.info("Torneo %s ya crawleado, se omite (resume).", tno)
                continue

            logger.info("Crawleando torneo %s — %s", tno, tournament.get("Name"))
            try:
                matches = self.fetch_tournament_matches(tno)
            except Exception:
                logger.exception("Fallo crawleando torneo %s — se deja pendiente para el siguiente resume.", tno)
                continue

            self._write_tournament(tournament, matches)
            self.state.completed_tournament_ids.add(tno)
            self.state.save(self.state_path)

    def _write_tournament(self, tournament: Dict, matches: List[Dict]) -> None:
        path = self.output_dir / f"tournament_{tournament['No']}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_type": "tournament", **tournament}) + "\n")
            for m in matches:
                fh.write(json.dumps({"_type": "match", **m}) + "\n")


def iter_raw_tournament_files(output_dir: Path) -> Iterable[Path]:
    yield from sorted(Path(output_dir).glob("tournament_*.jsonl"))
