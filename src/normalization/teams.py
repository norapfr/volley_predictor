"""
Normalización de nombres de selecciones — Fase 1.

Distintas fuentes (FIVB VIS, Kaggle, etc) nombran a la misma selección de
formas distintas ("USA" / "United States" / "Estados Unidos" / "USA (US)").
Este módulo resuelve cualquier nombre de origen a un team_id canónico único.

Dos niveles de resolución, en orden:
1. `team_aliases_seed.csv` — correcciones puntuales validadas a mano
   (tiene prioridad; útil para casos raros que el nivel 2 resuelve mal).
2. `country_lookup.resolve_country` — resolución automática vía ISO
   (pycountry) + excepciones FIVB conocidas. Cubre ~200 selecciones sin
   tener que mantener un CSV gigante a mano.

Equipos que ni así se resuelven (sección 28, "tests obligatorios: equipos
desconocidos") nunca se ignoran en silencio: se registran y se pueden
inspeccionar con `unresolved_names()`.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.normalization.country_lookup import resolve_country


def _normalize_key(name: str) -> str:
    """Normaliza un nombre para comparación: minúsculas, sin acentos, sin espacios extra."""
    stripped = unicodedata.normalize("NFKD", name.strip().lower())
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return " ".join(stripped.split())


class TeamNormalizer:
    """Resuelve nombres de equipo de cualquier fuente a un team_id canónico."""

    def __init__(self) -> None:
        self._alias_to_id: Dict[str, str] = {}
        self._unresolved: List[Tuple[str, str]] = []  # (source, source_name) — solo las que fallan del todo
        self._auto_resolved: List[Tuple[str, str, str]] = []  # (source, source_name, team_id) vía ISO/pycountry

    @classmethod
    def from_csv(cls, path: str | Path) -> "TeamNormalizer":
        normalizer = cls()
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                normalizer.add_alias(
                    canonical_team_id=row["canonical_team_id"],
                    source=row["source"],
                    source_name=row["source_name"],
                )
        return normalizer

    def add_alias(self, canonical_team_id: str, source: str, source_name: str) -> None:
        key = self._key(source, source_name)
        self._alias_to_id[key] = canonical_team_id
        # También registrar el propio canonical_team_id como alias válido de sí mismo.
        self._alias_to_id.setdefault(self._key(source, canonical_team_id), canonical_team_id)

    @staticmethod
    def _key(source: str, name: str) -> str:
        return f"{source.strip().lower()}::{_normalize_key(name)}"

    def resolve(self, source: str, name: str) -> str | None:
        """Devuelve el team_id canónico, o None si el nombre no se pudo resolver por ningún nivel."""
        key = self._key(source, name)
        team_id = self._alias_to_id.get(key)
        if team_id is not None:
            return team_id

        team_id = resolve_country(name)
        if team_id is not None:
            self._alias_to_id[key] = team_id  # cachear para el resto de esta ejecución
            self._auto_resolved.append((source, name, team_id))
            return team_id

        self._unresolved.append((source, name))
        return None

    def resolve_many(self, pairs: Iterable[Tuple[str, str]]) -> List[str | None]:
        return [self.resolve(source, name) for source, name in pairs]

    def unresolved_names(self) -> List[Tuple[str, str]]:
        """Lista de (source, source_name) que no pudieron resolverse por ningún nivel — revisión manual."""
        return list(self._unresolved)

    def auto_resolved_names(self) -> List[Tuple[str, str, str]]:
        """Lista de (source, source_name, team_id) resueltos automáticamente vía ISO — para auditar/validar."""
        return list(self._auto_resolved)

    def clear_unresolved(self) -> None:
        self._unresolved.clear()

