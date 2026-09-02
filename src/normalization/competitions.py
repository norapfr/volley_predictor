"""Catálogo de competiciones — carga configs/competitions_seed.csv y resuelve (nombre, gender) -> competition_id."""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path
from typing import Dict, Tuple

from src.schema import Competition, Gender


def _key(name: str, gender: str) -> str:
    norm = unicodedata.normalize("NFKD", name.strip().lower())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    return f"{gender.strip().lower()}::{' '.join(norm.split())}"


class CompetitionCatalog:
    def __init__(self) -> None:
        self._by_key: Dict[str, Competition] = {}
        self._by_id: Dict[str, Competition] = {}

    @classmethod
    def from_csv(cls, path: str | Path) -> "CompetitionCatalog":
        catalog = cls()
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                comp = Competition(
                    competition_id=row["competition_id"],
                    name=row["name"],
                    gender=Gender(row["gender"]),
                    level=row["level"],
                )
                catalog._by_key[_key(comp.name, comp.gender.value)] = comp
                catalog._by_id[comp.competition_id] = comp
        return catalog

    def resolve(self, name: str, gender: str) -> Competition | None:
        return self._by_key.get(_key(name, gender))

    def get(self, competition_id: str) -> Competition | None:
        return self._by_id.get(competition_id)
