"""
Resuelve un nombre de país/selección (tal y como lo escribe FIVB VIS, Kaggle,
etc) a un código ISO 3166-1 alpha-3 — usado como `team_id` canónico.

Por qué esto en vez de un CSV de alias mantenido a mano para ~220 países:
mantener manualmente el alias de cada selección del mundo no escala y es
una fuente constante de "equipo desconocido". `pycountry` (paquete estándar
con la base de datos ISO) resuelve la gran mayoría de nombres de forma
automática y determinista. Solo hace falta un diccionario pequeño de
excepciones para los nombres que FIVB usa y que no coinciden con el nombre
ISO oficial (Chinese Taipei, Costa de Marfil, RD Congo, etc), o para
selecciones ya desaparecidas que aparecen en partidos históricos
(Checoslovaquia, URSS, Alemania del Este/Oeste, Yugoslavia...).

El CSV `configs/team_aliases_seed.csv` se sigue usando primero (para
correcciones puntuales que alguien haya validado a mano) y tiene prioridad
sobre esta resolución automática.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pycountry

# Sufijos entre paréntesis que VIS a veces añade al nombre del equipo para
# distinguir escuadras del mismo país en partidos de práctica/exhibición
# (p.ej. "Germany (RED)" vs "Germany (WHITE)" en un torneo con dos combinados
# del mismo país). Se elimina el sufijo para resolver el país base — si hiciera
# falta distinguir ambas escuadras como equipos distintos en el futuro, habría
# que tratarlo en una capa aparte, no en la resolución de país.
_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")

# Excepciones confirmadas: nombres que pycountry.search_fuzzy no resuelve
# (o resuelve mal) tal y como los usa FIVB, más selecciones históricas ya
# desaparecidas (partidos antiguos en el dataset). team_id inventado con
# prefijo "X" para las que no tienen código ISO vigente, siguiendo la
# convención habitual de datasets deportivos históricos.
MANUAL_OVERRIDES: Dict[str, str] = {
    "chinese taipei": "TPE",
    "ivory coast": "CIV",
    "cote d'ivoire": "CIV",
    "côte d'ivoire": "CIV",
    "dr congo": "COD",
    "democratic republic of the congo": "COD",
    "congo dr": "COD",
    "republic of congo": "COG",
    "congo": "COG",
    "cape verde": "CPV",
    "cabo verde": "CPV",
    "korea": "KOR",
    "korea republic": "KOR",
    "republic of korea": "KOR",
    "north korea": "PRK",
    "dpr korea": "PRK",
    "chinese hong kong": "HKG",
    "hong kong, china": "HKG",
    "macau": "MAC",
    "macau, china": "MAC",
    "usa": "USA",
    "united states": "USA",
    "brasil": "BRA",
    "brésil": "BRA",
    "alemania": "DEU",
    "deutschland": "DEU",
    "italia": "ITA",
    "espana": "ESP",
    "españa": "ESP",
    "polska": "POL",
    "francia": "FRA",
    "japon": "JPN",
    "japón": "JPN",
    "great britain": "GBR",
    "great britain and n. ireland": "GBR",
    "vatican": "VAT",
    "kosovo": "XKX",  # sin código ISO oficial; XKX es el cuasi-estándar usado en deporte/estadística
    "chinese taipei (tpe)": "TPE",
    "st. vincent and the grenadines": "VCT",
    "saint vincent and the grenadines": "VCT",
    "st. kitts and nevis": "KNA",
    "saint kitts and nevis": "KNA",
    "st. lucia": "LCA",
    "saint lucia": "LCA",
    "us virgin islands": "VIR",
    "u.s. virgin islands": "VIR",
    "british virgin islands": "VGB",
    "myanmar": "MMR",
    "burma": "MMR",
    "laos": "LAO",
    "brunei": "BRN",
    "eswatini": "SWZ",
    "swaziland": "SWZ",
    "czechia": "CZE",
    # Nombres de VIS que ya no coinciden con el nombre ISO vigente, o erratas
    # observadas en datos reales del crawl.
    "turkey": "TUR",  # ISO renombró oficialmente a "Türkiye"; VIS sigue usando "Turkey"
    "türkiye": "TUR",
    "turkiye": "TUR",
    "turkez": "TUR",  # errata observada en VIS
    "u.s.a.": "USA",
    "maldive islands": "MDV",  # nombre antiguo de Maldivas usado por VIS
    "moldovia": "MDA",  # errata observada en VIS (Moldova)
    "samoa, western": "WSM",
    "netherlands antilles": "ANT",  # territorio disuelto en 2010; sin ISO vigente
    "china, people's rep. of": "CHN",
    "democratic republic of congo": "COD",  # variante sin "the" de la ya existente
    "hongkong": "HKG",  # VIS a veces lo escribe sin espacio
    "kuweit": "KWT",  # errata observada en VIS (Kuwait)
    "macao, china": "MAC",
    "mali republic": "MLI",
    "nertherlands": "NLD",  # errata observada en VIS (Netherlands)
    "saudia arabia": "SAU",  # errata observada en VIS (Saudi Arabia)
    # Selecciones históricas (partidos anteriores a su disolución) — sin ISO vigente.
    "soviet union": "XSU",
    "ussr": "XSU",
    "czechoslovakia": "XCS",
    "yugoslavia": "XYU",
    "west germany": "XFRG",
    "east germany": "XGDR",
    "serbia and montenegro": "XSCG",
}


def resolve_country(name: str) -> Optional[str]:
    """
    Devuelve el código ISO alpha-3 (o pseudo-código histórico) para `name`,
    o None si no se pudo resolver por ningún método.

    Prueba varias variantes del nombre (tal cual, sin sufijo entre paréntesis,
    sin puntos) contra las excepciones manuales y pycountry, en ese orden.
    """
    candidates: List[str] = [name]

    stripped = _TRAILING_PAREN_RE.sub("", name).strip()
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    no_dots = name.replace(".", "").strip()
    if no_dots and no_dots not in candidates:
        candidates.append(no_dots)

    stripped_no_dots = stripped.replace(".", "").strip()
    if stripped_no_dots and stripped_no_dots not in candidates:
        candidates.append(stripped_no_dots)

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in MANUAL_OVERRIDES:
            return MANUAL_OVERRIDES[key]

    for candidate in candidates:
        try:
            return pycountry.countries.lookup(candidate).alpha_3
        except LookupError:
            continue

    for candidate in candidates:
        try:
            return pycountry.countries.search_fuzzy(candidate)[0].alpha_3
        except LookupError:
            continue

    return None
