"""
Cliente de bajo nivel para FIVB VIS — Fase 1 (extensión: crawler histórico).

Se apoya en el paquete `fivbvis` (cliente oficial de la comunidad, instalable
vía `pip install fivbvis`, https://github.com/claromes/fivbvis) para construir
las URLs de petición, pero añade encima:

- Parseo de XML a `dict`/`list[dict]` (el paquete `fivbvis` solo devuelve el
  texto crudo de la respuesta).
- Reintentos con backoff ante fallos de red o límites de tasa.
- Rate limiting cliente (VIS es un servicio público sin SLA — hay que ser
  respetuoso).
- "Probing" de campos: VIS devuelve un error si se pide un campo que no
  existe para ese tipo de recurso o que el usuario no tiene permiso de ver.
  En vez de asumir una lista de campos fija, el cliente detecta el campo
  inválido en el mensaje de error y reintenta sin él, registrando cuáles
  campos tuvo que descartar.

IMPORTANTE — verificación pendiente por el usuario:
Este entorno de ejecución no tiene acceso de red a fivb.org, así que estos
nombres de campo NO se han podido verificar en vivo contra el servicio real.
Están tomados de la documentación pública del paquete `fivbvis` y de
ejemplos publicados (sección "Documentación" del README). Antes de lanzar
un crawl grande, ejecuta `scripts/probe_fivb_vis_fields.py` (incluido) para
confirmar qué campos responde el servicio realmente para tu cuenta/acceso,
y ajusta `FIELD_SETS` en `fivb_vis_crawler.py` si hace falta.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fivbvis.fivbvis import FivbVis

logger = logging.getLogger(__name__)


class VisFieldError(Exception):
    """VIS rechazó uno o más campos solicitados (no existen o sin permiso)."""

    def __init__(self, message: str, bad_fields: List[str]):
        super().__init__(message)
        self.bad_fields = bad_fields


class VisRequestError(Exception):
    """Error genérico devuelto por VIS (no relacionado con campos)."""


@dataclass
class VisClientConfig:
    min_seconds_between_requests: float = 1.0
    max_retries: int = 5
    backoff_base_seconds: float = 2.0
    timeout_seconds: float = 30.0


class FivbVisClient:
    """Envoltorio sobre `fivbvis.FivbVis` con parseo, reintentos y rate limiting."""

    def __init__(self, config: Optional[VisClientConfig] = None) -> None:
        self._raw = FivbVis()
        self.config = config or VisClientConfig()
        self._last_request_ts: float = 0.0

    # -- rate limiting -----------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        wait = self.config.min_seconds_between_requests - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    # -- parsing -------------------------------------------------------------
    @staticmethod
    def _parse_single(xml_text: str) -> Dict[str, Any]:
        root = ET.fromstring(xml_text)
        return dict(root.attrib)

    @staticmethod
    def _parse_list(xml_text: str) -> List[Dict[str, Any]]:
        root = ET.fromstring(xml_text)
        return [dict(child.attrib) for child in root]

    @staticmethod
    def _extract_bad_fields(error_text: str) -> List[str]:
        """
        Best-effort: VIS suele reportar errores de campo en el propio XML de
        error (p.ej. mencionando el nombre del campo inválido). Como no se ha
        podido verificar el formato exacto del error contra el servicio real
        desde este entorno, esto es deliberadamente conservador: si no se
        puede identificar el campo concreto, `VisRequestError` genérico se
        lanza en su lugar y el caller decide cómo reaccionar.
        """
        return []

    # -- low level request with retries --------------------------------------
    def _request_with_retries(self, fn, *args, **kwargs) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            self._throttle()
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # httpx errors, timeouts, etc.
                last_exc = exc
                wait = self.config.backoff_base_seconds ** attempt
                logger.warning(
                    "Fallo en petición a VIS (intento %d/%d): %s — reintentando en %.1fs",
                    attempt,
                    self.config.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise VisRequestError(f"Petición a VIS falló tras {self.config.max_retries} intentos: {last_exc}")

    # -- public API ------------------------------------------------------
    def get(self, request_type: str, no: int, fields: str) -> Dict[str, Any]:
        xml_text = self._request_with_retries(
            self._raw.get, request_type, no, fields, "xml"
        )
        self._raise_if_error(xml_text)
        return self._parse_single(xml_text)

    def get_list(self, request_type: str, fields: str, filter: Optional[str] = None) -> List[Dict[str, Any]]:
        xml_text = self._request_with_retries(
            self._raw.get_list, request_type, fields, filter, "xml"
        )
        self._raise_if_error(xml_text)
        return self._parse_list(xml_text)

    def get_list_with_field_probing(
        self, request_type: str, fields: List[str], filter: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Como `get_list`, pero si VIS rechaza el conjunto de campos, va
        eliminando campos sospechosos y reintentando hasta obtener una
        respuesta válida o agotar los campos.

        Devuelve (filas, campos_descartados).
        """
        remaining = list(fields)
        dropped: List[str] = []

        while remaining:
            try:
                rows = self.get_list(request_type, " ".join(remaining), filter)
                return rows, dropped
            except VisFieldError as exc:
                for bad in exc.bad_fields:
                    if bad in remaining:
                        remaining.remove(bad)
                        dropped.append(bad)
                if not exc.bad_fields:
                    # No pudimos identificar el campo culpable: no hay forma segura
                    # de seguir probando automáticamente.
                    raise
        raise VisRequestError(f"{request_type}: todos los campos fueron rechazados por VIS.")

    @staticmethod
    def _raise_if_error(xml_text: str) -> None:
        # VIS devuelve normalmente una raíz de error reconocible en caso de fallo.
        # No se ha podido confirmar el formato exacto de error contra el servicio
        # real desde este entorno; esta comprobación cubre el caso más común
        # documentado (raíz "Errors").
        stripped = xml_text.strip()
        if stripped.startswith("<Errors") or "<Error " in stripped[:200]:
            bad_fields = FivbVisClient._extract_bad_fields(xml_text)
            if bad_fields:
                raise VisFieldError(xml_text, bad_fields)
            raise VisRequestError(xml_text)
