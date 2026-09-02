"""
Verificación en vivo de campos de FIVB VIS — ejecutar DESDE TU MÁQUINA
(este entorno de desarrollo no tiene acceso de red a fivb.org).

No se pudo confirmar contra el servicio real qué campos exactos acepta
GetVolleyTournamentList / GetVolleyMatchList (sección 3.1 de la spec, docs
oficiales solo enlazadas, no accesibles desde aquí). Este script hace
peticiones mínimas y de bajo riesgo para que lo compruebes tú y ajustes
`TOURNAMENT_FIELDS` / `MATCH_FIELDS` en `src/ingestion/fivb_vis_crawler.py`
si hace falta.

Uso:
    python scripts/probe_fivb_vis_fields.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.fivb_vis_client import FivbVisClient, VisClientConfig, VisRequestError
from src.ingestion.fivb_vis_crawler import MATCH_FIELDS, TOURNAMENT_FIELDS, is_in_scope


def main() -> None:
    client = FivbVisClient(VisClientConfig(min_seconds_between_requests=1.5))

    print("1) Probando GetVolleyTournamentList con campos:", TOURNAMENT_FIELDS)
    try:
        rows, dropped = client.get_list_with_field_probing("GetVolleyTournamentList", TOURNAMENT_FIELDS)
        print(f"   OK — {len(rows)} torneos recibidos. Campos descartados: {dropped}")
    except VisRequestError as exc:
        print(f"   FALLÓ: {exc}")
        print("   Ajusta TOURNAMENT_FIELDS en fivb_vis_crawler.py según el error de VIS.")
        return

    in_scope = [r for r in rows if is_in_scope(r.get("Name", ""))]
    print(f"   De esos, {len(in_scope)} caen dentro de nuestro alcance (VNL, Mundial, JJOO, etc).")
    if not in_scope:
        print("   Ningún torneo en alcance encontrado — revisa is_in_scope() / los nombres reales de VIS.")
        return

    # Prioriza un torneo reciente y, a ser posible, de Nations League (suele tener muchos partidos).
    def sort_key(t):
        is_vnl = "nations league" in t.get("Name", "").lower()
        season = t.get("Season", "0")
        return (is_vnl, season)

    in_scope.sort(key=sort_key, reverse=True)
    sample = in_scope[0]
    print(f"   Ejemplo de torneo elegido para la prueba: {sample}")

    print(f"\n2) Probando GetVolleyMatchList para torneo {sample['No']} ({sample.get('Name')}) con campos:", MATCH_FIELDS)
    try:
        matches, dropped = client.get_list_with_field_probing(
            "GetVolleyMatchList", MATCH_FIELDS, filter=f"NoTournament='{sample['No']}'"
        )
        print(f"   OK — {len(matches)} partidos recibidos. Campos descartados: {dropped}")
        if matches:
            print("   Ejemplo de partido:", matches[0])
        else:
            print("   0 partidos con NoTournament — puede que el filtro no aplique igual a indoor.")
            print("   Probando sin filtro (trae de más, solo para diagnóstico)...")
            unfiltered, dropped2 = client.get_list_with_field_probing("GetVolleyMatchList", MATCH_FIELDS, filter=None)
            print(f"   Sin filtro: {len(unfiltered)} partidos totales recibidos. Campos descartados: {dropped2}")
            if unfiltered:
                print("   Ejemplo de partido (sin filtrar por torneo):", unfiltered[0])
                print("   -> Si ese partido SÍ tiene un campo tipo 'NoTournament' con el valor del torneo,")
                print("      dime cómo se llama exactamente ese campo en la salida para corregir el filtro.")
    except VisRequestError as exc:
        print(f"   FALLÓ: {exc}")
        print("   Ajusta MATCH_FIELDS en fivb_vis_crawler.py según el error de VIS.")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()