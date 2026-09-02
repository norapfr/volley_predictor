# Volley Predictor — Selecciones Nacionales

Predictor probabilístico de partidos de voleibol entre selecciones nacionales
senior (masculino y femenino). Ver la especificación completa en
`docs/spec.md` (copiada del documento original para Codex).

## Estado: Fase 1 — Dataset y schema ✅

Lo implementado en esta fase:

- **Schema** (`src/schema.py`): modelos pydantic para `competitions`,
  `teams`, `team_aliases`, `matches`, `sets`, `team_match_stats` y
  `team_ratings`, con validaciones (marcador de sets consistente con el
  ganador, sin clubes, un equipo no puede jugar contra sí mismo, etc).
- **Ingestión** (`src/ingestion/`): interfaz `RawMatchSource` +
  implementación `KaggleVNLSource` para CSVs estilo Kaggle. No descarga
  datos (Kaggle/FIVB no son accesibles desde este entorno de ejecución);
  lee un CSV ya obtenido por el usuario.
- **Normalización** (`src/normalization/`): `TeamNormalizer` resuelve
  alias de equipos ("United States" / "Estados Unidos" / "USA (US)" → `USA`)
  de forma insensible a mayúsculas/acentos, y reporta equipos desconocidos
  en vez de fallar en silencio. `CompetitionCatalog` resuelve nombre +
  género → `competition_id`.
- **Limpieza** (`src/cleaning/matches.py`): pipeline `build_matches` que
  parsea fechas, resuelve equipos y competición, deduplica (por
  `source_match_id` y por clave natural `gender+date+equipos`), y separa
  partidos válidos de filas rechazadas con motivo.
- **Validación / leakage** (`src/evaluation/leakage.py`): utilidad
  `assert_no_future_leakage` que implementa el test crítico de la spec
  (`feature(match_T)` no puede depender de partidos en o después de `T`).
  Se usará directamente en Fase 3 sobre las tablas de features reales.
- **Datos semilla** (`configs/`): catálogo de competiciones y un conjunto
  inicial de alias de equipos (ampliar según vayan apareciendo fuentes).
- **Tests** (`tests/`, 15 tests, todos en verde): normalización, schema,
  pipeline de limpieza end-to-end (incluye duplicados y equipos
  desconocidos) y la utilidad de leakage.

## Fuente principal: crawler histórico de FIVB VIS

Además de la ingestión de CSVs de Kaggle, hay un crawler para FIVB VIS
(`src/ingestion/fivb_vis_*.py`), que es la fuente prioritaria según la spec
(sección 3.1). Se apoya en el cliente comunitario
[`fivbvis`](https://github.com/claromes/fivbvis) (`pip install fivbvis`).

**Cómo funciona:**

1. `FivbVisCrawler.discover_tournaments()` pide TODOS los torneos vía
   `GetVolleyTournamentList` y filtra en cliente por nombre (`is_in_scope`)
   contra el alcance de la sección 2 — excluye clubes, categorías juveniles
   y beach volleyball.
2. Para cada torneo dentro de alcance, pide sus partidos vía
   `GetVolleyMatchList` con `filter="NoTournament='<id>'"`.
3. Cada torneo se guarda como `data/raw/fivb_vis/tournament_<id>.jsonl`, y
   `state.json` lleva el registro de qué torneos ya se completaron — el
   crawl se puede cortar y reanudar sin perder trabajo ni re-pedir de más.
4. `src/ingestion/fivb_vis_source.py` traduce lo crudo a `RawMatchRow` +
   filas de `sets`, que pasan por el **mismo** `build_matches` que Kaggle —
   una sola capa de limpieza/normalización/anti-leakage para todas las
   fuentes.

**⚠️ Importante — verificar antes de un crawl grande:** este entorno de
desarrollo no tiene acceso de red a `fivb.org`, así que los nombres de
campo en `TOURNAMENT_FIELDS` / `MATCH_FIELDS` (`fivb_vis_crawler.py`) y el
filtro `NoTournament` (confirmado en la documentación pública solo para
Beach Volleyball) **no se han podido verificar en vivo**. Antes de lanzar
el crawl completo, ejecuta desde tu máquina:

```bash
python scripts/probe_fivb_vis_fields.py
```

Esto hace peticiones mínimas y te dice si algún campo falla, para ajustar
las listas antes de un crawl largo. El cliente (`FivbVisClient`) ya hace
"field probing" automático: si VIS rechaza un campo, lo descarta y
reintenta, dejando constancia de qué se perdió.

**Lanzar el crawl completo:**

```bash
python scripts/run_fivb_vis_crawl.py
# más lento pero más respetuoso con el servicio:
python scripts/run_fivb_vis_crawl.py --seconds-between-requests 2
```

Esto deja `data/processed/matches.csv` y `data/processed/sets.csv`, más un
reporte de equipos sin resolver para ampliar
`configs/team_aliases_seed.csv`.

**Kaggle/GitHub como validación cruzada, no como fuente principal:** una
vez tengas `matches.csv` de VIS, compara marcadores y fechas contra los
CSVs de Kaggle (sección 3.2) y contra `volleyball_athletes` del paquete R
[`olympicAthletes`](https://github.com/moderndive/olympicAthletes) para
los Juegos Olímpicos (da participación/medallero, no partido a partido,
pero sirve para contrastar qué selecciones jugaron cada edición). Cualquier
discrepancia sistemática entre VIS y Kaggle es una señal de que hay que
revisar el mapeo de campos, no de que uno de los dos esté "mal" por
defecto.

## Cómo ejecutar

```bash
pip install -e ".[dev]"
pytest -v
python scripts/run_ingestion_demo.py            # usa el fixture sintético
python scripts/run_ingestion_demo.py --csv ruta/al/dataset_kaggle.csv
```

## Importante — sobre los datos

Este entorno no tiene acceso de red a `kaggle.com` ni a `fivb.org`, así que
la Fase 1 se validó con un CSV **sintético** (`tests/fixtures/sample_vnl_men.csv`,
claramente de prueba, no datos reales). Para poblar el dataset maestro de
verdad:

1. Descarga manualmente los CSVs de Kaggle listados en la spec (sección 3.2).
2. Pásalos por `KaggleVNLSource` + `build_matches` (ver `scripts/run_ingestion_demo.py`
   como plantilla).
3. Revisa el reporte de "equipos no resueltos" y amplía
   `configs/team_aliases_seed.csv` según haga falta.
4. Para FIVB VIS hace falta implementar un `RawMatchSource` nuevo que
   consulte el web service (`GetVolleyMatchList` / `GetVolleyMatch`); el
   contrato de `RawMatchSource` en `src/ingestion/base.py` ya está listo
   para esa implementación.

## Siguiente paso (Fase 2)

Baselines: Elo, Logistic Regression, Bradley-Terry + walk-forward
validation, sobre los `matches` ya limpios y normalizados de esta fase.

## Estructura

```text
volley_predictor/
  data/{raw,interim,processed}
  src/{ingestion,cleaning,normalization,features,ratings,models,calibration,evaluation,prediction,api}
  models/{men,women}
  configs/        # catálogos semilla (competiciones, alias de equipos)
  scripts/        # demos ejecutables
  tests/
  docs/spec.md    # especificación original
```
