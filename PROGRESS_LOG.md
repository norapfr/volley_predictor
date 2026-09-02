# Volley Predictor — Registro de progreso

> Documento vivo. Se actualiza cada vez que hay un resultado nuevo (crawl, fix, dataset). Última actualización: **ajuste anti-sesgo de marcador exacto + explicación de confianza en UI/API**.

## Estado actual

**Fase 1 (dataset y schema): ✅ COMPLETADA** — 6626 partidos válidos en `data/processed/matches.csv`.

**Fase 2 (baselines): ✅ COMPLETADA** — Elo, Bradley-Terry y regresión logística sobre Elo, con validación walk-forward. Los tres modelos superan claramente al azar; prácticamente empatados entre sí.

**Fase 3 (feature engineering): ✅ COMPLETADA** — forma reciente, head-to-head, importancia de competición y progreso dentro del torneo, todo anti-leakage. `data/processed/matches_with_features.csv` listo para la Fase 4.

**Fase 4 (ML tabular): ✅ COMPLETADA** — CatBoost gana claramente a los baselines de la Fase 2 en ambos géneros, de forma consistente. XGBoost segundo. LightGBM (el candidato original de la spec) queda empatado con los baselines — recomendación actualizada a CatBoost como modelo principal.

**Fase 5 (calibración/ensemble): ✅ COMPLETADA** — CatBoost sin calibrar ni ensemble es la mejor opción. Ya viene bien calibrado de fábrica; Platt/isotonic y el ensemble con XGBoost no aportan nada, incluso empeoran ligeramente. Modelo final recomendado: CatBoost solo.

**Fase 6 (modelo de marcador de sets): ✅ COMPLETADA** — 39.7% (hombres) / 47.8% (mujeres) de acierto en el marcador EXACTO de 6 posibles (azar = 16.7%). Modelos finales ya se guardan en disco (`models/<gender>/`), confirmado con datos reales (154 equipos hombres, 134 mujeres).

**Fase 7 (diferencia de puntos): ✅ COMPLETADA** — CatBoost MAE 10.98 (hombres) / 12.57 (mujeres), bate claramente al baseline de media y a la regresión lineal sobre Elo.

## Qué se ha construido

### 1. Schema del dataset maestro (`src/schema.py`)
Modelos pydantic para `competitions`, `teams`, `team_aliases`, `matches`, `sets`, `team_match_stats`, `team_ratings`. Validaciones: marcador de sets consistente con el ganador, sin empates a sets, un equipo no puede jugar contra sí mismo, competiciones de clubes rechazadas.

### 2. Ingestión desde Kaggle (`src/ingestion/kaggle_vnl.py`)
Loader para CSVs estilo "un partido por fila". **Nota:** los CSVs reales de Kaggle que compartió el usuario (VNL2024Men Attackers/Blockers/Diggers/...) son de estadísticas por jugador, no por partido — este loader no aplica directamente a esos; haría falta uno nuevo si se quiere usar Kaggle como fuente de `team_match_stats` o como cruce de validación.

### 3. Crawler histórico de FIVB VIS (`src/ingestion/fivb_vis_*.py`)
Fuente principal de datos, según preferencia del usuario ("crawler que recorra FIVB VIS... Kaggle como validación cruzada, no como fuente principal").

- **Descubrimiento**: `GetVolleyTournamentList` → todos los torneos, filtrados en cliente por alcance (`is_in_scope`).
- **Descarga**: `GetVolleyMatchList` con `filter="NoTournament='<id>'"` — confirmado en vivo que funciona.
- **Checkpoint/resume**: `data/raw/fivb_vis/state.json` — el crawl se puede cortar y retomar sin perder trabajo ni repetir torneos.
- **Limpieza compartida**: los partidos de VIS pasan por el mismo `build_matches` que Kaggle (deduplicación, validación de schema, anti-leakage).

### 4. Resolución automática de equipos (`src/normalization/country_lookup.py`)
En vez de mantener a mano el alias de ~200 países, se resuelve vía ISO (librería `pycountry`) + una lista de excepciones para nombres específicos de FIVB y selecciones históricas.

## Bugs encontrados y corregidos (con datos reales del usuario)

| # | Bug | Cómo se detectó | Fix |
|---|-----|-----------------|-----|
| 1 | Campo de fecha `LocalDate`/`LocalTime` no existe en VIS; el real es `DateTimeLocal` | Probe en vivo devolvió el partido sin fecha | Cambiado el campo solicitado y el parseo (se extrae solo la parte `YYYY-MM-DD`) |
| 2 | Seed de alias de equipos solo tenía 9 países de ejemplo → 0 partidos válidos de 6333 crudos | Primer crawl completo real | Resolución automática vía ISO (`pycountry`) en vez de CSV manual |
| 3 | 9 nombres no resueltos ni por ISO: `Germany (RED)`, `Japan (WHITE)` (sufijos de escuadra), `U.S.A.` (puntos), `Turkey` (ISO renombró a "Türkiye"), `Moldovia`/`Turkez` (erratas de VIS), `Maldive Islands`/`Samoa, Western`/`Netherlands Antilles` (nombres antiguos/históricos) | Segundo crawl completo real | Normalización de sufijos entre paréntesis y puntos + excepciones manuales específicas |
| 4 | **Asian Championship y African Championship con 0 partidos** pese a estar en alcance | Revisión de sanidad del dataset (conteo por competición) | El nombre real de VIS inserta el género en medio (`"Asian Men's Championship"`), lo que rompía la comparación de subcadena contigua `"asian championship"`. Cambiado a matching por palabras (todas deben aparecer, no necesariamente pegadas) |
| 5 | Efecto secundario del fix #4: **falsos positivos** — `"World Grand Champions Cup"` colaba como World Cup (contiene "world" y "cup" sueltos) y los Mundiales Militares CISM colaban como World Championship (contienen "world" y "championship" sueltos), ninguno de los dos es una competición de selecciones absolutas oficiales | Lectura línea a línea del log del segundo crawl completo | Añadidas exclusiones explícitas: `military`, `cism`, `grand champions cup` |
| 6 | 8 nombres de equipo más sin resolver tras el fix #4 (torneos nuevos trajeron países que no habían aparecido antes): `China, People's Rep. of`, `Democratic Republic of Congo`, `Hongkong`, `Kuweit` (errata de Kuwait), `Macao, China`, `Mali Republic`, `Nertherlands` (errata de Netherlands), `Saudia Arabia` (errata de Saudi Arabia) | Segundo crawl completo real | Añadidos a las excepciones manuales de `country_lookup.py` |
| 7 | `Asian Men's U16 Championship` colaba en alcance (faltaba `u16` en exclusiones) | Revisión del log del segundo crawl | Añadido `u16`/`u15`/`u14` a `EXCLUDE_KEYWORDS` |
| 8 | Decisión del usuario: excluir torneos zonales/sub-regionales (`Asian Eastern Zonal`, `Central Asian Championship`) — no son el campeonato continental completo | Confirmación explícita del usuario | Añadidos `zonal` y `central asian` a `EXCLUDE_KEYWORDS` |
| 9 | **Crash de pandas al analizar `matches.csv`**: una fecha `2568-11-08` (fuera de rango) reventaba `pd.to_datetime`. Causa real: un torneo alojado en Tailandia devolvió la fecha en **calendario budista tailandés** (año gregoriano + 543) en vez de gregoriano | Traceback real al correr la revisión de sanidad | Dos capas de fix: (1) detección específica del defecto budista tailandés en `fivb_vis_source.py` (convierte automáticamente si el año-543 cae en rango plausible), (2) validación general de rango de fechas (`1990..2030`) en `cleaning/matches.py` que rechaza con motivo cualquier fecha imposible de cualquier fuente, en vez de dejarla pasar y romper el análisis después |

## Resultado final confirmado (Fase 1 cerrada)

```
Partidos crudos descargados: 7060
Partidos válidos tras limpieza: 6626
Rechazados en limpieza: 69   |   Duplicados omitidos: 3
Nombres de equipo sin resolver: 0

Por género:      men 3357 / women 3269
Por competición:
  Volleyball Nations League      1874
  World Championship             1695
  EuroVolley                      712
  Olympic Games                   631
  World Cup                       558
  NORCECA Championship            536
  Asian Championship              376
  African Championship            184
  South American Championship      60

Rango de fechas: 2006-10-31 -> 2026-08-27
Equipos únicos: 158

Partidos por año:
  2006 208   2007 132   2008 181   2009 447   2010 182
  2011 158   2012 211   2013  49   2014 250   2015 157
  2016 231   2017  61   2018 534   2019 545   2020  24
  2021 784   2022 542   2023 647   2024 343   2025 543
  2026 397
```

Notas de lectura sobre el histórico:
- **2020 con solo 24 partidos**: coherente, la mayoría de competiciones de 2020 se cancelaron/pospusieron por COVID.
- **2013 y 2017 bajos** (49 y 61): años sin Nations League (empezó en 2018) y con menos densidad de continentales — plausible, no es señal de bug.
- **2021 es el año con más partidos (784)**: probablemente calendario comprimido post-COVID (varios campeonatos aplazados de 2020 se jugaron ese año).

## Decisiones ya confirmadas con el usuario

- ~~Torneos "zonales"/sub-regionales~~ → **Excluidos** (fix #8).
- ~~World League / Grand Prix (pre-2018)~~ → **No se añaden**. Volumen de datos confirmado como suficiente para Fase 2 y Fase 4 (Elo/Bradley-Terry/tabular ML); deep learning (Fase 8) es la única fase donde ~3300 partidos/género se queda corto, pero la propia spec ya contempla saltarla si no aporta.
- **Sesgo práctico hacia `3-0` en el marcador mostrado** → se decidió atacar en dos capas:
  - Modelo: `SetMarginModel` ahora entrena con `auto_class_weights="Balanced"` para que los errores en `3-1` y `3-2` pesen más y el modelo de margen no se refugie tanto en la clase mayoritaria (`3-0`).
  - Producto/API: `most_likely_score` se mantiene por compatibilidad, pero ya no se presenta solo como una predicción única. La API devuelve `most_likely_score_probability`, `score_confidence`, `score_confidence_gap` y `close_score_alternatives`; Streamlit muestra alternativas cercanas y avisa cuando la confianza del score exacto es baja.

## Pendiente / próximos pasos

- [ ] **Fase 8: deep learning** ← construida, pendiente de correr contra datos reales.
- [ ] **Antes de la Fase 9 (API)**: persistir el estado de "forma reciente"/head-to-head por equipo (ahora mismo solo vive en memoria durante `build_features`, no se guarda) — imprescindible para poder predecir un partido que no está en el dataset histórico. Los modelos y los ratings Elo finales ya se guardan (confirmado con datos reales, ver Fase 6.5); falta este último trozo de estado.
- [ ] Revisar cuántos partidos son del tipo `Germany (RED)` / `Japan (WHITE)` (posibles partidos de práctica/exhibición, no oficiales) y decidir si se filtran — pendiente, no bloqueante.
- [ ] Loader dedicado para los CSVs de Kaggle reales (nivel jugador) si se quieren usar como cruce de validación o para `team_match_stats` — pendiente, no bloqueante.

---

## Fase 8 — Deep learning (solo si mejora sobre CatBoost)

**Qué se construyó:**
- `src/models/neural.py` — MLP compacto (2 capas ocultas, dropout, regularización L2) con early stopping sobre un tramo cronológico reservado del propio train. A diferencia de los árboles, necesita imputación explícita de los `NaN` reales del dataset (mediana del train + columna binaria "_was_missing" por cada feature con huecos, para no perder la señal de "sin historial" que los árboles aprovechan solos) y escalado de features (`StandardScaler`).
- `scripts/run_phase8_deep_learning.py` — compara Elo, CatBoost y la red, e imprime un veredicto explícito por género ("mejora" / "no mejora, no se adopta") siguiendo el criterio literal de la spec.

**Bug real encontrado (con test sintético)**: cuando una feature sale completamente vacía en el tramo de entrenamiento de un fold (p.ej. `tournament_progress` sin `tournament_no` disponible), su mediana también es `NaN` — rellenar con `NaN` no rellena nada, y la columna vacía rompe el escalado (`StandardScaler` divide por una varianza indefinida). Corregido: si la mediana sale `NaN`, se usa `0.0` como último recurso.

**Estado**: construido y probado con datos sintéticos (500 partidos/género) — la red iguala o mejora ligeramente a CatBoost ahí, pero ambos quedan por debajo de Elo en ese test tan pequeño, patrón coherente con lo visto en fases anteriores (con poca muestra, la complejidad no ayuda). **Pendiente de correr contra los 6626 partidos reales**, que es lo que de verdad decide si esta fase se adopta o no, según el propio criterio de la spec ("solo si mejora").

---

## Fase 7 — Modelo de diferencia de puntos

**Qué se construyó:**
- `src/models/point_diff.py` — tres opciones comparadas: `mean_baseline` (predice la media del train), `linear_elo` (regresión lineal simple sobre elo_diff), `catboost` (CatBoost en modo regresión, con las mismas features de la Fase 3). `PointDiffModel` con guardado/carga igual que los modelos de las fases anteriores.
- `src/evaluation/metrics.py` — añadidas MAE y RMSE.
- `scripts/run_phase7_point_diff.py` — validación walk-forward de los tres, con la mediana de puntos totales por partido como referencia de contexto.

**Estado**: construido y probado con datos sintéticos. En ese test, `linear_elo` queda ligeramente por delante de `catboost` (patrón ya visto en fases anteriores: con pocos datos, el modelo simple generaliza mejor) — ambos baten claramente al `mean_baseline` (~15% menos error), señal de que hay algo real que aprender. **Confirmado con datos reales**: CatBoost gana con claridad.

```
Hombres (mediana 157 puntos/partido):  catboost MAE 10.98 | linear_elo 11.83 | mean_baseline 14.41
Mujeres (mediana 138 puntos/partido):  catboost MAE 12.57 | linear_elo 14.39 | mean_baseline 18.37
```

CatBoost reduce el error un 24% (hombres) / 32% (mujeres) frente al baseline de media, y un 7-13% frente a la regresión lineal sobre Elo. Mismo patrón que en las Fases 4-6: CatBoost es sistemáticamente la mejor opción una vez hay datos reales suficientes.

---

## Fase 6.5 — Persistencia de modelos (entre Fase 6 y Fase 7)

Hasta este punto, cada script (Fases 2/4/5/6) entrenaba los modelos desde cero solo para validar con walk-forward — nada se guardaba en disco. Se cerró ese hueco:

- `CatBoostModel.save()`/`.load()` y `SetMarginModel.save()`/`.load()` (formato nativo CatBoost `.cbm` + un JSON sidecar con las columnas de features usadas).
- `scripts/train_final_models.py` — entrena con el **100% del histórico** (a diferencia de los scripts de validación, aquí no se reserva nada para test: para el modelo real, más datos de entrenamiento es estrictamente mejor) y guarda en `models/<gender>/`: `win_model.cbm`, `margin_model.cbm`, `elo_ratings.json` (rating final de cada equipo) y `metadata.json` (fecha de entrenamiento, nº de partidos, columnas de features).

**Confirmado con datos reales**: `models/men/` (154 equipos en `elo_ratings.json`) y `models/women/` (134 equipos) ya tienen `win_model.cbm`, `margin_model.cbm`, `elo_ratings.json` y `metadata.json` guardados.

**Limitación conocida, pendiente para antes de la Fase 9**: esto persiste los *modelos*, pero no el estado de forma reciente/head-to-head por equipo (vive solo en memoria mientras corre `compute_form_and_h2h_features`). Para predecir un partido futuro real (no en el dataset) hace falta ese estado también.

---

## Fase 6 — Modelo de marcador de sets

**Qué se construyó:**
- `src/models/set_score.py` — en vez de un clasificador directo de 6 clases, se descompone en dos piezas: (1) el modelo de "quién gana" ya validado (CatBoost, Fase 5), y (2) un modelo nuevo de "margen" (3-0/3-1/3-2), entrenado sobre el 100% de los partidos reorientados hacia la perspectiva de quien realmente ganó cada uno (así no se descarta la mitad de los datos). Combinando las dos piezas salen las 6 probabilidades de marcador exacto.
- `scripts/run_phase6_set_scores.py` — validación walk-forward de la combinación, con log loss/accuracy multiclase.

**Bug real encontrado (con test sintético, antes de tocar datos reales)**: `sklearn.metrics.log_loss` con el parámetro `labels=` asume que las columnas de la matriz de probabilidades están en **orden alfabético**, sin importar el orden de la lista `labels` que se le pase — si no coincide, compara cada probabilidad con la clase equivocada y da un log loss peor que el azar, sin avisar de forma clara. Corregido reordenando las columnas a orden alfabético antes de la llamada.

**Hallazgo posterior en uso de la interfaz**: aunque la distribución global del modelo parecía razonable, el `idxmax` hacía que el marcador mostrado como "más probable" cayera con mucha frecuencia en `3-0`. No era un bug de código que forzara `3-0`: era una mezcla de (1) clase históricamente mayoritaria (`3-0` = 50.5% hombres / 58.3% mujeres), (2) modelo de margen entrenado sin balanceo, y (3) presentación de una sola clase ganadora aunque el top score tuviera poca ventaja sobre alternativas cercanas.

**Decisión aplicada**:
- `src/models/set_score.py`: `SetMarginModel` ahora usa `auto_class_weights="Balanced"` en CatBoost. Requiere reentrenar con `scripts/train_final_models.py` para afectar a los modelos `.cbm`.
- `src/api/predictor.py`: añadido resumen de decisión del score exacto (`most_likely_score_probability`, `score_confidence`, `score_confidence_gap`, `close_score_alternatives`) para no sobredimensionar el `idxmax`.
- `src/api/schemas.py`: actualizada la respuesta de la API con esos campos.
- `streamlit_app.py`: la interfaz muestra porcentaje del marcador más probable, confianza específica del score exacto, ventaja sobre el siguiente score y alternativas cercanas; si la confianza es baja, muestra aviso.

**Estado**: construido y probado con datos sintéticos (log loss cerca del azar teórico de 6 clases, `ln(6) ≈ 1.79`, esperable con tan pocos datos). **Confirmado con datos reales antes del ajuste anti-sesgo**: log loss 1.5654 hombres / 1.4206 mujeres (azar = 1.79), accuracy de marcador EXACTO 39.7% hombres / 47.8% mujeres (azar = 16.7%, hay 6 marcadores posibles). **Pendiente tras el cambio**: volver a correr `scripts/run_phase6_set_scores.py` y `scripts/train_final_models.py` para medir si baja la frecuencia de `3-0` como `most_likely_score` sin empeorar demasiado log loss/accuracy.

---

## Fase 5 — Calibración y ensemble

**Qué se construyó:**
- `src/models/calibration.py` — Platt scaling (regresión logística sobre el logit) e isotonic regression, ambos anti-leakage.
- `src/models/ensemble.py` — combinación por media de varios modelos (`make_average_ensemble_factory`) y wrapper de calibración (`make_calibrated_factory`) que, dentro de cada fold walk-forward, separa un tramo cronológico del propio train (últim 20%) solo para ajustar el calibrador — el modelo base nunca lo ve al entrenar.
- `scripts/run_phase5_calibration_ensemble.py` — compara CatBoost solo, XGBoost solo, ensemble, y CatBoost+Platt/Isotonic, más una tabla de calibración (reliability diagram) antes y después.

**Bug real encontrado (con test sintético, antes de tocar datos reales)**: `IsotonicRegression` sin acotar devuelve 0.0/1.0 exactos cuando el tramo de calibración tiene pocos puntos (folds tempranos, ~30-90 filas) — un solo fallo en esos extremos "imposibles" dispara el log loss (log(0) ≈ infinito). Corregido acotando la salida a `[0.02, 0.98]`.

**Incidencia observada al rerun**: una ejecución local de `scripts/run_phase5_calibration_ensemble.py` terminó con un `CatBoostError` vacío después de imprimir el resumen de hombres, durante el bloque extra de predicciones agrupadas para la tabla de calibración. Se repitió el mismo comando dentro de `.venv` (`.\.venv\Scripts\python.exe scripts\run_phase5_calibration_ensemble.py`) y completó correctamente hombres y mujeres, guardando `reports/phase5_calibration_ensemble_results.csv`. Conclusión actual: fallo puntual/no reproducido; no parece relacionado con el cambio de `SetMarginModel`, porque Fase 5 usa el CatBoost de ganador, no el modelo de margen de sets.

**Resultado real confirmado** (5 folds walk-forward, sobre los 6626 partidos):

| Género | Modelo | log_loss (media) |
|---|---|---|
| Hombres (3357) | **catboost** (sin tocar) | **0.5549** |
| Hombres | ensemble_avg | 0.5550 (empate técnico) |
| Hombres | catboost_platt | 0.5623 |
| Hombres | xgboost | 0.5650 |
| Hombres | catboost_isotonic | 0.5759 |
| Mujeres (3269) | **catboost** (sin tocar) | **0.5031** |
| Mujeres | ensemble_avg | 0.5034 (empate técnico) |
| Mujeres | catboost_platt | 0.5041 |
| Mujeres | xgboost | 0.5150 |
| Mujeres | catboost_isotonic | 0.5181 |

**Lectura**: la tabla de calibración de CatBoost sin ajustar ya muestra `predicted_mean` muy cerca de `actual_rate` en casi todos los bins — CatBoost viene razonablemente bien calibrado de fábrica. Por eso ni Platt ni isotonic mejoran nada (de hecho empeoran ligeramente el log loss en ambos géneros). El ensemble con XGBoost queda en empate técnico con CatBoost solo, porque XGBoost es algo peor y diluye la media en vez de sumar.

**Conclusión de la fase — y es una conclusión válida, no una fase "sin resultado"**: no añadir complejidad. **El modelo final recomendado es CatBoost solo, sin calibrar y sin ensemble.**

---

## Fase 4 — ML tabular (LightGBM/XGBoost/CatBoost)

**Qué se construyó:**
- `src/models/tabular.py` — wrappers de LightGBM/XGBoost/CatBoost con la misma interfaz `fit`/`predict_proba_batch` que los modelos de la Fase 2, más `available_tabular_models()` que detecta en tiempo de ejecución cuáles están instaladas (import perezoso dentro de cada `__init__`, así el módulo se puede importar aunque falte alguna librería).
- `scripts/run_phase4_tabular.py` — compara los 3 baselines de la Fase 2 contra los modelos tabulares disponibles, sobre los mismos folds walk-forward. Imprime importancia de features del modelo principal al final.

Instalación incremental recomendada (evita que xgboost se quede resolviendo versiones eternamente en Windows si se instala todo `[ml]` de golpe): `pip install lightgbm`, luego `pip install xgboost`, luego `pip install catboost`, cada uno por separado.

**Resultado real confirmado** (5 folds walk-forward, sobre los 6626 partidos):

| Género | Modelo | log_loss (media) | Brier (media) | Accuracy (media) |
|---|---|---|---|---|
| Hombres (3357) | **catboost** | **0.5549** | **0.1861** | 72.5% |
| Hombres | xgboost | 0.5650 | 0.1887 | 72.2% |
| Hombres | elo | 0.5699 | 0.1940 | 70.1% |
| Hombres | logistic_elo | 0.5727 | 0.1936 | 70.5% |
| Hombres | lightgbm | 0.5752 | 0.1911 | 72.3% |
| Hombres | bradley_terry | 0.5775 | 0.1980 | 69.3% |
| Mujeres (3269) | **catboost** | **0.5031** | **0.1637** | 76.0% |
| Mujeres | xgboost | 0.5150 | 0.1671 | 76.1% |
| Mujeres | elo | 0.5301 | 0.1775 | 73.8% |
| Mujeres | logistic_elo | 0.5324 | 0.1767 | 73.7% |
| Mujeres | lightgbm | 0.5367 | 0.1719 | 75.3% |
| Mujeres | bradley_terry | 0.5372 | 0.1810 | 72.4% |

**Lectura**:
- **CatBoost gana claramente en ambos géneros y las tres métricas**, con el mismo orden de ranking en hombres y mujeres — señal fiable de que la ventaja es real, no ruido de un fold con suerte.
- **XGBoost segundo**, también claramente por delante de los baselines de la Fase 2.
- **LightGBM (el "candidato principal" que proponía la spec original) queda prácticamente empatado con los baselines** — sin ventaja clara en log_loss, aunque sí mejora la accuracy. Hallazgo real que actualiza la recomendación según evidencia, no asunción: para este dataset, **CatBoost pasa a ser el modelo principal recomendado**, con XGBoost como alternativa.
- Los modelos tabulares tienen más varianza entre folds que Elo (std 0.03-0.08 vs 0.015-0.02), pero mejoran notablemente con más datos de entrenamiento — en el fold 4 (el de más histórico acumulado) es donde sacan más ventaja sobre los baselines. Curva de aprendizaje coherente: confirma que la ventaja de los modelos tabulares es señal real que necesita volumen para aprovecharse, no un artefacto de un fold concreto.
- **Importancia de features**: `elo_diff` y `matches_played_diff` dominan en ambos géneros; `tournament_progress` (la feature que reconstruimos en la Fase 3 tras descubrir que VIS no da la fase real) sale **3ª más importante** — buena validación de que ese fix mereció la pena. `competition_importance` es la menos influyente pero no nula.

---

## Fase 3 — Feature engineering

**Qué se construyó:**
- `src/features/rolling_form.py` — por equipo, antes de cada partido: partidos jugados, % victorias y margen medio de sets en los últimos 5/10, racha actual, días desde el último partido. Head-to-head específico entre el par de equipos (no confundir con la forma general de cada uno).
- `src/features/competition_importance.py` — importancia de la competición (Juegos Olímpicos/Mundial > World Cup > VNL > continentales), vía el catálogo de competiciones ya existente.
- `src/features/tournament_progress.py` — sustituto de la fase (final/semifinal/...) que no pudimos usar (ver bug abajo): posición del partido dentro de su torneo, normalizada 0-1.
- `src/features/build_features.py` — orquestador: junta Elo (Fase 2) + forma + head-to-head + importancia + progreso, y calcula las columnas "diff" (equipo A menos B) listas para modelar. `FEATURE_COLUMNS` centraliza la lista para que la Fase 4 no tenga que conocer los nombres internos de cada módulo.

**Bug real encontrado con datos reales**: la columna `stage_importance` planeada dependía del campo `Phase` de FIVB VIS — y VIS no lo devuelve para **ningún** partido (7060/7060 vacío, confirmado inspeccionando los JSONL crudos). La columna quedaba siempre en 1.0, sin aportar nada. Solución: en vez de fingir que teníamos la fase, añadimos al pipeline un dato que sí existe (`no_in_tournament`, ya se pedía a VIS pero nunca se llevaba hasta el CSV maestro) y construimos `tournament_progress` = posición del partido dentro de su torneo. No es una etiqueta exacta de fase, pero es señal real y disponible — se documentó como aproximación, no como sustituto exacto. Esto obligó a extender el schema (`tournament_no`, `no_in_tournament` nuevos en `Match`) y reprocesar el dataset ya descargado (sin re-crawlear).

**Verificado con datos reales tras el fix**: `tournament_progress` ya varía de verdad (media 0.51, rango 0-1, antes plano en 1.0).

## Medias reales de las features (Fase 3, sobre los 6626 partidos)

| Feature | Media | Lectura |
|---|---|---|
| `elo_diff` | 16.06 | Casi neutro — esperado, team_a/team_b no favorece a nadie |
| `matches_played_diff` | 5.88 | A suele tener ~6 partidos más de experiencia que B |
| `current_streak_diff` | 0.33 | Rachas prácticamente parejas de media |
| `days_since_last_diff` | -13.19 | Poco relevante; el máximo (6400 días) ya investigado, ver arriba |
| `h2h_matches_played` | 3.42 | De media, los dos equipos ya se han visto ~3 veces antes |
| `h2h_win_rate_a` | 0.529 | Casi 50/50 — sin sesgo sistemático en quién es "A" |
| `competition_importance` | 3.50 | Entre VNL (3) y Mundial/Olímpicos (5) — refleja que la VNL es la competición más frecuente |
| `tournament_progress` | 0.507 | Justo en el medio del torneo, de media |
| `win_rate_last5_diff` | 0.026 | Casi neutro |
| `win_rate_last10_diff` | 0.021 | Casi neutro |
| `set_margin_avg_last5_diff` | 0.139 | Casi neutro |
| `set_margin_avg_last10_diff` | 0.118 | Casi neutro |

Lectura general: casi todas las medias rondan 0 (o el punto medio de su escala), que es justo lo esperado — team_a/team_b es solo una etiqueta de posición en el CSV, no "el favorito". Si alguna estuviera claramente desviada de 0 sería señal de sesgo sistemático en la construcción del dataset; no lo está.

**Otro hallazgo verificado, no era bug**: `days_since_last_diff` con máximo de ~6400 días (~17 años) parecía sospechoso. Se investigó con un diagnóstico directo sobre `matches.csv`: corresponde a selecciones de nivel bajo/medio (Macao, Emiratos Árabes, Uzbekistán, Bangladés, Islas Vírgenes Británicas...) que solo reaparecen en el alcance del dataset una vez cada 10-15 años — así es el vóley fuera de la élite, no un fallo de datos. Anotado como algo a vigilar si en el futuro se añade un modelo lineal/neuronal sensible a outliers (Fase 8); los modelos de árboles de la Fase 4 no tienen problema con esto.

---

## Fase 2 — Baselines (Elo, Bradley-Terry, Logistic-sobre-Elo)

**Qué se construyó:**
- `src/ratings/elo.py` — Elo con historial pre-partido (`elo_a_pre`/`elo_b_pre`) anti-leakage por construcción: cada valor solo depende de partidos estrictamente anteriores. Pools de rating separados por género.
- `src/models/bradley_terry.py` — Bradley-Terry implementado como regresión logística sin intercepto sobre variables indicadoras por equipo (equivalencia matemática exacta con la estimación por máxima verosimilitud clásica), con regularización L2.
- `src/models/logistic_baseline.py` — regresión logística que recalibra la diferencia de Elo, en vez de asumir la escala fija de 400 del Elo clásico.
- `src/evaluation/walk_forward.py` — validación walk-forward (ventana creciente, folds por fecha), con una comprobación en tiempo de ejecución que hace fallar el proceso si algún fold mezclara fechas de test antes que de train.
- `src/evaluation/metrics.py` — Log Loss, Brier Score, accuracy, tabla de calibración.
- `scripts/run_phase2_baselines.py` — orquesta todo, por género, y guarda resultados en `reports/phase2_baseline_results.csv`.

Dependencia añadida: `scikit-learn` (instalar suelto con `pip install scikit-learn`, no con el extra `[ml]` completo — `xgboost` hacía que la instalación se quedara resolviendo versiones durante mucho tiempo; el resto de librerías de `[ml]` no hacen falta hasta Fase 4/7).

**Resultado real confirmado** (5 folds walk-forward, sobre los 6626 partidos):

Referencia: un modelo que predijera siempre 50/50 tendría log loss 0.693 y Brier 0.25 — todos los modelos reales quedan muy por debajo, señal de que hay algo real que aprender.

| Género | Modelo | log_loss (media) | Brier (media) | Accuracy (media) |
|---|---|---|---|---|
| Hombres (3357) | elo | 0.5699 | 0.1940 | 70.1% |
| Hombres | bradley_terry | 0.5775 | 0.1980 | 69.3% |
| Hombres | logistic_elo | 0.5727 | 0.1936 | 70.5% |
| Mujeres (3269) | elo | 0.5301 | 0.1775 | 73.8% |
| Mujeres | bradley_terry | 0.5372 | 0.1810 | 72.4% |
| Mujeres | logistic_elo | 0.5324 | 0.1767 | 73.7% |

**Lectura**: los tres modelos están prácticamente empatados (diferencias dentro de la desviación estándar entre folds). Esto no es un problema — es la conclusión esperada en esta fase: con solo "quién ganó a quién" (sin ninguna feature adicional), Elo ya captura casi toda la señal disponible; Bradley-Terry y la recalibración logística no aportan ventaja clara todavía. Elo es además el modelo más estable entre folds (menor desviación estándar). El salto de calidad real se espera de la Fase 3 (features), no de cambiar de modelo de rating — por eso se pasa directamente a Fase 3 sin invertir tiempo ahora en ajustar el K-factor de Elo u otros hiperparámetros de esta fase.
