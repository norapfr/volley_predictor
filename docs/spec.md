# Predictor de Vóley de Selecciones Nacionales — Especificación para Codex

> Proyecto: predictor probabilístico de partidos de voleibol entre selecciones nacionales senior, masculino y femenino. Sin clubes ni ligas.

## 1. Objetivo

Construir un sistema que, antes de cada partido, estime:

- P(team A gana)
- P(team B gana)
- P(3-0), P(3-1), P(3-2), P(2-3), P(1-3), P(0-3)
- marcador de sets más probable
- diferencia de puntos esperada
- Elo/rating de cada selección
- nivel de confianza
- factores explicativos

La prioridad es la **calidad y calibración de las probabilidades**, no únicamente accuracy.

---

## 2. Alcance

### Incluir

- Volleyball Nations League
- World Championship
- Olympic Games
- World Cup
- EuroVolley
- NORCECA
- South American Championship
- Asian Championship
- African Championship
- Otras competiciones oficiales internacionales de selecciones senior

FIVB mantiene un catálogo de competiciones y distingue tipos como Nations League, World Championship, World Cup y Continental Championship. [FIVB Competitions](https://www.fivb.com/volleyball/fivb-competitions/)

### Excluir

- Clubes
- Ligas nacionales
- Competiciones de clubes
- Selecciones juveniles en la V1

### Separación

Crear modelos independientes:

```text
models/men/
models/women/
```

Compartir infraestructura, pero no mezclar las distribuciones en el entrenamiento inicial.

---

# 3. Dataset

## 3.1 Fuente principal: FIVB VIS

La documentación oficial de FIVB indica que VIS permite a aplicaciones de terceros acceder a datos públicos de voleibol. [FIVB VIS Web Service](https://www.fivb.org/VisSDK/VisWebService/Introduction.html)

Recursos especialmente importantes:

- [VIS Web Service](https://www.fivb.org/VisSDK/VisWebService/webindex.html)
- [GetVolleyMatch](https://www.fivb.org/VisSDK/VisWebService/GetVolleyMatch.html)
- [GetVolleyMatchList](https://www.fivb.org/VisSDK/VisWebService/RequestList.html)
- [VolleyTournament](https://www.fivb.org/VisSDK/VisWebService/VolleyTournament.html)
- [VolleyMatchFilter](https://www.fivb.org/VisSDK/VisWebService/VolleyMatchFilter.html)
- [VolleyTournamentFilter](https://www.fivb.org/VisSDK/VisWebService/VolleyTournamentFilter.html)

VIS permite consultar partidos, torneos, equipos y estadísticas disponibles; el acceso exacto depende de los campos públicos expuestos por el servicio. No asumir que todas las estadísticas están disponibles para todos los torneos.

## 3.2 Datasets auxiliares

Utilizar datasets públicos como fuentes secundarias, validación y enriquecimiento.

### Kaggle

- VNL 2025 Men's:
  https://www.kaggle.com/datasets/owenhoag07/vnl-2025-mens

- VNL 2025 Player Data:
  https://www.kaggle.com/datasets/joshuali12/vnl-2025-player-data

- VNL 2024 Men's:
  https://www.kaggle.com/datasets/jonathanpmoyer/vnl-2024-mens-stats

- VNL 2024 Women's:
  https://www.kaggle.com/datasets/incdata/vnl-women-2024

No depender exclusivamente de Kaggle. El dataset maestro debe priorizar datos oficiales y conservar `source` y `source_match_id`.

---

# 4. Cobertura temporal

## MVP

```text
VNL 2021
VNL 2022
VNL 2023
VNL 2024
VNL 2025
VNL 2026
```

Masculino + femenino.

## Expansión

Añadir:

```text
Olympic Games
World Championship
World Cup
EuroVolley
NORCECA
South American Championship
Asian Championship
African Championship
otras competiciones senior oficiales
```

Objetivo orientativo:

```text
2015-2026
```

pero solo si la calidad y consistencia de los datos lo permiten.

---

# 5. Dataset maestro

## matches

```text
match_id
date
gender
competition
competition_id
stage
team_a
team_b
team_a_score
team_b_score
sets_a
sets_b
winner
venue
country_host
neutral
source
source_match_id
```

## sets

```text
match_id
set_number
team_a_points
team_b_points
```

## team_match_stats

```text
match_id
team
attack_attempts
attack_points
attack_errors
attack_efficiency
blocks
aces
serve_errors
reception_attempts
positive_reception
perfect_reception
digs
sideout
break_points
points
sets
```

No convertir `missing` en `0`.

## teams

```text
team_id
canonical_name
gender
country_code
continent
active
```

## team_aliases

```text
canonical_team_id
source
source_name
```

Ejemplo:

```text
USA
United States
Estados Unidos
USA (US)
```

deben mapear a una única selección.

---

# 6. Regla crítica: NO DATA LEAKAGE

Para un partido en fecha `T`, solo pueden utilizarse datos que hubieran estado disponibles antes de `T`.

Nunca usar:

- resultado del partido objetivo
- estadísticas del partido objetivo
- ranking posterior
- resultados futuros
- medias acumuladas que incluyan el partido objetivo
- datos de torneos futuros
- target encoding construido con información futura
- calibración entrenada con partidos posteriores
- stacking entrenado con datos que violen la frontera temporal

Crear tests automáticos de leakage.

Test fundamental:

```text
features(match_T) no puede depender de result(match_T)
```

---

# 7. Feature engineering

## 7.1 Elo

Crear:

- Elo global
- Elo reciente
- Elo con decaimiento temporal
- diferencia Elo
- Elo ajustado por margen si mejora el backtest

Elo será el baseline principal.

También probar Bradley-Terry.

## 7.2 Forma

Ventanas:

```text
3
5
10
20
365 días
```

Variables:

```text
win_rate
sets_win_rate
points_win_rate
set_difference
point_difference
```

## 7.3 Strength of Schedule

Calcular:

```text
average_opponent_elo_last_5
average_opponent_elo_last_10
strength_of_schedule
```

No asumir que 5 victorias contra rivales débiles equivalen a 5 victorias contra equipos top.

## 7.4 Estadísticas técnicas

Cuando estén disponibles:

### Ataque

```text
attack_efficiency
attack_points_per_set
attack_errors_per_set
```

### Bloqueo

```text
blocks_per_set
block_points
```

### Saque

```text
aces_per_set
serve_errors_per_set
serve_efficiency
```

### Recepción

```text
positive_reception_rate
perfect_reception_rate
reception_efficiency
```

### Defensa

```text
digs_per_set
```

### Juego

```text
sideout_rate
break_point_rate
```

Crear diferencias:

```text
metric_diff = team_a_metric - team_b_metric
```

## 7.5 Descanso

```text
days_since_last_match
matches_last_7d
matches_last_14d
matches_last_30d
back_to_back
```

## 7.6 Competición

```text
competition
competition_level
competition_importance
stage
knockout
quarterfinal
semifinal
final
```

No permitir que competición se convierta artificialmente en proxy de fuerza.

## 7.7 Localía

```text
home
away
neutral
host_country
```

En torneos internacionales asumir neutralidad cuando corresponda.

## 7.8 Head-to-head

Probar:

```text
h2h_last_3
h2h_last_5
h2h_win_rate
```

Eliminarlo si no mejora el rendimiento out-of-sample.

---

# 8. Modelos

## Baselines

### Elo

Baseline estadístico.

### Logistic Regression

Baseline interpretable.

### Bradley-Terry

Modelo probabilístico de fuerza relativa.

## ML tabular

### LightGBM

Candidato principal:

```text
binary classification
loss = binary logloss
```

Ventajas:

- excelente en datos tabulares
- relaciones no lineales
- rápido
- adecuado para datasets medianos
- SHAP

### XGBoost

Competidor y fuente de diversidad para ensemble.

### CatBoost

Competidor especialmente interesante para categóricas.

## Deep Learning

### MLP

Arquitectura inicial:

```text
Input
↓
Dense 128
↓
Dropout
↓
Dense 64
↓
Dense 32
↓
Output
```

### LSTM/GRU

Representar los últimos N partidos de cada selección como secuencia.

### Transformer temporal

Solo probar si los modelos tabulares ya están estabilizados y existe suficiente historial.

No asumir que Deep Learning será mejor.

---

# 9. Ensemble

Probar:

```text
Elo
LightGBM
XGBoost
CatBoost
MLP
```

Métodos:

- weighted averaging
- stacking

No fijar pesos manualmente.

Los pesos/meta-modelo deben aprenderse respetando el tiempo.

---

# 10. Modelo de sets

Clases:

```text
3-0
3-1
3-2
2-3
1-3
0-3
```

Probar dos estrategias.

### A. Multiclass

LightGBM/XGBoost multiclass.

### B. Generativa

Estimar probabilidades de ganar cada set y simular el partido.

Comparar ambas out-of-sample.

Las probabilidades deben sumar 1.

---

# 11. Modelo de puntos

Crear posteriormente un regresor para:

```text
point_difference
```

Probar:

- LightGBM Regression
- XGBoost Regression
- Random Forest Regression

Métricas:

```text
MAE
RMSE
```

---

# 12. Calibración

Obligatoria.

Probar:

- Platt scaling / sigmoid
- isotonic regression

Evaluar:

- reliability diagrams
- Brier Score
- Expected Calibration Error
- Log Loss

Objetivo:

> Si el modelo dice 70%, aproximadamente el 70% de esos casos debe producir victoria a largo plazo.

---

# 13. Validación

NO utilizar:

```python
train_test_split(..., shuffle=True)
```

como evaluación final.

Usar walk-forward / expanding window.

Ejemplo:

```text
TRAIN 2015-2020 → TEST 2021
TRAIN 2015-2021 → TEST 2022
TRAIN 2015-2022 → TEST 2023
TRAIN 2015-2023 → TEST 2024
TRAIN 2015-2024 → TEST 2025
TRAIN 2015-2025 → TEST 2026
```

Adaptar a los datos reales.

Todas las transformaciones, tuning, selección de features y calibración deben respetar la frontera temporal.

---

# 14. Métricas

## Prioridad

1. Log Loss
2. Brier Score
3. Calibration / ECE
4. Estabilidad entre años
5. Accuracy

## Secundarias

- ROC-AUC
- F1
- Precision
- Recall
- Confusion Matrix

## Sets

- multiclass Log Loss
- multiclass Brier
- accuracy
- calibration

## Puntos

- MAE
- RMSE

Accuracy nunca será la única métrica para seleccionar el modelo.

---

# 15. Ablation testing

Ejecutar:

```text
Modelo A = Elo

Modelo B = Elo + forma

Modelo C = Elo + forma + estadísticas

Modelo D = Elo + forma + estadísticas + descanso

Modelo E = todas las features
```

Comparar exclusivamente out-of-sample.

---

# 16. Explicabilidad

Usar:

- LightGBM feature importance
- permutation importance
- SHAP

Generar explicación global y por partido.

Ejemplo:

```text
Brasil 68.4%

Factores positivos:
+ Elo
+ ataque reciente
+ sideout
+ fuerza de calendario

Factores negativos:
- bloqueo rival
```

No presentar SHAP como causalidad.

---

# 17. Concept drift

Analizar:

```text
2015-2018
2019-2022
2023-2026
```

Comparar:

- expanding window
- rolling window
- temporal decay

Elegir según rendimiento temporal.

---

# 18. Dos modos de predictor

## Basic pre-match

Debe funcionar con:

- Elo
- ranking
- forma
- fuerza del rival
- competición
- descanso

## Full pre-match

Añadir:

- ataque
- bloqueo
- saque
- recepción
- defensa
- sideout
- break points
- otras estadísticas disponibles

Esto permite trabajar incluso con competiciones nuevas sin estadísticas completas.

---

# 19. Jugadores

NO usar jugadores como unidad principal en V1.

Después de demostrar que el modelo de equipos funciona, investigar:

- convocatoria
- bajas
- disponibilidad
- cambios de roster
- rendimiento individual

Solo añadir si mejora el backtest.

---

# 20. Cuotas de apuestas

No usar cuotas como feature en V1.

Pueden utilizarse posteriormente como benchmark externo.

---

# 21. LLM

No utilizar GPT/Claude/Gemini para producir directamente probabilidades.

Un LLM puede utilizarse para:

- resumir noticias
- estructurar texto
- generar explicaciones
- extraer información textual

pero la probabilidad debe provenir del modelo estadístico/ML.

---

# 22. Arquitectura

```text
volley_predictor/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── src/
│   ├── ingestion/
│   ├── cleaning/
│   ├── normalization/
│   ├── features/
│   ├── ratings/
│   ├── models/
│   ├── calibration/
│   ├── evaluation/
│   ├── prediction/
│   └── api/
│
├── models/
│   ├── men/
│   └── women/
│
├── notebooks/
├── tests/
├── configs/
├── reports/
└── README.md
```

---

# 23. Stack

```text
Python 3.12+
numpy
pandas
polars
scikit-learn
lightgbm
xgboost
catboost
scipy
statsmodels
shap
mlflow
pytest
pydantic
fastapi
postgresql
pytorch
```

PyTorch únicamente para modelos neuronales.

---

# 24. PostgreSQL

Tablas:

```text
teams
team_aliases
competitions
matches
sets
team_match_stats
players
player_match_stats
team_ratings
predictions
model_runs
```

---

# 25. MLflow

Registrar:

- modelo
- features
- periodo de entrenamiento
- periodo de test
- hiperparámetros
- Log Loss
- Brier
- ECE
- Accuracy
- ROC-AUC
- artifact
- versión del dataset
- commit Git

Nunca sobrescribir experimentos.

---

# 26. API

```text
POST /predict
```

Input:

```json
{
  "gender": "men",
  "team_a": "Brazil",
  "team_b": "Italy",
  "competition": "VNL",
  "date": "2026-08-20"
}
```

Output:

```json
{
  "team_a_win_probability": 0.684,
  "team_b_win_probability": 0.316,
  "set_probabilities": {},
  "expected_point_difference": 6.8,
  "elo_a": 1850,
  "elo_b": 1710,
  "confidence": 0.684,
  "explanations": []
}
```

---

# 27. Orden de implementación

## Fase 1 — Dataset

- repositorio
- schema
- ingestión
- limpieza
- normalización
- validación

## Fase 2 — Baseline

- Elo
- Logistic Regression
- Bradley-Terry
- walk-forward

## Fase 3 — Features

- forma
- strength of schedule
- estadísticas
- descanso
- competición

## Fase 4 — ML

- LightGBM
- XGBoost
- CatBoost

## Fase 5 — Probabilidades

- calibración
- ensemble
- SHAP

## Fase 6 — Sets

- multiclass
- generativo/simulación

## Fase 7 — Puntos

- regresión

## Fase 8 — Deep Learning

- MLP
- LSTM/GRU
- Transformer solo si aporta

## Fase 9 — API

- FastAPI

## Fase 10 — Dashboard

- selección
- competición
- predicción
- explicación
- histórico

---

# 28. Tests obligatorios

Testear:

- normalización de selecciones
- duplicados
- cálculo Elo
- rolling statistics
- features temporales
- ausencia de leakage
- probabilidades que sumen 1
- calibración
- API
- equipos desconocidos

Test crítico:

```text
feature(match_T) no puede depender de result(match_T)
```

---

# 29. Literatura

## Vóley + ML

- Scientific Reports — Women's volleyball + gradient boosting:
  https://www.nature.com/articles/s41598-025-26344-y

- PubMed:
  https://pubmed.ncbi.nlm.nih.gov/41309801/

- PMC:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12660672/

- Machine learning in sports analytics — volleyball:
  https://researchoutput.ncku.edu.tw/en/publications/machine-learning-in-sports-analytics-volleyball-match-outcome-pre/

- Research sobre predicción de vóley:
  https://labs.sciety.org/articles/by?article_doi=10.21203%2Frs.3.rs-9967646%2Fv1

## Rating

- Bradley-Terry-Elo:
  https://arxiv.org/abs/1701.08055

## Gradient Boosting

- XGBoost:
  https://arxiv.org/abs/1603.02754

- LightGBM:
  https://proceedings.neurips.cc/paper/6907-a-highly-efficient-gradient-boosting-decision-tree.pdf

- CatBoost:
  https://arxiv.org/abs/1706.09516

## Explicabilidad

- SHAP:
  https://shap.readthedocs.io/

---

# 30. Ruta de aprendizaje

## Nivel 1

- Python
- NumPy
- Pandas/Polars
- SQL
- probabilidad
- estadística
- Logistic Regression

## Nivel 2

- árboles
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- overfitting
- regularización

## Nivel 3

- Elo
- Bradley-Terry
- rolling statistics
- strength of schedule
- time decay
- walk-forward validation

## Nivel 4

- Log Loss
- Brier Score
- calibration
- Platt scaling
- isotonic regression
- reliability diagrams
- uncertainty

## Nivel 5

- PyTorch
- MLP
- RNN
- LSTM
- GRU
- Transformer temporal

---

# 31. Qué NO estudiar todavía

No priorizar:

- Reinforcement Learning
- GANs
- Diffusion
- fine-tuning de LLMs
- Computer Vision
- GNNs

Solo investigar estas tecnologías si aparece una necesidad real.

---

# 32. Prompt maestro para Codex

Copia este bloque directamente como prompt inicial de Codex:

```text
QUIERO CONSTRUIR DESDE CERO UN SISTEMA PROFESIONAL DE MACHINE LEARNING PARA PREDECIR PARTIDOS DE VOLEIBOL ENTRE SELECCIONES NACIONALES.

ALCANCE:
- Solo selecciones nacionales senior.
- Masculino y femenino con modelos independientes.
- NO clubes ni ligas.
- VNL, World Championship, Olympic Games, World Cup, EuroVolley, NORCECA, South American Championship, Asian Championship, African Championship y otras competiciones oficiales.
- El sistema debe generalizar entre competiciones.

OBJETIVO:
Dado team_a, team_b, gender, competition, date y stage, devolver:
- P(team_a gana)
- P(team_b gana)
- P(3-0), P(3-1), P(3-2), P(2-3), P(1-3), P(0-3)
- marcador de sets más probable
- diferencia de puntos esperada
- Elo/rating
- confianza
- factores explicativos

REGLA ABSOLUTA:
NO DATA LEAKAGE.
Para un partido en T solo usar información disponible antes de T.
No usar estadísticas del partido objetivo, rankings futuros, resultados futuros, acumulados que incluyan T, ni transformaciones que vean el futuro.
Crear tests automáticos de leakage.

DATOS:
Prioridad:
1. FIVB / Volleyball World.
2. Datasets públicos de Kaggle.
3. Repositorios académicos.
4. Otras fuentes abiertas verificables.

MVP:
VNL 2021-2026 masculino y femenino.

Después:
Olympics + World Championship + World Cup + continentales.
Objetivo orientativo: 2015-2026 si la calidad lo permite.

DATA MODEL:
Crear matches, sets, team_match_stats, teams, team_aliases, competitions, team_ratings.
Players/player_match_stats son opcionales y NO forman parte del núcleo de V1.

FEATURES:
- Elo global/recent/time-decay/margin-adjusted.
- Bradley-Terry baseline.
- win rate 3/5/10/20.
- set/point win rate.
- set/point differential.
- strength of schedule.
- attack efficiency.
- blocks/set.
- aces/set.
- serve errors.
- reception efficiency.
- digs/set.
- sideout.
- break points.
- days since last match.
- congestion 7/14/30d.
- competition/stage.
- neutral/home/host.
- h2h 3/5 solo si mejora el backtest.

MODELOS:
Baselines:
- Elo
- Logistic Regression
- Bradley-Terry

Tabular:
- LightGBM (candidato principal)
- XGBoost
- CatBoost

Neural:
- MLP
- LSTM/GRU
- Transformer temporal solo si demuestra mejora

ENSEMBLE:
Comparar weighted averaging y stacking.
No fijar pesos manualmente.
Todo entrenamiento de pesos debe ser temporalmente correcto.

SETS:
Crear modelo multiclass y modelo generativo/simulación.
Comparar out-of-sample.

POINTS:
Regresión para point_difference.
Métricas MAE/RMSE.

CALIBRATION:
- sigmoid/Platt
- isotonic
- reliability diagrams
- Brier
- ECE
- Log Loss

VALIDATION:
NO random train/test para evaluación final.
Usar walk-forward / expanding window.
Ejemplo:
2015-2020 -> 2021
2015-2021 -> 2022
...
Adaptar a datos reales.

METRICS:
Prioridad:
1 Log Loss
2 Brier
3 Calibration
4 estabilidad temporal
5 Accuracy
Además ROC-AUC, F1, Precision, Recall.
Para sets: multiclass log loss/Brier/accuracy/calibration.
Para puntos: MAE/RMSE.

ABLATION:
A Elo
B Elo + forma
C Elo + forma + estadísticas
D Elo + forma + estadísticas + descanso
E todas

EXPLICABILITY:
LightGBM feature importance + permutation importance + SHAP.
No presentar SHAP como causalidad.

STACK:
Python 3.12+, numpy, pandas/polars, scikit-learn, lightgbm, xgboost, catboost, scipy, statsmodels, shap, mlflow, pytest, pydantic, fastapi, postgresql, PyTorch solo para neural.

STRUCTURE:
volley_predictor/
data/raw interim processed
src/ingestion cleaning normalization features ratings models calibration evaluation prediction api
models/men women
notebooks
tests
configs
reports

ROADMAP:
Fase 1 dataset/schema/cleaning/normalization
Fase 2 Elo + baselines + backtest
Fase 3 features
Fase 4 LightGBM/XGBoost/CatBoost
Fase 5 calibration + ensemble + SHAP
Fase 6 sets
Fase 7 points
Fase 8 deep learning si aporta
Fase 9 API
Fase 10 dashboard

REGLAS:
- No inventar datos.
- No inventar métricas.
- No afirmar que el modelo funciona sin backtest out-of-sample.
- No usar clubes.
- No usar cuotas de apuestas como feature en V1.
- No usar LLM para generar directamente probabilidades.
- No avanzar de fase si hay tests fallando o leakage.
- Registrar dataset version, Git commit, hyperparameters y métricas con MLflow.
- Después de cada fase: mostrar archivos creados, tests, métricas, problemas y siguiente paso.
```

---

# 33. Arquitectura final

```text
DATOS FIVB / VOLLEYBALL WORLD
             +
       DATASETS AUXILIARES
             ↓
       DATASET MAESTRO
             ↓
       NORMALIZACIÓN
             ↓
     FEATURE ENGINEERING
             ↓
    ┌────────┼─────────┐
    ↓        ↓         ↓
   ELO    LIGHTGBM   XGBOOST
    │        │         │
    │      CATBOOST    │
    │        │         │
    └────────┼─────────┘
             ↓
          ENSEMBLE
             ↓
        CALIBRACIÓN
             ↓
      ┌──────┼───────┐
      ↓      ↓       ↓
   GANADOR  SETS   PUNTOS
      ↓      ↓       ↓
      └──────┼───────┘
             ↓
         PREDICTOR
             ↓
       API / DASHBOARD
```

---

# 34. Criterio de éxito

El proyecto NO se considera terminado porque "acierta muchos partidos".

Se considera terminado cuando demuestra mediante backtesting temporal que:

1. supera los baselines;
2. mantiene rendimiento en años distintos;
3. produce probabilidades calibradas;
4. no presenta leakage;
5. es reproducible;
6. puede generalizar entre competiciones.

Si:

```text
LightGBM + Elo > Transformer
```

se mantiene:

```text
LightGBM + Elo
```

La complejidad solo se justifica cuando produce una mejora medible y reproducible.
