<div align="center">

# Volley Predictor

Professional volleyball match prediction system for senior national teams.

Sistema profesional de predicción de partidos de voleibol entre selecciones nacionales senior.

[English](#english) | [Español](#español)

</div>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img alt="CatBoost" src="https://img.shields.io/badge/CatBoost-ML-FFCC00?style=for-the-badge">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-Arrays-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-Metrics-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Ready-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img alt="pytest" src="https://img.shields.io/badge/pytest-Tested-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white">
</p>

<p align="center">
  <img alt="Matches" src="https://img.shields.io/badge/6%2C626-valid_matches-ff6b35?style=flat-square">
  <img alt="Men accuracy" src="https://img.shields.io/badge/Men_accuracy-72.5%25-2ea44f?style=flat-square">
  <img alt="Women accuracy" src="https://img.shields.io/badge/Women_accuracy-76.0%25-2ea44f?style=flat-square">
  <img alt="Exact score" src="https://img.shields.io/badge/Exact_score-39.7%25_men_%2F_47.8%25_women-blue?style=flat-square">
</p>

---

## English

Volley Predictor is an end-to-end machine learning project that predicts international volleyball matches for men's and women's national teams. It includes data ingestion from FIVB VIS, cleaning and normalization, Elo ratings, feature engineering, walk-forward validation, trained CatBoost models, a FastAPI-ready prediction layer, and a Streamlit dashboard.

The app is designed for realistic future match prediction. The interface does not allow manual match dates because the model uses the latest saved team state. This avoids misleading historical simulations and prevents predictions from changing just because a user chooses a different future date.

### Key Results

| Area | Result |
|---|---:|
| Valid historical matches | 6,626 |
| Men's matches | 3,357 |
| Women's matches | 3,269 |
| Date range | 2006-10-31 to 2026-08-27 |
| Unique teams | 158 |
| Men's trained teams | 154 |
| Women's trained teams | 134 |
| Final winner model | CatBoost |

### Winner Prediction Performance

Walk-forward validation was used so every test fold is evaluated on matches that happen after the training data.

| Gender | Model | Log Loss | Brier Score | Accuracy |
|---|---|---:|---:|---:|
| Men | CatBoost | 0.5549 | 0.1861 | 72.5% |
| Women | CatBoost | 0.5031 | 0.1637 | 76.0% |

CatBoost outperformed Elo, Bradley-Terry, logistic regression on Elo, LightGBM, and XGBoost in the final comparison.

### Exact Score and Point Difference

The system predicts more than just the winner:

| Output | Meaning | Result / Definition |
|---|---|---|
| Win probability | Probability that each team wins | `P(team_a)` and `P(team_b) = 1 - P(team_a)` |
| Most likely score | Most probable set score | One of `3-0`, `3-1`, `3-2`, `2-3`, `1-3`, `0-3` |
| Exact score accuracy | Correct exact score prediction | 39.7% men / 47.8% women |
| Lead over next score | Difference between the most likely score and the second most likely score | Shown in percentage points (`pp`) |
| Expected total points diff | Expected full-match point margin | `team_a_total_points - team_b_total_points` |
| Point difference MAE | Average absolute error for total point margin | 10.98 men / 12.57 women |

Example: an expected total points diff of `+6.4` means Team A is projected to finish about 6.4 total points ahead across the full match. It is not Elo and it is not points per set.

### What The Dashboard Shows

- Team A and Team B win probabilities.
- A probability bar comparing both teams.
- Most likely set score and exact score confidence.
- Lead over next score, which shows how decisive the exact-score prediction is.
- Elo rating for each team.
- Expected total points difference for the full match.
- A "Why this prediction?" section with readable matchup context.

The "Why this prediction?" section is based on descriptive features such as Elo difference, current streak, head-to-head history, recent win rate, competition importance, and expected total points difference. It is not a SHAP-style causal explanation of CatBoost internals.

### Features Used By The Models

- Elo difference.
- Matches played difference.
- Current streak difference.
- Days since last match difference.
- Head-to-head matches played.
- Head-to-head win rate.
- Competition importance.
- Tournament progress.
- Win rate over the last 5 and 10 matches.
- Average set margin over the last 5 and 10 matches.

### Tech Stack

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-Data_processing-150458?style=flat-square&logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-Numerical_features-013243?style=flat-square&logo=numpy&logoColor=white">
  <img alt="CatBoost" src="https://img.shields.io/badge/CatBoost-Gradient_boosting-FFCC00?style=flat-square">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-Baselines_%26_metrics-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
  <img alt="Pydantic" src="https://img.shields.io/badge/Pydantic-Schemas-E92063?style=flat-square&logo=pydantic&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Prediction_layer-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Interactive_UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
  <img alt="pytest" src="https://img.shields.io/badge/pytest-Test_suite-0A9EDC?style=flat-square&logo=pytest&logoColor=white">
</p>

The project also includes a FIVB VIS ingestion pipeline, walk-forward evaluation, leakage checks, model persistence, and a Streamlit interface for portfolio-ready demonstrations.

### Run Locally

For the dashboard:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

For full development, including training and evaluation:

```bash
pip install -e ".[dev,ml]"
pytest -v
```

Main training/evaluation scripts:

```bash
python scripts/run_phase2_baselines.py
python scripts/run_phase4_tabular.py
python scripts/run_phase5_calibration_ensemble.py
python scripts/run_phase6_set_scores.py
python scripts/run_phase7_point_diff.py
python scripts/train_final_models.py
```

### Project Structure

```text
volley_predictor/
  streamlit_app.py                 # Streamlit dashboard
  src/api/                         # Prediction API layer
  src/features/                    # Feature engineering
  src/models/                      # ML models
  src/ratings/                     # Elo ratings
  src/ingestion/                   # FIVB VIS and CSV ingestion
  src/cleaning/                    # Dataset cleaning
  src/normalization/               # Country/team normalization
  src/evaluation/                  # Metrics, leakage checks, validation
  models/{men,women}/              # Trained model artifacts
  data/processed/                  # Cleaned datasets
  configs/                         # Competition and team config
  tests/                           # Test suite
  PROGRESS_LOG.md                  # Detailed development log
```

---

## Español

Volley Predictor es un proyecto completo de machine learning para predecir partidos internacionales de voleibol entre selecciones nacionales masculinas y femeninas. Incluye ingestión de datos desde FIVB VIS, limpieza y normalización, ratings Elo, feature engineering, validación walk-forward, modelos CatBoost entrenados, una capa de predicción compatible con FastAPI y una interfaz en Streamlit.

La aplicación está pensada para predecir partidos futuros de forma realista. La interfaz no permite elegir manualmente la fecha del partido porque el modelo usa el último estado guardado de cada equipo. Así se evita simular partidos pasados de forma engañosa y también se evita que la predicción cambie solo por escoger otra fecha futura.

### Resultados principales

| Area | Resultado |
|---|---:|
| Partidos históricos válidos | 6,626 |
| Partidos masculinos | 3,357 |
| Partidos femeninos | 3,269 |
| Rango de fechas | 2006-10-31 a 2026-08-27 |
| Equipos únicos | 158 |
| Equipos masculinos entrenados | 154 |
| Equipos femeninos entrenados | 134 |
| Modelo final de ganador | CatBoost |

### Rendimiento del modelo de ganador

Se usó validación walk-forward: cada bloque de test ocurre después de los datos usados para entrenar.

| Género | Modelo | Log Loss | Brier Score | Accuracy |
|---|---|---:|---:|---:|
| Hombres | CatBoost | 0.5549 | 0.1861 | 72.5% |
| Mujeres | CatBoost | 0.5031 | 0.1637 | 76.0% |

CatBoost fue el modelo final porque superó a Elo, Bradley-Terry, regresión logística sobre Elo, LightGBM y XGBoost en la comparación final.

### Marcador exacto y diferencia de puntos

El sistema predice más que el ganador:

| Número en la interfaz | Qué significa | Resultado / Definición |
|---|---|---|
| Probabilidad de victoria | Probabilidad de ganar de cada equipo | `P(team_a)` y `P(team_b) = 1 - P(team_a)` |
| Marcador más probable | Marcador de sets más probable | Uno de `3-0`, `3-1`, `3-2`, `2-3`, `1-3`, `0-3` |
| Acierto de marcador exacto | Acierto prediciendo el marcador exacto | 39.7% hombres / 47.8% mujeres |
| Lead over next score | Ventaja del marcador mas probable sobre el segundo | Se muestra en puntos porcentuales (`pp`) |
| Expected total points diff | Diferencia esperada de puntos totales del partido | `puntos_totales_team_a - puntos_totales_team_b` |
| MAE de diferencia de puntos | Error absoluto medio del margen total de puntos | 10.98 hombres / 12.57 mujeres |

Ejemplo: si `Expected total points diff` muestra `+6.4`, significa que Team A se proyecta unos 6.4 puntos totales por encima de Team B en todo el partido. No es Elo y no son puntos por set.

### Qué muestra la interfaz

- Probabilidad de victoria de Team A y Team B.
- Barra comparativa de probabilidad entre ambos equipos.
- Marcador de sets más probable y confianza del marcador exacto.
- `Lead over next score`, que indica si el marcador exacto es claro o esta muy cerca de otros resultados.
- Elo de cada equipo.
- Diferencia esperada de puntos totales del partido completo.
- Sección "Why this prediction?" con contexto legible del enfrentamiento.

La sección "Why this prediction?" se basa en estadísticas descriptivas: diferencia de Elo, racha actual, historial directo, forma reciente, importancia de la competición y diferencia esperada de puntos totales. No es una explicación causal interna tipo SHAP del modelo CatBoost.

### Variables usadas por los modelos

- Diferencia de Elo.
- Diferencia de partidos jugados.
- Diferencia de racha actual.
- Diferencia de días desde el último partido.
- Partidos directos entre ambos equipos.
- Porcentaje de victorias en el head-to-head.
- Importancia de la competicion.
- Progreso dentro del torneo.
- Win rate en los últimos 5 y 10 partidos.
- Margen medio de sets en los últimos 5 y 10 partidos.

### Tecnologías

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-Procesamiento_de_datos-150458?style=flat-square&logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-Features_numericas-013243?style=flat-square&logo=numpy&logoColor=white">
  <img alt="CatBoost" src="https://img.shields.io/badge/CatBoost-Gradient_boosting-FFCC00?style=flat-square">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-Baselines_y_metricas-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
  <img alt="Pydantic" src="https://img.shields.io/badge/Pydantic-Schemas-E92063?style=flat-square&logo=pydantic&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Capa_de_prediccion-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Interfaz_interactiva-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
  <img alt="pytest" src="https://img.shields.io/badge/pytest-Suite_de_tests-0A9EDC?style=flat-square&logo=pytest&logoColor=white">
</p>

El proyecto también incluye pipeline de ingestión desde FIVB VIS, validación walk-forward, checks anti-leakage, persistencia de modelos y una interfaz Streamlit lista para demo de portfolio.

### Ejecutar localmente

Para abrir la interfaz:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Para desarrollo completo, entrenamiento y evaluacion:

```bash
pip install -e ".[dev,ml]"
pytest -v
```

Scripts principales:

```bash
python scripts/run_phase2_baselines.py
python scripts/run_phase4_tabular.py
python scripts/run_phase5_calibration_ensemble.py
python scripts/run_phase6_set_scores.py
python scripts/run_phase7_point_diff.py
python scripts/train_final_models.py
```

### Estructura

```text
volley_predictor/
  streamlit_app.py                 # Dashboard Streamlit
  src/api/                         # Capa de predicción
  src/features/                    # Feature engineering
  src/models/                      # Modelos ML
  src/ratings/                     # Ratings Elo
  src/ingestion/                   # Ingestión FIVB VIS y CSV
  src/cleaning/                    # Limpieza del dataset
  src/normalization/               # Normalizacion de paises/equipos
  src/evaluation/                  # Métricas, leakage, validación
  models/{men,women}/              # Modelos entrenados
  data/processed/                  # Datasets limpios
  configs/                         # Configuracion de competiciones/equipos
  tests/                           # Suite de tests
  PROGRESS_LOG.md                  # Registro detallado del desarrollo
```
