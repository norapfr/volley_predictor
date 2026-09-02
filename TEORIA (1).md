# Volley Predictor — Guía de teoría

> Compañero de estudio de `PROGRESS_LOG.md`. Ese documento cuenta QUÉ se hizo y QUÉ resultados dio; este cuenta el PORQUÉ — la teoría detrás de cada decisión, organizada por fase y por archivo, para que sepas defender cualquier parte del proyecto si te preguntan.

---

## Fase 1 — Dataset y schema

### `src/schema.py` — Validación de datos con Pydantic
**Teoría**: un modelo de datos (`Match`, `Competition`, `TeamRating`...) no es solo tipado — cada `Field` con validación (`ge=0`, enums para `Winner`/`Stage`) actúa como un contrato: si un dato no lo cumple, falla en el momento de entrada, no 3 pasos después de forma críptica. Esto se llama **"fail fast"** — cuanto antes falle un dato malo, más barato es arreglarlo.

**Decisión clave**: el validador `_consistent_result` en `Match` comprueba que el marcador de sets sea coherente con el ganador declarado (nunca un 3-3, nunca ganador contradictorio). Esto es una **invariante de dominio** — una regla que viene del mundo real (el vóley se juega al mejor de 5), no de la base de datos.

### `src/ingestion/*.py` — Patrón "fuente → formato común"
**Teoría**: cada fuente de datos (FIVB VIS, Kaggle) tiene su propio formato crudo. En vez de que cada fuente "sepa" cómo convertirse directamente al dataset maestro, todas convergen a un formato intermedio común (`RawMatchRow`) — este es el **patrón adaptador** (adapter pattern): aísla el "ruido" de cada fuente para que la limpieza (`cleaning/matches.py`) solo se escriba una vez, no una por fuente.

### `src/cleaning/matches.py` — Limpieza tolerante a fallos
**Teoría**: `build_matches` nunca deja que una fila mala tumbe el proceso entero — acumula errores en `rejected` con motivo legible. Esto es **degradación elegante** (graceful degradation): un dataset de 6626 partidos válidos con 69 rechazados documentados es mucho más útil que un proceso que revienta al primer error y no produce nada.

### `src/normalization/country_lookup.py` — Resolución de entidades
**Teoría**: "USA", "United States" y "Estados Unidos" son la misma entidad con distinto nombre superficial — esto se llama **resolución de entidades** (entity resolution), un problema clásico de calidad de datos. La decisión de usar ISO 3166 (vía `pycountry`) en vez de un diccionario manual es una elección de **escalabilidad**: mantener alias a mano para ~200 países no escala; delegar en un estándar internacional sí.

### `src/evaluation/leakage.py` — Anti-leakage como principio, no como parche
**Teoría del proyecto entero**: "data leakage" es cuando un modelo, durante el entrenamiento, tiene acceso (directo o indirecto) a información que en producción no tendría. La forma más traicionera de leakage es **temporal**: usar en la fecha T información de fecha > T. `assert_no_future_leakage` no es un detalle de la Fase 1 — es la regla que gobierna Elo, features, walk-forward y calibración en todas las fases siguientes.

---

## Fase 2 — Baselines (Elo, Bradley-Terry, Logistic-sobre-Elo)

### `src/ratings/elo.py` — El sistema Elo
**Teoría**: Elo modela la fuerza de un equipo como un número (`rating`) y predice el resultado con una función logística:

```
P(A gana) = 1 / (1 + 10^((rating_B − rating_A) / 400))
```

Después de cada partido, el rating se actualiza proporcionalmente a la **sorpresa** del resultado:

```
nuevo_rating_A = rating_A + K × (resultado_real − resultado_esperado)
```

- Si gana el favorito → `resultado_real ≈ resultado_esperado` → apenas cambia el rating.
- Si gana el que no se esperaba → gran diferencia → el rating salta.

`K` (el "K-factor") controla cuánto peso tiene cada partido nuevo frente al historial — un K alto hace que el rating reaccione rápido pero sea ruidoso; un K bajo lo hace estable pero lento para detectar mejoras reales de un equipo.

**Por qué es la base anti-leakage de todo lo demás**: `compute_elo_history` guarda `elo_a_pre`/`elo_b_pre` — el rating tal y como estaba **antes** de cada partido. Cualquier feature derivada de ahí hereda la garantía de no-leakage gratis.

### `src/models/bradley_terry.py` — Bradley-Terry como regresión logística
**Teoría**: Bradley-Terry asigna a cada equipo una fuerza latente `s_i`, y modela:

```
P(A gana a B) = sigmoid(s_A − s_B)
```

Esto es **matemáticamente idéntico** a una regresión logística sin intercepto, con una variable indicadora por equipo (+1 para A, −1 para B). Por eso el código no reimplementa el algoritmo iterativo clásico (Zermelo) — usa `sklearn.LogisticRegression` directamente. La **regularización L2** (`C` en sklearn) actúa como un *prior* que evita que un equipo con pocos partidos y 100% de victorias reciba una fuerza infinita (sobreajuste extremo con poca muestra).

### `src/models/logistic_baseline.py` — Recalibrar en vez de asumir
**Teoría**: la fórmula de Elo asume una escala fija (400). `EloLogisticModel` deja que una regresión logística **aprenda** esa escala de los datos reales, en vez de asumirla — es el mismo principio de "no asumas lo que puedes medir" que rige todo el proyecto.

### `src/evaluation/walk_forward.py` — Por qué nunca k-fold aleatorio
**Teoría**: en un problema con orden temporal, un split aleatorio (k-fold clásico) permite que el modelo entrene con partidos de 2023 y sea evaluado en partidos de 2020 — eso es leakage por definición (en producción nunca tendrías datos del futuro). **Walk-forward validation** con ventana creciente (*expanding window*) resuelve esto: cada fold entrena solo con lo estrictamente anterior al bloque de test.

### `src/evaluation/metrics.py` — Por qué Log Loss antes que Accuracy
**Teoría**: accuracy solo mide "¿acertó el ganador?" — no dice nada sobre si el nivel de confianza tenía sentido. Un modelo que dice "90% seguro" y acierta solo el 60% de las veces tiene mala **calibración**, aunque su accuracy parezca decente.

- **Log Loss** = `−mean(y·log(p) + (1−y)·log(1−p))` — penaliza exponencialmente estar "seguro y equivocado" (decir 99% y fallar es mucho peor que decir 60% y fallar).
- **Brier Score** = `mean((p − y)²)` — el equivalente del error cuadrático medio para probabilidades; más suave que Log Loss, menos sensible a errores extremos.
- **Calibración** (`calibration_table`): agrupa las predicciones en bins de probabilidad y compara la media predicha con la tasa real de aciertos en ese bin — si un modelo dice "70%" cien veces y acierta 70 de esas, está bien calibrado.

---

## Fase 3 — Feature engineering

### `src/features/rolling_form.py` — Ventanas móviles anti-leakage
**Teoría**: una "ventana móvil" (*rolling window*) resume el pasado reciente de una entidad (aquí, un equipo) en un número — % de victorias en los últimos 5 partidos, racha actual, margen medio de sets. La parte crítica no es el cálculo en sí, es el **orden de operaciones**: calcular la feature de la fila *antes* de actualizar el historial con esa misma fila. Invertir ese orden es la forma más común de leakage en proyectos de series temporales.

**Head-to-head** es distinto de "forma general": dos equipos pueden tener Elo parecido pero un historial desequilibrado entre ellos (estilos de juego que chocan mal). Es una feature de **interacción**, no de cada equipo por separado.

### `src/features/competition_importance.py` — Codificación ordinal
**Teoría**: "importancia del torneo" no es una categoría sin orden (como el país) — es una escala (una final importa más que un partido de grupos). Por eso se usa **codificación ordinal** (números que respetan el orden: 1, 2, 3...) en vez de one-hot encoding, que trataría "final" y "fase de grupos" como categorías sin relación entre sí.

### `src/features/tournament_progress.py` — Cuando el dato ideal no existe, usa el mejor proxy disponible
**Teoría**: se quería usar la fase real del partido (`Phase` de VIS), pero la fuente no la da. En vez de fabricar un dato falso, se usó un **proxy honesto**: la posición del partido dentro del torneo (`no_in_tournament / total`). Un proxy no es lo mismo que el dato real — se documenta como aproximación, nunca se presenta como si fuera la fase exacta.

### `src/features/build_features.py` — Features "diff" simétricas
**Teoría**: al modelo no le importa "cuánto vale el equipo A" en términos absolutos — le importa la diferencia relativa frente a B (`elo_a − elo_b`, no `elo_a` y `elo_b` por separado). Reducir a diferencias también evita que el modelo tenga que aprender una simetría que ya sabemos que existe (equivale a la mitad de los parámetros a estimar).

---

## Fase 4 — ML tabular (LightGBM / XGBoost / CatBoost)

### `src/models/tabular.py` — Gradient Boosting
**Teoría**: los tres son variantes de **gradient boosting sobre árboles de decisión** — en vez de un único árbol grande, se entrenan cientos de árboles pequeños en secuencia, donde cada árbol nuevo corrige los errores del conjunto anterior (no del dato original, sino del *residuo* — lo que el ensemble hasta ahora predijo mal).

**Por qué manejan NaN sin imputar** (a diferencia de la red neuronal de la Fase 8): un árbol de decisión, en cada nodo, puede aprender "si el valor es NaN, ve por la rama izquierda" como una decisión más — no necesita que el NaN se convierta en un número antes.

**Hiperparámetros usados** (`max_depth=4`, `num_leaves`, `min_child_samples`) — todos limitan la complejidad del árbol individual. Con ~1000-3000 filas de entrenamiento por fold, un árbol profundo memoriza en vez de generalizar (**sobreajuste**, *overfitting*): aprende el ruido específico de esas filas, no el patrón real.

**Resultado real**: CatBoost ganó con claridad y consistencia (mismo ranking en ambos géneros) — evidencia de que la ventaja es señal real, no azar de un fold con suerte. LightGBM, pese a ser el "candidato principal" de la spec original, quedó empatado con los baselines — la lección aquí es **confiar en el experimento, no en la expectativa previa**.

### Guardado/carga de modelos (`save`/`load` en `tabular.py`, `set_score.py`, `point_diff.py`)
**Teoría**: un modelo entrenado en memoria desaparece al cerrar el proceso — **serializar** (`save_model` de CatBoost, formato binario `.cbm`) lo convierte en un archivo reutilizable. Se guarda también un JSON "sidecar" con la lista de columnas de features usadas — sin eso, cargar el modelo no basta: hay que saber en qué orden y con qué nombres esperaba las columnas de entrada.

---

## Fase 5 — Calibración y ensemble

### `src/models/calibration.py` — Platt Scaling e Isotonic Regression
**Teoría**: un modelo puede discriminar bien (separar ganadores de perdedores) y aun así estar mal calibrado. La calibración corrige la probabilidad de salida SIN tocar el modelo:

- **Platt Scaling**: ajusta una regresión logística 1-D sobre el *logit* de la probabilidad cruda — asume que la relación entre "lo que dice el modelo" y "lo que pasa de verdad" es una curva logística suave. Pocos parámetros (2), poco riesgo de sobreajuste incluso con datos limitados.
- **Isotonic Regression**: ajusta una función monótona no paramétrica (sin asumir ninguna forma concreta) — más flexible, pero necesita más datos para no sobreajustar. Con pocos puntos de calibración puede colapsar a predicciones extremas (0.0/1.0 exactos) — bug real que se encontró y se corrigió acotando la salida a `[0.02, 0.98]`.

**Resultado real**: CatBoost ya venía razonablemente bien calibrado de fábrica (la tabla `predicted_mean` vs `actual_rate` ya estaba cerca) — ni Platt ni isotonic aportaron nada. Esto es en sí un hallazgo válido: **no toda fase necesita "ganar" con una técnica nueva; a veces la conclusión correcta es "no hace falta".**

### `src/models/ensemble.py` — Combinar modelos, con cuidado con el leakage
**Teoría del ensemble por media**: si dos modelos cometen errores distintos (no perfectamente correlacionados), promediar sus predicciones suele reducir la varianza del error combinado — es la misma lógica que diversificar una cartera de inversión.

**El detalle anti-leakage**: `make_calibrated_factory` separa, dentro de cada fold de walk-forward, un tramo cronológico del propio `train_df` solo para ajustar el calibrador — el modelo base nunca ve esas filas al entrenar. Si el calibrador se ajustara sobre las mismas filas que entrenaron el modelo, aprendería a "confiar" en el sobreajuste del modelo base en vez de corregirlo.

**Resultado real**: el ensemble CatBoost+XGBoost quedó en empate técnico con CatBoost solo — porque XGBoost es algo peor, así que promediar diluye la ventaja de CatBoost en vez de sumarla. Un ensemble solo ayuda cuando los componentes son comparablemente buenos.

---

## Fase 6 — Modelo de marcador de sets

### `src/models/set_score.py` — Descomposición de probabilidad
**Teoría**: en vez de un clasificador directo de 6 clases (3-0, 3-1, 3-2, 2-3, 1-3, 0-3), se descompuso el problema usando la **regla de la probabilidad condicional**:

```
P(A gana 3-1) = P(A gana) × P(margen=3-1 | A gana)
```

Esto tiene dos ventajas: (1) reutiliza el modelo de "quién gana" ya validado en vez de aprenderlo de cero, y (2) el modelo de margen es más simple (3 clases en vez de 6), lo que necesita menos datos para generalizar bien.

**Reorientación de features** (`orient_features`): para entrenar el modelo de margen con el 100% de los partidos (no solo "los que ganó A"), cada partido se reescribe desde la perspectiva de quien realmente ganó. Esto exige distinguir tres tipos de features:
- **Antisimétricas** (`elo_diff`, etc.): cambian de signo si se invierte la perspectiva.
- **Tipo tasa** (`h2h_win_rate_a`): se usa el complementario (`1 − x`), no el negativo, porque son probabilidades en `[0, 1]`, no diferencias.
- **Simétricas** (`competition_importance`, etc.): no dependen de quién es "A" — se dejan igual.

**Bug real encontrado**: `sklearn.metrics.log_loss` con el parámetro `labels=` asume orden **alfabético** de las columnas de la matriz de probabilidades, sin importar el orden que se le pase en `labels`. Si no coincide, compara cada probabilidad con la clase equivocada — lección: nunca asumas el comportamiento de una función de una librería sin comprobarlo, aunque tenga una interfaz aparentemente clara.

---

## Fase 7 — Modelo de diferencia de puntos

### `src/models/point_diff.py` — De clasificación a regresión
**Teoría**: predecir "quién gana" es clasificación (categoría discreta); predecir "por cuántos puntos" es **regresión** (número continuo) — cambia el tipo de modelo (`CatBoostRegressor` en vez de `CatBoostClassifier`) y las métricas de evaluación.

- **MAE** (Error Absoluto Medio) = `mean(|y_true − y_pred|)` — en las mismas unidades que el dato (puntos), fácil de interpretar ("de media nos equivocamos por 11 puntos").
- **RMSE** (Raíz del Error Cuadrático Medio) = `sqrt(mean((y_true − y_pred)²))` — al elevar al cuadrado, penaliza más los errores grandes que el MAE. Si RMSE >> MAE, es señal de que hay algunos errores muy grandes arrastrando la media.

**Baseline `linear_elo`**: una regresión lineal simple de `point_diff` sobre `elo_diff` sirve de referencia intermedia entre "no saber nada" (`mean_baseline`) y el modelo completo — si CatBoost no le gana con margen claro a un modelo de una sola variable, la complejidad extra no se está aprovechando.

**Resultado real**: CatBoost redujo el error un 24-32% frente a la media, y un 7-13% frente al modelo lineal — la ventaja de usar todas las features (no solo Elo) es real aquí, a diferencia de la Fase 5 donde añadir complejidad no ayudó.

---

## Fase 8 — Deep learning (solo si mejora)

### `src/models/neural.py` — Red neuronal (MLP)
**Teoría de la arquitectura**: un *Multi-Layer Perceptron* (MLP) es la red neuronal más simple — capas de neuronas totalmente conectadas, cada una aplicando una transformación lineal seguida de una función de activación no lineal (`ReLU` aquí). Sin la no-linealidad, apilar capas sería matemáticamente equivalente a una sola capa lineal — la no-linealidad es lo que le da a la red capacidad de aprender patrones complejos.

**Por qué necesita lo que los árboles no necesitan**:
- **Imputación de NaN**: una red no puede "aprender una rama para NaN" como un árbol — hay que rellenar el hueco con algo (aquí, la mediana del train) y, para no perder la señal de "esto faltaba", se añade una columna binaria `_was_missing` por cada feature con huecos.
- **Escalado de features** (`StandardScaler`): las redes se entrenan por descenso de gradiente, que converge mal si las features están en escalas muy distintas (Elo en cientos, tasas en `[0,1]`) — escalar todo a media 0 y desviación 1 estabiliza el entrenamiento.

**Regularización — por qué tantos frenos**:
- **Dropout**: en cada paso de entrenamiento, apaga aleatoriamente un porcentaje de neuronas — evita que la red dependa demasiado de una combinación específica de neuronas (fuerza redundancia, como estudiar sin depender de una sola chuleta).
- **Weight decay** (regularización L2 sobre los pesos): penaliza pesos grandes, empujando al modelo hacia soluciones más simples.
- **Early stopping**: se reserva un tramo cronológico del train como validación interna, y se para de entrenar en cuanto el error de validación deja de mejorar (`patience` épocas sin mejora) — sin esto, con ~1000-3000 filas por fold, la red memoriza en pocas épocas.

**Optimización**: `Adam` es el optimizador — una variante de descenso de gradiente que adapta la tasa de aprendizaje por parámetro automáticamente, generalmente más rápido de converger que el descenso de gradiente clásico. `BCELoss` (Binary Cross-Entropy) es matemáticamente la misma función que el Log Loss de `metrics.py` — es la pérdida "correcta" para clasificación binaria con salida en `[0,1]`.

**Bug real encontrado**: cuando una feature sale completamente vacía en un fold pequeño (`tournament_progress` sin `tournament_no`), su mediana también es `NaN` — rellenar con `NaN` no rellena nada, y esa columna vacía rompe el escalado (varianza indefinida). Corregido con un `0.0` de último recurso cuando la mediana sale `NaN`.

**El criterio de esta fase, explícito en la spec**: "solo si mejora". Con ~3300 partidos por género, es matemáticamente esperable que una red no le gane a un modelo de árboles bien ajustado — las redes suelen necesitar mucho más volumen de datos para aprovechar su capacidad. El veredicto real lo da el experimento contra el dataset real, no la intuición.

---

## Fase 9 — API

### `src/api/predictor.py` — Separar la lógica del framework web
**Teoría**: `predict_match(...)` no importa nada de FastAPI — solo trabaja con modelos y DataFrames. Esto es el principio de **separación de responsabilidades** (*separation of concerns*): la lógica de negocio (cómo se calcula una predicción) y la capa de transporte (cómo llega esa predicción por HTTP) son cosas distintas que cambian por razones distintas. La ventaja práctica es que `predict_match` se puede probar con `pytest` normal, sin levantar un servidor — y se puede reutilizar en cualquier otro sitio (como el dashboard de la Fase 10, que la llama directamente sin pasar por HTTP en absoluto).

**Carga perezosa de modelos** (`_get_bundle` en `app.py`): los modelos no se cargan al arrancar la API, sino la primera vez que alguien los pide. Esto es un patrón de **inicialización perezosa** (*lazy initialization*) — si solo hay modelos entrenados para un género, la API arranca igualmente en vez de fallar por completo.

**`GenderModelBundle`**: agrupa los 6 artefactos que hacen falta para predecir (2 modelos CatBoost + 1 regresor + 3 JSON de estado) en un solo objeto, cargado una sola vez. Evitar cargar un modelo desde disco en cada petición no es un detalle menor — con tráfico real, eso sería el cuello de botella más caro de toda la API.

**Nivel de confianza — una decisión de diseño, no una fórmula estándar**: se define en función de cuánto historial real tienen AMBOS equipos (`min(matches_played_a, matches_played_b)`), no de lo "segura" que esté la predicción. Es una elección deliberada: un modelo puede dar 95% de probabilidad con muy pocos datos y estar "seguro" sin motivo — la confianza aquí mide la fiabilidad de la fuente de información, no la convicción del modelo.

### Bug real: `None` de Python vs. `NaN` numérico en una fila suelta
**Teoría del bug**: pandas infiere el tipo de una columna mirando TODOS sus valores. Con miles de filas (el histórico de entrenamiento), una columna con algún hueco se infiere como `float64` con `NaN` automáticamente. Con una única fila (un partido nuevo a predecir), si ese único valor es `None` de Python, pandas no tiene con qué comparar y la deja como tipo `object` — y las operaciones numéricas (restas, negaciones) fallan sobre ese tipo. La lección: **el mismo dato puede comportarse distinto según cuántas filas lo rodeen** — un motivo más para probar el camino de inferencia (una fila) con un test específico, no asumir que "si funciona con el histórico, funciona igual con un partido suelto".

---

## Fase 10 — Dashboard y despliegue

### `streamlit_app.py` — Por qué Streamlit y no una API + frontend separados
**Teoría de la decisión**: Streamlit convierte un script de Python normal en una aplicación web interactiva, sin escribir HTML/CSS/JS — cada vez que el usuario interactúa (cambia un desplegable, pulsa un botón), Streamlit **re-ejecuta el script entero de arriba a abajo**. Esto es distinto de un framework web tradicional (que reacciona a eventos concretos) y explica por qué el cacheo es tan importante aquí.

**`@st.cache_resource`**: sin esto, cada interacción del usuario recargaría los modelos CatBoost desde disco desde cero — carga una vez por sesión de servidor, no por clic. Es la misma idea que `GenderModelBundle` en la Fase 9 (cargar una vez, no en cada petición), adaptada al modelo de ejecución de Streamlit.

**Por qué llama a `predictor.py` directamente en vez de hacer peticiones HTTP a la API**: un despliegue gratuito (Streamlit Community Cloud) ejecuta un único proceso — no hay forma sencilla de tener la API de FastAPI corriendo *a la vez* que el dashboard en el mismo contenedor gratuito. Reutilizar la lógica de predicción directamente (gracias a que la Fase 9 la separó del framework web) evita ese problema por completo.

### Despliegue gratuito — la parte que no es "teoría de ML" pero es igual de importante
**El detalle que rompe despliegues sin que el código tenga ningún bug**: un servicio de despliegue gratuito parte de una copia limpia del repositorio de GitHub en un contenedor que no tiene NADA de tu ordenador — ni tus archivos locales, ni tu caché de pip, ni los modelos que entrenaste ayer. Si un artefacto (aquí, `models/<gender>/*.cbm` y los JSON de estado) no está commiteado en el repo, el despliegue arranca sin él. Por eso el `.gitignore` de este proyecto excluye explícitamente `data/raw/` (regenerable, pesado) pero **incluye `models/`** — la excepción más importante de todo el archivo, y la contraria a lo que recomienda casi cualquier guía genérica de ".gitignore para proyectos de ML".

**`requirements.txt` mínimo, verificado, no asumido**: en vez de asumir qué librerías hacen falta para servir el dashboard, se rastrearon los imports reales de toda la cadena que toca `streamlit_app.py`, y se comprobó en un entorno virtual limpio (sin nada más instalado) que la app funciona solo con esas 6 librerías. Este rastreo reveló algo no evidente: `scikit-learn` hace falta aunque el dashboard no lo use directamente, porque `point_diff.py` importa `LinearRegression` **a nivel de módulo** (se ejecuta con solo importar el archivo, no hace falta usar esa clase) — mientras que `lightgbm`/`xgboost` NO hacen falta pese a estar en el mismo paquete de modelos, porque sus imports están dentro de `__init__` (solo se disparan si de verdad instancias esas clases, cosa que la API nunca hace). Mismo patrón de todo el proyecto: **comprobar, no asumir** — aquí aplicado a dependencias de despliegue, no a resultados de modelos.

---

## Conceptos transversales (aparecen en casi todas las fases)

- **Leakage temporal**: usar en el momento T información que en producción no existiría hasta después de T. Es el error más peligroso en un proyecto de series temporales porque los números de validación pueden parecer excelentes y ser completamente falsos.
- **Walk-forward validation**: la única forma de validar honestamente un modelo con orden temporal — cada fold entrena solo con el pasado del bloque que evalúa.
- **Overfitting (sobreajuste)**: cuando un modelo aprende el ruido específico de los datos de entrenamiento en vez del patrón general — se combate con regularización (L2, dropout), profundidad limitada, y evaluando siempre en datos que el modelo no vio entrenar.
- **Baseline como ancla**: cada fase compara el modelo nuevo contra algo más simple (Elo, la media, un modelo lineal) — sin esa ancla, un número de métrica no dice nada sobre si el modelo es bueno de verdad.
- **"Compruébalo, no lo asumas"**: el patrón que más se repite en este proyecto — LightGBM no ganó pese a ser el "candidato principal" de la spec; la calibración no mejoró nada pese a la expectativa de que "siempre ayuda"; la red neuronal se acepta o descarta según el resultado real, no según lo esperado de antemano; el `requirements.txt` de despliegue se verificó en un entorno aislado en vez de asumir qué hacía falta.
- **Import perezoso vs. import a nivel de módulo**: un `import` dentro de una función/método solo se ejecuta si esa función se llama; un `import` al principio de un archivo se ejecuta siempre que el archivo se importa, lo uses o no. La diferencia parece un detalle de estilo, pero decide qué dependencias hacen falta de verdad para desplegar (Fase 10) y por qué el código de modelos tabulares (Fase 4) no revienta si falta alguna librería.
