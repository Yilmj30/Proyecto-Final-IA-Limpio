# jeorregog1 - Modelado, Arquitectura y Metodología

Notas de trabajo para el bloque de **jeorregog1**.

## Confirmación del enfoque del proyecto

Sí, el proyecto quedó alineado con la **opción 1** que habíamos propuesto:

- **Tema:** predicción de deserción estudiantil.
- **Dataset:** Higher Education Predictors of Student Retention / Student Dropout (Kaggle).
- **Tipo de problema:** clasificación binaria de riesgo de deserción.
- **Modelo principal acordado por el equipo:** **LightGBM**.
- **Salida de tu bloque:** un modelo LightGBM tuneado y listo para que el Integrante 3 lo use en la evaluación final y la comunicación de resultados.

## Alcance de jeorregog1

Tu bloque queda así:

- Construir el **baseline**.
- Entrenar el **modelo principal LightGBM**.
- Hacer **tuning** sin romper la separación train/val/test.
- Generar artefactos de **validación y tuning**.
- Redactar las **secciones 3 y 4 del informe**.

La imagen de reparto y la división textual del proyecto se pueden reconciliar sin solaparte con el Integrante 3:

- El texto formal de la división menciona baseline + modelo principal + tuning.
- La imagen resume el bloque de forma más amplia, pero para evitar mezclar responsabilidades conviene cerrar tu parte en el **módulo predictivo**.

La forma práctica de organizar tu parte es:

1. baseline
2. LightGBM base
3. LightGBM tuneado
4. handoff del mejor modelo al Integrante 3

## Sección 3 del informe - Arquitectura del sistema

### Versión final sugerida

La arquitectura del sistema se diseñó como un flujo modular orientado a la predicción del riesgo de deserción estudiantil a partir de variables académicas, demográficas, financieras e institucionales. En la primera capa, el sistema recibe los datos previamente depurados y transformados durante la etapa de preprocesamiento. Esta fase garantiza consistencia en la codificación de variables, escalado, partición de los conjuntos de datos y trazabilidad experimental, de modo que el módulo de modelado opere sobre entradas homogéneas y reproducibles.

En la segunda capa se implementó el motor predictivo del proyecto, compuesto por un baseline lineal de referencia y un modelo principal basado en LightGBM. El baseline permite establecer un punto de comparación claro para medir la ganancia real del modelo más avanzado. Por su parte, LightGBM fue seleccionado como modelo principal debido a su buen desempeño en datos tabulares, su capacidad para capturar relaciones no lineales entre variables y su eficiencia para explorar interacciones complejas sin requerir un preprocesamiento excesivo adicional.

En la tercera capa, el sistema produce probabilidades de riesgo y predicciones binarias de deserción. Estos resultados se almacenan junto con los hiperparámetros seleccionados, las métricas de validación y los artefactos de handoff. De esta manera, el bloque de modelado queda desacoplado de la evaluación final, permitiendo que el Integrante 3 utilice el mejor candidato encontrado como insumo directo para la comparación formal de modelos, la evaluación sobre el conjunto de prueba y la comunicación final de resultados.

### Componentes clave que debes mencionar

- **Entrada del sistema:** conjuntos `X_train`, `X_val`, `y_train` y `y_val` generados en la etapa de preprocesamiento.
- **Baseline de referencia:** `logreg_balanced`, usado para medir si el modelo principal aporta una mejora real.
- **Modelo principal:** `LightGBM` con manejo de desbalance mediante `class_weight='balanced'`.
- **Salida del módulo:** probabilidades, predicción binaria, métricas de validación, mejores hiperparámetros y artefactos de handoff.

### Figura sugerida para esta sección

- `Datos procesados -> Baseline / LightGBM -> Predicción de deserción -> Artefactos de handoff`

## Sección 4 del informe - Metodología

### Versión final sugerida

El proceso de modelado se desarrolló utilizando exclusivamente los conjuntos de entrenamiento y validación construidos durante la etapa de preprocesamiento. El conjunto de prueba se mantuvo completamente aislado para evitar data leakage y preservar una evaluación final imparcial, la cual corresponde al bloque del Integrante 3. Bajo este esquema, el conjunto `train` se utilizó para el ajuste inicial de los modelos, mientras que el conjunto `val` se destinó a la comparación de candidatos y a la selección del modelo final.

Como línea base se entrenaron dos referencias sencillas: un `DummyClassifier` con estrategia `prior`, que establece el piso mínimo del problema, y una `LogisticRegression` balanceada, utilizada como baseline lineal competitivo. Posteriormente se implementó un modelo `LightGBM` base con `class_weight='balanced'`, decisión coherente con el desbalance observado en la variable objetivo durante el análisis exploratorio. Sobre este modelo se aplicó un proceso de ajuste de hiperparámetros mediante `RandomizedSearchCV`, utilizando un `PredefinedSplit` para respetar la separación entre entrenamiento y validación durante la búsqueda.

La métrica principal de selección fue `F1`, complementada con `accuracy`, `precision`, `recall` y `AUC-ROC`. Esta decisión metodológica responde a que el problema de deserción estudiantil presenta desbalance de clases y, en consecuencia, no resulta suficiente optimizar únicamente accuracy. Desde una perspectiva aplicada, la métrica F1 permite balancear precisión y recall, mientras que recall adquiere una relevancia especial al buscar identificar correctamente a los estudiantes en riesgo.

El modelo finalmente seleccionado fue `lightgbm_tuned`, que alcanzó un `F1 = 0.8057` sobre validación, superando de forma marginal a `logreg_balanced` (`F1 = 0.8054`). Aunque la diferencia es pequeña, el resultado respalda la elección de LightGBM como modelo principal del proyecto, ya que ofrece un desempeño competitivo y conserva la ventaja de modelar relaciones no lineales en datos tabulares.

### Resultados de validación que puedes reportar

- `lightgbm_tuned`: `accuracy = 0.8765`, `precision = 0.8173`, `recall = 0.7944`, `f1 = 0.8057`, `roc_auc = 0.9235`
- `logreg_balanced`: `accuracy = 0.8705`, `precision = 0.7807`, `recall = 0.8318`, `f1 = 0.8054`, `roc_auc = 0.9259`
- `lightgbm_default`: `accuracy = 0.8614`, `precision = 0.8020`, `recall = 0.7570`, `f1 = 0.7788`, `roc_auc = 0.9215`
- `dummy_prior`: `accuracy = 0.6777`, `precision = 0.0000`, `recall = 0.0000`, `f1 = 0.0000`, `roc_auc = 0.5000`

### Hiperparámetros del modelo final

- `subsample = 0.9`
- `reg_lambda = 0.5`
- `reg_alpha = 0.0`
- `num_leaves = 127`
- `n_estimators = 300`
- `min_child_samples = 10`
- `max_depth = 5`
- `learning_rate = 0.1`
- `colsample_bytree = 0.7`

### Decisiones metodológicas que debes defender

- Problema tratado como **clasificación binaria** de riesgo de deserción.
- Separación estricta de conjuntos:
  - `train` para ajuste inicial
  - `val` para comparación y tuning
  - `test` reservado para evaluación final
- Manejo del desbalance con `class_weight='balanced'`.
- Uso de un baseline para medir la ganancia real del modelo principal.
- Selección final basada en `F1` como métrica principal.

## Qué debes completar después de correr el notebook 03

- Nombre exacto del baseline usado en el informe.
- Mejores hiperparámetros encontrados.
- Métricas de validación del modelo tuneado.
- Comparación numérica en validación contra baseline.
- Ruta del modelo guardado para entregarlo al Integrante 3.
- Resumen corto de handoff en JSON con rutas y mejor candidato.

## Figuras sugeridas para tu bloque

- `fig8_validation_model_comparison.png`: comparación de métricas entre candidatos en validación.

## Secuencia honesta de commits para tu bloque

Sin falsear fechas ni “simular” trabajo, una secuencia limpia puede ser:

1. `chore(modeling): update requirements for the LightGBM notebook`
2. `feat(modeling): add notebook 03 with baselines and LightGBM workflow`
3. `docs(modeling): add architecture and methodology notes for jeorregog1`
4. `feat(modeling): add tuned LightGBM artifacts for handoff`

## Artefactos de handoff sugeridos

Además del modelo `.pkl`, es útil dejar:

- `metrics_validation_candidates.csv`
- `lightgbm_best_params.json`
- `lightgbm_tuning_results.csv`
- `modeling_handoff_summary.json`

Ese último archivo le facilita al Integrante 3 identificar rápido:

- cuál fue el mejor candidato por validación
- cuál fue la métrica de selección
- dónde están los parámetros finales
- dónde está el modelo serializado

## Resultados de la corrida de 03_modeling

La corrida final del notebook ya se ejecuto correctamente en el entorno local con Python 3.12 y dejo estos resultados principales:

- Mejor candidato por `f1` en validacion: `lightgbm_tuned`
- `f1`: `0.8057`
- `accuracy`: `0.8765`
- `precision`: `0.8173`
- `recall`: `0.7944`
- `roc_auc`: `0.9235`

Comparacion directa contra el baseline mas competitivo:

- `logreg_balanced`: `f1 = 0.8054`
- `lightgbm_tuned`: `f1 = 0.8057`

La mejora del modelo tuneado sobre la regresion logistica balanceada es pequena, pero suficiente para justificar la seleccion final si el criterio principal del proyecto es `f1`.

## Handoff final

Artefactos listos para el Integrante 3:

- `models/metrics_validation_candidates.csv`
- `models/lightgbm_best_params.json`
- `models/lightgbm_tuning_results.csv`
- `models/modeling_handoff_summary.json`
- `data/processed/fig8_validation_model_comparison.png`

El modelo serializado `models/checkpoints/lightgbm_dropout_model.pkl` se genero correctamente de forma local, pero no queda versionado porque esta ignorado en `.gitignore`.
