* Una descripción clara del `DAG`, explicando la funcionalidad de cada tarea y cómo se relacionan entre sí.
  * Diagrama de flujo del *pipeline* completo.
  * Una representación visual del `DAG` en la interfaz de `Airflow`.
  * Explicación de cómo se diseñó la lógica para integrar futuros datos, detectar *drift* y reentrenar el modelo.
  * Todos estos puntos deben estar contenidos en un archivo markdown junto a la carpeta de esta sección.


## Descripción del DAG

Este DAG implementa un pipeline completo de Machine Learning con las siguientes características:

1. **Preparación de datos**:
   - Carga y limpieza de datos
   - Transformaciones y agregaciones
   - División en train/val/test

2. **Entrenamiento**:
   - Optimización de hiperparámetros con Optuna
   - Tracking completo con MLflow
   - Evaluación en test set
   - Interpretabilidad con SHAP

3. **Monitoreo**:
   - Detección automática de data drift
   - Tests estadísticos (Kolmogorov-Smirnov)
   - Generación de reportes

4. **Reentrenamiento**:
   - Decisión automática basada en drift
   - Reentrenamiento condicional
   - Actualización de modelos

5. **Trazabilidad**:
   - Registro de versiones de librerías
   - Metadata de modelos
   - Logs completos en MLflow

ESTRUCTURA DE CARPETAS GENERADA:
```
/opt/airflow/dags/{execution_date}/
├── raw/                    # Datos originales
├── transformed/            # Datos transformados
├── splits/                 # División train/val/test
├── preprocessed/           # Datos preprocesados
├── mlruns/                 # Experimentos MLflow
├── models/                 # Modelos entrenados
│   ├── KNN_optimo.joblib
│   ├── preprocessor.joblib
│   └── model_metadata.json
├── drift_reports/          # Reportes de drift
├── interpretability/       # Explicaciones SHAP
└── evaluation/             # Métricas de test
```

VARIABLES DE ENTORNO REQUERIDAS:
- BASE_PATH: Ruta base para almacenar outputs

EJECUCIÓN:
- Manual (schedule_interval=None)
- Sin backfill (catchup=False)

"""