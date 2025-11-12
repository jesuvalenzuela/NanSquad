from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator


from decision_functions import (
    create_folders,
    prepare_data,
    split_data,
    preprocess_data,
    optimize_model,
    evaluate_model,
    detect_drift,
    should_retrain,
    retrain_model,
    generate_interpretability,
    save_library_versions
)


# Ruta base para almacenar outputs
BASE_PATH = '/opt/airflow/dags'      # Guardar en dags
MODEL_NAME = "product_priority_pipeline"

# Argumentos por defecto del DAG
default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 0. Inicializar un DAG con fecha de inicio el 1 de octubre de 2024, ejecución manual y **sin backfill**
with DAG(
    dag_id='ml_pipeline_with_drift_detection',
    default_args=default_args,
    start_date=datetime(2024, 10, 1),
    schedule_interval=None,     # Ejecución manual
    catchup=False       # Sin backfill
) as dag:
    
    # ========================================
    # 1. INICIO Y PREPARACIÓN DE DATOS
    # ========================================

    start = EmptyOperator(task_id='start_pipeline')

    crear_carpetas = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders,
        op_kwargs={'base_path': BASE_PATH}
    )

    preparar_datos = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data
    )

    split_datos = PythonOperator(
        task_id='split_data',
        python_callable=split_data
    )

    preprocesar = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    # ========================================
    # 2. ENTRENAMIENTO Y OPTIMIZACIÓN
    # ========================================

    optimizar = PythonOperator(
        task_id='optimize_model',
        python_callable=optimize_model,
        op_kwargs={
            'target_column': 'priority',
            'n_trials': 50,
            'model_name': MODEL_NAME
        }
    )
    
    evaluar = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model
    )
    
    interpretar = PythonOperator(
        task_id='generate_interpretability',
        python_callable=generate_interpretability,
        op_kwargs={'n_samples': 500}
    )
    
    # ========================================
    # FASE 3: DRIFT DETECTION Y REENTRENAMIENTO
    # ========================================

    detectar_drift = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift,
        op_kwargs={'significance_level': 0.05}
    )

    decidir_reentrenamiento = BranchPythonOperator(
        task_id='should_retrain',
        python_callable=should_retrain
    )
    
    reentrenar = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model,
        op_kwargs={
            'target_column': 'priority',
            'n_trials': 30,  # Menos trials para reentrenamiento
        }
    )
    
    skip_reentrenamiento = EmptyOperator(
        task_id='skip_retrain'
    )

    # ========================================
    # 4. FINALIZACIÓN
    # ========================================
    
    guardar_versiones = PythonOperator(
        task_id='save_library_versions',
        python_callable=save_library_versions,
        trigger_rule='none_failed' # Se ejecuta aunque se haya saltado el reentrenamiento
    )

    end = EmptyOperator(
        task_id='end_pipeline',
        trigger_rule='none_failed'
    )

    # ========================================
    # DEFINICIÓN DE DEPENDENCIAS
    # ========================================

    # Flujo principal: preparación y entrenamiento
    start >> crear_carpetas >> preparar_datos >> split_datos >> preprocesar
    
    # Entrenamiento y evaluación
    preprocesar >> optimizar >> [evaluar, interpretar]
    
    # Drift detection después de evaluación
    [evaluar, interpretar] >> detectar_drift
    
    # Branching para reentrenamiento
    detectar_drift >> decidir_reentrenamiento
    decidir_reentrenamiento >> [reentrenar, skip_reentrenamiento]
    
    # Convergencia y finalización
    [reentrenar, skip_reentrenamiento] >> guardar_versiones >> end

