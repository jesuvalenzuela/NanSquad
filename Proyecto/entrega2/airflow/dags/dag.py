from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.models.baseoperator import cross_downstream

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'plugins'))

from decision_functions import (
    assert_folders,
    check_historical_data,
    check_new_data,
    extend_dataset,
    decide_if_train,
    prepare_data,
    split_data,
    preprocess_data,
    should_optimize_hyperparameters,
    load_best_params,
    optimize_model,
    evaluate_and_interpret_model,
    train_final_model
)

# Ruta base para almacenar outputs
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
BASE_PATH = AIRFLOW_HOME
DATA_PATH = os.path.join(AIRFLOW_HOME, 'data')

MODEL_NAME = "product_priority_model"

# === Definición del DAG ===
assert_folders(AIRFLOW_HOME)

# Argumentos por defecto del DAG
default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Inicializar un DAG con fecha de inicio el 31 de diciembre de 2024, ejecución manual y sin backfill
with DAG(
    dag_id='product_purchase_prediction',
    default_args=default_args,
    start_date=datetime(2024, 12, 31),
    schedule='@weekly',        # Reentrenamiento semanal
    catchup=False       # Sin backfill
) as dag:
    
    # =======================================================
    # 1. INICIO Y PREPARACIÓN DE DATOS
    # =======================================================

    # === Iniciar ===
    start = EmptyOperator(task_id='start_pipeline')

    # === Revisar si existe dataset histórico ===
    revisar_historicos = BranchPythonOperator(
    task_id='check_historical_data',
    python_callable=check_historical_data,
    op_kwargs={'data_path': DATA_PATH}
    )

    # Si no existe dataset histórico: copiar raw
    copiar_raw_a_historico = BashOperator(
        task_id='copy_raw',
        bash_command=f"cp {DATA_PATH}/raw/transacciones.parquet {DATA_PATH}/historical_raw/transacciones.parquet",
    )
    
    # Si existe dataset histórico: no hacer nada
    pasar_1 = EmptyOperator(task_id='pass_1')

    # === Revisar si hay nuevos datos ===
    revisar_nuevos_datos = BranchPythonOperator(
        task_id='check_new_data',
        python_callable=check_new_data,
        op_kwargs={'data_path': DATA_PATH},
        trigger_rule='none_failed'
    )

    # Si hay nuevos datos: extender dataset
    incorporar_nuevos_datos = PythonOperator(
        task_id='extend_dataset',
        python_callable=extend_dataset,
        op_kwargs={'data_path': DATA_PATH}
    )
    
    # Si no hay nuevos datos: no hacer nada
    pasar_2 = EmptyOperator(task_id='pass_2')

    # =======================================================
    # 2. PROCESAMIENTO DE DATOS 
    # =======================================================

    # === Decidir si es necesario procesar datos y entrenar modelo ===
    decidir_entrenamiento = PythonOperator(
        task_id='decide_training',
        python_callable=decide_if_train,
        op_kwargs={'base_path': BASE_PATH,
                   'model_name': MODEL_NAME},
        trigger_rule='none_failed'
    )

    no_entrenar = EmptyOperator(task_id='not_train')

    # === Formateo inicial ===
    preparar_datos = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
        op_kwargs={'data_path': DATA_PATH}
    )

    # === Holdout ===
    split_datos = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs={'data_path': DATA_PATH}
    )
    
    # === Preprocesamiento ===
    preprocesar = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        op_kwargs={'base_path': BASE_PATH}
    )

    # =======================================================
    # 3. OPTIMIZACIÓN, EVALUACIÓN E INTERPRETACIÓN DEL MODELO
    # =======================================================

    # === Decidir si optimizar o cargar hiperparámetros ===
    decidir_optimizacion = BranchPythonOperator(
        task_id='check_if_should_optimize',
        python_callable=should_optimize_hyperparameters,
        op_kwargs={
            'base_path': BASE_PATH,
            'reoptimize_weeks': 4  # Re-optimizar cada 4 semanas
        }
    )

    # === Cargar hiperparámetros óptimos existentes ===
    cargar_params = PythonOperator(
        task_id='load_best_params',
        python_callable=load_best_params,
        op_kwargs={'base_path': BASE_PATH}
    )

    # === Optimizar parámetros (solo si es necesario) ===
    optimizar = PythonOperator(
        task_id='optimize_model',
        python_callable=optimize_model,
        op_kwargs={
            'base_path': BASE_PATH,
            'target_column': 'priority',
            'n_trials': 30,  # Reducido de 50 a 30
            'model_name': MODEL_NAME
        }
    )

    # === Evaluar e interpretar modelo ===
    evaluar_interpretar = PythonOperator(
        task_id='evaluate_and_interpret',
        python_callable=evaluate_and_interpret_model,
        op_kwargs={
            'base_path': BASE_PATH,
            'target_column': 'priority',
            'n_shap_samples':500,
            'model_name': MODEL_NAME
        },
        trigger_rule='none_failed'  # Se ejecuta después de cualquiera de las dos ramas
    )

    # === Entrenar con todos los datos ===
    entrenar_modelo_final = PythonOperator(
        task_id='train_final_model',
        python_callable=train_final_model,
        op_kwargs={
            'base_path': BASE_PATH,
            'model_name': MODEL_NAME
        },
        trigger_rule='none_failed'  # Se ejecuta después de cualquiera de las dos ramas
    )
    
    # =======================================================
    # 4. FINALIZACIÓN
    # =======================================================
    """guardar_versiones = PythonOperator(
        task_id='save_library_versions',
        python_callable=save_library_versions,
        trigger_rule='none_failed' # Se ejecuta aunque se haya saltado el entrenamiento
    )"""

    end = EmptyOperator(
        task_id='end_pipeline',
        trigger_rule='none_failed'
    )

    # =======================================================
    # DEFINICIÓN DE DEPENDENCIAS
    # =======================================================
    # 1. Primera ejecución: copia datos raw a histórico, optimiza hiperparámetros, entrena modelo
    # 2. Datos nuevos: extiende dataset, carga hiperparámetros existentes, reentrena modelo
    # 3. Re-optimización periódica: cada 4 semanas se re-optimizan hiperparámetros

    # Inicio y primera bifurcacion
    start >> revisar_historicos >> [copiar_raw_a_historico, pasar_1] >> revisar_nuevos_datos

    # Segunda bifurcación
    revisar_nuevos_datos >> [incorporar_nuevos_datos, pasar_2] >> decidir_entrenamiento

    # Tercera bifurcacion (decidir si entrenar)
    decidir_entrenamiento >> [preparar_datos, no_entrenar]

    # Cuarta bifurcación (decidir si optimizar o cargar hiperparámetros)
    preparar_datos >> split_datos >> preprocesar >> decidir_optimizacion
    decidir_optimizacion >> [optimizar, cargar_params]

    # Ambas ramas convergen en evaluación y entrenamiento final
    cross_downstream([optimizar, cargar_params], [evaluar_interpretar, entrenar_modelo_final])
    [evaluar_interpretar, entrenar_modelo_final] >> end

    # Si se toma la rama "no_entrenar", se pasa directo a la tarea final
    no_entrenar >> end
    

