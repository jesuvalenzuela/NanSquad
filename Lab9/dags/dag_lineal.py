from datetime import datetime
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface

# Ruta base para almacenar outputs
BASE_PATH = '/opt/airflow/dags'      # Guardar en dags
MODEL_NAME = "hiring_rf_pipeline"

# 0. Inicializar un DAG con fecha de inicio el 1 de octubre de 2024, ejecución manual y **sin backfill**
with DAG(
    dag_id='hiring_lineal',
    start_date=datetime(2024, 10, 1),
    schedule_interval=None,     # Ejecución manual
    catchup=False       # Sin backfill
) as dag:
    
    # 1. Comenzar con un marcador de posición que indique el inicio del pipeline
    start = EmptyOperator(task_id='start_pipeline')

    # 2. Crear una carpeta correspondiente a la ejecución del pipeline y cree las subcarpetas `raw`, `splits` y `models` mediante la función `create_folders()
    crear_carpetas = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders,
        op_kwargs={'base_path': BASE_PATH}
    )

    # Ruta a carpeta de ejecución (La funcion create_folders la retorna)
    execution_path = "{{ ti.xcom_pull(task_ids='create_folders') }}"

    # 3. Descargar datos
    descargar_datos = BashOperator(
        task_id='download_data',
        bash_command=(
            f'curl -o {BASE_PATH}/{{{{ ds }}}}/raw/data_1.csv '
            'https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv'
        )
    )

    # 4. Split de datos
    split_datos = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs={'execution_path': execution_path}
    )

    # 5. Preprocesamiento y entrenamiento
    entrenar_modelo = PythonOperator(
        task_id='preprocess_and_train',
        python_callable=preprocess_and_train,
        op_kwargs={'execution_path': execution_path,
                   'model_name': MODEL_NAME}
    )

    # Ruta a ejecución (La funcion preprocess_and_train la retorna)
    model_path = "{{ ti.xcom_pull(task_ids='preprocess_and_train') }}"

    # 6. Montar una interfaz en gradio
    deploy_gradio = PythonOperator(
        task_id='gradio_interface',
        python_callable=gradio_interface,
        op_kwargs={'model_path': model_path}
    )

    # Definir dependencias
    start >> crear_carpetas >> descargar_datos >> split_datos >> entrenar_modelo >> deploy_gradio