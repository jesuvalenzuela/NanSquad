from datetime import datetime
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

from decision_functions import (create_folders,
                                read_parquet_files,
                                prepare_data,
                                split_data,
                                preprocess_data,
                                optimize_model
                                ) # split_data, preprocess_and_train, gradio_interface

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Ruta base para almacenar outputs
BASE_PATH = '/opt/airflow/dags'      # Guardar en dags
MODEL_NAME = "product_priority_pipeline"


# 0. Inicializar un DAG con fecha de inicio el 1 de octubre de 2024, ejecución manual y **sin backfill**
with DAG(
    dag_id='decision',
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

    # 3. Cargar datos
    leer_datos = PythonOperator(
        task_id='read_data',
        python_callable=read_parquet_files,
        op_kwargs={'base_path': BASE_PATH}
    )

    # 4. Preparar datos
    preparar_datos = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data
    )

    # 5. Split de datos
    split_datos = PythonOperator(
        task_id='split_data',
        python_callable=split_data
    )

    # 6. Preprocesamiento
    preprocesar = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    # 7. Optimización de parámetros
    optimizar = PythonOperator(
        task_id='optimize_model',
        python_callable=optimize_model
    )

    # 8. Montar una interfaz en gradio
    """deploy_gradio = PythonOperator(
        task_id='gradio_interface',
        python_callable=gradio_interface
    )"""

    # Definir dependencias
    start >> crear_carpetas >> leer_datos >> preparar_datos >> split_datos >> preprocesar >> optimizar

