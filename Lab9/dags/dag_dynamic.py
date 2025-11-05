from datetime import datetime, timezone
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from hiring_dynamic_functions import create_folders, load_and_merge, split_data, train_model, evaluate_models

# Ruta base para almacenar outputs
BASE_PATH = '/opt/airflow/dags'
MODEL_NAME = "hiring_rf_pipeline"

# Función de decisión
def decidir_descarga(**context):
    execution_date = context['execution_date']
    fecha_limite = datetime(2024, 11, 1, tzinfo=timezone.utc)
    
    if execution_date < fecha_limite:
        return 'download_data_1_only'
    else:
        return 'download_both_data'
    

# 1. Inicializar un DAG con fecha de inicio el 1 de octubre de 2024, el cual se ejecuta el día 5 de cada mes a las 15:00 UTC
with DAG(
    dag_id='monthly_hiring_pipeline',
    start_date=datetime(2024, 10, 1, tzinfo=timezone.utc),
    schedule_interval='0 15 5 * *',     # Día 5 de cada mes a las 15:00 UTC
    catchup=True        # Backfill habilitado
) as dag:
    
    # 2. Comenzar con un marcador de posición que indique el inicio del pipeline
    start = EmptyOperator(task_id='start_pipeline')

    # 3. Crear una carpeta correspondiente a la ejecución del pipeline y cree las subcarpetas `raw`, `splits` y `models` mediante la función `create_folders()
    crear_carpetas = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders,
        op_kwargs={'base_path': BASE_PATH}
    )

    # Ruta a carpeta de ejecución (La funcion create_folders la retorna)
    execution_path = "{{ ti.xcom_pull(task_ids='create_folders') }}"

    # 4. Descargar datos
    branch_task = BranchPythonOperator(
    task_id='branching',
    python_callable=decidir_descarga,
    provide_context=True
    )

    # Rama 1: Solo data_1.csv (antes del 1 nov 2024)
    download_data_1_only = BashOperator(
        task_id='download_data_1_only',
        bash_command=(
            f'curl -o {BASE_PATH}/{{{{ ds }}}}/raw/data_1.csv '
            'https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv'
        )
    )

    # Rama 2: Ambos archivos (desde 1 nov 2024)
    download_both_data = BashOperator(
        task_id='download_both_data',
        bash_command=(
            f'curl -o {BASE_PATH}/{{{{ ds }}}}/raw/data_1.csv '
            'https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv && '
            f'curl -o {BASE_PATH}/{{{{ ds }}}}/raw/data_2.csv '
            'https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv'
        )
    )

    # 5. Merge
    merge_datos = PythonOperator(
        task_id='merge_data',
        python_callable=load_and_merge,
        op_kwargs={'execution_path': execution_path},
        trigger_rule=TriggerRule.ONE_SUCCESS
    )

    # 6. Split de datos
    split_datos = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs={'execution_path': execution_path}
    )

    # 7. Entrenamiento de modelos

    # Entrenamiento 1: Random Forest
    train_rf = PythonOperator(
        task_id='train_random_forest',
        python_callable=train_model,
        op_kwargs={
            'execution_path': execution_path,
            'clf': RandomForestClassifier(random_state=42, n_estimators=100),
            'model_name': 'random_forest_model'
        }
    )

    # Entrenamiento 2: Logistic Regression
    train_lr = PythonOperator(
        task_id='train_logistic_regression',
        python_callable=train_model,
        op_kwargs={
            'execution_path': execution_path,
            'clf': LogisticRegression(random_state=42, max_iter=600),
            'model_name': 'logistic_regression_model'
        }
    )

    # Entrenamiento 3: Gradient Boosting
    train_gb = PythonOperator(
        task_id='train_gradient_boosting',
        python_callable=train_model,
        op_kwargs={
            'execution_path': execution_path,
            'clf': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'model_name': 'gradient_boosting_model'
        }
    )

    # 8. Evaluar modelos
    evaluar_modelos = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,
        op_kwargs={'execution_path': execution_path}
    )
   
    # Definir dependencias
    start >> crear_carpetas >> branch_task >> [download_data_1_only, download_both_data] >> merge_datos
    merge_datos >> split_datos >> [train_rf, train_lr, train_gb] >> evaluar_modelos