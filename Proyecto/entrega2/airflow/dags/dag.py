from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
#from airflow.models import Variable
from airflow.models import TaskInstance

from decision_functions import (
    extend_dataset,
    prepare_data,
    split_data,
    preprocess_data,
    optimize_model,
    evaluate_and_interpret_model,
    train_final_model,
    #detect_drift,
    #should_retrain,
    #retrain_model,
    save_library_versions
)


# Ruta base para almacenar outputs
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
BASE_PATH = AIRFLOW_HOME
DATA_PATH = os.path.join(AIRFLOW_HOME, 'data')
#RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
#NEW_DATA_PATH = os.path.join(DATA_PATH, 'new_data')
#PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')


#BASE_PATH = '/opt/airflow/dags'      # Guardar en dags
#DATA_PATH = '/opt/airflow/data'      # Guardar en dags
MODEL_NAME = "product_priority_model"

# === Funciones auxiliares ===
def assert_folders(base_path='/opt/airflow'):
    """
    Corrobora que, dentro de la carpeta 'airflow', existen las siguientes carpetas:
        - data/raw
        - data/new
        - data/historical_raw
        - data/transformed
        - data/splits
        - data/preprocessed
        - mlruns
        - models
    """   
    # 1. Corroborar que exiten todos los directorios "base"
    data_path = os.path.join(base_path, 'data')
    os.makedirs(data_path, exist_ok=True)
    data_subfolders = ['raw', 'new', 'historical_raw', 'transformed', 'preprocessed', 'splits']
    for subfolder in data_subfolders:
        subfolder_path = os.path.join(data_path, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

    mlruns_path = os.path.join(base_path, 'mlruns')
    os.makedirs(mlruns_path, exist_ok=True)

    models_path = os.path.join(base_path, 'models')
    os.makedirs(models_path, exist_ok=True)

    return base_path

assert_folders(AIRFLOW_HOME)



# Funciones de decisión
def check_historical_data(data_path):
    """Corrobora si existe registro de datos historicos"""

    path_historical = os.path.join(data_path, 'historical_raw', 'transacciones.parquet')
    if os.path.exists(path_historical):
        return 'pass_1'
    else:
        return 'copy_raw'
    
def check_new_data(data_path):
    """Corrobora si existen datos nuevos"""

    path_new = os.path.join(data_path, 'new', 'transacciones.parquet')
    if os.path.exists( path_new):
        return 'extend_dataset'
    else:
        return 'pass_2'

def decide_if_train(**context):
    """Decide si es necesario entrenar el modelo
        - Entrena si: se agregaron nuevos datos o 
        """
    dag_run = context['dag_run']
    
    # Obtener estados de las tareas
    copy_raw_instance = dag_run.get_task_instance('copy_raw')
    extension_instance = dag_run.get_task_instance('extend_dataset')
    
    # Corroborar si alguna corrio con exito
    if copy_raw_instance and extension_instance:
        copy_raw_success = copy_raw_instance.state == 'success'
        extension_success = extension_instance.state == 'success'

        if copy_raw_success or extension_success:
            return 'prepare_data'
        else:
            return 'not_train'
    else:
        print(f"One of the tasks does not exist!!")


# Argumentos por defecto del DAG
default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Inicializar un DAG con fecha de inicio el 31 de diciembre de 2024, ejecución manual y **sin backfill**
with DAG(
    dag_id='ml_pipeline_with_drift_detection',
    default_args=default_args,
    start_date=datetime(2024, 12, 31),
    schedule_interval='@weekly',  # Reentrenamiento semanal
    catchup=False       # Sin backfill
) as dag:
    
    # ========================================
    # 1. INICIO Y PREPARACIÓN DE DATOS
    # ========================================

    # Iniciar
    start = EmptyOperator(task_id='start_pipeline')

    # Revisar si existe dataset histórico
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

    # ================

    # Revisar si hay nuevos datos
    revisar_nuevos_datos = BranchPythonOperator(
        task_id='check_new_data',
        python_callable=check_new_data,
        op_kwargs={'data_path': DATA_PATH}
    )

    # Si hay nuevos datos: extender dataset
    incorporar_nuevos_datos = PythonOperator(
        task_id='extend_dataset',
        python_callable=extend_dataset,
        op_kwargs={'data_path': DATA_PATH}
    )
    
    # Si no hay nuevos datos: no hacer nada
    pasar_2 = EmptyOperator(task_id='pass_2')

    # ========================================
    # 2. PROCESAMIENTO DE DATOS 
    # ========================================
    
    decidir_entrenamiento = PythonOperator(
        task_id='decide_training',
        python_callable=decide_if_train,
        trigger_rule='none_failed'
    )

    no_entrenar = EmptyOperator(task_id='not_train')

    preparar_datos = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
        p_kwargs={'data_path': DATA_PATH}
    )

    split_datos = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        p_kwargs={'data_path': DATA_PATH}
    )

    preprocesar = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        p_kwargs={'base_path': BASE_PATH}
    )

    # ========================================
    # 2. ENTRENAMIENTO Y OPTIMIZACIÓN
    # ========================================

    optimizar = PythonOperator(
        task_id='optimize_model',
        python_callable=optimize_model,
        op_kwargs={
            'base_path': BASE_PATH,
            'target_column': 'priority',
            'n_trials': 50,
            'model_name': MODEL_NAME
        }
    )
    
    evaluar_interpretar = PythonOperator(
        task_id='evaluate_and_interpret',
        python_callable=evaluate_and_interpret_model,
        op_kwargs={
            'base_path': BASE_PATH,
            'target_column': 'priority',
            'n_shap_samples':500,
            'model_name': MODEL_NAME
        }
    )

    entrenar_modelo_final = PythonOperator(
        task_id='evaluate_and_interpret',
        python_callable=train_final_model,
        op_kwargs={
            'base_path': BASE_PATH,
            'model_name': MODEL_NAME
        }
    )
    
    # ========================================
    # FASE 3: DRIFT DETECTION Y REENTRENAMIENTO
    # ========================================

    """detectar_drift = PythonOperator(
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
    )"""

    # ========================================
    # 4. FINALIZACIÓN
    # ========================================
    
    guardar_versiones = PythonOperator(
        task_id='save_library_versions',
        python_callable=save_library_versions,
        trigger_rule='none_failed' # Se ejecuta aunque se haya saltado el entrenamiento
    )

    end = EmptyOperator(
        task_id='end_pipeline',
        trigger_rule='none_failed'
    )

    # ========================================
    # DEFINICIÓN DE DEPENDENCIAS
    # ========================================
    # se asume que si no hay nada en historico, es la primera vez que se corre, por lo que tampoco hay modelo, datos preprocesados, etc
    # si hay historico, se asume que ya hay un modelo entrenado, por que lo si no hay datos nuevos no se reentrena
    # si hay historico y datos nuevos, se reentrena el modelo

    # Inicio y primera bifuracion
    start >> revisar_historicos >> [copiar_raw_a_historico, pasar_1] >> revisar_nuevos_datos


    # Segunda bifurcación
    revisar_nuevos_datos >> [incorporar_nuevos_datos, pasar_2] >> decidir_entrenamiento
    # detectar entrenamiento revisa los outputs de funciones anteriores, y entrena si no habia datos historicos o si se detectaron nuevos datos
    
    # Tercera bifurcacion
    decidir_entrenamiento >> [preparar_datos, no_entrenar]

    # Si se toma la rama "preparar datos", se sigue todo el flujo de preparacion, entrenamiento, etc.
    preparar_datos >> split_datos >> preprocesar >> optimizar >> [evaluar_interpretar, entrenar_modelo_final] >> guardar_versiones
    
    # Si se toma la rama "pasar_3", se pasa directo a la tarea final
    no_entrenar >> guardar_versiones 

    guardar_versiones >> end
    
    # Branching para reentrenamiento
    #detectar_drift >> decidir_reentrenamiento
    #decidir_reentrenamiento >> [reentrenar, skip_reentrenamiento]
    
    # Convergencia y finalización
    #[reentrenar, skip_reentrenamiento] >> guardar_versiones >> end

