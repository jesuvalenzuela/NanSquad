import os
from pathlib import Path
from datetime import datetime

def create_folders(base_path ='/opt/airflow/data', **kwargs):
    """
    Crea una carpeta, la cual utiliza la fecha de ejecución como nombre.
    Adicionalmente, dentro de esta carpeta crea las siguientes subcarpetas:
        - raw
        - splits
        - models
    """
    # Obtener la fecha de ejecución desde el contexto de Airflow
    execution_date = kwargs['execution_date']
    
    # Formatear la fecha como string (YYYY-MM-DD)
    date_folder = execution_date.strftime('%Y-%m-%d')
    
    # Crear la carpeta principal con la fecha
    main_folder = os.path.join(base_path, date_folder)
    os.makedirs(main_folder, exist_ok=True)
    
    # Crear las subcarpetas
    subfolders = ['raw', 'splits', 'models']
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

    return main_folder
