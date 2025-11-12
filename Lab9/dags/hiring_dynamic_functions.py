import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import joblib

def create_folders(base_path ='', **kwargs):
    """
    Crea una carpeta, la cual utiliza la fecha de ejecución como nombre.
    Adicionalmente, dentro de esta carpeta crea las siguientes subcarpetas:
        - raw
        - preprocessed
        - splits
        - models
    """
    # Obtener la fecha de ejecución desde el contexto de Airflow
    execution_date = kwargs['ds']
    #execution_date = execution_date.strftime('%Y-%m-%d')       # Formatear la fecha como string (YYYY-MM-DD)

    # Crear la carpeta principal con la fecha
    main_folder = os.path.join(base_path, execution_date)
    os.makedirs(main_folder, exist_ok=True)
    
    # Crear las subcarpetas
    subfolders = ['raw', 'preprocessed', 'splits', 'models']
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

    return main_folder

def load_and_merge(execution_path, out_name="merged_data.csv"):
    """
    Lee desde la carpeta `raw` los archivos `data_1.csv`y `data_2.csv` en caso de estar disponibles.
    Luego concatena estos y genera un nuevo archivo resultante, guardándolo en la carpeta `preprocessed`.
    """
    # Paths
    raw_path = os.path.join(execution_path, 'raw')

    # Lista para almacenar dataframes disponibles
    dfs = []
    
    # Intentar cargar cada archivo
    for filename in ["data_1.csv", "data_2.csv"]:
        filepath = os.path.join(raw_path, filename)

        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            dfs.append(df)
            print(f"{filename} cargado")
        else:
            print(f"{filename} no encontrado")

    # Concatenar si hay datos disponibles
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(execution_path, 'preprocessed', out_name)
        merged_df.to_csv(output_path, index=False)
        print(f"Archivo fusionado guardado: {output_path}")
        return output_path
    else:
        print("No se encontraron archivos para procesar")
        return None
    

def split_data(execution_path,
               merged_file_name="merged_data.csv",
               target_column="HiringDecision",
               test_size=0.2,
               seed=42):
    """Lee la data guardada en la carpeta `preprocessed` y realiza un hold out sobre esta data.
    Crea un conjunto de entrenamiento y uno de prueba. Mantiene una semilla y 20% para el conjunto de prueba.
    Guarda los conjuntos resultantes en la carpeta `splits`."""

    # Rutas
    data_path = os.path.join(execution_path, 'preprocessed', merged_file_name)
    splits_path = os.path.join(execution_path, 'splits')

    # Cargar datos
    df = pd.read_csv(data_path)

    # Separar features y target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Hold-out estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )
    
    # Reconstruir dataframes completos
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Guardar splits
    train_df.to_csv(os.path.join(splits_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(splits_path, 'test.csv'), index=False)

    return splits_path


def train_model(execution_path,
                clf,
                model_name,
                target_column="HiringDecision"):
    """Recibe un modelo de clasificación.
        - Comienza leyendo el conjunto de entrenamiento desde la carpeta `splits`
        - Crea y aplicar un `Pipeline` con una etapa de preprocesamiento
        - Añade una etapa de entrenamiento utilizando un modelo que ingrese a la función.
        - Crea un archivo joblib con el pipeline entrenado
    """
    # Leer conjunto de entrenamiento
    splits_path = os.path.join(execution_path, 'splits')
    train_df = pd.read_csv(os.path.join(splits_path, 'train.csv'))
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]

    # Crear pipeline
    encoding_columns = ["EducationLevel", "PreviousCompanies", "RecruitmentStrategy"]
    scaler_columns = ["Age", "ExperienceYears", "DistanceFromCompany", "InterviewScore", "SkillScore", "PersonalityScore"]

    # Nota: se agrega scaler para mayor flexibilidad ante modelos sensibles a escalas, lo cual no era necesario cuando solo usabamos RF
    preprocesador = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(sparse_output=False, drop='first'), encoding_columns),
                      ("scaler", StandardScaler(), scaler_columns)
                      ],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    pipe = Pipeline(steps=[("pre", preprocesador), ("clf", clf)])

    # Entrenar
    pipe.fit(X_train, y_train)

    # Guardar pipeline entrenado
    model_path = os.path.join(execution_path, 'models', f'{model_name}.joblib')
    joblib.dump(pipe, model_path)

    return model_path


def evaluate_models(execution_path, target_column="HiringDecision"):
    """
    - Recibe modelos entrenados desde la carpeta `models`
    - Evalúa su desempeño mediante `accuracy` en el conjunto de prueba
    - Selecciona el mejor modelo obtenido
    - Guarda el mejor modelo como archivo `.joblib`
    - Imprime el nombre del modelo seleccionado y el accuracy obtenido
    """
    # Identificar modelos en carpeta models
    models_path = os.path.join(execution_path, 'models')
    models = os.listdir(models_path)

    # Filtrar solo archivos .joblib
    model_files = [f for f in models if f.endswith('.joblib')]

    # Cargar datos de prueba
    splits_path = os.path.join(execution_path, 'splits')
    test_df = pd.read_csv(os.path.join(splits_path, 'test.csv'))
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    # Evaluar cada modelo
    best_accuracy = 0
    best_model_name = None
    best_model = None

    for model_file in model_files:
        model_path = os.path.join(models_path, model_file)
        model = joblib.load(model_path)
        
        # Predecir y calcular accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy {model_file}: {accuracy:.4f}")
        
        # Actualizar mejor modelo si es necesario
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_file
            best_model = model
    
    # Guardar mejor modelo
    best_model_path = os.path.join(execution_path, 'models', 'best_model.joblib')
    joblib.dump(best_model, best_model_path)
    
    # Imprimir resultados
    print(f"Mejor modelo: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f}")
