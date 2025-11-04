import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import joblib

import gradio as gr

# === 1.1.1. Creado de carpetas ===
def create_folders(base_path ='', **kwargs):
    """
    Crea una carpeta, la cual utiliza la fecha de ejecución como nombre.
    Adicionalmente, dentro de esta carpeta crea las siguientes subcarpetas:
        - raw
        - splits
        - models
    """
    # Obtener la fecha de ejecución desde el contexto de Airflow
    execution_date = kwargs['ds']

    # Formatear la fecha como string (YYYY-MM-DD)
    #execution_date = execution_date.strftime('%Y-%m-%d')

    # Crear la carpeta principal con la fecha
    main_folder = os.path.join(base_path, execution_date)
    os.makedirs(main_folder, exist_ok=True)
    
    # Crear las subcarpetas
    subfolders = ['raw', 'splits', 'models']
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

    return main_folder


# === 1.1.2. Holdout ===
def split_data(execution_path,
               target_column="HiringDecision",
               test_size=0.2,
               seed=42):
    """
    Lee data_1.csv desde raw/, aplica hold-out estratificado y guarda
    train.csv y test.csv en splits/.

    Parámetros:
    - execution_path: ruta a la carpeta de ejecución del run.
    - target_column: nombre de columna target (y) en el df
    - test_size: fracción para test (default 0.20).
    - seed: semilla (default 42).
    """

    # Rutas
    raw_path = os.path.join(execution_path, 'raw', 'data_1.csv')
    splits_path = os.path.join(execution_path, 'splits')

    # Cargar datos
    df = pd.read_csv(raw_path)

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

    return X_train, X_test, y_train, y_test


# === 1.1.3. Preprocesamiento + entrenamiento + guardado del pipeline ===

def preprocess_and_train(execution_path,
                         target_column="HiringDecision",
                         seed=42,
                         model_name="hiring_rf_pipeline"):
    """
    Esta función:
        - Lee los set de entrenamiento y prueba de la carpeta `splits`
        - Crea y aplica un `Pipeline` con una etapa de preprocesamiento
        - Entrena un modelo RandomForest
        - Crea un archivo `joblib` con el pipeline entrenado** en la carpeta `models`
        - Imprime el accuracy en el conjunto de prueba y el f1-score de la clase positiva (contratado)
        
    El preprocesamiento realizado, de acuerdo a lo observado en el archivo `data_1_report.html` es el siguiente:
        (a) Variables numéricas: Age, ExperienceYears, DistanceFromCompany, InterviewScore, SkillScore y PersonalityScore.
            - Escalado: No Requerido. Si bien el reporte muestra que estas variables tienen rangos y distribuciones muy diferentes,
            los modelos Random Forest no se ven afectados por la escala de las features.
        
        (b) Variables categóricas: Gender, EducationLevel, PreviousCompanies y RecruitmentStrategy.
            - Encoding: Gender ya está en formato binario (0 y 1); puede dejarse como está. EducationLevel, PreviousCompanies,
            RecruitmentStrategy tienen baja cardinalidad (2 a 5 valores únicos); utilizar One-Hot Encoding (con drop_first=True
            para evitar multicolinealidad).

        (c) Variable Objetivo: HiringDecision
            - El reporte muestra que es categórica con 2 valores únicos (0 y 1). No requiere transformación.

        (d) El reporte indica que no hay celdas faltantes (0.0%). No se necesita imputación.
"""
    # Leer los set de entrenamiento y prueba
    splits_path = os.path.join(execution_path, 'splits')

    train_df = pd.read_csv(os.path.join(splits_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(splits_path, 'test.csv'))

    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    # Crear pipeline
    encoding_columns = ["EducationLevel", "PreviousCompanies", "RecruitmentStrategy"]

    preprocesador = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(sparse_output=False, drop='first'), encoding_columns)],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    clf = RandomForestClassifier(n_estimators=300, random_state=seed)

    pipe = Pipeline(steps=[("pre", preprocesador), ("clf", clf)])

    # Entrenar
    pipe.fit(X_train, y_train)

    # Evaluar
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_pos = f1_score(y_test, y_pred, pos_label=1)

    print(f"Accuracy (test): {acc:.4f}")
    print(f"F1 clase positiva (contratado): {f1_pos:.4f}")

    # Guardar pipeline entrenado
    model_path = os.path.join(execution_path, 'models', f'{model_name}.joblib')
    joblib.dump(pipe, model_path)

    return model_path


# === 1.1.4. Interfaz gradio ===
def predict(file, model_path):

    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}


def gradio_interface(model_path):
    """
    Despliega modelo en gradio.

    Nota: En lugar de dejar fija la ruta a model path, como se sugería, se dejó como input de la función
    para ingresar el output de preprocess_and_train, que es justamente la ruta donde se guardo el modelo.
    """
    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(share=True)
