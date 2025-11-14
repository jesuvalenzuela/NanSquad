import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

import joblib
import optuna
import mlflow
import shap

import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, chi2_contingency

#import gradio as gr

# Configurar logging
import logging
logging.getLogger('mlflow').setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ===========================
# ====== SETUP INICIAL ======
# ===========================
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

# ==================================
# ====== PREPARACIÓN DE DATOS ======
# ==================================

# === Revisar si existen registros historicos ===
# Supuestos: Los datos en bruto "aparecen magicamente" en AIRFLOW_HOME/data/raw
# La primera vez que se corre el script, estos se copian a AIRFLOW_HOME/data/historical_raw
# Por lo tanto, si AIRFLOW_HOME/data/historical_raw/transacciones.parquet no existe, es porque es primera vez que se ejecuta el Pipeline
def check_historical_data(data_path):
    """Función de branching: corrobora si existe registro de datos historicos."""
    path_historical = os.path.join(data_path, 'historical_raw', 'transacciones.parquet')
    if os.path.exists(path_historical):
        return 'pass_1'
    else:
        return 'copy_raw'

# === Revisar si hay datos nuevos ===
# Supuesto: estos "aparecen" en AIRFLOW_HOME/data/new
def check_new_data(data_path):
    """Función de branching: corrobora si existen datos nuevos."""

    path_new = os.path.join(data_path, 'new', 'transacciones.parquet')
    if os.path.exists( path_new):
        return 'extend_dataset'
    else:
        return 'pass_2'
    
def extend_dataset(data_path):
    """Agrega nuevas observaciones al dataset historico"""
    path_new = os.path.join(data_path, 'new', 'transacciones.parquet')
    path_historical = os.path.join(data_path, 'historical_raw', 'transacciones.parquet')

    new_rows = pd.read_parquet(path_new)
    df_old = pd.read_parquet(path_historical)

    df_extended = pd.concat([df_old, new_rows], ignore_index=True)
    df_extended.to_parquet(path_historical)
    return path_historical


# ====================================
# ====== PROCESAMIENTO DE DATOS ======
# ====================================

# === Decisión de entrenamiento ===
# Solo se realiza la preparación, split y preprocesamiento de datos si se va entrenar (o reentrenar) el modelo
def decide_if_train(**context):
    """
    Función de branching: decide si es necesario entrenar el modelo.
    Entrena si se agregaron nuevos datos o es primera vez que se corre el pipeline.
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
        return None

# === Formateo inicial de los datos ===
# Supuesto: Solo se agregan nuevas transacciones. Es decir, no se agregan nuevos clientes ni productos.
def read_raw_parquet_files(data_path):
    """Carga 3 archivos parquet desde el directorio de trabajo."""
    transacciones_path = os.path.join(data_path, 'historical_raw', 'transacciones.parquet')
    clientes_path = os.path.join(data_path, 'raw', 'clientes.parquet')
    productos_path = os.path.join(data_path, 'raw', 'productos.parquet')

    # Cargar archivos
    dataframes = []

    for file_path in [transacciones_path, clientes_path, productos_path]:
        # Corroborar que archivo existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        dataframes.append(pd.read_parquet(file_path))

    return dataframes

def calculate_client_quantiles(group, global_quantiles):
    """Calcular cuantiles para un grupo de observaciones."""
    if len(group) >= 5:     # Si ha comprado al menos 5 veces (cualquier producto)
        return group.quantile([0.2, 0.4, 0.6, 0.8])
    else:
        # Pocos datos: usar cuantiles globales
        return pd.Series(global_quantiles, index=[0.2, 0.4, 0.6, 0.8])

def assign_priority(row,
                    client_quantiles,
                    global_quantiles,
                    client_col='customer_id',
                    items_col='weekly_items'
                    ):
    """Asigna string de prioridad según cuantiles."""
    customer_id = row[client_col]
    items_value = row[items_col]
    
    # Obtener cuantiles para este cliente (o usar globales)
    if customer_id in client_quantiles:
        q20, q40, q60, q80 = client_quantiles[customer_id]
    else:
        q20, q40, q60, q80 = global_quantiles
    
    # Asignar etiqueta según cuantil del cliente
    if items_value <= q20:
        return 'Very Low'
    elif items_value <= q40:
        return 'Low'
    elif items_value <= q60:
        return 'Medium'
    elif items_value <= q80:
        return 'High'
    else:
        return 'Very High'
    
def prepare_data(data_path):
    """
    Lee los datos en bruto y los prepara para el modelamiento.
    """
    # Leer datos
    transformed_path = os.path.join(data_path, 'transformed')
    transacciones_0, clientes_0, productos_0 = read_raw_parquet_files(data_path)

    # 1. Eliminar duplicados
    transacciones = transacciones_0.drop_duplicates()
    clientes = clientes_0.drop_duplicates()
    productos = productos_0.drop_duplicates()

    # 2. Cruce de información
    # Usamos left join, ya que solo nos interesan productos que han sido transados.
    df_t = pd.merge(transacciones, productos, on='product_id', how='left')

    # Guardar productos unicos
    productos_unicos = df_t[['product_id', 'brand', 'sub_category', 'segment', 'package', 'size']].copy().drop_duplicates()
    productos_unicos.to_csv(os.path.join(transformed_path, 'unique_products.csv'), index=False)
    
    # Para clientes también usamos left join, ya que solo nos interesan aquellos que compran.
    df_c = pd.merge(transacciones, clientes, on='customer_id', how='left')

    # Guardar clientes unicos
    clientes_unicos = df_c[['customer_id', 'customer_type', 'num_deliver_per_week']].copy().drop_duplicates()
    clientes_unicos.to_csv(os.path.join(transformed_path, 'unique_clients.csv'), index=False)

    # Juntar todo
    # Usamos inner join para preservar transacciones que tienen tanto informacion de clientes como productos.
    df = pd.merge(df_t, df_c, on='transaction_id', how='inner')

    # 3. Corregir tipo de dato
    # Convertir columnas con tipo 'object' a 'category':
    columnas_object = df.select_dtypes(include='object').columns.values
    for col in columnas_object:
        df[col] = df[col].astype('category')

    # 4. Agregación a escala semanal
    # El dataset solo abarca 1 año, por lo que no hay informacion de variabilidad entre años
    # Por lo tanto, para la agregación solo consideramos semanas (sin agregar por año)
    iso_calendar = df['purchase_date'].dt.isocalendar()
    df['week'] = iso_calendar['week']       

    # Cómo agregar
    group_cols = ['customer_id', 'week', 'product_id']
    agg_dict = {'items': 'sum'}

    other_cols = [col for col in df.columns if col not in group_cols + ['items']]
    for col in other_cols:
        agg_dict[col] = 'first'

    # Agregación
    weekly_data = df.groupby(group_cols).agg(agg_dict).reset_index()
    weekly_data.rename(columns={'items': 'weekly_items'}, inplace=True)

    # 5. Creación de variable objetivo
    # Creación de etiquetas
    client_col='customer_id'
    items_col='weekly_items'

    client_quantiles = {}
    global_quantiles = weekly_data[items_col].quantile([0.2, 0.4, 0.6, 0.8]).values     # Cuantiles globales como fallback

    # Calcular cuantiles por cliente
    client_quantiles_df = weekly_data.groupby(client_col)[items_col].apply(
        lambda x: calculate_client_quantiles(x, global_quantiles))
    
    # Convertir a diccionario para acceso rápido
    for customer_id, quantiles in client_quantiles_df.groupby(level=0):
        client_quantiles[customer_id] = quantiles.values
    
    # Aplicar función a cada fila
    weekly_data['priority'] = weekly_data.apply(
        lambda row: assign_priority(row, client_quantiles, global_quantiles),
        axis=1)
    
    # 6. Eliminar columnas que no se usaran
    weekly_data = weekly_data.drop(columns=['X', 'Y', 'order_id', 'region_id', 'zone_id', 'num_visit_per_week', 'category', 'purchase_date', 'weekly_items'])

    # 7. Guardar
    weekly_data.to_csv(os.path.join(transformed_path, 'weekly_data.csv'), index=False)

    return transformed_path

# ====== Holdout ======
def split_data(data_path, random_state=42):
    """Separar datos preparados en conjuntos de entrenamiento, validación y prueba."""
    transformed_data_path = os.path.join(data_path, 'transformed', 'weekly_data.csv')
    weekly_data = pd.read_csv(transformed_data_path)

    # Para conservar el orden temporal, ordenamos el dataset y luego separamos usando 'shuffle = False'
    df_sorted = weekly_data.sort_values(by = ['week'])

    # Determinar variables predictoras y objetivo
    X = df_sorted.drop(columns=['priority'])
    y = df_sorted['priority']

    # Primero, separamos el conjunto de test, tomando el 10% final de los datos totales
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False, random_state=random_state)

    # Luego, separamos los datos restantes en conjuntos de entrenamiento (80%) y validación (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, shuffle=False, random_state=random_state)
    
    # Reconstruir dataframes completos
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Guardar splits
    splits_path = os.path.join(data_path, 'splits')
    train_df.to_csv(os.path.join(splits_path, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(splits_path, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(splits_path, 'test.csv'), index=False)

    return splits_path

# ====== Preprocesamiento ======
class CategoricalImputerByType(BaseEstimator, TransformerMixin):
    """
    Imputador categórico usando moda por tipo de cliente
    """
    def __init__(self, group_col='customer_type'):
        self.group_col = group_col
        self.mode_by_group = {}
        self.global_modes = {}
        
    def fit(self, X, y=None):
        # Calcular modas por grupo usando SOLO entrenamiento
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != self.group_col]
        
        for col in categorical_cols:
            # Moda por tipo de cliente
            self.mode_by_group[col] = X.groupby(self.group_col)[col].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
            ).to_dict()
            
            # Moda global como fallback
            mode_global = X[col].mode()
            self.global_modes[col] = mode_global.iloc[0] if len(mode_global) > 0 else 'Unknown'
            
        return self
    
    def transform(self, X):
        X = X.copy()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != self.group_col]
        
        for col in categorical_cols:
            # Aplicar moda por tipo de cliente
            for group, mode_val in self.mode_by_group[col].items():
                mask = (X[self.group_col] == group) & (X[col].isna())
                if not pd.isna(mode_val):
                    X.loc[mask, col] = mode_val
            
            # Fallback para valores aún nulos
            X[col].fillna(self.global_modes[col], inplace=True)
            
        return X
    
    def set_output(self, transform='default'):
        return self

class IdEncoder(BaseEstimator, TransformerMixin):
    """
    Codifica columnas de IDs (enteros) asignando un número único a cada valor.
    Maneja de forma robusta los IDs que no se vieron durante el entrenamiento.
    """
    def __init__(self, columns):
        self.columns = columns
        # Diccionario para guardar un LabelEncoder por cada columna
        self.label_encoders = {}

    def fit(self, X, y=None):
        """
        Aprende los IDs únicos de cada columna en el set de entrenamiento.
        """
        for col in self.columns:
            if col in X.columns:
                # Se convierte a string para tratar los IDs como categorías discretas
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.label_encoders[col] = le
        return self

    def transform(self, X):
        """
        Aplica la codificación aprendida. Los IDs no vistos se asignan a una nueva categoría.
        """
        X_copy = X.copy()
        for col in self.columns:
            if col in X.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                
                # Convertimos la columna a string para el procesamiento
                current_col_str = X_copy[col].astype(str)
                
                # Identificar valores que no estaban en el set de entrenamiento ('fit')
                unseen_mask = ~current_col_str.isin(le.classes_)
                
                if unseen_mask.any():
                    # Añadir una categoría para 'no vistos' si no existe
                    unseen_category_name = 'UNSEEN_ID'
                    if unseen_category_name not in le.classes_:
                        le.classes_ = np.append(le.classes_, unseen_category_name)
                    
                    # Asignar los valores no vistos a esta nueva categoría
                    current_col_str.loc[unseen_mask] = unseen_category_name

                # Aplicar la transformación y reemplazar la columna original
                X_copy[col] = le.transform(current_col_str)
                
        return X_copy
    
    def set_output(self, transform='default'):
        return self

# Crear el pipeline utilizando las clases definidas
def create_pipeline(numerical_cols=['num_deliver_per_week', 'size', 'week'],
                    categorical_cols=['customer_type', 'brand', 'sub_category', 'segment', 'package'],
                    id_cols=['customer_id', 'product_id']):
    """
    Genera un pipeline completo.
    Se hace limpieza de Nan y codificación de variables.
    Se procesa numericas, categoricas e ids por separado.
    No se imputan outliers, pues se presentan principalmente en variables de id.
    """
    # Pipeline
    full_pipeline = Pipeline([

        # Pasos pre-procesamiento
        ('preprocess', ColumnTransformer([
            ('num', Pipeline([('impute', SimpleImputer()),      # Si bien no hay Nan tras el join, se incluye por robustez
                              ('scale', StandardScaler())
                              ]), 
             numerical_cols),
            ('cat', Pipeline([('impute', CategoricalImputerByType()),
                             ('encode', OneHotEncoder(drop='first', sparse_output=False))       # drop='first' para evitar multicolinealidad
                             ]), 
             categorical_cols),
             # No aplicamos StandardScaler a los IDs codificados:
            ('ids_encoder', Pipeline([('ids', IdEncoder(columns=id_cols))
                                      ]),
             id_cols)
        ], remainder='passthrough',
        verbose_feature_names_out=False)),
    ])
    
    return full_pipeline

def preprocess_data(base_path, target_column='priority'):
    """Ejecutar el preprocesamiento de las features."""
    # Cargar datos
    data_path = os.path.join(base_path, 'data')

    train_df = pd.read_csv(os.path.join(data_path, 'splits', 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_path, 'splits', 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'splits', 'test.csv'))

    # Separar en target y features
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    # Crear pipeline
    pipeline_preprocesamiento = create_pipeline()
    pipeline_preprocesamiento.set_output(transform="pandas")

    # Transformar features
    pipeline_preprocesamiento.fit(X_train)
    X_train_clean = pipeline_preprocesamiento.transform(X_train)
    X_val_clean = pipeline_preprocesamiento.transform(X_val)
    X_test_clean = pipeline_preprocesamiento.transform(X_test)

    # Reconstruir dataframes completos
    train_df_clean = pd.concat([X_train_clean, y_train], axis=1)
    val_df_clean = pd.concat([X_val_clean, y_val], axis=1)
    test_df_clean = pd.concat([X_test_clean, y_test], axis=1)

    # Guardar splits
    preprocessed_path = os.path.join(data_path, 'preprocessed')
    train_df_clean.to_csv(os.path.join(preprocessed_path, 'train.csv'), index=False)
    val_df_clean.to_csv(os.path.join(preprocessed_path, 'val.csv'), index=False)
    test_df_clean.to_csv(os.path.join(preprocessed_path, 'test.csv'), index=False)

    preprocessor_path = os.path.join(base_path, 'models', f'preprocessor.joblib')
    joblib.dump(pipeline_preprocesamiento, preprocessor_path)

    return preprocessed_path

# ==================================================================
# ====== OPTIMIZACIÓN, EVALUACIÓN E INTERPRETACIÓN DEL MODELO ======
# ==================================================================

# ====== Optimización ======
def optimize_model(base_path, target_column='priority', n_trials=50, model_name='KNN_optimo', **context):
    """Optimiza parámetros de clasificador K-Neighbors con Optuna y registra en MLFlow."""
    data_path = os.path.join(base_path, 'data')
    preprocessed_path = os.path.join(data_path, 'preprocessed')

    # Configurar MFlow
    mlruns_path = os.path.join(base_path, 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlruns_path}")

    # Cargar datos: Optimizamos con set de validación
    train_df = pd.read_csv(os.path.join(preprocessed_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(preprocessed_path, 'val.csv'))

    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column]

    # Configurar experimento MLFlow
    execution_date = context['ds']
    experiment_name = f"{model_name}_optimization_{execution_date}"
    mlflow.set_experiment(experiment_name)

    # Función objetivo para Optuna
    def objective(trial):
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 75),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),      # 1=Manhattan, 2=Euclid
            "leaf_size": trial.suggest_int("leaf_size", 15, 60),
            "algorithm": trial.suggest_categorical("algorithm", ["auto","ball_tree","kd_tree"])
        }
        
        # Nombre interpretable para el run
        run_name = f"KNN_nn{params['n_neighbors']}_w{params['weights']}_p{params['p']}"
        
        # Registrar en MLFlow
        with mlflow.start_run(run_name=run_name):
            clf = KNeighborsClassifier(**params)
            clf.fit(X_train, y_train)
            
            # Métricas en validación
            y_pred_val = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred_val, average="macro")

            # Registrar en MLflow
            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", f1)
            mlflow.sklearn.log_model(clf, "model")
        
        return f1
    
    # Ejecutar optimización
    study = optuna.create_study(direction='maximize', study_name=f"{model_name}_study")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Obtener mejores parámetros
    best_params = study.best_params
    return best_params

# ====== Evaluación e interpretación ======
def evaluate_and_interpret_model(base_path, target_column='priority', model_name='KNN_optimo', n_shap_samples=500, **context):
    """Evalúa el modelo en el conjunto de test y registra métricas"""
    best_params = context['ti'].xcom_pull(task_ids='optimize_model')

    # Cargar todos los datos
    preprocessed_path = os.path.join(base_path, 'data', 'preprocessed')
    train_df = pd.read_csv(os.path.join(preprocessed_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(preprocessed_path, 'val.csv'))
    test_df = pd.read_csv(os.path.join(preprocessed_path, 'test.csv'))
    
    # Preparar datasets
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    # Configurar MFlow
    mlruns_path = os.path.join(base_path, 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    
    experiment_name = "Model_Evaluation_and_Interpretation"
    mlflow.set_experiment(experiment_name)

    execution_date = context['ds']
    with mlflow.start_run(run_name=f"model_evaluation_and_interpretation_{execution_date}"):
        # Entrenar con mejores parámetros
        best_model = KNeighborsClassifier(**best_params)
        best_model.fit(X_train, y_train)
        
        # Evaluar en todos los sets
        metrics = {}
        
        # Train metrics
        y_pred_train = best_model.predict(X_train)
        metrics['train_f1'] = f1_score(y_train, y_pred_train, average="macro")
        metrics['train_accuracy'] = accuracy_score(y_train, y_pred_train)
        metrics['train_precision'] = precision_score(y_train, y_pred_train, average="macro", zero_division=0)
        metrics['train_recall'] = recall_score(y_train, y_pred_train, average="macro", zero_division=0)
        
        # Validation metrics
        y_pred_val = best_model.predict(X_val)
        metrics['valid_f1'] = f1_score(y_val, y_pred_val, average="macro")
        metrics['valid_accuracy'] = accuracy_score(y_val, y_pred_val)
        metrics['valid_precision'] = precision_score(y_val, y_pred_val, average="macro", zero_division=0)
        metrics['valid_recall'] = recall_score(y_val, y_pred_val, average="macro", zero_division=0)
        
        # Test metrics
        y_pred_test = best_model.predict(X_test)
        metrics['test_f1'] = f1_score(y_test, y_pred_test, average="macro")
        metrics['test_accuracy'] = accuracy_score(y_test, y_pred_test)
        metrics['test_precision'] = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
        metrics['test_recall'] = recall_score(y_test, y_pred_test, average="macro", zero_division=0)
        
        # Registrar todas las métricas
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        
        # Guardar classification report
        report = classification_report(y_test, y_pred_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Guardar como artefacto temporal
        report_path = os.path.join(base_path, 'temp_classification_report.csv')
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)
        os.remove(report_path)  # Limpiar archivo temporal
        
        # Interpretabilidad con SHAP
        try:
            X_sample = X_test.sample(min(n_shap_samples, len(X_test)), random_state=42)
            
            # KNN usa KernelExplainer
            explainer = shap.KernelExplainer(
                best_model.predict_proba,
                shap.sample(X_train, min(100, len(X_train)))
            )
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            shap_path = os.path.join(base_path, 'temp_shap_summary.png')
            plt.savefig(shap_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            mlflow.log_artifact(shap_path)
            os.remove(shap_path)  # Limpiar archivo temporal
            
        except Exception as e:
            print(f"Error generando SHAP: {str(e)}")
            mlflow.log_param("shap_error", str(e))
        
        # Guardar modelo entrenado con train set
        model_train_path = os.path.join(base_path, 'models', f'{model_name}_train.joblib')
        joblib.dump(best_model, model_train_path)
        mlflow.log_artifact(model_train_path)

    return mlruns_path

# ====== Entrenar modelo final con todos los datos ======
def train_final_model(base_path, target_column='priority', model_name='KNN_optimo', **context):
    """Entrena el modelo con todos los datos, usando los parámetros optimos encontrados en la optimizacion."""
    best_params = context['ti'].xcom_pull(task_ids='optimize_model')

    # Configurar MLflow
    mlruns_path = os.path.join(base_path, 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    
    # Cargar todos los datos una sola vez
    preprocessed_path = os.path.join(base_path, 'data', 'preprocessed')
    train_df = pd.read_csv(os.path.join(preprocessed_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(preprocessed_path, 'val.csv'))
    test_df = pd.read_csv(os.path.join(preprocessed_path, 'test.csv'))
    
    # Preparar datasets
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]
    
    # Concatenar todos los datos para el entrenamiento final
    X_all = pd.concat([X_train, X_val, X_test], ignore_index=True)
    y_all = pd.concat([y_train, y_val, y_test], ignore_index=True)

    experiment_name = "Final_Model"
    mlflow.set_experiment(experiment_name)

    execution_date = context['ds']
    with mlflow.start_run(run_name=f"final_model_{execution_date}"):

        final_model = KNeighborsClassifier(**best_params)
        final_model.fit(X_all, y_all)
        
        # Guardar modelo final
        model_final_path = os.path.join(base_path, 'models', f'{model_name}.joblib')
        joblib.dump(final_model, model_final_path)
        
        # Registrar el modelo final en MLflow
        mlflow.sklearn.log_model(
            final_model, 
            "model_full_data",
            registered_model_name=model_name
        )

# ==================================================
# ====== CIERRE Y PREPARACIÓN PARA DESPLIEGUE ======
# ==================================================
def save_library_versions():
    # Guardar versiones de librerías

    with open("library_versions.txt", "w") as f:
        f.write(f"python: {sys.version}\n")
        f.write(f"optuna: {optuna.__version__}\n")
        f.write(f"mlflow: {mlflow.__version__}\n")
        f.write(f"shap: {shap.__version__}\n")
        f.write(f"pandas: {pd.__version__}\n")
        f.write(f"numpy: {np.__version__}\n")
        f.write(f"matplotlib: {plt.matplotlib.__version__}\n")
        f.write(f"joblib: {joblib.__version__}\n")
        f.write(f"sklearn: {Pipeline.__module__.split('.')[0]}\n")
        #f.write(f"gradio: {gr.__version__}\n")

# ==================================================
# ====== Detección de drift (no implementado) ======
# ==================================================
def detect_drift(significance_level=0.05, **context):
    """
    Detecta data drift comparando distribuciones entre train y nuevos datos.
    Retorna True si se detecta drift significativo.
    """
    base_path = context['ti'].xcom_pull(task_ids='create_folders')
    preprocessed_path = os.path.join(base_path, 'preprocessed')
    drift_reports_path = os.path.join(base_path, 'drift_reports')
    
    # Cargar datos
    train_df = pd.read_csv(os.path.join(preprocessed_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(preprocessed_path, 'test.csv'))
    
    drift_detected = False
    drift_results = {}
    
    # Identificar columnas numéricas (excluir target)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'priority' in numeric_cols:
        numeric_cols.remove('priority')
    
    # Test de Kolmogorov-Smirnov para variables numéricas
    for col in numeric_cols:
        if col in test_df.columns:
            statistic, p_value = ks_2samp(train_df[col], test_df[col])
            
            drift_results[col] = {
                'test': 'KS',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'drift': p_value < significance_level
            }
            
            if p_value < significance_level:
                drift_detected = True
                print(f"Drift detectado en '{col}': p-value={p_value:.4f}")
    
    # Guardar reporte
    drift_report = {
        'timestamp': datetime.now().isoformat(),
        'drift_detected': drift_detected,
        'significance_level': significance_level,
        'features': drift_results
    }
    
    import json
    report_path = os.path.join(drift_reports_path, 'drift_report.json')
    with open(report_path, 'w') as f:
        json.dump(drift_report, f, indent=2)
    
    # Configurar MLflow y registrar
    mlruns_path = os.path.join(base_path, 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlruns_path}")

    experiment_name = "Drift_Detection"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"drift_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_metric("drift_detected", int(drift_detected))
        mlflow.log_metric("n_features_with_drift", sum(1 for r in drift_results.values() if r['drift']))
        mlflow.log_artifact(report_path)

    # Pushear resultado para decision
    context['ti'].xcom_push(key='drift_detected', value=drift_detected)
    
    return drift_detected


# ====== Interfaz gradio ======
def prepare_data_for_prediction(week, customer_id, base_path):
    """
    Lee el input del usuario y lo prepara para la predicción.
    Es decir, genera un dataframe con las siguientes columnas:
    ['customer_id', 'week', 'product_id', 'customer_type', 'num_deliver_per_week',
    'brand', 'sub_category', 'segment', 'package', 'size'] considerando todos los productos
    y el cliente y semana ingresados.
    """
    # Leer datos
    transformed_path = os.path.join(base_path, "data", "transformed")
    clientes = pd.read_csv(os.path.join(transformed_path, 'unique_clients.csv'))
    productos = pd.read_csv(os.path.join(transformed_path, 'unique_products.csv'))

    # rescatar información de productos:
    model_input = productos.copy()

    # generar columnas para inputs de usuario:
    model_input['customer_id'] = customer_id
    model_input['week'] = week

    # rescatar información de clientes:
    model_input = pd.merge(model_input, clientes,
                           on='customer_id',
                           how='left')

     # Convertir columnas con tipo 'object' a dtype 'category':
    columnas_object = model_input.select_dtypes(include='object').columns.values
    for col in columnas_object:
        model_input[col] = model_input[col].astype('category')

    return model_input[['customer_id', 'week', 'product_id', 'customer_type', 'num_deliver_per_week',
                        'brand', 'sub_category', 'segment', 'package', 'size']]


def predict(week, customer_id, base_path, model_name='KNN_optimo'):
    """Entrega prediccion para la semana siguiente para el cliente especificado."""

    model_path = os.path.join(base_path, 'models', f'{model_name}.joblib')

    # transformar
    input_data = prepare_data_for_prediction(week, customer_id, base_path)

    # preprocesar
    preprocessor_path = os.path.join(base_path, 'models', f'preprocessor.joblib')
    preprocessor = joblib.load(preprocessor_path)
    data_clean = preprocessor.transform(input_data)

    model = joblib.load(model_path)
    predictions = model.predict(data_clean)
    products = input_data['product_id'].values

    result = dict(zip(products, predictions))
    print(f"""Predicción para cliente {customer_id}:
          {result}""")
    return result

# Calcular número de "próxima semana"
def calculate_week_number(**context):
    """
    Calcula el número de semana basándose en:
    - Semana 52: fecha de inicio del DAG (31-dic-2024)
    - Incrementa +1 por cada semana desde entonces
    """
    execution_date = context['execution_date']
    
    # Fecha de inicio (semana 52)
    start_date = datetime(2024, 12, 31)
    
    # Calcular semanas transcurridas desde el inicio
    weeks_elapsed = (execution_date - start_date).days // 7
    
    # Semana base (52) + semanas transcurridas + 1
    next_week = 52 + weeks_elapsed + 1

    # Guardar en XCom para usar en la tarea de predicción
    context['task_instance'].xcom_push(key='next_week', value=next_week)
    
    return next_week


def predict_next_week_all_customers(base_path, model_name='KNN_optimo', **context):
    ti = context['task_instance']
    next_week = ti.xcom_pull(task_ids='calculate_week', key='next_week')

    clients_path = os.path.join(base_path, 'data', 'transformed', 'unique_clients.csv')
    clients_df = pd.read_csv(clients_path)
    client_ids = clients_df['customer_id'].values

    predictions = {}
    for client in client_ids:
        predictions[client] = predict(next_week, client, base_path, model_name)

    return predictions

def gradio_interface(**context):
    """
    Despliega modelo en gradio.
    """
    base_path = context['ti'].xcom_pull(task_ids='create_folders')
    model_path = context['ti'].xcom_pull(task_ids='optimize_model')

    interface = gr.Interface(
        fn=lambda file: predict_next_week(file, model_path, base_path),
        inputs=gr.File(label="Ingresa un ID de cliente"),
        outputs="json",
        title="Product Priority Prediction",
        description="Ingresa un ID de cliente para obtener una predicción de la prioridad de compra para cada producto."
    )
    interface.launch(share=True)
