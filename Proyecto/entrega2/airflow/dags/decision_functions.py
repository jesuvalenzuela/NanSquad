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

import gradio as gr

# ====== Crear carpetas ======
def create_folders(base_path='', **context):
    """
    Crea una carpeta, la cual utiliza la fecha de ejecución como nombre.
    Adicionalmente, dentro de esta carpeta crea las siguientes subcarpetas:
        - raw
        - transformed
        - splits
        - preprocessed
        - mlruns
        - models
        - drift_reports
        - interpretability
    """
    # Obtener la fecha de ejecución desde el contexto de Airflow
    execution_date = context['ds']
    #execution_date = execution_date.strftime('%Y-%m-%d')       # Formatear la fecha como string (YYYY-MM-DD)

    # Crear la carpeta principal con la fecha
    main_folder = os.path.join(base_path, execution_date)
    os.makedirs(main_folder, exist_ok=True)
    
    # Crear las subcarpetas
    subfolders = ['raw',
                  'transformed',
                  'preprocessed',
                  'splits',
                  'mlruns',
                  'models',
                  'drift_reports',
                  'interpretability',
                  'evaluation']
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

    return main_folder

# ====== Preparar datos ======
def read_parquet_files(raw_path):
    """Carga 3 archivos parquet desde el directorio de trabajo."""
    
    # Cargar archivos
    files = ['transacciones.parquet', 'clientes.parquet', 'productos.parquet']
    dataframes = []

    for file in files:
        file_path = os.path.join(raw_path, file)
        # Corroborar que archivo existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        dataframes.append(pd.read_parquet(file_path))

    return dataframes


def calculate_client_quantiles(group, global_quantiles):
    """Calcular cuantiles para un grupo de observaciones"""
    if len(group) >= 5:  # Si ha comprado al menos 5 veces (cualquier producto)
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
    """Asigna string de prioridad según cuantiles"""
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
    

def prepare_data(**context):
    """
    Lee los datos en bruto y los prepara para el modelamiento.
    Supuesto: estos aparecen 'magicamente' en la carpeta raw.
    """
    # Leer datos
    main_folder = context['ti'].xcom_pull(task_ids='create_folders')
    raw_path = os.path.join(main_folder, 'raw')
    transformed_path = os.path.join(main_folder, 'transformed')

    transacciones_0, clientes_0, productos_0 = read_parquet_files(raw_path)

    # 1. Eliminar duplicados
    transacciones = transacciones_0.drop_duplicates()
    clientes = clientes_0.drop_duplicates()
    productos = productos_0.drop_duplicates()

    # 2. Cruce de información
    df_t = pd.merge(transacciones, productos, on='product_id', how='left')

    # Guardar productos unicos
    productos_unicos = df_t[['product_id', 'brand', 'sub_category', 'segment', 'package', 'size']].copy().drop_duplicates()
    productos_unicos.to_csv(os.path.join(transformed_path, 'unique_products.csv'), index=False)
    
    df_c = pd.merge(transacciones, clientes, on='customer_id', how='left')

    # Guardar clientes unicos
    clientes_unicos = df_c[['customer_id', 'customer_type', 'num_deliver_per_week']].copy().drop_duplicates()
    clientes_unicos.to_csv(os.path.join(transformed_path, 'unique_clients.csv'), index=False)

    # juntar todo
    df = pd.merge(df_t, df_c, on='transaction_id', how='inner')

    # 3. Corregir tipo de dato
    # Convertir columnas con tipo 'object' a 'category':
    columnas_object = df.select_dtypes(include='object').columns.values
    for col in columnas_object:
        df[col] = df[col].astype('category')

    # 4. Agregación a escala semanal
    iso_calendar = df['purchase_date'].dt.isocalendar()
    df['week'] = iso_calendar['week']       # solo consideramos semanas ya que el df solo abarca 1 año

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
    
    # 6. Eliminar columnas asociadas que no se usaran
    weekly_data = weekly_data.drop(columns=['X', 'Y', 'order_id', 'region_id', 'zone_id', 'num_visit_per_week', 'category', 'purchase_date', 'weekly_items'])

    # 7. Guardar
    weekly_data.to_csv(os.path.join(transformed_path, 'weekly_data.csv'), index=False)

    return transformed_path


# ====== Holdout ======
def split_data(**context):
    """Separar datos preparados en conjuntos de entrenamiento, validación y prueba."""
    main_folder = context['ti'].xcom_pull(task_ids='create_folders')
    transformed_path = f"{main_folder}/transformed/weekly_data.csv"
    weekly_data = pd.read_csv(transformed_path)

    # Para conservar el orden temporal, ordenamos el dataset y luego separamos usando 'shuffle = False'
    df_sorted = weekly_data.sort_values(by = ['week'])

    # Determinar variables predictoras y objetivo
    X = df_sorted.drop(columns=['priority'])
    y = df_sorted['priority']

    # Primero, separamos el conjunto de test, tomando el 10% final de los datos totales
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False, random_state=42)

    # Luego, separamos los datos restantes en conjuntos de entrenamiento (80%) y validación (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, shuffle=False, random_state=42)
    
    # Reconstruir dataframes completos
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Guardar splits
    splits_path = os.path.join(main_folder, 'splits')
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

def preprocess_data(target_column='priority', **context):
    """Ejecutar el preprocesamiento de las features."""
    # Cargar datos
    main_folder = context['ti'].xcom_pull(task_ids='create_folders')
    splits_path = os.path.join(main_folder, 'splits')
    train_df = pd.read_csv(os.path.join(splits_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(splits_path, 'val.csv'))
    test_df = pd.read_csv(os.path.join(splits_path, 'test.csv'))

    # Separar en target y features
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    # Crear pipeline
    pipeline_preprocesamiento = create_pipeline()
    pipeline_preprocesamiento.set_output(transform="pandas")

    # Fit solo en train, transform en train/val/test
    pipeline_preprocesamiento.fit(X_train)
    X_train_clean = pipeline_preprocesamiento.transform(X_train)
    X_val_clean = pipeline_preprocesamiento.transform(X_val)
    X_test_clean = pipeline_preprocesamiento.transform(X_test)

    # Reconstruir dataframes completos
    train_df_clean = pd.concat([X_train_clean, y_train], axis=1)
    val_df_clean = pd.concat([X_val_clean, y_val], axis=1)
    test_df_clean = pd.concat([X_test_clean, y_test], axis=1)

    # Guardar splits
    preprocessed_path = os.path.join(main_folder, 'preprocessed')
    train_df_clean.to_csv(os.path.join(preprocessed_path, 'train.csv'), index=False)
    val_df_clean.to_csv(os.path.join(preprocessed_path, 'val.csv'), index=False)
    test_df_clean.to_csv(os.path.join(preprocessed_path, 'test.csv'), index=False)

    preprocessor_path = os.path.join(main_folder, 'models', f'preprocessor.joblib')
    joblib.dump(pipeline_preprocesamiento, preprocessor_path)

    return preprocessed_path


def get_best_model(experiment_id):
    """Retorna modelo con mayor F1-Score"""
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model   

def optimize_model(target_column='priority', n_trials=50, model_name='KNN_optimo', **context):
    """Optimiza parámetros de clasificador K-Neighbors con Optuna y registra en MLFlow."""
    main_folder = context['ti'].xcom_pull(task_ids='create_folders')
    preprocessed_path = os.path.join(main_folder, 'preprocessed')

    # Configurar MFlow
    mlruns_path = os.path.join(main_folder, 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlruns_path}")

    # Cargar datos
    train_df = pd.read_csv(os.path.join(preprocessed_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(preprocessed_path, 'val.csv'))

    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_val, y_val = val_df.drop(columns=[target_column]), val_df[target_column]

    import logging
    logging.getLogger('mlflow').setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Configurar experimento MLFlow
    experiment_name = f"KNN_Optimization"
    mlflow.set_experiment(experiment_name)
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id        # para get_best_model()

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
            # Modelo
            clf = KNeighborsClassifier(**params)
            
            # Entrenar y predecir
            clf.fit(X_train, y_train)
            y_pred_val = clf.predict(X_val)

            # Calcular métricas
            f1 = f1_score(y_val, y_pred_val, average="macro")
            accuracy = accuracy_score(y_val, y_pred_val)
            precision = precision_score(y_val, y_pred_val, average="macro", zero_division=0)
            recall = recall_score(y_val, y_pred_val, average="macro", zero_division=0)

            # Registrar en MLflow
            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", f1)
            mlflow.log_metric("valid_accuracy", accuracy)
            mlflow.log_metric("valid_precision", precision)
            mlflow.log_metric("valid_recall", recall)
            mlflow.sklearn.log_model(clf, "model")

            # Tags útiles
            mlflow.set_tags({
                "model_type": "KNeighborsClassifier",
                "framework": "sklearn",
                "optimization": "optuna"
            })
        
        return f1
    
    # Ejecutar optimización
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Obtener mejor modelo usando get_best_model()
    best_model = get_best_model(experiment_id)

    # Guardar mejor modelo entrenado
    model_path = os.path.join(main_folder, 'models', f'{model_name}.joblib')
    joblib.dump(best_model, model_path)
    
    return model_path


# ====== Evaluación en test set ======
def evaluate_model(target_column='priority', **context):
    """Evalúa el modelo en el conjunto de test y registra métricas"""
    main_folder = context['ti'].xcom_pull(task_ids='create_folders')
    preprocessed_path = os.path.join(main_folder, 'preprocessed')
    model_path = context['ti'].xcom_pull(task_ids='optimize_model')
    evaluation_path = os.path.join(main_folder, 'evaluation')

    # Configurar MFlow
    mlruns_path = os.path.join(main_folder, 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    
    # Cargar datos y modelo
    test_df = pd.read_csv(os.path.join(preprocessed_path, 'test.csv'))
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]
    
    model = joblib.load(model_path)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    f1 = f1_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    
    # Guardar métricas
    metrics = {
        'test_f1': f1,
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    metrics_path = os.path.join(evaluation_path, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Registrar en MLflow
    experiment_name = "Model_Evaluation"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"test_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_path)
        
        # Guardar classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(evaluation_path, 'classification_report.csv')
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)
    
    return evaluation_path


# ====== Detección de drift ======
def detect_drift(significance_level=0.05, **context):
    """
    Detecta data drift comparando distribuciones entre train y nuevos datos.
    Retorna True si se detecta drift significativo.
    """
    main_folder = context['ti'].xcom_pull(task_ids='create_folders')
    preprocessed_path = os.path.join(main_folder, 'preprocessed')
    drift_reports_path = os.path.join(main_folder, 'drift_reports')
    
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
    mlruns_path = os.path.join(main_folder, 'mlruns')
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


# ====== Interpretabilidad con SHAP ======
def generate_interpretability(n_samples=500, **context):
    """Genera explicaciones SHAP del modelo"""
    main_folder = context['ti'].xcom_pull(task_ids='create_folders')
    preprocessed_path = os.path.join(main_folder, 'preprocessed')
    model_path = context['ti'].xcom_pull(task_ids='optimize_model')
    interpretability_path = os.path.join(main_folder, 'interpretability')
    
    # Cargar datos y modelo
    test_df = pd.read_csv(os.path.join(preprocessed_path, 'test.csv'))
    X_test = test_df.drop(columns=['priority'])
    
    model = joblib.load(model_path)
    
    # Crear explainer SHAP (usar subset para velocidad)
    X_sample = X_test.sample(min(n_samples, len(X_test)), random_state=42)

    try:
        # KNN usa KernelExplainer
        explainer = shap.KernelExplainer(
            model.predict_proba,
            shap.sample(X_test, 100)
        )
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        summary_path = os.path.join(interpretability_path, 'shap_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Registrar en MLflow
        mlruns_path = os.path.join(main_folder, 'mlruns')
        mlflow.set_tracking_uri(f"file://{mlruns_path}")

        experiment_name = "Model_Interpretability"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"shap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_artifact(summary_path)
            mlflow.log_param("n_samples", len(X_sample))
        
        return interpretability_path
        
    except Exception as e:
        print(f"Error generando SHAP: {str(e)}")
        print("Continuando sin interpretabilidad...")
        return None


# ====== Decisión de reentrenamiento ======
def should_retrain(**context):
    """
    Decide si se debe reentrenar basado en:
    1. Detección de drift
    2. Degradación de métricas
    3. Tiempo desde último entrenamiento
    """
    # Obtener resultado de drift
    drift_detected = context['ti'].xcom_pull(
        task_ids='detect_drift', 
        key='drift_detected'
    )
    
    # Solo consideramos drift
    should_retrain_flag = drift_detected
    
    if should_retrain_flag:
        print("REENTRENAMIENTO NECESARIO")
    else:
        print("No se requiere reentrenamiento")
    
    return 'retrain_model' if should_retrain_flag else 'skip_retrain'


# ====== Reentrenamiento ======
def retrain_model(target_column='priority', n_trials=30, **context):
    """Reentrena el modelo con datos actualizados"""
    print("Iniciando reentrenamiento del modelo...")
    
    # Reutilizar la función de optimización pero con menos trials
    return optimize_model(
        target_column=target_column,
        n_trials=n_trials,
        model_name='KNN_retrained',
        **context
    )


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
        f.write(f"gradio: {gr.__version__}\n")


# ====== Interfaz gradio ======
def prepare_data_for_prediction(week, customer_id, main_folder):
    """
    Lee el input del usuario y lo prepara para la predicción.
    Es decir, genera un dataframe con las siguientes columnas:
    ['customer_id', 'week', 'product_id', 'customer_type', 'num_deliver_per_week',
    'brand', 'sub_category', 'segment', 'package', 'size'] considerando todos los productos
    y el cliente y semana ingresados.
    """
    # Leer datos
    transformed_path = f"{main_folder}/transformed"
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


def predict_next_week(customer_id, model_path, main_folder):
    """Entrega prediccion para la semana siguiente para el cliente especificado."""

    # transformar
    transformed_path = f"{main_folder}/transformed"
    weekly_data =  pd.read_csv(os.path.join(transformed_path, 'weekly_data.csv'))

    max_week = weekly_data['week'].max()
    input_data = prepare_data_for_prediction(max_week + 1, customer_id, main_folder)

    # preprocesar
    preprocessor_path = os.path.join(main_folder, 'models', f'preprocessor.joblib')
    preprocessor = joblib.load(preprocessor_path)
    data_clean = preprocessor.transform(input_data)

    model = joblib.load(model_path)
    predictions = model.predict(data_clean)
    products = input_data['product_id'].values

    result = dict(zip(products, predictions))
    print(f"""Predicción para cliente {customer_id}:
          {result}""")
    return result


def gradio_interface(**context):
    """
    Despliega modelo en gradio.
    """
    main_folder = context['ti'].xcom_pull(task_ids='create_folders')
    model_path = context['ti'].xcom_pull(task_ids='optimize_model')

    interface = gr.Interface(
        fn=lambda file: predict_next_week(file, model_path, main_folder),
        inputs=gr.File(label="Ingresa un ID de cliente"),
        outputs="json",
        title="Product Priority Prediction",
        description="Ingresa un ID de cliente para obtener una predicción de la prioridad de compra para cada producto."
    )
    interface.launch(share=True)
