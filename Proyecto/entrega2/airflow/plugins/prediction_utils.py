"""
Funciones de predicción puras (sin dependencias de Airflow).
Pueden usarse tanto en el DAG como en la API FastAPI.
"""

import os
from datetime import datetime
import pandas as pd
import joblib

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


def predict(week, customer_id, base_path, model_name='product_priority_model'):
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

def calculate_week_number(execution_date=None):
    """
    Calcula el número de semana basándose en:
    - Semana 52: fecha de inicio (31-dic-2024)
    - Incrementa +1 por cada semana desde entonces
    
    Args:
        execution_date: datetime object. Si es None, usa la fecha actual.
    
    Returns:
        int: Número de semana
    """
    if execution_date is None:
        execution_date = datetime.now()
    
    # Fecha de inicio (semana 52)
    start_date = datetime(2024, 12, 31)
    
    # Calcular semanas transcurridas desde el inicio
    weeks_elapsed = (execution_date - start_date).days // 7
    
    # Semana base (52) + semanas transcurridas + 1
    next_week = 52 + weeks_elapsed + 1
    
    return next_week

def predict_next_week_all_customers(base_path, model_name='product_priority_model'):
    next_week = calculate_week_number()

    clients_path = os.path.join(base_path, 'data', 'transformed', 'unique_clients.csv')
    clients_df = pd.read_csv(clients_path)
    client_ids = clients_df['customer_id'].values

    predictions = {}
    for client in client_ids:
        predictions[client] = predict(next_week, client, base_path, model_name)

    return predictions


def predictions_to_dataframe(predictions_dict):
    """
    Convierte el diccionario de predicciones a DataFrame para facilitar visualización.
    
    Args:
        predictions_dict: dict retornado por predict_next_week_all_customers()
    
    Returns:
        pd.DataFrame con columnas: customer_id, product_id, week, priority_score
    """
    rows = []
    for customer_id, pred in predictions_dict.items():
        # Ajusta según la estructura de tu predicción
        if isinstance(pred, dict):
            # Si pred tiene estructura {top_products: [...], priority_scores: [...]}
            for product, score in zip(pred.get('top_products', []), 
                                     pred.get('priority_scores', [])):
                rows.append({
                    'customer_id': customer_id,
                    'product_id': product,
                    'week': pred.get('week', 0),
                    'priority_score': score
                })
        else:
            # Ajusta según tu estructura real
            rows.append({
                'customer_id': customer_id,
                'prediction': str(pred)
            })
    
    return pd.DataFrame(rows)