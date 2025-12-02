"""
Funciones de predicción puras (sin dependencias de Airflow).
Pueden usarse tanto en el DAG como en la API FastAPI.
"""

import os
from datetime import datetime, timedelta
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


def predict(week, customer_id, base_path, model_name='product_priority_model',
            model=None, preprocessor=None):
    """Entrega prediccion para la semana siguiente para el cliente especificado."""

    # transformar
    input_data = prepare_data_for_prediction(week, customer_id, base_path)

    # preprocesar (cargar solo si no se pasó como parámetro)
    if preprocessor is None:
        preprocessor_path = os.path.join(base_path, 'models', f'preprocessor.joblib')
        preprocessor = joblib.load(preprocessor_path)
    data_clean = preprocessor.transform(input_data)

    # predecir (cargar solo si no se pasó como parámetro)
    if model is None:
        model_path = os.path.join(base_path, 'models', f'{model_name}.joblib')
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

def calculate_next_week_from_historical_data(base_path):
    """
    Calcula el número de semana del día siguiente al último dato disponible.
    Usa numeración consecutiva desde 01-ene-2024 (semana 1 = 01-ene a 07-ene).

    Args:
        base_path: Ruta base donde se encuentra la carpeta 'data'

    Returns:
        tuple: (week_number, next_day) - Número de semana a predecir y fecha del día siguiente
    """
    # Leer datos históricos
    historical_path = os.path.join(base_path, 'data', 'historical_raw', 'transacciones.parquet')
    df = pd.read_parquet(historical_path)

    # Obtener fecha máxima
    last_date = pd.to_datetime(df['purchase_date']).max()

    # Calcular el día siguiente al último dato
    next_day = last_date + timedelta(days=1)

    # Usar numeración consecutiva desde 01-ene-2024
    # Semana 1 = días 0-6 (01-ene a 07-ene)
    # Semana 2 = días 7-13, etc.
    start_date = datetime(2024, 1, 1)
    days_elapsed = (next_day - start_date).days
    week_number = days_elapsed // 7 + 1

    print(f"Última fecha en datos históricos: {last_date.strftime('%Y-%m-%d')}")
    print(f"Día siguiente (inicio predicción): {next_day.strftime('%Y-%m-%d')}")
    print(f"Semana calculada para predicciones: {week_number}")

    return week_number, next_day

def predict_next_week_all_customers(base_path, model_name='product_priority_model', week_number=None):
    # Si no se proporciona week_number, calcular usando fecha actual (comportamiento por defecto)
    if week_number is None:
        next_week = calculate_week_number()
    else:
        next_week = week_number

    # Cargar modelo y preprocessor UNA SOLA VEZ (no en cada iteración)
    model_path = os.path.join(base_path, 'models', f'{model_name}.joblib')
    preprocessor_path = os.path.join(base_path, 'models', 'preprocessor.joblib')

    print(f"Cargando modelo desde {model_path}...")
    model = joblib.load(model_path)
    print(f"Cargando preprocessor desde {preprocessor_path}...")
    preprocessor = joblib.load(preprocessor_path)

    clients_path = os.path.join(base_path, 'data', 'transformed', 'unique_clients.csv')
    clients_df = pd.read_csv(clients_path)
    client_ids = clients_df['customer_id'].values

    print(f"Generando predicciones para {len(client_ids)} clientes...")

    predictions = {}
    for i, client in enumerate(client_ids):
        if i % 100 == 0:  # Progress logging cada 100 clientes
            print(f"Procesando cliente {i}/{len(client_ids)}...")
        predictions[client] = predict(next_week, client, base_path, model_name,
                                     model=model, preprocessor=preprocessor)

    print(f"Predicciones completadas para {len(predictions)} clientes.")
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


def generate_and_save_predictions(base_path, model_name='product_priority_model'):
    """
    Genera predicciones para la semana del día siguiente al último dato disponible,
    filtra 'Very High' y 'High' y guarda en CSV.

    Args:
        base_path: Ruta base de AIRFLOW_HOME
        model_name: Nombre del modelo a usar para predicciones

    Returns:
        str: Ruta del archivo CSV generado
    """
    print("=" * 70)
    print("GENERANDO PREDICCIONES PARA PRÓXIMA SEMANA")
    print("=" * 70)

    # 1. Calcular número de semana y fecha del día siguiente
    week_to_predict, next_day = calculate_next_week_from_historical_data(base_path)

    # 2. Formatear fecha para el nombre del archivo (día siguiente al último dato)
    date_str = next_day.strftime('%d-%m-%y')  # Formato: 30-12-24 o 01-01-25

    print(f"Generando predicciones para semana {week_to_predict} (desde: {next_day.strftime('%d-%m-%Y')})")

    # 3. Generar predicciones pasando el número de semana
    predictions = predict_next_week_all_customers(base_path, model_name, week_number=week_to_predict)

    # 4. Filtrar 'Very High' y 'High'
    rows = []
    for customer_id, products_dict in predictions.items():
        for product_id, priority in products_dict.items():
            if priority in ['Very High', 'High']:
                rows.append([customer_id, product_id])

    # 5. Crear carpeta predictions si no existe
    predictions_path = os.path.join(base_path, 'data', 'predictions')
    os.makedirs(predictions_path, exist_ok=True)

    # 6. Guardar CSV sin headers
    filename = f'predictions_{date_str}.csv'
    filepath = os.path.join(predictions_path, filename)

    df = pd.DataFrame(rows, columns=['customer_id', 'product_id'])
    df.to_csv(filepath, index=False, header=False)

    print("=" * 70)
    print(f"✓ Predicciones guardadas en: {filepath}")
    print(f"✓ Total combinaciones 'Very High' + 'High': {len(rows)}")
    print("=" * 70)

    return filepath