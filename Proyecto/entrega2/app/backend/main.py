"""
Backend FastAPI para sistema de predicción de productos prioritarios.
Endpoints:
    - POST /upload-data: Recibe archivo de transacciones nuevas
    - POST /predict: Genera predicciones para próxima semana
"""

import os
from pathlib import Path

from datetime import datetime, timedelta

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Product Priority Prediction API")

# Configurar CORS para permitir requests desde Gradio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths relativos al contenedor

import sys

####
# Detectar si estamos en Docker o desarrollo local
if os.path.exists("/app/airflow"):  # Estamos en Docker
    AIRFLOW_HOME = "/app/airflow"
else:  # Estamos en desarrollo local
    ROOT_DIR = Path(__file__).resolve().parents[1]
    project_dir = os.path.dirname(ROOT_DIR)
    AIRFLOW_HOME = os.path.join(project_dir, "airflow")

print(f"AIRFLOW_HOME: {AIRFLOW_HOME}")
print(f"Existe AIRFLOW_HOME: {os.path.exists(AIRFLOW_HOME)}")

AIRFLOW_DATA_NEW = Path(AIRFLOW_HOME) / "data" / "new"
AIRFLOW_DATA_TRANSFORMED = Path(AIRFLOW_HOME) / "data" / "transformed"
AIRFLOW_MODELS = Path(AIRFLOW_HOME) / "models"
AIRFLOW_PLUGINS = Path(AIRFLOW_HOME) / "plugins"
MODEL_PATH = AIRFLOW_MODELS / "product_priority_model.joblib"
PREPROCESSOR_PATH = AIRFLOW_MODELS / "preprocessor.joblib"

# Insert the parent directory at the beginning of sys.path
sys.path.insert(0, str(AIRFLOW_PLUGINS))

try:
    from prediction_utils import predict_next_week_all_customers
except ImportError as e:
    print(f"Warning: No se pudo importar prediction_utils: {e}")
    predict_next_week_all_customers = None

@app.get("/")
def read_root():
    """Health check."""
    return {"status": "ok", "service": "Product Priority API"}


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """
    Recibe archivo parquet con transacciones nuevas y lo guarda en airflow/data/new.
    
    Columnas esperadas: customer_id, product_id, order_id, purchase_date, items
    """
    try:
        # Validar extensión
        if not file.filename.endswith(".parquet"):
            raise HTTPException(
                status_code=400,
                detail="Solo se aceptan archivos .parquet"
            )
        
        # Crear directorio si no existe
        AIRFLOW_DATA_NEW.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo
        file_path = AIRFLOW_DATA_NEW / "transacciones.parquet"
        content = await file.read()
        
        # Validar que sea parquet válido
        try:
            df = pd.read_parquet(pd.io.common.BytesIO(content))
            
            # Validar columnas requeridas
            required_cols = ["customer_id", "product_id", "order_id", 
                           "purchase_date", "items"]
            missing_cols = set(required_cols) - set(df.columns)
            
            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Columnas faltantes: {missing_cols}"
                )
            
            # Guardar archivo
            with open(file_path, "wb") as f:
                f.write(content)
                
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error al procesar parquet: {str(e)}"
            )
        
        return {
            "status": "success",
            "message": f"Archivo guardado en {file_path}",
            "rows": len(df),
            "note": "Ejecuta tu DAG de Airflow para reentrenar el modelo"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def generate_predictions():
    """
    Carga el modelo entrenado y genera predicciones para todos los clientes
    en la próxima semana.

    Retorna DataFrame con predicciones en formato JSON.
    """
    try:
        # Verificar que existe el modelo
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Modelo no encontrado en {MODEL_PATH}. "
                       "Ejecuta el DAG primero."
            )

        # Verificar que existen los archivos de datos transformados
        if not AIRFLOW_DATA_TRANSFORMED.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Datos transformados no encontrados. Ejecuta el DAG primero."
            )

        # Verificar función de predicción
        if predict_next_week_all_customers is None:
            raise HTTPException(
                status_code=500,
                detail="Función de predicción no disponible. Verifica instalación de dependencias."
            )

        # Generar predicciones usando la función del plugin de Airflow
        predictions_dict = predict_next_week_all_customers(
            base_path=AIRFLOW_HOME,
            model_name="product_priority_model"
        )

        # Convertir el diccionario a formato lista para el frontend
        # predictions_dict tiene estructura: {customer_id: {product_id: priority_prediction}}
        predictions_list = []
        for customer_id, products_pred in predictions_dict.items():
            # products_pred es un dict {product_id: priority_prediction}
            for product_id, priority in products_pred.items():
                predictions_list.append({
                    "customer_id": str(customer_id),
                    "product_id": str(product_id),
                    "priority": int(priority) if isinstance(priority, (int, float)) else priority
                })

        return {
            "status": "success",
            "predictions": predictions_list,
            "total_predictions": len(predictions_list),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar predicciones: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)