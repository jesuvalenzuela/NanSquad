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
# Iteramos hatsa llegar al path del proyecto
current_file_path = os.path.abspath(__file__)
backend_dir = os.path.dirname(current_file_path)
app_dir = os.path.dirname(backend_dir)
project_dir = os.path.dirname(app_dir)

AIRFLOW_HOME = os.path.join(project_dir, "airflow")

AIRFLOW_DATA_NEW = os.path.join(AIRFLOW_HOME, "data/new")
AIRFLOW_MODELS = os.path.join(AIRFLOW_HOME, "models")
AIRFLOW_PLUGINS = os.path.join(AIRFLOW_HOME, "plugins")
MODEL_PATH = Path(AIRFLOW_MODELS) / "product_priority_model.joblib"

# Insert the parent directory at the beginning of sys.path
sys.path.insert(0, AIRFLOW_PLUGINS)

try:
    from prediction_utils import predict_next_week_all_customers  # Ajusta el nombre según tu función
except ImportError as e:
    print(f"Warning: No se pudo importar prediction_function: {e}")
    predict_next_week = None

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
        
        # Cargar modelo
        model = joblib.load(MODEL_PATH)

        predictions = predict_next_week_all_customers(base_path=AIRFLOW_HOME, model_name="product_priority_model")
        
        return {
            "status": "success",
            "predictions": predictions,
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
    uvicorn.run(app, host="0.0.0.0", port=8000)