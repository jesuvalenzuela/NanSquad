"""
API REST para predicción de potabilidad del agua usando XGBoost.
Ejecutar con: python main.py
"""

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# Configuración
app = FastAPI(title="API de Predicción de Potabilidad del Agua")
MODEL_PATH = "models/best_xgboost_model.pkl"

# Cargar modelo
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Modelo cargado desde {MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar modelo: {e}")
    model = None


# Modelos de datos
class WaterQualityInput(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


class PredictionResponse(BaseModel):
    potabilidad: int


# Endpoints
@app.get("/")
def home():
    """
    Descripción del modelo y la API.
    """
    return {
        "modelo": "XGBoost optimizado con Optuna",
        "problema": "Clasificación binaria: determinar si el agua es potable",
        "entrada": "JSON con 9 parámetros físico-químicos: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity",
        "salida": "JSON con campo 'potabilidad' (0: No potable, 1: Potable)",
        "uso": "POST /potabilidad/ - Ver documentación en /docs"
    }


@app.post("/potabilidad/", response_model=PredictionResponse)
def predecir_potabilidad(data: WaterQualityInput):
    """
    Predice si el agua es potable.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta optimize.py primero."
        )
    
    try:
        # Convertir a array numpy
        features = np.array([[
            data.ph,
            data.Hardness,
            data.Solids,
            data.Chloramines,
            data.Sulfate,
            data.Conductivity,
            data.Organic_carbon,
            data.Trihalomethanes,
            data.Turbidity
        ]])
        
        # Predecir
        prediction = model.predict(features)
        
        return PredictionResponse(potabilidad=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Ejecutar servidor
if __name__ == "__main__":
    print("\n" + "="*50)
    print("API corriendo en http://localhost:8000")
    print("Documentación: http://localhost:8000/docs")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)