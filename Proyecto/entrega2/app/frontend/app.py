"""
Frontend Gradio para sistema de predicción de productos prioritarios.
Permite:
    - Subir datos nuevos para reentrenar modelo
    - Generar predicciones para próxima semana
"""

import os
import requests
import pandas as pd
import gradio as gr


# URL del backend (desde variable de entorno o default)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def upload_new_data(file):
    """
    Sube archivo parquet con transacciones nuevas al backend.
    """
    if file is None:
        return "Error: Debes seleccionar un archivo"
    
    try:
        # Leer archivo
        with open(file.name, "rb") as f:
            files = {"file": (file.name, f, "application/octet-stream")}
            response = requests.post(
                f"{BACKEND_URL}/upload-data",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            return f"""✅ **Datos subidos exitosamente**
            
- Registros procesados: {data.get('rows', 'N/A')}
- Ubicación: {data.get('message', '')}

📝 **Próximo paso:** Ejecuta el DAG de Airflow para reentrenar el modelo con estos datos.
"""
        else:
            return f"Error: {response.json().get('detail', 'Error desconocido')}"
            
    except Exception as e:
        return f"Error al conectar con el backend: {str(e)}"


def generate_predictions():
    """
    Genera predicciones para todos los clientes en la próxima semana.
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get("predictions", [])
            
            if not predictions:
                return None, "⚠️ No se generaron predicciones"
            
            # Convertir a DataFrame para visualización
            df = pd.DataFrame(predictions)
            
            message = f"""✅ **Predicciones generadas exitosamente**
            
- Total de predicciones: {data.get('total_predictions', len(predictions))}
- Timestamp: {data.get('timestamp', 'N/A')}

Resultados en la tabla inferior.
"""
            return df, message
            
        else:
            error_detail = response.json().get('detail', 'Error desconocido')
            return None, f"Error: {error_detail}"
            
    except Exception as e:
        return None, f"Error al conectar con el backend: {str(e)}"


# ============================================================================
# INTERFAZ GRADIO
# ============================================================================

with gr.Blocks(title="Predicción de Productos Prioritarios") as app:
    
    gr.Markdown("""
    # 🎯 Sistema de Predicción de Productos Prioritarios
    
    Esta aplicación permite interactuar con el modelo de predicción de productos 
    prioritarios para la venta de bebestibles por cliente y semana.
    """)
    
    with gr.Tabs():
        
        # ========== PESTAÑA 1: SUBIR DATOS ==========
        with gr.Tab("📤 Subir Datos Nuevos"):
            gr.Markdown("""
            ### Instrucciones
            
            1. **Prepara tu archivo**: Debe ser formato `.parquet` con las columnas:
               - `customer_id`: ID del cliente
               - `product_id`: ID del producto
               - `order_id`: ID de la orden
               - `purchase_date`: Fecha de compra
               - `items`: Cantidad de items
            
            2. **Sube el archivo**: Usa el botón inferior para seleccionar tu archivo
            
            3. **Reentrena el modelo**: Una vez subido, ejecuta manualmente el DAG de 
               Airflow para que el modelo incorpore estos nuevos datos
            """)
            
            with gr.Row():
                file_input = gr.File(
                    label="Selecciona archivo transacciones.parquet",
                    file_types=[".parquet"]
                )
            
            upload_button = gr.Button("Subir Datos", variant="primary")
            upload_output = gr.Markdown()
            
            upload_button.click(
                fn=upload_new_data,
                inputs=file_input,
                outputs=upload_output
            )
        
        # ========== PESTAÑA 2: PREDICCIONES ==========
        with gr.Tab("Generar Predicciones"):
            gr.Markdown("""
            ### Instrucciones
            
            1. **Asegúrate** de que el modelo esté entrenado (el DAG debe haberse 
               ejecutado al menos una vez)
            
            2. **Genera predicciones**: Haz clic en el botón para obtener predicciones 
               de todos los clientes para la próxima semana
            
            3. **Descarga resultados**: Puedes exportar la tabla a CSV usando el botón 
               de descarga en la esquina superior derecha de la tabla
            """)
            
            predict_button = gr.Button("Generar Predicciones", variant="primary")
            predict_output = gr.Markdown()
            predictions_table = gr.DataFrame(
                label="Predicciones para Próxima Semana",
                wrap=True
            )
            
            predict_button.click(
                fn=generate_predictions,
                outputs=[predictions_table, predict_output]
            )
    
    gr.Markdown("""
    ---
    ### Notas Adicionales
    
    - **Modelo**: El modelo se carga desde `/airflow/models/product_priority_model.joblib`
    - **Datos nuevos**: Se guardan en `/airflow/data/new/transacciones.parquet`
    - **Reentrenamiento**: Se debe ejecutar el DAG manualmente después de subir datos nuevos
    """)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
