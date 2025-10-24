# Imagen base ligera de Python
FROM python:3.10-slim

# Evitar archivos pyc y mejorar logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar compiladores (necesarios para XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de archivos del proyecto
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Crear volumen para persistencia de modelos
VOLUME ["/app/models"]

# Crear usuario sin privilegios
RUN useradd -m appuser
USER appuser

# Comando para iniciar la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
