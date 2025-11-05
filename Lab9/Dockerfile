FROM python:3.10-slim

ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True

WORKDIR /opt/airflow

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
    
RUN pip install --no-cache-dir \
    "apache-airflow==2.10.3" \
    pandas \
    scikit-learn \
    joblib \
    gradio

RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/plugins

EXPOSE 8080 7860

RUN echo '#!/bin/bash\n\
airflow db init\n\
airflow users create \\\n\
    --username admin \\\n\
    --firstname Admin \\\n\
    --lastname User \\\n\
    --role Admin \\\n\
    --email admin@example.com \\\n\
    --password admin 2>/dev/null || true\n\
airflow webserver --port 8080 &\n\
airflow scheduler\n\
' > /opt/airflow/start.sh && chmod +x /opt/airflow/start.sh

CMD ["/opt/airflow/start.sh"]
