FROM apache/airflow:3.3.1-python3.12

# Airflow itself only needs the ingestion libraries
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" pymongo==4.18.1

# dbt and the ML code get their own virtualenvs: no dependency clashes with Airflow
RUN python -m venv /opt/airflow/venvs/dbt \
 && /opt/airflow/venvs/dbt/bin/pip install --no-cache-dir dbt-bigquery==1.12.1

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/airflow/venvs/ml \
 && /opt/airflow/venvs/ml/bin/pip install --no-cache-dir -r /tmp/requirements.txt
