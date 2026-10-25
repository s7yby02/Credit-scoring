"""Production credit scoring pipeline: End-to-end orchestration from raw data to drift monitoring.

Pipeline Flow:
  [ingest_transactions, ingest_applications, ingest_outcomes] (Parallel)
                           │
                           ▼
                      [dbt_build] (19 data tests & marts models)
                           │
                           ▼
                     [parity_test] (Zero Training-Serving Skew check)
                           │
                           ▼
                      [train_model] (XGBoost monotonic training & Champion promotion)
                           │
                           ▼
                      [drift_check] (Population Stability Index / PSI check)
"""
from __future__ import annotations

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param, dag, get_current_context, task

from pipelines.ingestion import load_table


@dag(
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    params={"full_refresh": Param(True, type="boolean")},
    tags=["credit-scoring", "p1", "production"],
)
def credit_scoring_pipeline():
    @task
    def ingest(table: str) -> int:
        context = get_current_context()
        run_date = context.get("logical_date") or pendulum.now("UTC")
        start = context.get("data_interval_start") or run_date.subtract(days=7)
        end = context.get("data_interval_end") or run_date
        return load_table(table, run_date.date().isoformat(), context["params"]["full_refresh"], start, end)

    # 1. Ingestion layer (MongoDB -> BigQuery raw.*)
    ingest_tasks = [
        ingest.override(task_id=f"ingest_{t}")(t)
        for t in ("transactions", "applications", "outcomes")
    ]

    # 2. Transformation & Quality layer (dbt build with 19 tests)
    dbt_build = BashOperator(
        task_id="dbt_build",
        bash_command="cd /opt/project/dbt/credit_scoring && /opt/airflow/venvs/dbt/bin/dbt build --profiles-dir .",
    )

    # 3. Guardrail: Parity test between BigQuery SQL and Python
    parity_test = BashOperator(
        task_id="parity_test",
        bash_command="cd /opt/project && /opt/airflow/venvs/ml/bin/python -m pytest -q tests/test_parity.py",
    )

    # 4. Model Training, Evaluation and Champion Promotion in MLflow
    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/project && /opt/airflow/venvs/ml/bin/python training/train.py",
    )

    # 5. Production Drift Monitoring (PSI check)
    drift_check = BashOperator(
        task_id="drift_check",
        bash_command="cd /opt/project && /opt/airflow/venvs/ml/bin/python monitoring/drift.py",
    )

    # Topological execution graph
    ingest_tasks >> dbt_build >> parity_test >> train_model >> drift_check


credit_scoring_pipeline()
