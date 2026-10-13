"""Step 3 version: ingestion only (MongoDB -> BigQuery raw). Step 9 adds the rest."""
from __future__ import annotations

import pendulum
from airflow.sdk import Param, dag, get_current_context, task

from pipelines.ingestion import load_table


@dag(
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    params={"full_refresh": Param(True, type="boolean")},
    tags=["credit-scoring"],
)
def credit_scoring_pipeline():
    @task
    def ingest(table: str) -> int:
        context = get_current_context()
        run_date = context.get("logical_date") or pendulum.now("UTC")  # manual runs may have no logical date
        start = context.get("data_interval_start") or run_date.subtract(days=7)
        end = context.get("data_interval_end") or run_date
        return load_table(table, run_date.date().isoformat(), context["params"]["full_refresh"], start, end)

    for table in ("transactions", "applications", "outcomes"):
        ingest.override(task_id=f"ingest_{table}")(table)


credit_scoring_pipeline()
