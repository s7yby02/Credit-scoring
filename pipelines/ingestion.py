"""Extract documents from MongoDB and load them into BigQuery raw tables with load jobs."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from google.cloud import bigquery
from pymongo import MongoClient

F = bigquery.SchemaField
LOAD_COLUMNS = [F("ingestion_date", "DATE", mode="REQUIRED"), F("_loaded_at", "TIMESTAMP", mode="REQUIRED")]
SCHEMAS = {
    "transactions": [
        F("transaction_id", "STRING", mode="REQUIRED"),
        F("account_id", "STRING", mode="REQUIRED"),
        F("user_id", "STRING", mode="REQUIRED"),
        F("booked_at", "TIMESTAMP"),
        F("booking_date", "DATE"),
        F("amount", "FLOAT64"),  # in production, prefer NUMERIC for money
        F("currency", "STRING"),
        F("description", "STRING"),
        F("balance_after", "FLOAT64"),
        F("fetched_at", "TIMESTAMP"),
        *LOAD_COLUMNS,
    ],
    "applications": [
        F("application_id", "STRING", mode="REQUIRED"),
        F("user_id", "STRING"),
        F("account_id", "STRING"),
        F("application_date", "DATE"),
        F("requested_amount", "INT64"),
        F("partner_id", "STRING"),
        *LOAD_COLUMNS,
    ],
    "outcomes": [
        F("application_id", "STRING", mode="REQUIRED"),
        F("defaulted", "BOOL"),
        F("outcome_date", "DATE"),
        *LOAD_COLUMNS,
    ],
}
CLUSTERING = {"transactions": ["account_id"], "applications": ["account_id"], "outcomes": None}


def iter_rows(table: str, start: datetime | None, end: datetime | None, ingestion_date: str, loaded_at: str) -> Iterator[dict]:
    db = MongoClient(os.getenv("MONGO_URI", "mongodb://mongo:27017"))[os.getenv("MONGO_DB", "credit_scoring")]
    load_info = {"ingestion_date": ingestion_date, "_loaded_at": loaded_at}
    if table == "transactions":
        query = {} if start is None else {"fetched_at": {"$gte": start, "$lt": end}}
        for account in db.accounts.find(query, batch_size=100):
            for tx in account["transactions"]:  # flatten the nested array: one row per transaction
                yield {
                    **tx,
                    "account_id": account["_id"],
                    "user_id": account["user_id"],
                    "currency": account["currency"],
                    "fetched_at": account["fetched_at"].isoformat(),
                    **load_info,
                }
    elif table == "applications":
        for app in db.applications.find():
            yield {
                "application_id": app["_id"],
                "user_id": app["user_id"],
                "account_id": app["account_id"],
                "application_date": app["application_date"].date().isoformat(),
                "requested_amount": app["requested_amount"],
                "partner_id": app["partner_id"],
                **load_info,
            }
    else:
        for outcome in db.outcomes.find():
            yield {
                "application_id": outcome["application_id"],
                "defaulted": outcome["defaulted"],
                "outcome_date": outcome["outcome_date"].date().isoformat(),
                **load_info,
            }


def load_table(table: str, ingestion_date: str, full_refresh: bool, start=None, end=None) -> int:
    project, location = os.environ["GCP_PROJECT"], os.getenv("BQ_LOCATION", "EU")
    client = bigquery.Client(project=project, location=location)
    dataset = bigquery.Dataset(f"{project}.raw")
    dataset.location = location
    client.create_dataset(dataset, exists_ok=True)

    table_id = f"{project}.raw.{table}"
    partitioning = bigquery.TimePartitioning(field="ingestion_date")
    target = bigquery.Table(table_id, schema=SCHEMAS[table])
    target.time_partitioning = partitioning
    target.clustering_fields = CLUSTERING[table]
    client.create_table(target, exists_ok=True)

    incremental = table == "transactions" and not full_refresh
    loaded_at = datetime.now(timezone.utc).isoformat()
    rows = iter_rows(table, start if incremental else None, end if incremental else None, ingestion_date, loaded_at)

    path = Path(tempfile.mkdtemp()) / f"{table}.jsonl"  # stream to disk: never hold all rows in memory
    n_rows = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n_rows += 1
    if n_rows == 0:
        print(f"{table}: nothing new to load")
        return 0

    # Incremental runs overwrite only their own partition (table$YYYYMMDD)
    destination = f"{table_id}${ingestion_date.replace('-', '')}" if incremental else table_id
    job_config = bigquery.LoadJobConfig(
        schema=SCHEMAS[table],
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        time_partitioning=partitioning,
        clustering_fields=CLUSTERING[table],
    )
    with open(path, "rb") as f:
        job = client.load_table_from_file(f, destination, job_config=job_config)
    job.result()
    if job.output_rows != n_rows:
        raise ValueError(f"{table}: extracted {n_rows} rows but BigQuery loaded {job.output_rows}")
    print(f"{table}: {n_rows} rows loaded into {destination}")
    return n_rows
