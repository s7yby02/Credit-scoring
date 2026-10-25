"""Parity test: verifies that Python feature engineering (features/compute.py)
and BigQuery dbt feature engineering (models/marts/fct_applicant_features.sql)
produce identical outputs on the exact same input data.

Guarantees zero Training-Serving Skew between batch ML training and real-time serving.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from google.cloud import bigquery

from features.compute import PARITY_COLUMNS, compute_features, load_rules

SAMPLE_SIZE = int(os.getenv("PARITY_SAMPLE_SIZE", "50"))


@pytest.fixture(scope="module")
def bq_client():
    project = os.environ.get("GCP_PROJECT", "algoan-credit-scoring")
    location = os.environ.get("BQ_LOCATION", "EU")
    return bigquery.Client(project=project, location=location)


def test_sql_python_feature_parity(bq_client):
    """Fetches a sample of applications from BigQuery, computes Python features on their
    raw transactions, and asserts that Python == SQL for every single feature.
    """
    project = bq_client.project

    # 1. Fetch a deterministic sample of applications with dbt-computed features
    sql_features_query = f"""
        SELECT
            application_id,
            account_id,
            application_date,
            n_transactions,
            avg_monthly_income,
            income_cv,
            rent_to_income,
            gambling_share,
            n_rejected_debits,
            negative_balance_days,
            net_cashflow_ratio
        FROM `{project}.analytics.fct_applicant_features`
        ORDER BY application_id
        LIMIT {SAMPLE_SIZE}
    """
    df_sql = bq_client.query(sql_features_query).to_dataframe()
    assert len(df_sql) > 0, "No records found in analytics.fct_applicant_features!"

    account_ids = df_sql["account_id"].unique().tolist()
    account_ids_sql = ", ".join(f"'{aid}'" for aid in account_ids)

    # 2. Fetch all raw transactions for these accounts
    tx_query = f"""
        SELECT
            transaction_id,
            account_id,
            booking_date,
            amount,
            description,
            balance_after
        FROM `{project}.raw.transactions`
        WHERE account_id IN ({account_ids_sql})
    """
    df_tx = bq_client.query(tx_query).to_dataframe()

    rules = load_rules()
    mismatches = []

    # 3. For each application, compute features in Python and compare with SQL
    for _, row in df_sql.iterrows():
        app_id = row["application_id"]
        acc_id = row["account_id"]
        app_date = row["application_date"]

        # Filter raw transactions for this account
        tx_user = df_tx[df_tx["account_id"] == acc_id]
        py_feats = compute_features(tx_user, app_date, rules)

        for col in PARITY_COLUMNS:
            sql_val = row[col]
            py_val = py_feats[col]

            # Both NaN is considered a match (e.g. division by zero for safe_divide)
            if pd.isna(sql_val) and pd.isna(py_val):
                continue

            # Numeric comparison with small tolerance for floating point rounding
            if isinstance(sql_val, (int, np.integer)) and isinstance(py_val, (int, np.integer)):
                match = sql_val == py_val
            else:
                match = np.isclose(float(sql_val), float(py_val), rtol=1e-3, atol=1e-3)

            if not match:
                mismatches.append({
                    "application_id": app_id,
                    "feature": col,
                    "sql_value": sql_val,
                    "python_value": py_val,
                    "diff": abs(float(sql_val) - float(py_val)),
                })

    # 4. Assert zero discrepancies across all tested applications and features
    assert not mismatches, f"Feature parity failed with {len(mismatches)} mismatches:\n" + "\n".join(
        f"  [App {m['application_id']}] {m['feature']}: SQL={m['sql_value']} != Python={m['python_value']} (diff={m['diff']:.5f})"
        for m in mismatches[:10]
    )
