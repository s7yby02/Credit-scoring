"""Feature computation used by real-time serving and by the parity test.

Mirrors dbt/credit_scoring/models/marts/fct_applicant_features.sql.
If you change one, change the other: tests/test_parity.py fails when they drift apart.
"""
from __future__ import annotations

import math
import re
from datetime import date
from pathlib import Path

import pandas as pd

# Same CSV as the dbt seed: one source of truth for categorization rules
RULES_PATH = Path(__file__).resolve().parents[1] / "dbt" / "credit_scoring" / "seeds" / "category_rules.csv"
INCOME_CATEGORIES = {"salary", "freelance_income", "benefits"}
LOOKBACK_DAYS = 180
N_MONTHS = 6

FEATURE_COLUMNS = [
    "avg_monthly_income",
    "income_cv",
    "rent_to_income",
    "gambling_share",
    "n_rejected_debits",
    "negative_balance_days",
    "net_cashflow_ratio",
]
PARITY_COLUMNS = ["n_transactions", *FEATURE_COLUMNS]

Rule = tuple[re.Pattern, str, str]


def load_rules(path: Path = RULES_PATH) -> list[Rule]:
    rules = pd.read_csv(path).sort_values("priority")
    return [(re.compile(r.pattern), r.direction, r.category) for r in rules.itertuples(index=False)]


def categorize(description: str, amount: float, rules: list[Rule]) -> str:
    text = description.upper()
    for pattern, direction, category in rules:
        sign_ok = direction == "any" or (direction == "credit" and amount > 0) or (direction == "debit" and amount < 0)
        if sign_ok and pattern.search(text):
            return category
    return "other"


def safe_divide(a: float, b: float) -> float:
    """Behaves like BigQuery SAFE_DIVIDE: NaN (NULL) instead of an error."""
    if b == 0 or pd.isna(a) or pd.isna(b):
        return math.nan
    return a / b


def compute_features(transactions: pd.DataFrame, application_date: date | str, rules: list[Rule] | None = None) -> dict:
    """transactions needs: transaction_id, booking_date, amount, description, balance_after."""
    rules = load_rules() if rules is None else rules
    app_date = pd.Timestamp(application_date).normalize()

    tx = transactions.drop_duplicates("transaction_id").copy()
    tx["booking_date"] = pd.to_datetime(tx["booking_date"])
    tx["days_before"] = (app_date - tx["booking_date"]).dt.days
    # Point-in-time window: the 180 days strictly before the application date
    tx = tx[(tx["days_before"] >= 1) & (tx["days_before"] <= LOOKBACK_DAYS)].copy()
    tx["category"] = [categorize(d, a, rules) for d, a in zip(tx["description"], tx["amount"])]
    tx["month_bucket"] = (tx["days_before"] - 1) // 30  # 0 = most recent 30 days ... 5 = oldest

    income = tx[tx["category"].isin(INCOME_CATEGORIES)]
    monthly_income = income.groupby("month_bucket")["amount"].sum().reindex(range(N_MONTHS), fill_value=0.0)
    avg_monthly_income = float(monthly_income.mean())

    total_spend = float((-tx.loc[tx["amount"] < 0, "amount"]).sum())
    avg_monthly_rent = float((-tx.loc[tx["category"] == "rent", "amount"]).sum()) / N_MONTHS
    gambling_spend = float((-tx.loc[tx["category"] == "gambling", "amount"]).sum())
    total_inflows = float(tx.loc[tx["amount"] > 0, "amount"].sum())

    return {
        "n_transactions": int(len(tx)),
        "avg_monthly_income": avg_monthly_income,
        "income_cv": safe_divide(float(monthly_income.std(ddof=0)), avg_monthly_income),  # STDDEV_POP
        "rent_to_income": safe_divide(avg_monthly_rent, avg_monthly_income),
        "gambling_share": safe_divide(gambling_spend, total_spend),
        "n_rejected_debits": int((tx["category"] == "rejected_debit").sum()),
        "negative_balance_days": int(tx.loc[tx["balance_after"] < 0, "booking_date"].nunique()),
        "net_cashflow_ratio": safe_divide(float(tx["amount"].sum()), total_inflows),
    }
