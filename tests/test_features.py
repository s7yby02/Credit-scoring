"""Unit tests for the Python feature extraction pipeline (features/compute.py)."""
from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from features.compute import (
    categorize,
    compute_features,
    load_rules,
    safe_divide,
)


@pytest.fixture(scope="module")
def rules():
    return load_rules()


def test_safe_divide():
    assert safe_divide(10.0, 2.0) == 5.0
    assert math.isnan(safe_divide(10.0, 0.0))
    assert math.isnan(safe_divide(math.nan, 5.0))
    assert math.isnan(safe_divide(5.0, math.nan))


def test_categorize(rules):
    # Credits
    assert categorize("VIR RECU SALAIRE OCTOBRE", 2500.0, rules) == "salary"
    assert categorize("VIREMENT CAF PRESTATIONS", 300.0, rules) == "benefits"
    assert categorize("VIR CLIENT MALT FACTURE 42", 1200.0, rules) == "freelance_income"

    # Debits
    assert categorize("PRLV SEPA LOYER FONCIA", -800.0, rules) == "rent"
    assert categorize("BETCLIC PARIS SPORTIFS", -50.0, rules) == "gambling"
    assert categorize("COMMISSION FRAIS REJET PRELEVEMENT", -20.0, rules) == "rejected_debit"

    # Unknown description fallback
    assert categorize("ACHAT DIVERS QUINCAILLERIE XYZ", -15.0, rules) == "other"

    # Sign sensitivity: a debit labeled "SALAIRE" should not be categorized as salary
    assert categorize("VIR EMIS SALAIRE ASSISTANTE", -500.0, rules) != "salary"


def test_point_in_time_window(rules):
    """Transactions on or after application_date, or older than 180 days, must be ignored."""
    app_date = date(2025, 4, 1)

    tx_data = pd.DataFrame([
        # Within window (180 days before 2025-04-01 -> approx 2024-10-03 to 2025-03-31)
        {"transaction_id": "tx1", "booking_date": "2025-03-15", "amount": 2000.0, "description": "SALAIRE", "balance_after": 2000.0},
        {"transaction_id": "tx2", "booking_date": "2025-02-15", "amount": 2000.0, "description": "SALAIRE", "balance_after": 1500.0},
        # Future transaction: strictly on or after application date (leakage!)
        {"transaction_id": "tx_future", "booking_date": "2025-04-01", "amount": 5000.0, "description": "SALAIRE", "balance_after": 6500.0},
        # Expired transaction: older than 180 days (e.g. 200 days before)
        {"transaction_id": "tx_old", "booking_date": "2024-08-01", "amount": 3000.0, "description": "SALAIRE", "balance_after": 3000.0},
    ])

    feats = compute_features(tx_data, app_date, rules)

    # Only tx1 and tx2 should be kept
    assert feats["n_transactions"] == 2
    # Income averaged over 6 months: (2000 + 2000) / 6 = 666.67
    assert pytest.approx(feats["avg_monthly_income"], abs=1e-2) == 4000.0 / 6.0


def test_empty_transactions(rules):
    """An applicant with zero transactions in the 180-day window."""
    app_date = date(2025, 4, 1)
    empty_tx = pd.DataFrame(columns=["transaction_id", "booking_date", "amount", "description", "balance_after"])

    feats = compute_features(empty_tx, app_date, rules)

    assert feats["n_transactions"] == 0
    assert feats["avg_monthly_income"] == 0.0
    assert feats["n_rejected_debits"] == 0
    assert feats["negative_balance_days"] == 0
    assert math.isnan(feats["rent_to_income"])
    assert math.isnan(feats["gambling_share"])
