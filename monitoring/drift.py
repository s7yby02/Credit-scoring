"""Production drift monitoring: computes PSI between the training baseline and recent applications.

Compares recent production features and model score distribution against the reference profile
stored in MLflow with the @champion model.

Exit code 0 if all PSI < 0.25.
Raises DriftDetected (exit code 1) if any feature or model score drifts >= 0.25.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from features.compute import FEATURE_COLUMNS, compute_features, load_rules
from monitoring.psi import ALERT, WARN, proportions, psi

MODEL_NAME = os.getenv("MODEL_NAME", "credit-scoring")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
PROD_DIR = Path(os.getenv("PROD_DIR", "data/prod"))


class DriftDetected(RuntimeError):
    """Raised when one or more features exceed the critical drift threshold (PSI >= 0.25)."""
    pass


def load_champion_reference_profile() -> tuple[dict, str, any]:
    """Downloads reference_profile.json and loads the model from MLflow for @champion."""
    client = MlflowClient()
    try:
        version_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except MlflowException as e:
        raise RuntimeError(f"Could not find model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}' in MLflow: {e}")

    run_id = version_info.run_id
    version = version_info.version

    # Download reference profile JSON stored during train.py
    local_path = client.download_artifacts(run_id, "monitoring/reference_profile.json")
    with open(local_path) as f:
        reference_profile = json.load(f)

    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{version}")
    return reference_profile, version, model


def compute_prod_features_and_scores(prod_dir: Path, model: any) -> tuple[pd.DataFrame, np.ndarray]:
    """Computes features and model scores for all accounts in the production folder."""
    apps_file = prod_dir / "applications.jsonl"
    accounts_file = prod_dir / "accounts.jsonl"

    if not apps_file.exists() or not accounts_file.exists():
        raise FileNotFoundError(f"Missing production batch files in '{prod_dir}'. Run 'make prod' first.")

    with open(apps_file) as f:
        apps = {a["_id"]: a for a in map(json.loads, f)}

    rules = load_rules()
    rows = []

    with open(accounts_file) as f:
        for line in f:
            account = json.loads(line)
            # Link via account_id or matching application
            acc_id = account["_id"]
            # Find application belonging to this account
            app = next((a for a in apps.values() if a["account_id"] == acc_id), None)
            if not app:
                continue

            tx = pd.DataFrame(account["transactions"])
            feats = compute_features(tx, app["application_date"], rules)
            rows.append(feats)

    df_feats = pd.DataFrame(rows)
    X = df_feats[FEATURE_COLUMNS].astype(float)
    scores = model.predict_proba(X)[:, 1]
    return df_feats, scores


def monitor_drift(prod_dir: Path = PROD_DIR) -> dict[str, float]:
    print(f"[*] Loading Champion model and reference profile from MLflow ({MODEL_NAME}@{MODEL_ALIAS})...")
    ref_profile, version, model = load_champion_reference_profile()
    ref_features = ref_profile["features"]

    print(f"[*] Processing production batch from {prod_dir}...")
    df_feats, prod_scores = compute_prod_features_and_scores(prod_dir, model)
    n_prod = len(df_feats)
    print(f"[*] Evaluated {n_prod} production applications against Champion v{version}.\n")

    psi_scores = {}
    drifted_features = []

    # Check each feature
    for col in FEATURE_COLUMNS:
        ref_spec = ref_features.get(col)
        if not ref_spec:
            continue
        edges = ref_spec["edges"]
        expected = np.array(ref_spec["proportions"])
        actual = proportions(df_feats[col].to_numpy(dtype=float), edges)
        psi_val = psi(expected, actual)
        psi_scores[col] = psi_val

    # Check model output score
    if "score" in ref_features:
        score_spec = ref_features["score"]
        expected_score = np.array(score_spec["proportions"])
        actual_score = proportions(prod_scores, score_spec["edges"])
        psi_scores["model_score"] = psi(expected_score, actual_score)

    # Pretty print summary table
    print(f"{'Feature / Metric':<25} | {'PSI':<8} | {'Status':<10}")
    print("-" * 50)

    for name, val in psi_scores.items():
        if val < WARN:
            status = "\033[92mSTABLE [OK]\033[0m"
        elif val < ALERT:
            status = "\033[93mWARNING\033[0m"
        else:
            status = "\033[91mALERT [DRIFT]\033[0m"
            drifted_features.append((name, val))

        print(f"{name:<25} | {val:<8.4f} | {status}")

    print("-" * 50)

    if drifted_features:
        drift_msg = ", ".join(f"{name} (PSI={val:.3f})" for name, val in drifted_features)
        raise DriftDetected(f"Critical population drift detected on: {drift_msg} (threshold >= {ALERT})")

    print("\n[+] All features and model scores are STABLE (PSI < 0.25). Model is performing reliably.")
    return psi_scores


if __name__ == "__main__":
    try:
        monitor_drift()
    except DriftDetected as e:
        print(f"\n[!] ALERT TRIGGERED: {e}", file=sys.stderr)
        sys.exit(1)
