"""Train the credit-scoring model, log it to MLflow, promote it if it beats the champion.

Run from the project root:  PYTHONPATH=. python training/train.py
"""
from __future__ import annotations

import os
import subprocess

import mlflow
import pandas as pd
import xgboost as xgb
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score

from features.compute import FEATURE_COLUMNS
from monitoring.psi import build_reference_profile

MODEL_NAME = os.getenv("MODEL_NAME", "credit-scoring")
SPLIT_DATE = pd.Timestamp(os.getenv("SPLIT_DATE", "2025-04-01"))  # train before, test after
MIN_GINI = float(os.getenv("MIN_GINI", "0.50"))
PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
}
# Risk may only go up with these signals (and down with income): easier to explain to a lender
MONOTONE = {"avg_monthly_income": -1, "gambling_share": 1, "n_rejected_debits": 1, "negative_balance_days": 1}


def load_training_data() -> pd.DataFrame:
    from google.cloud import bigquery

    project = os.environ["GCP_PROJECT"]
    client = bigquery.Client(project=project, location=os.getenv("BQ_LOCATION", "EU"))
    return client.query(f"select * from `{project}.analytics.training_dataset`").to_dataframe()


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.getenv("GIT_COMMIT", "unknown")


def champion_gini(X_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    """Score the current champion on the SAME test set, so the comparison is fair."""
    try:
        champion = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@champion")
    except MlflowException:
        return None  # first run: no champion yet
    return gini(y_test, champion.predict_proba(X_test)[:, 1])


def main(df: pd.DataFrame | None = None) -> None:
    df = load_training_data() if df is None else df
    df["application_date"] = pd.to_datetime(df["application_date"])
    train, test = df[df["application_date"] < SPLIT_DATE], df[df["application_date"] >= SPLIT_DATE]
    X_train, y_train = train[FEATURE_COLUMNS].astype(float), train["defaulted"].astype(int)
    X_test, y_test = test[FEATURE_COLUMNS].astype(float), test["defaulted"].astype(int)

    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run():
        model = xgb.XGBClassifier(**PARAMS, monotone_constraints=MONOTONE)
        model.fit(X_train, y_train)
        p_train = model.predict_proba(X_train)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        metrics = {
            "test_auc": roc_auc_score(y_test, p_test),
            "test_gini": gini(y_test, p_test),
            "train_gini": gini(y_train, p_train),
            "test_default_rate": float(y_test.mean()),
            "n_train": len(train),
            "n_test": len(test),
        }
        mlflow.log_params({**PARAMS, "split_date": SPLIT_DATE.date().isoformat(), "monotone": str(MONOTONE)})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "git_commit": git_commit(),
            "data_max_application_date": df["application_date"].max().date().isoformat(),
            "features": ",".join(FEATURE_COLUMNS),
        })
        # Training distributions, stored WITH the model, used later for drift monitoring
        mlflow.log_dict(build_reference_profile(X_train, p_train), "monitoring/reference_profile.json")
        model_info = mlflow.xgboost.log_model(
            model,
            name="model",
            signature=infer_signature(X_test, p_test),
            registered_model_name=MODEL_NAME,
        )

    client = MlflowClient()
    version = model_info.registered_model_version
    current = champion_gini(X_test, y_test)
    promote = metrics["test_gini"] >= MIN_GINI and (current is None or metrics["test_gini"] >= current)
    client.set_model_version_tag(MODEL_NAME, version, "promoted", str(promote))
    if promote:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)

    champion = "none" if current is None else f"{current:.3f}"
    print(f"v{version}: test Gini {metrics['test_gini']:.3f} (champion {champion}) -> promoted={promote}")


if __name__ == "__main__":
    main()
