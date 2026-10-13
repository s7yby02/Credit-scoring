"""Real-time scoring API: raw bank transactions in, default probability out.

Run locally from the project root:  PYTHONPATH=. serve run serving/config.yaml
(no `from __future__ import annotations` here: FastAPI needs the real types)
"""
import math
import os
from datetime import date

import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel
from ray import serve

from features.compute import FEATURE_COLUMNS, compute_features, load_rules

MODEL_NAME = os.getenv("MODEL_NAME", "credit-scoring")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MIN_TRANSACTIONS = 30  # below this, the history is too thin to score

api = FastAPI(title="Credit scoring API")


class Transaction(BaseModel):
    transaction_id: str
    booking_date: date
    amount: float
    description: str
    balance_after: float


class ScoreRequest(BaseModel):
    application_id: str
    application_date: date
    transactions: list[Transaction]


def risk_band(p: float) -> str:
    return "A" if p < 0.05 else "B" if p < 0.10 else "C" if p < 0.20 else "D"


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.5})
@serve.ingress(api)
class CreditScorer:
    def __init__(self) -> None:
        # Resolve the alias once at startup: every replica serves one known version
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        self.model_version = version.version
        self.model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{version.version}")
        self.rules = load_rules()

    @api.get("/health")
    def health(self) -> dict:
        return {"status": "ok", "model": MODEL_NAME, "model_version": self.model_version}

    @api.post("/score")
    def score(self, body: ScoreRequest) -> dict:
        tx = pd.DataFrame([t.model_dump() for t in body.transactions], columns=list(Transaction.model_fields))
        features = compute_features(tx, body.application_date, self.rules)
        X = pd.DataFrame([features])[FEATURE_COLUMNS].astype(float)
        p = float(self.model.predict_proba(X)[0, 1])
        enough_history = features["n_transactions"] >= MIN_TRANSACTIONS
        return {
            "application_id": body.application_id,
            "default_probability": round(p, 4),
            "risk_band": risk_band(p) if enough_history else "insufficient_history",
            "model_version": self.model_version,
            # JSON has no NaN: missing features are returned as null
            "features": {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in features.items()},
        }


app = CreditScorer.bind()
