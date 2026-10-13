"""Population Stability Index (PSI): how much a distribution moved since training.

Rule of thumb in credit scoring: < 0.10 stable, 0.10-0.25 watch, > 0.25 significant shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 10
WARN, ALERT = 0.10, 0.25


def make_edges(values: np.ndarray, n_bins: int = N_BINS) -> list[float]:
    """Decile edges computed on the reference (training) data."""
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return []
    return np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1])).tolist()


def proportions(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Share of values per bin. Missing values get their own last bin."""
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(np.asarray(edges, dtype=float), values, side="right")
    idx[np.isnan(values)] = len(edges) + 1
    counts = np.bincount(idx, minlength=len(edges) + 2)
    return counts / max(len(values), 1)


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference_profile(X: pd.DataFrame, scores: np.ndarray) -> dict:
    """Bin edges + proportions for every feature and for the model score."""
    columns = {c: X[c].to_numpy(dtype=float) for c in X.columns}
    columns["score"] = np.asarray(scores, dtype=float)
    profile = {}
    for name, values in columns.items():
        edges = make_edges(values)
        profile[name] = {"edges": edges, "proportions": proportions(values, edges).tolist()}
    return {"n_rows": len(X), "features": profile}
