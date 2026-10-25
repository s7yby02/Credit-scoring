"""Unit tests for the Population Stability Index (PSI) calculation (monitoring/psi.py)."""
from __future__ import annotations

import numpy as np
import pytest

from monitoring.psi import (
    ALERT,
    WARN,
    build_reference_profile,
    make_edges,
    proportions,
    psi,
)


def test_psi_identical_distributions():
    """Identical expected and actual distributions must yield a PSI of 0.0."""
    expected = np.array([0.1] * 10)
    actual = np.array([0.1] * 10)
    assert pytest.approx(psi(expected, actual), abs=1e-5) == 0.0


def test_psi_small_shift():
    """A small fluctuation should remain well below the warning threshold (WARN = 0.10)."""
    expected = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    actual = np.array([0.09, 0.11, 0.10, 0.09, 0.11, 0.10, 0.09, 0.11, 0.10, 0.10])
    val = psi(expected, actual)
    assert val < WARN, f"Expected small PSI < {WARN}, got {val}"


def test_psi_massive_drift():
    """A massive population shift must trigger the alert threshold (ALERT = 0.25)."""
    expected = np.array([0.10] * 10)
    # Severe shift: 70% of the population concentrated in the lowest decile
    actual = np.array([0.70, 0.10, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
    val = psi(expected, actual)
    assert val >= ALERT, f"Expected alert PSI >= {ALERT}, got {val}"


def test_proportions_handles_nan():
    """Missing values (NaN) must be counted in their own bucket without crashing."""
    values = np.array([10.0, 20.0, np.nan, 40.0, np.nan])
    edges = [15.0, 25.0, 35.0]
    props = proportions(values, edges)
    # Total proportions must sum to 1.0
    assert pytest.approx(float(props.sum()), abs=1e-6) == 1.0
    # Exactly 2 NaNs out of 5 values -> 40% in the NaN bucket
    assert pytest.approx(float(props[-1]), abs=1e-6) == 2.0 / 5.0
