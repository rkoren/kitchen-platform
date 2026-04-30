"""Tests for kitchen.evaluate."""

import numpy as np
import pandas as pd
import pytest

from kitchen.evaluate import brier_score, log_loss


def test_brier_perfect():
    assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)


def test_brier_random():
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_brier_worst():
    assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


def test_brier_accepts_numpy():
    y_true = np.array([1, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.8])
    assert brier_score(y_true, y_prob) == pytest.approx(np.mean((y_prob - y_true) ** 2))


def test_brier_accepts_series():
    assert brier_score(pd.Series([1, 0]), pd.Series([1.0, 0.0])) == pytest.approx(0.0)


# ── log_loss ──────────────────────────────────────────────────────────────────

def test_log_loss_perfect():
    assert log_loss([1, 0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-6)


def test_log_loss_random():
    assert log_loss([1, 0], [0.5, 0.5]) == pytest.approx(0.6931, abs=1e-3)


def test_log_loss_clamps_extremes():
    assert np.isfinite(log_loss([1, 0], [1.0, 0.0]))


def test_log_loss_accepts_numpy():
    y_true = np.array([1, 0])
    y_prob = np.array([0.8, 0.2])
    assert np.isfinite(log_loss(y_true, y_prob))
