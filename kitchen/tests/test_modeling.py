"""Tests for kitchen.modeling: train_val_split, classification_metrics, regression_metrics."""
# pylint: disable=redefined-outer-name  # standard pytest fixture injection pattern

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kitchen.modeling import (
    classification_metrics,
    regression_metrics,
    train_val_split,
)

# ---------------------------------------------------------------------------
# train_val_split
# ---------------------------------------------------------------------------


@pytest.fixture()
def binary_df():
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame(
        {
            "feature_a": rng.standard_normal(n),
            "feature_b": rng.standard_normal(n),
            "target": rng.integers(0, 2, n),
        }
    )


def test_split_sizes_default(binary_df):
    train, val = train_val_split(binary_df, target_col="target")
    assert len(train) == 80
    assert len(val) == 20
    assert len(train) + len(val) == len(binary_df)


def test_split_custom_val_size(binary_df):
    train, val = train_val_split(binary_df, target_col="target", val_size=0.3)
    assert len(val) == 30
    assert len(train) == 70


def test_split_reproducible(binary_df):
    train_a, _ = train_val_split(binary_df, target_col="target", seed=7)
    train_b, _ = train_val_split(binary_df, target_col="target", seed=7)
    pd.testing.assert_frame_equal(train_a.reset_index(drop=True), train_b.reset_index(drop=True))


def test_split_different_seeds_differ(binary_df):
    _, val_a = train_val_split(binary_df, target_col="target", seed=1)
    _, val_b = train_val_split(binary_df, target_col="target", seed=2)
    assert not val_a.index.equals(val_b.index)


def test_split_stratify_preserves_balance(binary_df):
    train, val = train_val_split(binary_df, target_col="target", stratify=True)
    overall_rate = binary_df["target"].mean()
    assert abs(train["target"].mean() - overall_rate) < 0.1
    assert abs(val["target"].mean() - overall_rate) < 0.1


def test_split_no_stratify(binary_df):
    train, val = train_val_split(binary_df, target_col="target", stratify=False)
    assert len(train) + len(val) == len(binary_df)


def test_split_preserves_columns(binary_df):
    train, val = train_val_split(binary_df, target_col="target")
    assert list(train.columns) == list(binary_df.columns)
    assert list(val.columns) == list(binary_df.columns)


def test_split_no_row_overlap(binary_df):
    train, val = train_val_split(binary_df, target_col="target")
    assert set(train.index).isdisjoint(set(val.index))


# ---------------------------------------------------------------------------
# classification_metrics
# ---------------------------------------------------------------------------


@pytest.fixture()
def perfect_binary():
    y = np.array([0, 1, 0, 1, 1, 0])
    return y, y, None


@pytest.fixture()
def imperfect_binary():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])  # 2 wrong
    y_proba = np.array([0.1, 0.9, 0.6, 0.8, 0.4, 0.2])
    return y_true, y_pred, y_proba


def test_classification_perfect_accuracy(perfect_binary):
    y_true, y_pred, _ = perfect_binary
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(1.0)


def test_classification_accuracy_value(imperfect_binary):
    y_true, y_pred, _ = imperfect_binary
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(4 / 6)


def test_classification_f1_present(imperfect_binary):
    y_true, y_pred, _ = imperfect_binary
    m = classification_metrics(y_true, y_pred)
    assert "f1" in m
    assert 0.0 <= m["f1"] <= 1.0


def test_classification_no_proba_no_auc(imperfect_binary):
    y_true, y_pred, _ = imperfect_binary
    m = classification_metrics(y_true, y_pred)
    assert "roc_auc" not in m
    assert "log_loss" not in m


def test_classification_with_proba_has_auc_and_logloss(imperfect_binary):
    y_true, y_pred, y_proba = imperfect_binary
    m = classification_metrics(y_true, y_pred, y_proba=y_proba)
    assert "roc_auc" in m
    assert "log_loss" in m
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert m["log_loss"] > 0.0


def test_classification_returns_floats(imperfect_binary):
    y_true, y_pred, y_proba = imperfect_binary
    m = classification_metrics(y_true, y_pred, y_proba=y_proba)
    for k, v in m.items():
        assert isinstance(v, float), f"{k} is not a float"


def test_classification_macro_average():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    m = classification_metrics(y_true, y_pred, average="macro")
    assert "f1" in m


# ---------------------------------------------------------------------------
# regression_metrics
# ---------------------------------------------------------------------------


def test_regression_perfect():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_regression_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 2.5, 3.5])
    m = regression_metrics(y_true, y_pred)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["mae"] == pytest.approx(0.5)
    assert "r2" in m


def test_regression_keys():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y + 0.1)
    assert set(m.keys()) == {"rmse", "mae", "r2"}


def test_regression_returns_floats():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y + 0.1)
    for k, v in m.items():
        assert isinstance(v, float), f"{k} is not a float"


def test_regression_rmse_positive():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    m = regression_metrics(y_true, y_pred)
    assert m["rmse"] > 0.0
    assert m["mae"] > 0.0
