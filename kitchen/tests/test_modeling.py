"""Tests for kitchen.modeling helpers."""
# pylint: disable=redefined-outer-name  # standard pytest fixture injection pattern

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kitchen.modeling import (
    classification_metrics,
    clip_predictions,
    clip_proba,
    cross_validate,
    regression_metrics,
    set_seed,
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


# ---------------------------------------------------------------------------
# cross_validate
# ---------------------------------------------------------------------------


@pytest.fixture()
def cv_binary_df():
    """100-row balanced binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 4))
    # Simple linearly separable-ish target so a logistic model does OK
    y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 0).astype(int)
    return pd.DataFrame(X, columns=["a", "b", "c", "d"]).assign(target=y)


@pytest.fixture()
def cv_regression_df():
    """100-row continuous regression dataset."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 3))
    y = X[:, 0] * 2 + rng.standard_normal(n) * 0.3
    return pd.DataFrame(X, columns=["x1", "x2", "x3"]).assign(target=y)


def _lr_factory():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=500, random_state=0)


def _ridge_factory():
    from sklearn.linear_model import Ridge

    return Ridge()


# --- return shape and key naming ---


def test_cv_returns_mean_and_std_keys(cv_binary_df):
    result = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
    )
    assert "accuracy_mean" in result
    assert "accuracy_std" in result
    assert "f1_mean" in result
    assert "f1_std" in result


def test_cv_all_values_are_floats(cv_binary_df):
    result = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
    )
    for k, v in result.items():
        assert isinstance(v, float), f"{k} is not a float"


def test_cv_result_has_no_extra_keys(cv_binary_df):
    """Keys should be exactly {metric}_mean and {metric}_std — no raw fold data."""
    result = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
    )
    for key in result:
        assert key.endswith("_mean") or key.endswith("_std"), f"unexpected key: {key}"


# --- metric values are sane ---


def test_cv_accuracy_in_range(cv_binary_df):
    result = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
    )
    assert 0.0 <= result["accuracy_mean"] <= 1.0
    assert result["accuracy_std"] >= 0.0


def test_cv_std_is_nonnegative(cv_binary_df):
    result = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
    )
    for key in result:
        if key.endswith("_std"):
            assert result[key] >= 0.0, f"{key} is negative"


# --- reproducibility ---


def test_cv_same_seed_reproducible(cv_binary_df):
    r1 = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
        seed=7,
    )
    r2 = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
        seed=7,
    )
    assert r1 == r2


def test_cv_different_seeds_differ(cv_binary_df):
    r1 = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
        seed=1,
    )
    r2 = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
        seed=99,
    )
    # With different fold splits, means should differ (highly unlikely to be equal)
    assert r1["accuracy_mean"] != r2["accuracy_mean"]


# --- n_splits ---


def test_cv_default_five_folds_produces_sensible_std(cv_binary_df):
    """5-fold CV on 100 rows — std should be small but non-zero."""
    result = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
    )
    # std > 0 for a noisy dataset; < 0.3 for a reasonably well-behaved one
    assert 0.0 <= result["accuracy_std"] < 0.3


def test_cv_three_folds(cv_binary_df):
    result = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
        n_splits=3,
    )
    assert "accuracy_mean" in result


# --- return_proba ---


def test_cv_return_proba_adds_roc_auc_and_log_loss(cv_binary_df):
    result = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
        return_proba=True,
    )
    assert "roc_auc_mean" in result
    assert "log_loss_mean" in result


def test_cv_no_proba_no_roc_auc(cv_binary_df):
    result = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
        return_proba=False,
    )
    assert "roc_auc_mean" not in result
    assert "log_loss_mean" not in result


def test_cv_roc_auc_in_range(cv_binary_df):
    result = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
        return_proba=True,
    )
    assert 0.0 <= result["roc_auc_mean"] <= 1.0


# --- regression mode ---


def test_cv_regression_returns_rmse_mae_r2(cv_regression_df):
    result = cross_validate(
        cv_regression_df,
        target_col="target",
        estimator_fn=_ridge_factory,
        metric_fn=regression_metrics,
        stratify=False,
    )
    assert "rmse_mean" in result
    assert "mae_mean" in result
    assert "r2_mean" in result


def test_cv_regression_rmse_positive(cv_regression_df):
    result = cross_validate(
        cv_regression_df,
        target_col="target",
        estimator_fn=_ridge_factory,
        metric_fn=regression_metrics,
        stratify=False,
    )
    assert result["rmse_mean"] > 0.0


def test_cv_regression_r2_reasonable(cv_regression_df):
    """Ridge on a near-linear dataset should achieve R² > 0.5."""
    result = cross_validate(
        cv_regression_df,
        target_col="target",
        estimator_fn=_ridge_factory,
        metric_fn=regression_metrics,
        stratify=False,
    )
    assert result["r2_mean"] > 0.5


# --- stratify=False (KFold path) ---


def test_cv_kfold_no_stratify(cv_binary_df):
    """KFold path should still work for classification data."""
    result = cross_validate(
        cv_binary_df,
        target_col="target",
        estimator_fn=_lr_factory,
        metric_fn=classification_metrics,
        stratify=False,
    )
    assert "accuracy_mean" in result
    assert 0.0 <= result["accuracy_mean"] <= 1.0


# --- top-level import ---


def test_cv_importable_from_kitchen():
    from kitchen import cross_validate as cv_fn  # noqa: F401

    assert callable(cv_fn)


# ---------------------------------------------------------------------------
# clip_proba  (M-008)
# ---------------------------------------------------------------------------


def test_clip_proba_clips_zeros_and_ones():
    arr = np.array([0.0, 0.5, 1.0])
    result = clip_proba(arr)
    assert result[0] > 0.0
    assert result[2] < 1.0
    assert result[1] == pytest.approx(0.5)


def test_clip_proba_default_eps():
    arr = np.array([0.0, 1.0])
    result = clip_proba(arr)
    assert result[0] == pytest.approx(1e-6)
    assert result[1] == pytest.approx(1.0 - 1e-6)


def test_clip_proba_custom_eps():
    arr = np.array([0.0, 0.5, 1.0])
    result = clip_proba(arr, eps=0.01)
    assert result[0] == pytest.approx(0.01)
    assert result[2] == pytest.approx(0.99)


def test_clip_proba_already_valid_unchanged():
    arr = np.array([0.2, 0.5, 0.8])
    result = clip_proba(arr)
    np.testing.assert_array_almost_equal(result, arr)


def test_clip_proba_returns_ndarray():
    result = clip_proba([0.0, 0.5, 1.0])
    assert isinstance(result, np.ndarray)


def test_clip_proba_2d_matrix():
    """Multiclass probability matrix — each element clipped independently."""
    arr = np.array([[0.0, 0.5, 0.5], [0.33, 0.33, 0.34]])
    result = clip_proba(arr)
    assert result.shape == arr.shape
    assert result[0, 0] > 0.0
    assert result.min() >= 1e-6
    assert result.max() <= 1.0 - 1e-6


def test_clip_proba_preserves_shape():
    arr = np.zeros((10, 3))
    result = clip_proba(arr)
    assert result.shape == (10, 3)


def test_clip_proba_safe_for_log_loss():
    """Clipped probas should not produce -inf in log_loss."""
    from sklearn.metrics import log_loss

    y_true = np.array([0, 1, 0, 1])
    y_proba = clip_proba(np.array([0.0, 1.0, 0.3, 0.7]))
    loss = log_loss(y_true, y_proba)
    assert np.isfinite(loss)


# ---------------------------------------------------------------------------
# clip_predictions  (M-008)
# ---------------------------------------------------------------------------


def test_clip_predictions_lower_bound():
    arr = np.array([-1.0, 0.0, 2.0, 5.0])
    result = clip_predictions(arr, low=0.0)
    assert result.min() >= 0.0
    assert result[2] == pytest.approx(2.0)


def test_clip_predictions_upper_bound():
    arr = np.array([0.0, 3.0, 6.0])
    result = clip_predictions(arr, high=5.0)
    assert result.max() <= 5.0
    assert result[0] == pytest.approx(0.0)


def test_clip_predictions_both_bounds():
    arr = np.array([-2.0, 0.5, 3.0, 7.0])
    result = clip_predictions(arr, low=0.0, high=5.0)
    assert result.min() >= 0.0
    assert result.max() <= 5.0
    assert result[1] == pytest.approx(0.5)


def test_clip_predictions_returns_ndarray():
    result = clip_predictions([1.0, 2.0, 3.0], low=0.0, high=4.0)
    assert isinstance(result, np.ndarray)


def test_clip_predictions_no_bounds_raises():
    with pytest.raises(ValueError, match="low.*high"):
        clip_predictions(np.array([1.0, 2.0]))


def test_clip_predictions_preserves_in_range_values():
    arr = np.array([1.0, 2.0, 3.0])
    result = clip_predictions(arr, low=0.0, high=5.0)
    np.testing.assert_array_almost_equal(result, arr)


def test_clip_predictions_float_conversion():
    """Integer input is cast to float."""
    result = clip_predictions([1, 2, 10], low=0, high=5)
    assert result.dtype == float
    assert result[2] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# set_seed  (M-009)
# ---------------------------------------------------------------------------


def test_set_seed_numpy_reproducible():
    set_seed(0)
    a = np.random.rand(5)
    set_seed(0)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_seed_different_seeds_differ():
    set_seed(1)
    a = np.random.rand(5)
    set_seed(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)


def test_set_seed_python_random_reproducible():
    import random

    set_seed(7)
    a = [random.random() for _ in range(5)]
    set_seed(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_returns_none():
    assert set_seed(42) is None


def test_set_seed_no_torch_no_error():
    """Should not raise even when torch is absent."""
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"torch": None}):
        set_seed(42)  # must not raise


def test_set_seed_no_tensorflow_no_error():
    """Should not raise even when tensorflow is absent."""
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"tensorflow": None}):
        set_seed(42)  # must not raise


def test_set_seed_default_seed_is_42():
    """Calling set_seed() twice with default must be identical to set_seed(42) twice."""
    set_seed()
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_seed_importable_from_kitchen():
    from kitchen import set_seed as ss  # noqa: F401

    assert callable(ss)


def test_clip_proba_importable_from_kitchen():
    from kitchen import clip_proba as cp  # noqa: F401

    assert callable(cp)


def test_clip_predictions_importable_from_kitchen():
    from kitchen import clip_predictions as cpred  # noqa: F401

    assert callable(cpred)
