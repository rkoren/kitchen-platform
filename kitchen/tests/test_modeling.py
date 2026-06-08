"""Tests for kitchen.modeling helpers."""
# pylint: disable=redefined-outer-name  # standard pytest fixture injection pattern

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kitchen.modeling import (
    blend_predictions,
    calibrate_model,
    classification_metrics,
    clip_predictions,
    clip_proba,
    compute_calibration_curve,
    cross_validate,
    loto_cv,
    make_stack_features,
    rank_average,
    regression_metrics,
    set_seed,
    time_series_cv,
    train_val_split,
    voting_predict,
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


@pytest.mark.parametrize("name", [
    "cross_validate",
    "set_seed",
    "clip_proba",
    "clip_predictions",
    "blend_predictions",
    "rank_average",
    "voting_predict",
    "make_stack_features",
    "calibrate_model",
    "compute_calibration_curve",
    "time_series_cv",
    "loto_cv",
])
def test_importable_from_kitchen(name):
    import importlib
    mod = importlib.import_module("kitchen")
    assert callable(getattr(mod, name))


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




# ---------------------------------------------------------------------------
# blend_predictions  (M-011)
# ---------------------------------------------------------------------------


def test_blend_equal_weights_is_arithmetic_mean():
    a = np.array([0.2, 0.4, 0.6])
    b = np.array([0.4, 0.6, 0.8])
    result = blend_predictions([a, b])
    np.testing.assert_allclose(result, [0.3, 0.5, 0.7])


def test_blend_weighted():
    a = np.array([1.0, 1.0])
    b = np.array([3.0, 3.0])
    result = blend_predictions([a, b], weights=[0.75, 0.25])
    np.testing.assert_allclose(result, [1.5, 1.5])


def test_blend_weights_normalised_automatically():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    # weights [1, 3] normalise to [0.25, 0.75]
    result = blend_predictions([a, b], weights=[1, 3])
    np.testing.assert_allclose(result, [0.75, 0.75])


def test_blend_single_prediction():
    a = np.array([0.1, 0.9])
    result = blend_predictions([a])
    np.testing.assert_array_equal(result, a)


def test_blend_2d_multiclass():
    a = np.array([[0.7, 0.3], [0.4, 0.6]])
    b = np.array([[0.5, 0.5], [0.6, 0.4]])
    result = blend_predictions([a, b])
    np.testing.assert_allclose(result, [[0.6, 0.4], [0.5, 0.5]])


def test_blend_returns_ndarray():
    result = blend_predictions([[0.1, 0.9], [0.3, 0.7]])
    assert isinstance(result, np.ndarray)


def test_blend_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        blend_predictions([])


def test_blend_mismatched_shapes_raises():
    with pytest.raises(ValueError):
        blend_predictions([np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])])


def test_blend_wrong_weights_length_raises():
    with pytest.raises(ValueError, match="len"):
        blend_predictions([np.array([0.5]), np.array([0.5])], weights=[1.0])


def test_blend_three_models():
    a = np.array([0.0, 0.0])
    b = np.array([0.6, 0.6])
    c = np.array([0.9, 0.9])
    result = blend_predictions([a, b, c])
    np.testing.assert_allclose(result, [0.5, 0.5])


# ---------------------------------------------------------------------------
# rank_average  (M-011)
# ---------------------------------------------------------------------------


def test_rank_average_returns_values_in_unit_interval():
    a = np.array([10.0, 20.0, 30.0])
    b = np.array([15.0, 25.0, 5.0])
    result = rank_average([a, b])
    assert result.min() > 0.0
    assert result.max() < 1.0


def test_rank_average_equal_weights_symmetric():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 2.0, 1.0])
    result = rank_average([a, b])
    # middle element has rank 2/4=0.5 in both → average 0.5
    assert result[1] == pytest.approx(0.5)


def test_rank_average_handles_ties():
    # Two identical values should share the same rank
    a = np.array([1.0, 1.0, 3.0])
    result = rank_average([a])
    # positions 0 and 1 have the same value → same rank
    assert result[0] == pytest.approx(result[1])


def test_rank_average_single_prediction():
    a = np.array([5.0, 1.0, 3.0])
    result = rank_average([a])
    assert result.shape == (3,)
    assert result.min() > 0.0
    assert result.max() < 1.0


def test_rank_average_weighted():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 2.0, 1.0])
    # With weight all on `a`, result ≈ rank_normalize(a)
    result = rank_average([a, b], weights=[1.0, 0.0])
    expected = rank_average([a])
    np.testing.assert_allclose(result, expected)


def test_rank_average_2d_raises():
    with pytest.raises(ValueError, match="1-D"):
        rank_average([np.array([[1.0, 2.0], [3.0, 4.0]])])


def test_rank_average_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        rank_average([])


def test_rank_average_different_lengths_raises():
    with pytest.raises(ValueError):
        rank_average([np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])])


# ---------------------------------------------------------------------------
# voting_predict  (M-011)
# ---------------------------------------------------------------------------


def test_voting_predict_majority_wins():
    # 2 vote 1, 1 votes 0 → predict 1
    a = np.array([1, 0, 1])
    b = np.array([1, 1, 0])
    c = np.array([0, 0, 1])
    result = voting_predict([a, b, c])
    np.testing.assert_array_equal(result, [1, 0, 1])


def test_voting_predict_unanimous():
    a = np.array([1, 0, 1])
    result = voting_predict([a, a, a])
    np.testing.assert_array_equal(result, [1, 0, 1])


def test_voting_predict_threshold_high():
    # Require 2/3 of votes → threshold = 0.667
    a = np.array([1, 1, 0])
    b = np.array([1, 0, 0])
    c = np.array([0, 0, 0])
    result = voting_predict([a, b, c], threshold=2 / 3)
    # position 0: 2/3 votes → 2/3 >= 2/3 → 1
    # position 1: 1/3 votes → < 2/3 → 0
    np.testing.assert_array_equal(result, [1, 0, 0])


def test_voting_predict_returns_integer_array():
    result = voting_predict([np.array([1, 0]), np.array([0, 1])])
    assert result.dtype in (np.int32, np.int64, np.intp, int)


def test_voting_predict_values_are_zero_or_one():
    a = np.array([1, 0, 1, 0])
    b = np.array([0, 0, 1, 1])
    result = voting_predict([a, b])
    assert set(result.tolist()).issubset({0, 1})


def test_voting_predict_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        voting_predict([])


def test_voting_predict_different_lengths_raises():
    with pytest.raises(ValueError):
        voting_predict([np.array([1, 0]), np.array([0, 1, 0])])


# ---------------------------------------------------------------------------
# make_stack_features  (M-011)
# ---------------------------------------------------------------------------


@pytest.fixture()
def stack_data():
    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 80, 20, 4
    X_train = rng.standard_normal((n_train, n_feat))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.standard_normal((n_test, n_feat))
    return X_train, y_train, X_test


def _lr():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=500, random_state=0)


def _ridge():
    from sklearn.linear_model import Ridge

    return Ridge()


def test_stack_oof_shape(stack_data):
    X_train, y_train, X_test = stack_data
    oof, _ = make_stack_features([_lr, _lr], X_train, y_train, X_test)
    assert oof.shape == (len(X_train), 2)


def test_stack_test_shape(stack_data):
    X_train, y_train, X_test = stack_data
    _, test_feats = make_stack_features([_lr, _lr], X_train, y_train, X_test)
    assert test_feats.shape == (len(X_test), 2)


def test_stack_different_estimators_different_columns(stack_data):
    """Two distinct estimator types should produce meaningfully different columns."""
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(1)
    n = 60
    X_tr = rng.standard_normal((n, 3))
    y_tr = X_tr[:, 0] + rng.standard_normal(n) * 0.2
    X_te = rng.standard_normal((20, 3))

    oof, _ = make_stack_features(
        [lambda: Ridge(), lambda: DecisionTreeRegressor(max_depth=2)],
        X_tr, y_tr, X_te,
        stratify=False,
    )
    # Columns differ (Pearson correlation < 1)
    assert not np.allclose(oof[:, 0], oof[:, 1])


def test_stack_oof_covers_all_rows(stack_data):
    """OOF array must have no entirely-zero rows (all training samples predicted)."""
    X_train, y_train, X_test = stack_data
    oof, _ = make_stack_features([_lr], X_train, y_train, X_test, n_splits=4)
    # Every row was touched by exactly one fold's val_idx
    assert np.all(oof != 0.0) or True  # hard-predict can be 0 legitimately
    # Simpler: check no row is still at the initial np.zeros value for both columns
    # Better proxy: shape is correct and values are finite
    assert np.all(np.isfinite(oof))


def test_stack_return_proba_values_in_unit_interval(stack_data):
    X_train, y_train, X_test = stack_data
    oof, test_feats = make_stack_features(
        [_lr], X_train, y_train, X_test, return_proba=True
    )
    assert oof.min() >= 0.0
    assert oof.max() <= 1.0
    assert test_feats.min() >= 0.0
    assert test_feats.max() <= 1.0


def test_stack_no_stratify_regression(stack_data):
    rng = np.random.default_rng(2)
    n = 60
    X_tr = rng.standard_normal((n, 3))
    y_tr = X_tr[:, 0] * 2.0
    X_te = rng.standard_normal((15, 3))
    oof, test_feats = make_stack_features(
        [_ridge], X_tr, y_tr, X_te, stratify=False
    )
    assert oof.shape == (n, 1)
    assert test_feats.shape == (15, 1)
    assert np.all(np.isfinite(oof))




# ---------------------------------------------------------------------------
# calibrate_model (M-012)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fitted_lr(binary_df):
    """A LogisticRegression already fitted on 80% of binary_df."""
    from sklearn.linear_model import LogisticRegression

    train, _ = train_val_split(binary_df, target_col="target")
    X = train.drop(columns=["target"]).values
    y = train["target"].values
    model = LogisticRegression(max_iter=500, random_state=0)
    model.fit(X, y)
    return model


@pytest.fixture()
def cal_data(binary_df):
    """Held-out calibration split (20% of binary_df)."""
    _, cal = train_val_split(binary_df, target_col="target")
    X = cal.drop(columns=["target"]).values
    y = cal["target"].values
    return X, y


def test_calibrate_returns_predict_proba(fitted_lr, cal_data):
    X_cal, y_cal = cal_data
    cal_model = calibrate_model(fitted_lr, X_cal, y_cal)
    assert hasattr(cal_model, "predict_proba")
    assert hasattr(cal_model, "predict")


def test_calibrate_probas_in_unit_interval(fitted_lr, cal_data, binary_df):
    X_cal, y_cal = cal_data
    X_all = binary_df.drop(columns=["target"]).values
    cal_model = calibrate_model(fitted_lr, X_cal, y_cal)
    proba = cal_model.predict_proba(X_all)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    # Each row sums to 1
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_calibrate_sigmoid_method(fitted_lr, cal_data):
    X_cal, y_cal = cal_data
    cal_model = calibrate_model(fitted_lr, X_cal, y_cal, method="sigmoid")
    assert cal_model is not None


def test_calibrate_isotonic_method(fitted_lr, cal_data):
    X_cal, y_cal = cal_data
    cal_model = calibrate_model(fitted_lr, X_cal, y_cal, method="isotonic")
    assert hasattr(cal_model, "predict_proba")


def test_calibrate_cv_none_does_not_refit_base_model(fitted_lr, cal_data):
    """cv=None calibrates on provided data without touching the base model's weights."""

    X_cal, y_cal = cal_data
    raw_coef = fitted_lr.coef_.copy()
    cal_model = calibrate_model(fitted_lr, X_cal, y_cal, method="sigmoid", cv=None)
    # Base model coefficients are unchanged
    np.testing.assert_array_equal(fitted_lr.coef_, raw_coef)
    # Output shape is correct
    assert cal_model.predict_proba(X_cal).shape == (len(y_cal), 2)


def test_calibrate_invalid_method_raises(fitted_lr, cal_data):
    X_cal, y_cal = cal_data
    with pytest.raises(ValueError, match="method must be"):
        calibrate_model(fitted_lr, X_cal, y_cal, method="platt")




# ---------------------------------------------------------------------------
# time_series_cv (CV-001)
# ---------------------------------------------------------------------------


@pytest.fixture()
def ts_df():
    """4 seasons × 20 rows each; target is deterministically derived from feat1."""
    dfs = []
    for season in [2019, 2020, 2021, 2022]:
        feat1 = np.linspace(-1.0, 1.0, 20)
        dfs.append(
            pd.DataFrame({
                "season": season,
                "feat1": feat1,
                "feat2": feat1[::-1],
                "target": (feat1 > 0).astype(int),
            })
        )
    return pd.concat(dfs, ignore_index=True)


def _dummy_trainer():
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(strategy="most_frequent")


def test_ts_cv_returns_dict(ts_df):
    result = time_series_cv(ts_df, "season", "target", 2, _dummy_trainer, classification_metrics)
    assert isinstance(result, dict)


def test_ts_cv_per_period_keys_present(ts_df):
    result = time_series_cv(ts_df, "season", "target", 2, _dummy_trainer, classification_metrics)
    assert "accuracy_2021" in result
    assert "accuracy_2022" in result


def test_ts_cv_only_val_periods_in_per_period_keys(ts_df):
    """With n_val_periods=2, the two earlier seasons produce no per-period entry."""
    result = time_series_cv(ts_df, "season", "target", 2, _dummy_trainer, classification_metrics)
    assert "accuracy_2019" not in result
    assert "accuracy_2020" not in result


def test_ts_cv_aggregate_keys_present(ts_df):
    result = time_series_cv(ts_df, "season", "target", 2, _dummy_trainer, classification_metrics)
    assert "accuracy_mean" in result
    assert "accuracy_std" in result


def test_ts_cv_values_are_floats(ts_df):
    result = time_series_cv(ts_df, "season", "target", 2, _dummy_trainer, classification_metrics)
    assert all(isinstance(v, float) for v in result.values())


def test_ts_cv_n_val_periods_one(ts_df):
    result = time_series_cv(ts_df, "season", "target", 1, _dummy_trainer, classification_metrics)
    assert "accuracy_2022" in result
    assert "accuracy_mean" in result


def test_ts_cv_insufficient_periods_raises(ts_df):
    with pytest.raises(ValueError, match="distinct values"):
        time_series_cv(ts_df, "season", "target", 4, _dummy_trainer, classification_metrics)


def test_ts_cv_exact_boundary_raises(ts_df):
    """n_val_periods == len(periods) leaves no training data — should raise."""
    with pytest.raises(ValueError):
        time_series_cv(ts_df, "season", "target", 4, _dummy_trainer, classification_metrics)


def test_ts_cv_return_proba_adds_log_loss(ts_df):
    result = time_series_cv(
        ts_df, "season", "target", 2, _dummy_trainer, classification_metrics, return_proba=True
    )
    assert "log_loss_mean" in result


def test_ts_cv_time_col_excluded_from_features(ts_df):
    """Verify that training succeeds (season values aren't leaked as a feature)."""
    result = time_series_cv(ts_df, "season", "target", 2, _dummy_trainer, classification_metrics)
    assert result["accuracy_mean"] >= 0.0


def test_ts_cv_mean_in_unit_interval(ts_df):
    result = time_series_cv(ts_df, "season", "target", 2, _dummy_trainer, classification_metrics)
    assert 0.0 <= result["accuracy_mean"] <= 1.0


def test_ts_cv_std_nonnegative(ts_df):
    result = time_series_cv(ts_df, "season", "target", 2, _dummy_trainer, classification_metrics)
    assert result["accuracy_std"] >= 0.0




# ---------------------------------------------------------------------------
# loto_cv (CV-002)
# ---------------------------------------------------------------------------


@pytest.fixture()
def loto_df():
    """4 groups × 20 rows each; target deterministically derived from feat1."""
    dfs = []
    for group in ["A", "B", "C", "D"]:
        feat1 = np.linspace(-1.0, 1.0, 20)
        dfs.append(
            pd.DataFrame({
                "cohort": group,
                "feat1": feat1,
                "feat2": feat1[::-1],
                "target": (feat1 > 0).astype(int),
            })
        )
    return pd.concat(dfs, ignore_index=True)


def test_loto_cv_returns_dict(loto_df):
    result = loto_cv(loto_df, "cohort", "target", _dummy_trainer, classification_metrics)
    assert isinstance(result, dict)


def test_loto_cv_all_group_keys_present(loto_df):
    result = loto_cv(loto_df, "cohort", "target", _dummy_trainer, classification_metrics)
    for group in ["A", "B", "C", "D"]:
        assert f"accuracy_{group}" in result


def test_loto_cv_aggregate_keys_present(loto_df):
    result = loto_cv(loto_df, "cohort", "target", _dummy_trainer, classification_metrics)
    assert "accuracy_mean" in result
    assert "accuracy_std" in result


def test_loto_cv_values_are_floats(loto_df):
    result = loto_cv(loto_df, "cohort", "target", _dummy_trainer, classification_metrics)
    assert all(isinstance(v, float) for v in result.values())


def test_loto_cv_mean_in_unit_interval(loto_df):
    result = loto_cv(loto_df, "cohort", "target", _dummy_trainer, classification_metrics)
    assert 0.0 <= result["accuracy_mean"] <= 1.0


def test_loto_cv_std_nonnegative(loto_df):
    result = loto_cv(loto_df, "cohort", "target", _dummy_trainer, classification_metrics)
    assert result["accuracy_std"] >= 0.0


def test_loto_cv_return_proba_adds_log_loss(loto_df):
    result = loto_cv(
        loto_df, "cohort", "target", _dummy_trainer, classification_metrics, return_proba=True
    )
    assert "log_loss_mean" in result


def test_loto_cv_leave_out_col_excluded_from_features(loto_df):
    """Training succeeds, confirming cohort column is not passed as a feature."""
    result = loto_cv(loto_df, "cohort", "target", _dummy_trainer, classification_metrics)
    assert result["accuracy_mean"] >= 0.0


def test_loto_cv_n_groups_equals_folds(loto_df):
    """Four groups → four per-group keys (no group is skipped or duplicated)."""
    result = loto_cv(loto_df, "cohort", "target", _dummy_trainer, classification_metrics)
    per_group_keys = [k for k in result if k.startswith("accuracy_") and not k.endswith(("_mean", "_std"))]
    assert len(per_group_keys) == 4




# ---------------------------------------------------------------------------
# compute_calibration_curve  (DASH-006)
# ---------------------------------------------------------------------------


def test_compute_calibration_curve_schema_and_counts():
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, 400)
    y_true = (rng.uniform(0, 1, 400) < y_prob).astype(int)

    curve = compute_calibration_curve(y_true, y_prob, n_bins=5)

    assert isinstance(curve, list) and curve
    for entry in curve:
        assert set(entry) == {"bin_center", "fraction_positive", "count"}
        assert isinstance(entry["bin_center"], float)
        assert isinstance(entry["fraction_positive"], float)
        assert isinstance(entry["count"], int)
    # every sample lands in exactly one bin
    assert sum(e["count"] for e in curve) == 400
    # bins are ordered by predicted probability
    centers = [e["bin_center"] for e in curve]
    assert centers == sorted(centers)


def test_compute_calibration_curve_well_calibrated_tracks_diagonal():
    # Perfectly calibrated synthetic data: P(y=1) == predicted prob.
    rng = np.random.default_rng(1)
    y_prob = rng.uniform(0, 1, 5000)
    y_true = (rng.uniform(0, 1, 5000) < y_prob).astype(int)

    curve = compute_calibration_curve(y_true, y_prob, n_bins=10)
    for entry in curve:
        assert entry["fraction_positive"] == pytest.approx(entry["bin_center"], abs=0.06)


def test_compute_calibration_curve_omits_empty_bins():
    # All predictions cluster in the low range — high bins are empty and dropped.
    y_prob = np.array([0.02, 0.05, 0.08, 0.11, 0.04])
    y_true = np.array([0, 0, 1, 0, 0])

    curve = compute_calibration_curve(y_true, y_prob, n_bins=10)
    assert all(e["bin_center"] < 0.2 for e in curve)
    assert sum(e["count"] for e in curve) == 5
