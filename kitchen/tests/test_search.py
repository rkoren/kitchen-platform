"""Tests for kitchen.search.grid_search (SWEEP-002)."""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import pytest

from kitchen.modeling import regression_metrics
from kitchen.search import _resolve_metric_key, grid_search

# ── A deterministic, param-driven estimator ───────────────────────────────────
# Predicts 1 when the first feature exceeds `thresh`. On a dataset where the
# label IS (x0 > 0.5), thresh=0.5 is perfect and other thresholds degrade — so
# the best/worst combination is known in advance.


class ThresholdClassifier:
    def __init__(self, thresh: float = 0.5):
        self.thresh = thresh

    def fit(self, X, y):
        self._classes = np.unique(y)
        return self

    def predict(self, X):
        return (np.asarray(X)[:, 0] > self.thresh).astype(int)


@pytest.fixture()
def separable_df():
    """120-row binary dataset where y == (x0 > 0.5)."""
    rng = np.random.default_rng(0)
    x0 = rng.uniform(0, 1, 120)
    x1 = rng.standard_normal(120)
    y = (x0 > 0.5).astype(int)
    return pd.DataFrame({"x0": x0, "x1": x1, "target": y})


@pytest.fixture()
def mlflow_tmp(tmp_path):
    """Point MLflow at a throwaway local SQLite store; restore + clean up after."""
    prev_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlruns.db")
    mlflow.set_experiment("search-test")
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(prev_uri)


def _client():
    import mlflow.tracking

    return mlflow.tracking.MlflowClient()


def _exp_id():
    return _client().get_experiment_by_name("search-test").experiment_id


# ── Best-selection correctness ────────────────────────────────────────────────


def test_returns_best_params_higher_is_better(separable_df, mlflow_tmp):
    best = grid_search(
        trainer_fn=lambda p: ThresholdClassifier(**p),
        param_grid={"thresh": [0.5, 2.0]},
        df=separable_df,
        target_col="target",
        metric="accuracy",
    )
    assert best == {"thresh": 0.5}


def test_returns_worst_params_when_lower_is_better(separable_df, mlflow_tmp):
    # With accuracy + lower_is_better, the *worst* threshold should win.
    best = grid_search(
        trainer_fn=lambda p: ThresholdClassifier(**p),
        param_grid={"thresh": [0.5, 2.0]},
        df=separable_df,
        target_col="target",
        metric="accuracy",
        higher_is_better=False,
    )
    assert best == {"thresh": 2.0}


# ── Metric resolution ─────────────────────────────────────────────────────────


def test_metric_resolves_base_name(separable_df, mlflow_tmp):
    best = grid_search(
        trainer_fn=lambda p: ThresholdClassifier(**p),
        param_grid={"thresh": [0.5, 2.0]},
        df=separable_df,
        target_col="target",
        metric="accuracy",  # base name -> accuracy_mean
    )
    assert best == {"thresh": 0.5}


def test_metric_resolves_exact_mean_key(separable_df, mlflow_tmp):
    best = grid_search(
        trainer_fn=lambda p: ThresholdClassifier(**p),
        param_grid={"thresh": [0.5, 2.0]},
        df=separable_df,
        target_col="target",
        metric="accuracy_mean",  # exact CV key
    )
    assert best == {"thresh": 0.5}


def test_resolve_metric_key_raises_on_unknown():
    with pytest.raises(ValueError, match="not found"):
        _resolve_metric_key("nope", {"accuracy_mean": 0.9, "accuracy_std": 0.0})


def test_unknown_metric_raises_listing_available(separable_df, mlflow_tmp):
    with pytest.raises(ValueError, match="available keys"):
        grid_search(
            trainer_fn=lambda p: ThresholdClassifier(**p),
            param_grid={"thresh": [0.5]},
            df=separable_df,
            target_col="target",
            metric="does_not_exist",
        )


# ── Empty grid ────────────────────────────────────────────────────────────────


def test_empty_param_grid_raises(separable_df, mlflow_tmp):
    with pytest.raises(ValueError, match="no parameter combinations"):
        grid_search(
            trainer_fn=lambda p: ThresholdClassifier(**p),
            param_grid=[],  # ParameterGrid([]) -> no combos
            df=separable_df,
            target_col="target",
            metric="accuracy",
        )


# ── MLflow run structure ──────────────────────────────────────────────────────


def test_logs_one_nested_run_per_combination(separable_df, mlflow_tmp):
    grid_search(
        trainer_fn=lambda p: ThresholdClassifier(**p),
        param_grid={"thresh": [0.0, 0.5, 2.0]},
        df=separable_df,
        target_col="target",
        metric="accuracy",
    )
    trials = _client().search_runs(
        [_exp_id()], filter_string="tags.`sweep.trial` != ''"
    )
    assert len(trials) == 3
    # Each trial logged its swept param and the CV aggregate metric.
    for r in trials:
        assert "thresh" in r.data.params
        assert "accuracy_mean" in r.data.metrics


def test_parent_run_carries_best_score_and_sweep_tags(separable_df, mlflow_tmp):
    grid_search(
        trainer_fn=lambda p: ThresholdClassifier(**p),
        param_grid={"thresh": [0.5, 2.0]},
        df=separable_df,
        target_col="target",
        metric="accuracy",
        run_name="my-sweep",
    )
    parents = _client().search_runs(
        [_exp_id()],
        filter_string="tags.`mlflow.runName` = 'my-sweep'",
    )
    assert len(parents) == 1
    parent = parents[0]
    assert parent.data.tags["sweep.metric"] == "accuracy"
    assert parent.data.tags["sweep.n_trials"] == "2"
    assert parent.data.tags["sweep.best.thresh"] == "0.5"
    # Best score logged under the resolved metric key (perfect separation -> 1.0).
    assert parent.data.metrics["accuracy_mean"] == pytest.approx(1.0)


def test_nests_under_existing_active_run(separable_df, mlflow_tmp):
    with mlflow.start_run(run_name="outer") as outer:
        grid_search(
            trainer_fn=lambda p: ThresholdClassifier(**p),
            param_grid={"thresh": [0.5, 2.0]},
            df=separable_df,
            target_col="target",
            metric="accuracy",
        )
        outer_id = outer.info.run_id
    # Both trials should declare `outer` as their parent run.
    trials = _client().search_runs(
        [_exp_id()], filter_string="tags.`sweep.trial` != ''"
    )
    assert len(trials) == 2
    for r in trials:
        assert r.data.tags.get("mlflow.parentRunId") == outer_id


def test_starts_own_parent_when_no_active_run(separable_df, mlflow_tmp):
    assert mlflow.active_run() is None
    grid_search(
        trainer_fn=lambda p: ThresholdClassifier(**p),
        param_grid={"thresh": [0.5]},
        df=separable_df,
        target_col="target",
        metric="accuracy",
        run_name="solo-sweep",
    )
    # grid_search must not leak an active run.
    assert mlflow.active_run() is None
    parents = _client().search_runs(
        [_exp_id()], filter_string="tags.`mlflow.runName` = 'solo-sweep'"
    )
    assert len(parents) == 1


def test_exception_in_trainer_ends_parent_run(separable_df, mlflow_tmp):
    def boom(_p):
        raise RuntimeError("trainer blew up")

    assert mlflow.active_run() is None
    with pytest.raises(RuntimeError, match="blew up"):
        grid_search(
            trainer_fn=boom,
            param_grid={"thresh": [0.5]},
            df=separable_df,
            target_col="target",
            metric="accuracy",
        )
    # Parent run must be closed even though a trial raised.
    assert mlflow.active_run() is None


# ── Regression path (metric_fn override, stratify off) ─────────────────────────


def test_regression_metric_fn_with_lower_is_better(mlflow_tmp):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((120, 2))
    y = x[:, 0] * 3.0
    df = pd.DataFrame(x, columns=["a", "b"]).assign(target=y)

    from sklearn.linear_model import Ridge

    best = grid_search(
        trainer_fn=lambda p: Ridge(**p),
        param_grid={"alpha": [0.01, 1000.0]},
        df=df,
        target_col="target",
        metric="rmse",
        metric_fn=regression_metrics,
        higher_is_better=False,
        stratify=False,
    )
    # Light regularisation fits a near-linear target far better than alpha=1000.
    assert best == {"alpha": 0.01}


# ── Top-level re-export ────────────────────────────────────────────────────────


def test_grid_search_exported_from_kitchen():
    import kitchen

    assert hasattr(kitchen, "grid_search")
    assert hasattr(kitchen, "search")
    assert kitchen.search.grid_search is kitchen.grid_search
