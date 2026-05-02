"""Tests for FeatureBuilder, Trainer, and Evaluator contracts."""
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kitchen.steps import Evaluator, FeatureBuilder, Trainer, _resolve


# --- Concrete implementations for testing ---

class DoubleFeatures(FeatureBuilder):
    def build(self, raw: pd.DataFrame, params: dict) -> pd.DataFrame:
        return raw * 2


class ConstantTrainer(Trainer):
    def fit(self, df: pd.DataFrame, params: dict) -> object:
        return {"weights": [1, 2, 3]}


class AccuracyEvaluator(Evaluator):
    def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, float]:
        return {"accuracy": 0.95, "f1": 0.90}


# --- FeatureBuilder ---

def test_feature_builder_build():
    raw = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = DoubleFeatures().build(raw, params={})
    assert list(result["a"]) == [2, 4]


def test_feature_builder_run_passes_params_to_build():
    received = {}

    class ParamCapture(FeatureBuilder):
        def build(self, raw: pd.DataFrame, params: dict) -> pd.DataFrame:
            received.update(params)
            return raw

    store = MagicMock()
    store.load_csv.return_value = pd.DataFrame({"x": [1]})
    ParamCapture().run(store, params={"raw_file": "d.csv", "custom_key": "custom_val"})
    assert received.get("custom_key") == "custom_val"


def test_feature_builder_run_calls_store(tmp_path):
    store = MagicMock()
    store.load_csv.return_value = pd.DataFrame({"x": [1, 2]})
    DoubleFeatures().run(store, params={"raw_file": "data.csv", "processed_file": "features.parquet"})
    store.load_csv.assert_called_once_with("data.csv")
    store.save_parquet.assert_called_once()
    saved_df = store.save_parquet.call_args[0][0]
    assert list(saved_df["x"]) == [2, 4]


# --- Trainer ---

def test_trainer_fit():
    df = pd.DataFrame({"x": [1, 2]})
    model = ConstantTrainer().fit(df, {})
    assert model == {"weights": [1, 2, 3]}


def test_trainer_run_logs_and_saves(tmp_path):
    store = MagicMock()
    store.load_parquet.return_value = pd.DataFrame({"x": [1, 2]})
    store.models_dir = tmp_path

    tracker = MagicMock()
    tracker.run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    tracker.run.return_value.__exit__ = MagicMock(return_value=False)

    ConstantTrainer().run(store, tracker, params={"processed_file": "features.parquet"})
    tracker.run.assert_called_once()
    tracker.log_model.assert_called_once_with(
        {"weights": [1, 2, 3]}, artifact_path="model", flavour="sklearn"
    )


def test_trainer_model_flavour_override(tmp_path):
    class XGBTrainer(Trainer):
        model_flavour = "xgboost"
        def fit(self, df, params):
            return object()

    store = MagicMock()
    store.load_parquet.return_value = pd.DataFrame({"x": [1]})
    store.models_dir = tmp_path
    tracker = MagicMock()
    tracker.run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    tracker.run.return_value.__exit__ = MagicMock(return_value=False)

    XGBTrainer().run(store, tracker, params={})
    _, kwargs = tracker.log_model.call_args
    assert kwargs["flavour"] == "xgboost"


# --- Evaluator ---

def test_evaluator_evaluate():
    model = object()
    df = pd.DataFrame({"x": [1]})
    metrics = AccuracyEvaluator().evaluate(model, df)
    assert metrics["accuracy"] == pytest.approx(0.95)
    assert metrics["f1"] == pytest.approx(0.90)


def test_evaluator_run_writes_metrics_json(tmp_path):
    store = MagicMock()
    store.load_parquet.return_value = pd.DataFrame({"x": [1]})

    metrics_file = tmp_path / "metrics.json"
    AccuracyEvaluator().run(
        model=object(),
        store=store,
        params={"processed_file": "features.parquet", "metrics_file": str(metrics_file)},
    )

    data = json.loads(metrics_file.read_text())
    assert data["accuracy"] == pytest.approx(0.95)


def test_evaluator_run_returns_metrics():
    store = MagicMock()
    store.load_parquet.return_value = pd.DataFrame({"x": [1]})

    with patch("pathlib.Path.write_text"):
        result = AccuracyEvaluator().run(object(), store, params={})

    assert "accuracy" in result


# --- _resolve: flat vs nested params ---

def test_resolve_flat_params():
    params = {"processed_file": "flat.parquet"}
    assert _resolve(params, "processed_file", "default.parquet") == "flat.parquet"


def test_resolve_nested_features_section():
    params = {"features": {"processed_file": "nested.parquet"}}
    assert _resolve(params, "processed_file", "default.parquet") == "nested.parquet"


def test_resolve_nested_takes_priority_over_flat():
    params = {"processed_file": "flat.parquet", "features": {"processed_file": "nested.parquet"}}
    assert _resolve(params, "processed_file", "default.parquet") == "nested.parquet"


def test_resolve_falls_back_to_default():
    assert _resolve({}, "processed_file", "default.parquet") == "default.parquet"


def test_feature_builder_run_nested_params():
    store = MagicMock()
    store.load_csv.return_value = pd.DataFrame({"x": [1, 2]})
    params = {
        "features": {"raw_file": "train.csv", "processed_file": "features.parquet"},
        "model": {"target": "y"},
    }
    DoubleFeatures().run(store, params=params)
    store.load_csv.assert_called_once_with("train.csv")
    store.save_parquet.assert_called_once()


# --- Trainer metric contract (P0-008) ---

def test_trainer_run_logs_feature_importances_for_xgboost(tmp_path):
    """Trainer.run() calls _log_feature_importances after fit."""
    class XGBLikeModel:
        """Minimal stand-in for an XGBoost Booster."""
        def get_score(self, importance_type="gain"):
            return {"feature_a": 1.5, "feature_b": 0.8}

    class XGBLikeTrainer(Trainer):
        model_flavour = "xgboost"
        def fit(self, df, params):
            return XGBLikeModel()

    store = MagicMock()
    store.load_parquet.return_value = pd.DataFrame({"feature_a": [1], "feature_b": [2]})
    store.models_dir = tmp_path
    tracker = MagicMock()
    tracker.run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    tracker.run.return_value.__exit__ = MagicMock(return_value=False)

    with patch("mlflow.log_dict") as mock_log_dict:
        XGBLikeTrainer().run(store, tracker, params={})

    mock_log_dict.assert_called_once()
    logged = mock_log_dict.call_args[0][0]
    assert "feature_a" in logged


def test_trainer_run_logs_feature_importances_for_sklearn(tmp_path):
    """Trainer.run() logs importances for sklearn models with feature_names_in_."""
    import numpy as np

    class SklearnLikeModel:
        feature_importances_ = np.array([0.6, 0.4])
        feature_names_in_ = np.array(["col_a", "col_b"])

    class SklearnLikeTrainer(Trainer):
        def fit(self, df, params):
            return SklearnLikeModel()

    store = MagicMock()
    store.load_parquet.return_value = pd.DataFrame({"col_a": [1], "col_b": [2]})
    store.models_dir = tmp_path
    tracker = MagicMock()
    tracker.run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    tracker.run.return_value.__exit__ = MagicMock(return_value=False)

    with patch("mlflow.log_dict") as mock_log_dict:
        SklearnLikeTrainer().run(store, tracker, params={})

    mock_log_dict.assert_called_once()
    logged = mock_log_dict.call_args[0][0]
    assert "col_a" in logged
