"""Tests for FeatureBuilder, Trainer, and Evaluator contracts."""
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kitchen.steps import Evaluator, FeatureBuilder, Trainer


# --- Concrete implementations for testing ---

class DoubleFeatures(FeatureBuilder):
    def build(self, raw: pd.DataFrame) -> pd.DataFrame:
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
    result = DoubleFeatures().build(raw)
    assert list(result["a"]) == [2, 4]


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
    tracker.log_model.assert_called_once()


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
