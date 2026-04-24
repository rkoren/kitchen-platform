import mlflow
import pytest
from kitchen.tracking import Tracker, _flatten


def test_flatten_nested():
    assert _flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}


def test_flatten_already_flat():
    assert _flatten({"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_tracker_logs_run(tmp_path):
    uri = f"file://{tmp_path}"
    tracker = Tracker("test-exp", tracking_uri=uri)
    with tracker.run(run_name="r1", params={"lr": 0.1}):
        pass

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert len(runs) == 1
    assert runs.iloc[0]["params.lr"] == "0.1"


def test_tracker_log_metrics(tmp_path):
    uri = f"file://{tmp_path}"
    tracker = Tracker("test-exp", tracking_uri=uri)
    with tracker.run():
        tracker.log_metrics({"brier": 0.12, "auc": 0.88})

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert float(runs.iloc[0]["metrics.brier"]) == pytest.approx(0.12)


def test_tracker_log_model_unknown_flavour(tmp_path):
    uri = f"file://{tmp_path}"
    tracker = Tracker("test-exp", tracking_uri=uri)
    with tracker.run():
        with pytest.raises(ValueError, match="Unknown flavour"):
            tracker.log_model(object(), "model", flavour="tensorflow")
