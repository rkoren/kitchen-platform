"""Tests for kitchen.experiment."""

from unittest.mock import patch

from kitchen.experiment import ExperimentConfig, log_config


def test_experiment_config_defaults():
    config = ExperimentConfig(name="baseline")
    assert config.params == {}
    assert config.description == ""


def test_experiment_config_stores_params():
    config = ExperimentConfig(name="depth-5", params={"max_depth": 5, "eta": 0.01})
    assert config.params["max_depth"] == 5


def test_log_config_logs_params():
    config = ExperimentConfig(name="test-run", params={"lr": 0.01, "epochs": 10})
    with patch("kitchen.experiment.mlflow") as mock_mlflow:
        log_config(config)
        mock_mlflow.log_params.assert_called_once_with({"lr": 0.01, "epochs": 10})


def test_log_config_sets_name_tag():
    config = ExperimentConfig(name="my-run", params={})
    with patch("kitchen.experiment.mlflow") as mock_mlflow:
        log_config(config)
        mock_mlflow.set_tag.assert_any_call("experiment_name", "my-run")


def test_log_config_sets_description_tag():
    config = ExperimentConfig(name="x", params={}, description="testing depth")
    with patch("kitchen.experiment.mlflow") as mock_mlflow:
        log_config(config)
        mock_mlflow.set_tag.assert_any_call("description", "testing depth")


def test_log_config_skips_description_tag_when_empty():
    config = ExperimentConfig(name="x", params={})
    with patch("kitchen.experiment.mlflow") as mock_mlflow:
        log_config(config)
        calls = [c.args[0] for c in mock_mlflow.set_tag.call_args_list]
        assert "description" not in calls


def test_log_config_skips_log_params_when_empty():
    config = ExperimentConfig(name="x", params={})
    with patch("kitchen.experiment.mlflow") as mock_mlflow:
        log_config(config)
        mock_mlflow.log_params.assert_not_called()
