"""Tests for KitchenConfig and sub-models."""
import pytest
import yaml
from pydantic import ValidationError

from kitchen.config import DataConfig, KitchenConfig, MLflowConfig, MonitorConfig


# --- KitchenConfig top-level ---

def test_minimal_valid_config():
    cfg = KitchenConfig(experiment="my-exp")
    assert cfg.experiment == "my-exp"
    assert cfg.mlflow.tracking_uri == "sqlite:///mlruns.db"
    assert cfg.data is None
    assert cfg.monitor is None


def test_experiment_required():
    with pytest.raises(ValidationError, match="experiment"):
        KitchenConfig()


def test_project_sections_pass_through():
    cfg = KitchenConfig(experiment="x", model={"depth": 5}, train={"lr": 0.01})
    assert cfg.model_extra["model"] == {"depth": 5}
    assert cfg.model_extra["train"] == {"lr": 0.01}


def test_from_yaml(tmp_path):
    params = {"experiment": "test-exp", "mlflow": {"tracking_uri": "sqlite:///test.db"}}
    p = tmp_path / "params.yaml"
    p.write_text(yaml.dump(params))
    cfg = KitchenConfig.from_yaml(str(p))
    assert cfg.experiment == "test-exp"
    assert cfg.mlflow.tracking_uri == "sqlite:///test.db"


# --- DataConfig ---

def test_kaggle_source_valid():
    cfg = DataConfig(source="kaggle", competition="titanic")
    assert cfg.competition == "titanic"


def test_kaggle_source_missing_competition():
    with pytest.raises(ValidationError, match="competition"):
        DataConfig(source="kaggle")


def test_s3_source_valid():
    cfg = DataConfig(source="s3", bucket="my-bucket", prefix="raw/")
    assert cfg.bucket == "my-bucket"


def test_s3_source_missing_bucket():
    with pytest.raises(ValidationError, match="bucket"):
        DataConfig(source="s3")


def test_local_source_valid():
    cfg = DataConfig(source="local", path="/data")
    assert cfg.path == "/data"


def test_local_source_missing_path():
    with pytest.raises(ValidationError, match="path"):
        DataConfig(source="local")


def test_unknown_source_rejected():
    with pytest.raises(ValidationError):
        DataConfig(source="gcs")


def test_data_extra_fields_allowed():
    cfg = DataConfig(source="kaggle", competition="titanic", raw_file="train.csv")
    assert cfg.model_extra["raw_file"] == "train.csv"


# --- MLflowConfig ---

def test_mlflow_defaults():
    cfg = MLflowConfig()
    assert cfg.tracking_uri == "sqlite:///mlruns.db"
    assert cfg.artifact_bucket is None


def test_mlflow_custom_uri():
    cfg = MLflowConfig(tracking_uri="http://localhost:5000")
    assert "5000" in cfg.tracking_uri


# --- MonitorConfig ---

def test_monitor_with_bucket():
    cfg = MonitorConfig(report_bucket="my-bucket")
    assert cfg.report_bucket == "my-bucket"


def test_monitor_with_local_path():
    cfg = MonitorConfig(local_path="/tmp/report.html")
    assert cfg.local_path == "/tmp/report.html"


def test_monitor_with_both():
    cfg = MonitorConfig(report_bucket="b", local_path="/tmp/r.html")
    assert cfg.report_bucket == "b"
    assert cfg.local_path == "/tmp/r.html"


def test_monitor_missing_output_raises():
    with pytest.raises(ValidationError, match="report_bucket.*local_path|local_path.*report_bucket"):
        MonitorConfig()


def test_monitor_defaults():
    cfg = MonitorConfig(report_bucket="b")
    assert cfg.reference_file == "reference.parquet"
    assert cfg.report_key == "monitoring/drift_report.html"


# --- Nested in KitchenConfig ---

def test_full_config_with_all_sections():
    cfg = KitchenConfig(
        experiment="titanic",
        data={"source": "kaggle", "competition": "spaceship-titanic"},
        mlflow={"tracking_uri": "sqlite:///mlruns.db"},
        monitor={"report_bucket": "my-bucket"},
        run_name="baseline",
        model={"n_estimators": 100},
    )
    assert cfg.data.source == "kaggle"
    assert cfg.monitor.report_bucket == "my-bucket"
    assert cfg.model_extra["model"]["n_estimators"] == 100


def test_invalid_data_section_propagates():
    with pytest.raises(ValidationError, match="competition"):
        KitchenConfig(experiment="x", data={"source": "kaggle"})
