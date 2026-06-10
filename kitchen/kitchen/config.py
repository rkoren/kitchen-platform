"""Typed configuration model for kitchen params.yaml files.

Load and validate with::

    from kitchen.config import KitchenConfig
    cfg = KitchenConfig.from_yaml("params.yaml")

Or validate inline::

    cfg = KitchenConfig(**yaml.safe_load(open("params.yaml")))
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataConfig(BaseModel):
    """``data:`` section — describes where raw data lives."""

    model_config = ConfigDict(extra="allow")

    source: Literal["kaggle", "s3", "local"]
    competition: str | None = None  # required when source=kaggle
    bucket: str | None = None  # required when source=s3
    prefix: str = ""  # s3 key prefix
    path: str | None = None  # required when source=local

    @model_validator(mode="after")
    def _validate_source_fields(self) -> "DataConfig":
        if self.source == "kaggle" and not self.competition:
            raise ValueError("data.competition is required when source is 'kaggle'")
        if self.source == "s3" and not self.bucket:
            raise ValueError("data.bucket is required when source is 's3'")
        if self.source == "local" and not self.path:
            raise ValueError("data.path is required when source is 'local'")
        return self


class MLflowConfig(BaseModel):
    """``mlflow:`` section — experiment tracking backend."""

    model_config = ConfigDict(extra="allow")

    tracking_uri: str = "sqlite:///mlruns.db"
    artifact_bucket: str | None = None


class MonitorConfig(BaseModel):
    """``monitor:`` section — drift monitoring configuration."""

    model_config = ConfigDict(extra="allow")

    reference_file: str = "reference.parquet"
    current_file: str = "current.parquet"
    report_bucket: str = ""
    report_key: str = "monitoring/drift_report.html"
    local_path: str = ""
    # MON-007: per-column p-value threshold; fail the run when the drifted share
    # reaches max_drift_share (only enforced when fail_on_drift is true).
    drift_threshold: float = 0.05
    fail_on_drift: bool = False
    max_drift_share: float = 0.5
    # MON-006: log drift summary metrics + the HTML/JSON report to MLflow.
    log_to_mlflow: bool = False
    mlflow_experiment: str | None = None  # default: "<experiment>-monitoring"

    @model_validator(mode="after")
    def _require_output(self) -> "MonitorConfig":
        if not self.report_bucket and not self.local_path:
            raise ValueError(
                "monitor config must specify at least one of: "
                "report_bucket (S3 upload) or local_path (local file)."
            )
        return self


class SubmissionConfig(BaseModel):
    """``submission:`` section — Kaggle submission configuration."""

    model_config = ConfigDict(extra="allow")

    id_col: str = "Id"
    target_col: str = "target"
    competition: str | None = None
    message: str = "kitchen submit"
    sample_submission: str = "sample_submission.csv"


class ThresholdSpec(BaseModel):
    """Explicit min/max threshold for a single metric.

    Use ``min`` for higher-is-better metrics (accuracy, AUC).
    Use ``max`` for lower-is-better metrics (logloss, RMSE).
    Both can be set to define a valid range.
    """

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> "ThresholdSpec":
        if self.min is None and self.max is None:
            raise ValueError("ThresholdSpec must specify at least one of: min, max")
        return self


class KitchenConfig(BaseModel):
    """Top-level model for params.yaml.

    Framework-owned sections (``data``, ``mlflow``, ``monitor``) are typed and
    validated.  Project-defined sections (``model``, ``features``, ``train``,
    etc.) are passed through without validation.
    """

    model_config = ConfigDict(extra="allow")

    experiment: str
    data: DataConfig | None = None
    mlflow: MLflowConfig = MLflowConfig()
    monitor: MonitorConfig | None = None
    submission: SubmissionConfig | None = None
    run_name: str | None = None
    metrics_file: str = "metrics.json"
    thresholds: dict[str, float | ThresholdSpec] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str = "params.yaml") -> "KitchenConfig":
        """Load and validate a params.yaml file."""
        import yaml

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
