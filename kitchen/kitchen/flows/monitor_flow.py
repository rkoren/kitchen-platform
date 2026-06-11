"""Generic drift monitoring flow for kitchen competition projects.

Run from the project root:
    python -m kitchen.flows.monitor_flow

Configure via params.yaml under the ``monitor`` key::

    monitor:
      reference_file: reference.parquet   # loaded from data/processed/
      current_file: current.parquet       # loaded from data/processed/
      local_path: monitoring/drift.html   # write locally (optional)
      report_bucket: my-bucket            # upload to S3 (optional)
      report_key: monitoring/drift.html   # S3 key (default: monitoring/drift_report.html)
      drift_threshold: 0.05               # per-column p-value threshold (default 0.05)
      fail_on_drift: true                 # exit non-zero when drift exceeds the share gate
      max_drift_share: 0.5                # drifted-column share that trips the gate (default 0.5)
      log_to_mlflow: true                 # log drift metrics + report artifacts to MLflow
      mlflow_experiment: my-monitoring    # MLflow experiment (default: <experiment>-monitoring)

At least one of ``local_path`` or ``report_bucket`` must be provided. With
``fail_on_drift: true`` the report is still written, then the run fails if the
share of drifted columns reaches ``max_drift_share``.
"""

from __future__ import annotations

import logging

import yaml

from kitchen.monitoring import DriftReport
from kitchen.store import DataStore

_log = logging.getLogger(__name__)


class DriftThresholdExceeded(RuntimeError):
    """Raised (after the report is written) when drift exceeds the configured gate."""

    def __init__(self, message: str, report_path: str = "") -> None:
        super().__init__(message)
        self.report_path = report_path


def _validate_output(monitor_cfg: dict) -> None:
    """Fail immediately if no output destination is configured."""
    if not monitor_cfg.get("report_bucket") and not monitor_cfg.get("local_path"):
        raise ValueError(
            "No output configured for monitoring. "
            "Run with --local report.html, or add 'local_path' or 'report_bucket' "
            "under the 'monitor' key in params.yaml."
        )


def _load_reference(store: DataStore, filename: str) -> object:
    return store.load_parquet(filename, stage="processed")


def _load_current(store: DataStore, filename: str) -> object:
    return store.load_parquet(filename, stage="processed")


def _run_drift_report(
    reference: object, current: object, drift_threshold: float = 0.05
) -> DriftReport:
    return DriftReport(reference, current, drift_threshold=drift_threshold).run()


def _safe_metric_key(name: str) -> str:
    """Sanitize a column name into a valid MLflow metric key."""
    import re

    return re.sub(r"[^a-zA-Z0-9_\-. :/]", "_", str(name))


def _log_to_mlflow(report: DriftReport, monitor_cfg: dict, params: dict) -> None:
    """MON-006: log drift summary metrics + the HTML/JSON report to MLflow."""
    if not monitor_cfg.get("log_to_mlflow"):
        return
    import json
    import tempfile
    from pathlib import Path

    import mlflow

    from kitchen.tracking import Tracker, configure_from_env

    configure_from_env()
    experiment = monitor_cfg.get("mlflow_experiment") or f"{params.get('experiment', 'kitchen')}-monitoring"
    result = report.result

    metrics: dict[str, float] = {
        "n_columns": float(result.n_columns),
        "n_drifted": float(result.n_drifted),
        "share_drifted": float(result.share_drifted),
        "dataset_drift": 1.0 if result.dataset_drift else 0.0,
    }
    for col in result.columns:
        metrics[f"psi.{_safe_metric_key(col.column)}"] = float(col.psi)

    with Tracker(experiment).run(run_name="drift-monitor"):
        mlflow.set_tag("run_type", "monitoring")
        Tracker.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as td:
            html_path = Path(td) / "drift_report.html"
            html_path.write_text(report.as_html(), encoding="utf-8")
            mlflow.log_artifact(str(html_path))
            json_path = Path(td) / "drift.json"
            json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            mlflow.log_artifact(str(json_path))
    _log.info("Logged drift report to MLflow experiment %s", experiment)


def _enforce_drift_gate(report: DriftReport, monitor_cfg: dict, report_path: str) -> None:
    """MON-007: fail the run when drift exceeds the configured share gate."""
    if not monitor_cfg.get("fail_on_drift"):
        return
    max_share = float(monitor_cfg.get("max_drift_share", 0.5))
    result = report.result
    if result.share_drifted >= max_share:
        raise DriftThresholdExceeded(
            f"data drift exceeded threshold: {result.n_drifted}/{result.n_columns} "
            f"columns drifted (share {result.share_drifted:.0%} >= {max_share:.0%})",
            report_path=report_path,
        )


def _drift_summary(result: object) -> str:
    """One-line drift summary for stdout (CBB-008), e.g.::

    ``Drift: 7/48 columns drifted (14.6%) — dataset_drift=False``
    """
    return (
        f"Drift: {result.n_drifted}/{result.n_columns} columns drifted "
        f"({result.share_drifted:.1%}) — dataset_drift={result.dataset_drift}"
    )


def _save_report(report: DriftReport, monitor_cfg: dict) -> str:
    bucket = monitor_cfg.get("report_bucket", "")
    key = monitor_cfg.get("report_key", "monitoring/drift_report.html")
    local_path = monitor_cfg.get("local_path", "")

    if not bucket and not local_path:
        raise ValueError(
            "monitor config must specify at least one of: "
            "report_bucket (S3 upload) or local_path (local file). "
            "Add one of these keys under 'monitor' in params.yaml."
        )

    result = ""
    if local_path:
        report.save_html(local_path)
        _log.info("Drift report saved to %s", local_path)
        result = local_path
    if bucket:
        url = report.upload(bucket, key)
        _log.info("Drift report uploaded to %s", url)
        result = url
    return result


def monitor_pipeline(params_file: str = "params.yaml", local_path_override: str | None = None) -> str:
    """Run drift detection: load reference + current data, generate the drift report, save/upload.

    Plain Python (no Prefect) so monitoring doesn't depend on a running Prefect
    server or risk the stale ``~/.prefect/prefect.db`` Alembic-migration crash —
    the same SCF-014 treatment applied to ``train_flow`` (CBB-009).
    """
    with open(params_file, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    monitor_cfg = params.get("monitor", {})
    if local_path_override:
        monitor_cfg["local_path"] = local_path_override
    _validate_output(monitor_cfg)
    store = DataStore()

    reference = _load_reference(store, monitor_cfg.get("reference_file", "reference.parquet"))
    current = _load_current(store, monitor_cfg.get("current_file", "current.parquet"))
    report = _run_drift_report(reference, current, float(monitor_cfg.get("drift_threshold", 0.05)))
    out = _save_report(report, monitor_cfg)
    _log_to_mlflow(report, monitor_cfg, params)  # no-op unless monitor.log_to_mlflow
    _enforce_drift_gate(report, monitor_cfg, out)  # raises after the report is written
    # Success path: surface the drift summary on stdout (CBB-008). The fail-on-drift
    # path already carries the counts in the DriftThresholdExceeded message.
    print(_drift_summary(report.result))
    return out


if __name__ == "__main__":
    monitor_pipeline()
