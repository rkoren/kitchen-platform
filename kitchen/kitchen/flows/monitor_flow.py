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

At least one of ``local_path`` or ``report_bucket`` must be provided.
"""

from __future__ import annotations

import logging

import yaml
from prefect import flow, task

from kitchen.monitoring import DriftReport
from kitchen.store import DataStore

_log = logging.getLogger(__name__)


def _validate_output(monitor_cfg: dict) -> None:
    """Fail immediately if no output destination is configured."""
    if not monitor_cfg.get("report_bucket") and not monitor_cfg.get("local_path"):
        raise ValueError(
            "No output configured for monitoring. "
            "Run with --local report.html, or add 'local_path' or 'report_bucket' "
            "under the 'monitor' key in params.yaml."
        )


@task(name="load-reference")
def _load_reference(store: DataStore, filename: str) -> object:
    return store.load_parquet(filename, stage="processed")


@task(name="load-current")
def _load_current(store: DataStore, filename: str) -> object:
    return store.load_parquet(filename, stage="processed")


@task(name="drift-report")
def _run_drift_report(reference: object, current: object) -> DriftReport:
    return DriftReport(reference, current).run()


@task(name="save-report")
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


@flow(name="kitchen-monitor")
def monitor_pipeline(params_file: str = "params.yaml", local_path_override: str | None = None) -> str:
    """Run drift detection: load reference + current data, generate Evidently report, save/upload."""
    with open(params_file, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    monitor_cfg = params.get("monitor", {})
    if local_path_override:
        monitor_cfg["local_path"] = local_path_override
    _validate_output(monitor_cfg)
    store = DataStore()

    reference = _load_reference(store, monitor_cfg.get("reference_file", "reference.parquet"))
    current = _load_current(store, monitor_cfg.get("current_file", "current.parquet"))
    report = _run_drift_report(reference, current)
    return _save_report(report, monitor_cfg)


if __name__ == "__main__":
    monitor_pipeline()
