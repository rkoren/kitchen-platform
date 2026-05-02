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

import yaml
from prefect import flow, task, get_run_logger

from kitchen.monitoring import DriftReport
from kitchen.store import DataStore


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
    log = get_run_logger()
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
        log.info("Drift report saved to %s", local_path)
        result = local_path
    if bucket:
        url = report.upload(bucket, key)
        log.info("Drift report uploaded to %s", url)
        result = url
    return result


@flow(name="kitchen-monitor")
def monitor_pipeline(params_file: str = "params.yaml") -> str:
    """Run drift detection: load reference + current data, generate Evidently report, save/upload."""
    with open(params_file) as f:
        params = yaml.safe_load(f)

    monitor_cfg = params.get("monitor", {})
    store = DataStore()

    reference = _load_reference(store, monitor_cfg.get("reference_file", "reference.parquet"))
    current = _load_current(store, monitor_cfg.get("current_file", "current.parquet"))
    report = _run_drift_report(reference, current)
    return _save_report(report, monitor_cfg)


if __name__ == "__main__":
    monitor_pipeline()
