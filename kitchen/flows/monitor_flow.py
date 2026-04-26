import yaml
from prefect import flow, task

from kitchen.monitoring import DriftReport
from kitchen.store import DataStore


@task(name="load-reference")
def load_reference_data(store: DataStore, filename: str) -> object:
    return store.load_parquet(filename, stage="processed")


@task(name="load-current")
def load_current_data(store: DataStore, filename: str) -> object:
    return store.load_parquet(filename, stage="processed")


@task(name="drift-report")
def run_drift_report(reference: object, current: object) -> DriftReport:
    return DriftReport(reference, current).run()


@task(name="upload-report")
def upload_report(report: DriftReport, bucket: str, key: str) -> str:
    return report.upload(bucket, key)


@flow(name="monitor")
def monitor_pipeline(params_file: str = "params.yaml") -> str:
    with open(params_file) as f:
        params = yaml.safe_load(f)

    store = DataStore()
    monitor_cfg = params.get("monitor", {})

    reference = load_reference_data(store, monitor_cfg.get("reference_file", "reference.parquet"))
    current = load_current_data(store, monitor_cfg.get("current_file", "current.parquet"))
    report = run_drift_report(reference, current)

    bucket = monitor_cfg.get("report_bucket", params.get("mlflow", {}).get("artifact_bucket", ""))
    key = monitor_cfg.get("report_key", "monitoring/drift_report.html")
    return upload_report(report, bucket, key)


if __name__ == "__main__":
    monitor_pipeline()
