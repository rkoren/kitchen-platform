import boto3
import pandas as pd
import yaml
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report
from prefect import flow, task

from kitchen.store import DataStore


@task(name="load-reference")
def load_reference_data(store: DataStore, filename: str) -> pd.DataFrame:
    return store.load_parquet(filename, stage="processed")


@task(name="load-current")
def load_current_data(store: DataStore, filename: str) -> pd.DataFrame:
    return store.load_parquet(filename, stage="processed")


@task(name="drift-report")
def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column_mapping: ColumnMapping | None = None,
) -> Report:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    return report


@task(name="upload-report")
def upload_report(report: Report, bucket: str, key: str) -> str:
    html = report.get_html()
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=html.encode(), ContentType="text/html")
    return f"s3://{bucket}/{key}"


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
