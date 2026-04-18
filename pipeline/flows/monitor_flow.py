"""Prefect flow: drift detection and monitoring reports."""
from prefect import flow, task


@task
def load_reference_data():
    raise NotImplementedError


@task
def load_current_data():
    raise NotImplementedError


@task
def run_drift_report(reference, current):
    raise NotImplementedError


@task
def upload_report(report):
    raise NotImplementedError


@flow(name="monitor-pipeline")
def monitor_pipeline():
    reference = load_reference_data()
    current = load_current_data()
    report = run_drift_report(reference, current)
    upload_report(report)


if __name__ == "__main__":
    monitor_pipeline()
