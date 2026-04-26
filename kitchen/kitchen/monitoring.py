"""Drift monitoring wrapper around Evidently.

Usage::

    from kitchen.monitoring import DriftReport

    report = DriftReport(reference_df, current_df)
    report.run()
    url = report.upload(bucket="my-bucket", key="monitoring/report.html")

    # Optional column config:
    report = DriftReport(ref, cur, target="label", numerical=["age", "score"])
    report.run()
"""
from __future__ import annotations

import boto3
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report


class DriftReport:
    def __init__(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        *,
        target: str | None = None,
        numerical: list[str] | None = None,
        categorical: list[str] | None = None,
    ) -> None:
        """Prepare a drift report. Column mapping is only built when at least one column arg is given."""
        self.reference = reference
        self.current = current
        self._column_mapping: ColumnMapping | None = None
        if any(arg is not None for arg in (target, numerical, categorical)):
            self._column_mapping = ColumnMapping(
                target=target,
                numerical_features=numerical,
                categorical_features=categorical,
            )
        self._report: Report | None = None

    def run(self) -> DriftReport:
        """Compute the drift report. Returns self so calls can be chained."""
        self._report = Report(metrics=[DataDriftPreset()])
        self._report.run(
            reference_data=self.reference,
            current_data=self.current,
            column_mapping=self._column_mapping,
        )
        return self

    def as_html(self) -> str:
        """Render the report as an HTML string. Requires run() first."""
        if self._report is None:
            raise RuntimeError("Call run() before as_html()")
        return self._report.get_html()

    def upload(self, bucket: str, key: str) -> str:
        """Upload the HTML report to S3 and return the s3:// URI."""
        html = self.as_html()
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=html.encode(),
            ContentType="text/html",
        )
        return f"s3://{bucket}/{key}"
