"""Lightweight data-drift monitoring — no heavy third-party dependency.

Compares a *reference* dataset (e.g. the training distribution) against a
*current* dataset (recent production inputs) and flags per-column drift using
standard statistics: the two-sample Kolmogorov–Smirnov test for numerical
columns, the chi-square test for categorical columns, and the Population
Stability Index (PSI) as a magnitude measure for both. Renders a self-contained
HTML report and can upload it to S3.

Usage::

    from kitchen.monitoring import DriftReport

    report = DriftReport(reference_df, current_df)
    report.run()
    url = report.upload(bucket="my-bucket", key="monitoring/report.html")

    # Optional column config (otherwise dtypes are auto-detected):
    report = DriftReport(ref, cur, target="label", numerical=["age", "score"])
    report.run()
    if report.result.dataset_drift:
        ...

This replaces the previous Evidently-based implementation; the public API
(``run`` / ``as_html`` / ``save_html`` / ``upload``) is unchanged.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd

#: p-value below which a column's KS / chi-square test counts as drifted.
_DEFAULT_DRIFT_THRESHOLD = 0.05
#: Share of drifted columns at or above which the dataset is flagged as drifted.
_DATASET_DRIFT_SHARE = 0.5
_PSI_BINS = 10
_EPS = 1e-6


@dataclass
class _ColumnConfig:
    """Optional explicit column roles; otherwise dtypes are auto-detected."""

    target: str | None = None
    numerical: list[str] | None = None
    categorical: list[str] | None = None


@dataclass
class ColumnDrift:
    """Drift result for a single column."""

    column: str
    kind: str  # "numerical" | "categorical"
    test: str  # "ks" | "chi2"
    statistic: float
    p_value: float
    psi: float
    drifted: bool


@dataclass
class DriftResult:
    """Aggregate drift result; convenient for thresholding and MLflow logging."""

    columns: list[ColumnDrift] = field(default_factory=list)
    drift_threshold: float = _DEFAULT_DRIFT_THRESHOLD

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    @property
    def n_drifted(self) -> int:
        return sum(1 for c in self.columns if c.drifted)

    @property
    def share_drifted(self) -> float:
        return self.n_drifted / self.n_columns if self.columns else 0.0

    @property
    def dataset_drift(self) -> bool:
        return self.share_drifted >= _DATASET_DRIFT_SHARE

    def to_dict(self) -> dict:
        return {
            "n_columns": self.n_columns,
            "n_drifted": self.n_drifted,
            "share_drifted": self.share_drifted,
            "dataset_drift": self.dataset_drift,
            "columns": [c.__dict__ for c in self.columns],
        }


def _psi(ref_props: np.ndarray, cur_props: np.ndarray) -> float:
    """Population Stability Index between two proportion vectors."""
    ref_p = np.clip(ref_props, _EPS, None)
    cur_p = np.clip(cur_props, _EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _numerical_drift(ref: pd.Series, cur: pd.Series, threshold: float) -> tuple[str, float, float, float, bool]:
    from scipy import stats

    r = ref.dropna().to_numpy(dtype=float)
    c = cur.dropna().to_numpy(dtype=float)
    if r.size == 0 or c.size == 0:
        return "ks", 0.0, 1.0, 0.0, False
    try:
        stat, p = stats.ks_2samp(r, c)
    except Exception:  # noqa: BLE001 — never let one bad column break the report
        return "ks", 0.0, 1.0, 0.0, False

    # PSI over reference-quantile bins.
    edges = np.unique(np.quantile(r, np.linspace(0, 1, _PSI_BINS + 1)))
    psi = 0.0
    if edges.size >= 2:
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts = np.histogram(r, bins=edges)[0].astype(float)
        cur_counts = np.histogram(c, bins=edges)[0].astype(float)
        if ref_counts.sum() and cur_counts.sum():
            psi = _psi(ref_counts / ref_counts.sum(), cur_counts / cur_counts.sum())
    return "ks", float(stat), float(p), psi, bool(p < threshold)


def _categorical_drift(ref: pd.Series, cur: pd.Series, threshold: float) -> tuple[str, float, float, float, bool]:
    from scipy import stats

    cats = sorted(set(ref.dropna().unique()) | set(cur.dropna().unique()), key=str)
    if not cats:
        return "chi2", 0.0, 1.0, 0.0, False
    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts()
    r = np.array([float(ref_counts.get(cat, 0)) for cat in cats])
    c = np.array([float(cur_counts.get(cat, 0)) for cat in cats])

    psi = 0.0
    if r.sum() and c.sum():
        psi = _psi(r / r.sum(), c / c.sum())

    table = np.array([r, c])
    table = table[:, table.sum(axis=0) > 0]  # drop categories absent from both
    if table.shape[1] < 2 or r.sum() == 0 or c.sum() == 0:
        return "chi2", 0.0, 1.0, psi, False
    try:
        stat, p, _, _ = stats.chi2_contingency(table)
    except Exception:  # noqa: BLE001
        return "chi2", 0.0, 1.0, psi, False
    return "chi2", float(stat), float(p), psi, bool(p < threshold)


class DriftReport:
    def __init__(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        *,
        target: str | None = None,
        numerical: list[str] | None = None,
        categorical: list[str] | None = None,
        drift_threshold: float = _DEFAULT_DRIFT_THRESHOLD,
    ) -> None:
        """Prepare a drift report. Column mapping is only built when at least one column arg is given."""
        self.reference = reference
        self.current = current
        self.drift_threshold = drift_threshold
        self._column_mapping: _ColumnConfig | None = None
        if any(arg is not None for arg in (target, numerical, categorical)):
            self._column_mapping = _ColumnConfig(
                target=target, numerical=numerical, categorical=categorical
            )
        self._result: DriftResult | None = None

    def _classify_columns(self) -> tuple[list[str], list[str]]:
        common = [col for col in self.reference.columns if col in self.current.columns]
        cfg = self._column_mapping
        if cfg is not None and (cfg.numerical or cfg.categorical):
            num = [c for c in (cfg.numerical or []) if c in common]
            cat = [c for c in (cfg.categorical or []) if c in common]
            return num, cat
        num, cat = [], []
        for col in common:
            if pd.api.types.is_numeric_dtype(self.reference[col]):
                num.append(col)
            else:
                cat.append(col)
        return num, cat

    def run(self) -> DriftReport:
        """Compute the drift report. Returns self so calls can be chained."""
        numerical, categorical = self._classify_columns()
        columns: list[ColumnDrift] = []
        for col in numerical:
            test, stat, p, psi, drifted = _numerical_drift(
                self.reference[col], self.current[col], self.drift_threshold
            )
            columns.append(ColumnDrift(col, "numerical", test, stat, p, psi, drifted))
        for col in categorical:
            test, stat, p, psi, drifted = _categorical_drift(
                self.reference[col], self.current[col], self.drift_threshold
            )
            columns.append(ColumnDrift(col, "categorical", test, stat, p, psi, drifted))
        self._result = DriftResult(columns=columns, drift_threshold=self.drift_threshold)
        return self

    @property
    def result(self) -> DriftResult:
        """The computed :class:`DriftResult`. Requires run() first."""
        if self._result is None:
            raise RuntimeError("Call run() before accessing result")
        return self._result

    def as_html(self) -> str:
        """Render the report as a self-contained HTML string. Requires run() first."""
        if self._result is None:
            raise RuntimeError("Call run() before as_html()")
        res = self._result
        drift_color = "#c0392b" if res.dataset_drift else "#27ae60"
        status = "DRIFT DETECTED" if res.dataset_drift else "STABLE"
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        rows = []
        for c in sorted(res.columns, key=lambda c: (not c.drifted, c.column)):
            flag = "✓" if c.drifted else ""
            row_bg = "#fdecea" if c.drifted else "#ffffff"
            rows.append(
                f'<tr style="background:{row_bg}">'
                f"<td>{html.escape(str(c.column))}</td>"
                f"<td>{c.kind}</td><td>{c.test}</td>"
                f"<td>{c.statistic:.4f}</td><td>{c.p_value:.4f}</td><td>{c.psi:.4f}</td>"
                f'<td style="text-align:center">{flag}</td></tr>'
            )

        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Drift report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #2c3e50; }}
  .badge {{ display:inline-block; padding:.25rem .75rem; border-radius:4px;
            color:#fff; background:{drift_color}; font-weight:600; }}
  table {{ border-collapse: collapse; margin-top: 1rem; width: 100%; }}
  th, td {{ border: 1px solid #e1e4e8; padding: .4rem .6rem; text-align: left; }}
  th {{ background: #f6f8fa; }}
  .summary {{ margin-top: .5rem; color: #586069; }}
</style></head><body>
<h1>Data drift report</h1>
<p><span class="badge">{status}</span></p>
<p class="summary">{res.n_drifted} of {res.n_columns} columns drifted
   (share {res.share_drifted:.0%}; threshold p &lt; {res.drift_threshold}). Generated {generated}.</p>
<table>
  <tr><th>Column</th><th>Type</th><th>Test</th><th>Statistic</th><th>p-value</th><th>PSI</th><th>Drift</th></tr>
  {''.join(rows)}
</table>
</body></html>"""

    def save_html(self, path: str) -> None:
        """Write the HTML report to a local file. Requires run() first."""
        import pathlib

        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_text(self.as_html(), encoding="utf-8")

    def upload(self, bucket: str, key: str) -> str:
        """Upload the HTML report to S3 and return the s3:// URI."""
        html_doc = self.as_html()
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=html_doc.encode(),
            ContentType="text/html",
        )
        return f"s3://{bucket}/{key}"
