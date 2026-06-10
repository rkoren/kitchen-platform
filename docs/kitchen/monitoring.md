# Monitoring

Data drift is detected by a small in-house module (`kitchen.monitoring`) using
standard statistics — no heavy third-party dependency. Reports are HTML and stored
in `monitoring/`.

## What gets monitored

Per-column **data drift** between the training reference distribution and live data:

| Column type | Test | Magnitude |
|---|---|---|
| Numerical | Two-sample Kolmogorov–Smirnov | Population Stability Index (PSI) |
| Categorical | Chi-square | Population Stability Index (PSI) |

A column is flagged when its test p-value falls below the drift threshold
(default `0.05`); the dataset is flagged when at least half of the columns drift.

## How it works

The `monitor_flow.py` Prefect flow:

1. Loads the training dataset as the **reference** distribution
2. Loads the most recent batch of inference inputs as **current** data
3. Runs `DriftReport` to compute per-column drift (KS / chi-square / PSI)
4. Saves an HTML report to `monitoring/` and optionally uploads it to S3

## Using `DriftReport` directly

```python
from kitchen.monitoring import DriftReport

report = DriftReport(reference_df, current_df).run()          # auto-detects column types
report.save_html("monitoring/drift.html")

result = report.result                                        # DriftResult
if result.dataset_drift:
    print(f"{result.n_drifted}/{result.n_columns} columns drifted")
    result.to_dict()                                          # JSON-friendly, e.g. for MLflow

# Optional explicit column roles + custom threshold:
DriftReport(ref, cur, target="label", numerical=["age", "score"], drift_threshold=0.01)
```

## Failing on drift (CI gate)

Set `fail_on_drift` under the `monitor` key to turn monitoring into a pass/fail gate.
The report is always written first, then the run exits non-zero when the share of
drifted columns reaches `max_drift_share` — so a CI step can block on drift:

```yaml
monitor:
  reference_file: reference.parquet
  current_file: current.parquet
  local_path: monitoring/drift.html
  drift_threshold: 0.05      # per-column p-value below which a column counts as drifted
  fail_on_drift: true        # exit non-zero when the gate trips
  max_drift_share: 0.5       # share of drifted columns that trips the gate (default 0.5)
```

```bash
kitchen run monitor   # writes the report, then exits 1 if drift exceeds the gate
```

## Logging to MLflow

Set `log_to_mlflow` to record each monitor run in MLflow — the drift summary metrics
(`n_drifted`, `share_drifted`, `dataset_drift`, per-column `psi.*`) plus the HTML report
and a `drift.json` as artifacts, tagged `run_type=monitoring`. Runs go to a separate
experiment (`<experiment>-monitoring` by default) so they don't clutter the training
leaderboard:

```yaml
monitor:
  local_path: monitoring/drift.html
  log_to_mlflow: true
  mlflow_experiment: my-project-monitoring   # optional override
```

## Running manually

```bash
python -m kitchen.flows.monitor_flow
```

## Report output

Reports are saved to `monitoring/` as self-contained HTML files (a drift badge plus a
per-column table of test statistic, p-value, PSI, and drift flag):

```
monitoring/
└── data_drift_2024-01-15.html
```

## Alerting

<!-- TODO: document alerting strategy (SNS, Slack webhook, etc.) when drift exceeds threshold -->
