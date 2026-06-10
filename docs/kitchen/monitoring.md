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

## Scheduling drift checks

Monitoring is most useful on a cadence. Pick the trigger that matches your infrastructure
— all three run the same `kitchen run monitor` (which writes the report, optionally logs
to MLflow, and exits non-zero on drift when `fail_on_drift` is set).

### GitHub Actions cron (recommended)

No extra infrastructure — a scheduled workflow that runs the monitor and uploads the
report as an artifact. The job needs the reference and current data available, so fetch
it first (commit a small reference snapshot, `dvc pull`, or download from S3).

```yaml
# .github/workflows/monitor.yml
name: drift-monitor

on:
  schedule:
    - cron: "0 6 * * 1" # 06:00 UTC every Monday
  workflow_dispatch: # manual run from the Actions tab

jobs:
  monitor:
    runs-on: ubuntu-latest
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -e ".[dev]"
      # - run: dvc pull data/processed   # or download reference/current from S3
      - name: Run drift monitor
        run: kitchen run monitor # exits non-zero on drift if fail_on_drift is set
      - uses: actions/upload-artifact@v4
        if: always() # keep the report even when the run fails on drift
        with:
          name: drift-report
          path: monitoring/
```

### Prefect deployment

`kitchen.flows.monitor_flow` is already a Prefect `@flow`, so if you run a Prefect server
and work pool you can schedule it directly — create a deployment with a cron schedule and
serve it on the pool:

```python
from kitchen.flows.monitor_flow import monitor_pipeline

monitor_pipeline.serve(name="weekly-drift", cron="0 6 * * 1")
```

### EventBridge + Lambda (AWS-native)

For fully serverless scheduling, package the monitor as a Lambda (the same ECR image
pattern `recipes` provisions for serving) and trigger it with an EventBridge schedule
rule. Heaviest to set up; choose it when you already run on AWS and want no always-on
runner.

**Recommendation:** GitHub Actions cron for most projects — it needs no standing
infrastructure and keeps the report artifact and (optionally) the MLflow history in the
same place as the rest of the pipeline.

## Alerting

When `fail_on_drift` is set the scheduled run exits non-zero, so GitHub Actions surfaces
a failed check (and emails the configured recipients) on drift — the simplest alert. For
a push-style alert, add a step that fires on failure (e.g. a Slack incoming-webhook
`curl` guarded by `if: failure()`), or wire SNS/EventBridge on the Lambda path.
