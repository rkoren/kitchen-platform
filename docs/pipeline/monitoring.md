# Monitoring

Model and data health are monitored with [Evidently AI](https://www.evidentlyai.com). Reports are generated on a schedule and stored in `pipeline/monitoring/`.

## What gets monitored

| Report | Description |
|---|---|
| Data drift | Distribution shift between training reference and live data |
| Data quality | Missing values, out-of-range values, schema changes |
| Model performance | Accuracy, AUC degradation over time (requires ground truth labels) |

## How it works

The `monitor_flow.py` Prefect flow:

1. Loads the training dataset as the **reference** distribution
2. Loads the most recent batch of inference inputs as **current** data
3. Runs Evidently drift and quality reports
4. Saves HTML reports to `monitoring/` and optionally uploads to S3

<!-- TODO: add report upload + alerting on drift threshold breach -->

## Running manually

```bash
python pipeline/flows/monitor_flow.py
```

## Report output

<!-- TODO: add screenshot of an Evidently report once generated -->

Reports are saved to `pipeline/monitoring/` as HTML files:

```
monitoring/
├── data_drift_2024-01-15.html
└── data_quality_2024-01-15.html
```

## Alerting

<!-- TODO: document alerting strategy (SNS, Slack webhook, etc.) when drift exceeds threshold -->
