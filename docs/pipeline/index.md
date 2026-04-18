# pipeline

`pipeline` is an end-to-end MLOps pipeline for healthcare tabular data (Kaggle competitions). It covers the full model lifecycle: data versioning, experiment tracking, model serving, drift monitoring, and orchestration.

## Architecture

```
Kaggle data
    │
    ▼
[DVC] ingest ──► features ──► train ──► evaluate
                                  │
                             [MLflow] experiment tracking
                                  │
                             [Prefect] orchestration
                                  │
                             [FastAPI + Docker]
                                  │
                             Lambda (ECR)
                                  │
                             [Evidently] drift monitoring
```

## Stack

| Concern | Tool | Why |
|---|---|---|
| Data versioning | DVC + S3 | Git-native, file-level, ML-standard |
| Experiment tracking | MLflow | Open-source, self-hostable, AWS-native |
| Serving | FastAPI + Docker → Lambda | Portable containers, serverless deploy |
| Monitoring | Evidently AI | ML-specific drift detection for tabular data |
| Orchestration | Prefect | Python-native, lightweight, modern DX |

## Dataset

The pipeline targets healthcare tabular datasets from Kaggle competitions.

<!-- TODO: document the specific dataset(s) used -->
