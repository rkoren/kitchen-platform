# kitchen

`kitchen` is a reusable MLOps framework for Kaggle and modeling projects. It covers the full model lifecycle: data versioning, experiment tracking, model serving, drift monitoring, and orchestration.

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
                             [built-in] drift monitoring
```

## Stack

| Concern | Tool | Why |
|---|---|---|
| Data versioning | DVC + S3 | Git-native, file-level, ML-standard |
| Experiment tracking | MLflow | Open-source, self-hostable, AWS-native |
| Serving | FastAPI + Docker → Lambda | Portable containers, serverless deploy |
| Monitoring | Built-in (KS / χ² / PSI) | Dependency-free drift detection for tabular data |
| Orchestration | Prefect | Python-native, lightweight, modern DX |

## Dataset

kitchen aims to support any tabular Kaggle competition

<!-- TODO: document the specific dataset(s) used -->
