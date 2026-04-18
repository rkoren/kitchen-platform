# Experiment Tracking

Experiments are tracked with [MLflow](https://mlflow.org). Every training run logs parameters, metrics, and model artifacts.

## What gets tracked

| Item | Example |
|---|---|
| Parameters | `n_estimators`, `max_depth`, `learning_rate` |
| Metrics | `roc_auc`, `f1`, `precision`, `recall` |
| Artifacts | Trained model, feature importance plot, confusion matrix |
| Dataset version | DVC commit hash |

## MLflow server

<!-- TODO: document production MLflow server setup on AWS (EC2 + S3 artifact store) -->

For local development, run:

```bash
mlflow ui
# Open http://localhost:5000
```

For a shared remote server:

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export MLFLOW_S3_ENDPOINT_URL=...  # if using MinIO or custom S3
```

## Logging in training code

<!-- TODO: fill in once src/train/run.py is implemented -->

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({"roc_auc": score})
    mlflow.sklearn.log_model(model, "model")
```

## Model registry

<!-- TODO: document model promotion workflow (Staging → Production) -->
