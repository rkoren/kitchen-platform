# light-ml-platform

**[Documentation](https://rkoren.github.io/light-ml-platform)**

## Modules

### [`recipes/`](recipes/) — IaC/CI/CD CLI
CLI for YAML -> Terraform configs for AWS resources

```bash
recipes generate infra.yaml --out ./tf
```

### [`pipeline/`](pipeline/) — MLOps Pipeline
ML pipeline for data (Kaggle competitions), covering data versioning, experiment tracking, model serving, drift monitoring, and orchestration.

| Component | Tool |
|---|---|
| Data versioning | DVC + S3 |
| Experiment tracking | MLflow |
| Serving | FastAPI + Docker → Lambda (ECR) |
| Monitoring | Evidently AI |
| Orchestration | Prefect |
