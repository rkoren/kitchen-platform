# Quickstart

## Prerequisites

- Python 3.11+
- AWS CLI configured (`aws configure`)
- DVC with S3 remote access
- Docker (for model serving)

## Installation

```bash
cd light-ml-platform/pipeline
pip install -e .
```

## Configure DVC remote

```bash
dvc remote add -d myremote s3://your-bucket/dvc-store
dvc remote modify myremote region us-east-1
```

## Run the pipeline

```bash
# Run all stages
dvc repro

# Run a specific stage
dvc repro train
```

## Track experiments with MLflow

```bash
# Start the MLflow tracking server (local)
mlflow ui

# Or point to a remote server
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

<!-- TODO: add MLflow server setup instructions for AWS -->

## Serve the model locally

```bash
cd src/serve
uvicorn app:app --reload
# POST http://localhost:8000/predict
```

## Deploy to Lambda

<!-- TODO: document the ECR push + Lambda update steps -->

!!! note "Coming soon"
    Deployment instructions will be added once the serving module is complete.
