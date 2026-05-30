# From Scratch to Deployed Lambda

An annotated walkthrough of the complete kitchen-platform workflow: scaffold a project, train a model, and serve it on AWS Lambda. This is the long form of the [AWS Deployment Quickstart](aws-deployment.md) — every step is explained, not just listed.

Estimated time: 60–90 minutes (not counting model training time).

---

## What you'll build

A FastAPI inference service running on Lambda that:

- Accepts single predictions on `POST /predict`
- Accepts batch predictions on `POST /predict/batch`
- Reports model metadata on `GET /metadata`
- Validates inputs against a Pydantic schema
- Auto-redeploys via CI on every push to `main`

---

## Prerequisites

| Tool | Why |
|---|---|
| Python 3.11+ | kitchen requires 3.11+ |
| AWS CLI | Authenticate and interact with AWS |
| Terraform ≥ 1.0 | Apply the generated infrastructure configs |
| Docker | Build the Lambda container image |
| Git | kitchen promote needs git history |
| Kaggle account | If your data source is Kaggle |

Install kitchen and recipes:

```bash
git clone https://github.com/rkoren/kitchen-platform.git
pip install -e kitchen-platform/kitchen -e kitchen-platform/recipes
```

---

## Part 1 — Scaffold the project

```bash
kitchen init my-model \
  --source kaggle \
  --competition my-competition \
  --template baseline-xgb \
  --ci
```

This creates a `my-model/` directory with:

- `params.yaml` — single source of truth for all training config
- `src/features/run.py` — feature engineering stub
- `src/train/run.py` — XGBoost baseline (because `--template baseline-xgb`)
- `src/evaluate/run.py` — evaluation stub
- `src/serve/predictor.py` — inference stub with MLflow load example
- `.env.example` — credential template
- `.github/workflows/train-evaluate.yml` — CI workflow

```bash
cd my-model
cp .env.example .env
# Edit .env: add KAGGLE_USERNAME and KAGGLE_KEY
kitchen check    # should pass all checks
```

`kitchen check` verifies Python version, CLI tools, credential presence, and `params.yaml` validity. Fix anything it flags before continuing.

---

## Part 2 — Get data and build features

```bash
kitchen ingest                  # downloads data/raw/ from Kaggle
python src/features/run.py      # builds data/processed/features.parquet
```

Open `src/features/run.py` and implement the `build()` function. The stub shows the expected signature:

```python
def build(params: dict, store: DataStore) -> None:
    raw = store.load_raw("train.csv")           # reads data/raw/train.csv
    # ... feature engineering ...
    store.save_processed(features, "features.parquet")
```

`store` handles paths so you never hard-code `data/raw/` or `data/processed/`.

---

## Part 3 — Train and track experiments

```bash
kitchen run train
```

Every run is automatically logged to MLflow with parameters, metrics, and the trained model artifact. View runs:

```bash
kitchen ui          # opens MLflow UI in the browser
kitchen leaderboard # ranked table in the terminal
```

Iterate: edit hyperparameters in `params.yaml`, re-run `kitchen run train`, compare on `kitchen leaderboard`. When you have a run you're happy with:

```bash
kitchen promote val_accuracy    # registers best run as champion in MLflow registry
```

---

## Part 4 — Implement the predictor

Open `src/serve/predictor.py`. The stub shows how to load the champion from the registry:

```python
import mlflow
from pydantic import BaseModel

model = mlflow.sklearn.load_model("models:/my-model@champion")

FEATURES = ["feature_a", "feature_b", "feature_c"]

class RequestModel(BaseModel):
    feature_a: float
    feature_b: str
    feature_c: float

class ResponseModel(BaseModel):
    prediction: int
    probability: float

def predict(payload: dict) -> dict:
    import pandas as pd
    X = pd.DataFrame([payload])[FEATURES]
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    return {"prediction": pred, "probability": prob}
```

Test it locally before deploying:

```bash
kitchen serve local
# → http://localhost:8000

curl http://localhost:8000/health
curl http://localhost:8000/docs     # OpenAPI schema generated from your Pydantic models
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"feature_a": 1.5, "feature_b": "x", "feature_c": 0.3}'
```

If `predictor.py` has any issues (import errors, missing `predict`, wrong model path), `kitchen serve local` surfaces the error immediately with a full traceback — much faster than discovering it after a Lambda deploy.

---

## Part 5 — Provision AWS infrastructure

Create `infra.yaml` in the project root:

```yaml
name: my-model
region: us-east-1

resources:
  - type: ecr
    name: my-model-serve
    scan_on_push: true
    lambda_access: true        # allows Lambda to pull from this repo

  - type: iam_role
    name: my-model-exec
    service: lambda.amazonaws.com
    policies:
      - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      - arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess  # for MLflow artifacts

  - type: lambda
    name: my-model-serve
    role: my-model-exec
    ecr_repo: my-model-serve   # references the ECR resource above — no hard-coded URL
    memory: 1024               # model loading needs memory; tune down after benchmarking
    timeout: 30
    environment:
      KITCHEN_MODEL_NAME: my-model
      KITCHEN_PREDICTOR_DIR: /var/task/src/serve
```

Generate and apply:

```bash
recipes validate infra.yaml          # catches schema errors before Terraform sees them
recipes generate infra.yaml --out tf
cd tf && terraform init && terraform apply
cd ..
```

Terraform creates the ECR repository and Lambda function. The Lambda can't run yet — it has no image. That's the next step.

---

## Part 6 — Build and push the Docker image

The CI workflow (`kitchen init --ci`) scaffolds a `Dockerfile`. If you're doing this manually:

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
ECR_URL="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/my-model-serve"

# Log in to ECR (token expires after 12 hours)
aws ecr get-login-password --region ${REGION} \
  | docker login --username AWS --password-stdin ${ECR_URL}

# Build and push
docker build -t my-model-serve:latest .
docker tag my-model-serve:latest ${ECR_URL}:latest
docker push ${ECR_URL}:latest
```

Update the Lambda to use the new image:

```bash
aws lambda update-function-code \
  --function-name my-model-serve \
  --image-uri ${ECR_URL}:latest

aws lambda wait function-updated --function-name my-model-serve
```

---

## Part 7 — Enable a Function URL and test

In the AWS console, go to **Lambda → my-model-serve → Configuration → Function URL** and create a URL (auth type: `NONE` for public, `AWS_IAM` for authenticated). Copy the URL.

```bash
URL="https://abc123.lambda-url.us-east-1.on.aws"

curl ${URL}/health
# {"status": "ok"}

curl ${URL}/metadata
# {"model_name": "my-model", "model_version": null, "git_sha": null, "features": [...]}

curl -X POST ${URL}/predict \
     -H "Content-Type: application/json" \
     -d '{"feature_a": 1.5, "feature_b": "x", "feature_c": 0.3}'

# Batch
curl -X POST ${URL}/predict/batch \
     -H "Content-Type: application/json" \
     -d '{"items": [{"feature_a": 1.5, "feature_b": "x", "feature_c": 0.3}]}'
```

---

## Part 8 — Set up CI for automatic redeploys

The CI workflow scaffolded by `kitchen init --ci` includes a `deploy` job. What it does:

1. Builds the Docker image tagged with `$GITHUB_SHA` (immutable, traceable)
2. Pushes to ECR
3. Updates the Lambda function

Secrets/variables needed in GitHub:

| Name | Type | Value |
|---|---|---|
| `AWS_ACCOUNT_ID` | Variable | Your 12-digit AWS account ID |
| `AWS_REGION` | Variable | `us-east-1` |
| `ECR_REPO_NAME` | Variable | `my-model-serve` |
| `LAMBDA_FUNCTION_NAME` | Variable | `my-model-serve` |

For authentication, configure OIDC (no long-lived keys required):

1. Create an IAM OIDC identity provider for `token.actions.githubusercontent.com`
2. Create an IAM role `github-actions` that trusts the OIDC provider, with ECR push and Lambda update permissions
3. Set `vars.AWS_ACCOUNT_ID` in GitHub

After this, every merge to `main` automatically builds, pushes, and redeploys. The `$GITHUB_SHA` tag means every Lambda image is traceable to an exact commit — it shows up in `GET /metadata` as `git_sha`.

---

## What's next

- Retrain when your model gets stale: `kitchen run train --auto-promote --promote-metric val_accuracy`
- Monitor for drift: `kitchen run monitor` (configure `monitor:` in `params.yaml` first)
- Submit to Kaggle: `kitchen run submit` (Kaggle projects)
- See [Troubleshooting](troubleshooting.md) if anything went wrong
