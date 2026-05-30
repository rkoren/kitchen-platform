# Model Serving

The trained model is served via a [FastAPI](https://fastapi.tiangolo.com) application, packaged as a Docker container, and deployed to AWS Lambda via ECR.

## Architecture

```
ECR image → Lambda function URL
                │
         FastAPI + Mangum
                │
    ┌───────────┼────────────────┐
    ▼           ▼                ▼
GET /health  GET /metadata   POST /predict
                             POST /predict/batch
```

[Mangum](https://mangum.io) adapts the FastAPI ASGI app to the Lambda event/response format.

---

## API reference

### `GET /health`

Health check. Always returns 200 if the container is running.

```json
{"status": "ok"}
```

---

### `GET /metadata`

Returns model metadata. All fields may be `null` if not configured.

```json
{
  "model_name": "my-project",
  "model_version": "3",
  "git_sha": "a1b2c3d",
  "features": ["home_court", "elo_diff", "pace"]
}
```

| Field | Source |
|---|---|
| `model_name` | `KITCHEN_MODEL_NAME` environment variable |
| `model_version` | `KITCHEN_MODEL_VERSION` environment variable |
| `git_sha` | `GITHUB_SHA` env var → `GIT_SHA` env var → `git rev-parse --short HEAD` |
| `features` | `FEATURES` list exported from `predictor.py` |

Set `KITCHEN_MODEL_NAME` and `KITCHEN_MODEL_VERSION` in the Lambda environment (via the `environment:` block in your `recipes` spec, or manually in the console).

---

### `POST /predict`

Run inference on a single observation.

**Untyped (default)** — accepts any JSON object, returns any JSON object:

```json
// Request
{"home_court": 1, "elo_diff": 120.5, "pace": 68.2}

// Response
{"prediction": 1, "probability": 0.87}
```

**Typed** — when `predictor.py` exports both `RequestModel` and `ResponseModel` (Pydantic `BaseModel` subclasses), FastAPI validates the request and response and generates full OpenAPI schema:

```python
# src/serve/predictor.py
from pydantic import BaseModel

class RequestModel(BaseModel):
    home_court: int
    elo_diff: float
    pace: float

class ResponseModel(BaseModel):
    prediction: int
    probability: float

def predict(payload: dict) -> dict:
    ...
```

With typed models, invalid input returns **422 Unprocessable Entity** with a structured validation error.

---

### `POST /predict/batch`

Run inference on multiple observations in a single request. Calls `predict()` once per item; results preserve input order.

**Untyped:**

```json
// Request
{"items": [{"home_court": 1, "elo_diff": 120.5}, {"home_court": 0, "elo_diff": -45.2}]}

// Response
{"results": [{"prediction": 1, "probability": 0.87}, {"prediction": 0, "probability": 0.31}]}
```

**Typed:** when both models are present, each item in `items` is validated against `RequestModel` and each result is validated against `ResponseModel`.

**Batch size cap:** defaults to 1000 items. Override with the `KITCHEN_BATCH_MAX_ITEMS` environment variable. Requests exceeding the limit return **413 Request Entity Too Large**.

---

## Implementing `predictor.py`

`kitchen init` scaffolds `src/serve/predictor.py` with a working stub. The minimum required interface:

```python
def predict(payload: dict) -> dict:
    """Return a prediction for payload."""
    ...
```

A complete example loading the MLflow champion model:

```python
import mlflow
from pydantic import BaseModel

# Load champion at module startup — happens once per Lambda cold start
model = mlflow.sklearn.load_model("models:/my-project@champion")

# Optional: expose feature list on GET /metadata
FEATURES = ["home_court", "elo_diff", "pace", "fg_pct_diff"]

class RequestModel(BaseModel):
    home_court: int
    elo_diff: float
    pace: float
    fg_pct_diff: float

class ResponseModel(BaseModel):
    prediction: int
    probability: float

def predict(payload: dict) -> dict:
    import pandas as pd
    X = pd.DataFrame([payload])[FEATURES]
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][pred])
    return {"prediction": pred, "probability": prob}
```

---

## Local development

```bash
kitchen serve local
# Starts uvicorn at http://localhost:8000
# Reloads on predictor.py changes
```

Test the endpoints:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"home_court": 1, "elo_diff": 120.5, "pace": 68.2}'

# OpenAPI docs
open http://localhost:8000/docs
```

---

## Docker build

```bash
docker build -t my-project-serve .
docker run -p 9000:8080 \
  -e KITCHEN_MODEL_NAME=my-project \
  -e KITCHEN_MODEL_VERSION=3 \
  -e KITCHEN_PREDICTOR_DIR=/var/task/src/serve \
  my-project-serve
```

---

## Deploy to Lambda

See [AWS Deployment Quickstart](aws-deployment.md) for the full ECR push + Lambda update workflow.

The scaffolded CI pipeline (`kitchen init --ci`) handles this automatically on every push to `main`:

1. Build Docker image
2. Push to ECR
3. Update Lambda function to the new image digest

!!! tip
    Use `recipes` to provision the Lambda function and ECR repo — see the [recipes YAML spec reference](../recipes/yaml-spec.md) for the `ecr` + `lambda` resource configuration.
