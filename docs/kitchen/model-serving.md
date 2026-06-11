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

| Field | Source (first non-null wins) |
|---|---|
| `model_name` | `predictor.MODEL_NAME` → `KITCHEN_MODEL_NAME` env → `MLFLOW_MODEL_NAME` env |
| `model_version` | `predictor.MODEL_VERSION` → `KITCHEN_MODEL_VERSION` env |
| `git_sha` | `GITHUB_SHA` env var → `GIT_SHA` env var → `git rev-parse --short HEAD` |
| `features` | `FEATURES` list exported from `predictor.py` |

The most reliable way to populate the model identity is to export `MODEL_NAME` and `MODEL_VERSION` (strings) from `predictor.py` — then `/metadata` reflects exactly what the predictor loads, with no env wiring. Otherwise set `KITCHEN_MODEL_NAME` / `KITCHEN_MODEL_VERSION` (or rely on the `MLFLOW_MODEL_NAME` you already use for loading) in the Lambda environment via the `environment:` block in your `recipes` spec.

### Reserved environment variables

`kitchen serve local` and the Lambda loader **set** these — do not reuse them for your own predictor settings (pick a project-specific name instead):

| Variable | Set to |
|---|---|
| `KITCHEN_PREDICTOR_DIR` | the directory containing `predictor.py` (used to locate and import it) |
| `KITCHEN_MODEL_NAME` / `KITCHEN_MODEL_VERSION` | read by `/metadata` |

`kitchen serve local` prints the resolved predictor directory at startup so you can see what `KITCHEN_PREDICTOR_DIR` will be.

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
from kitchen.serve import lazy_model

# lazy_model defers the load to the first prediction instead of module import,
# so Lambda cold starts are faster; it loads once and caches thereafter.
# (Use mlflow.pyfunc.load_model for flavor-agnostic loading if you don't need
# predict_proba.)
model = lazy_model(lambda: mlflow.sklearn.load_model("models:/my-project@champion"))

# Optional: surface model identity + feature list on GET /metadata
MODEL_NAME = "my-project-model"
MODEL_VERSION = "champion"
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

---

## Exposing the API over HTTP

A Lambda function isn't reachable over HTTP until you put an endpoint in front of it.
There are two options; for a single model-serving function, a **function URL** is the
simplest and what the platform recommends.

### Function URL (recommended)

A [Lambda function URL](https://docs.aws.amazon.com/lambda/latest/dg/lambda-urls.html)
is a dedicated HTTPS endpoint on the function itself — no extra infrastructure. Declare
it on the `lambda` resource in your `recipes` spec rather than clicking through the
console:

```yaml
- type: lambda
  name: my-project-serve
  role: my-project-exec
  ecr_repo: my-project-serve
  function_url: true              # provisions the HTTPS endpoint
  function_url_auth: AWS_IAM      # default; use NONE for a public endpoint
```

`recipes` emits the function URL as the `<name>_url` Terraform output, so after `recipes
apply` you get the endpoint directly:

```bash
terraform output my_project_serve_url
# https://abc123.lambda-url.us-east-1.on.aws/
```

**Auth modes:**

- `AWS_IAM` (default) — callers sign requests with SigV4. Secure by default; use for
  internal/service-to-service calls. With curl: `--aws-sigv4 "aws:amz:<region>:lambda"`
  plus `--user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY"` (add `-H "x-amz-security-token: $AWS_SESSION_TOKEN"` for temporary creds).
- `NONE` — public endpoint, no auth. `recipes` also adds the required
  `lambda:InvokeFunctionUrl` permission. Only use behind your own auth, or for demos.

```bash
FUNCTION_URL=$(terraform output -raw my_project_serve_url)
curl ${FUNCTION_URL}health
curl -X POST ${FUNCTION_URL}predict -H 'content-type: application/json' \
  -d '{"instances": [{"feature_a": 1.0, "feature_b": 2.0}]}'
```

Runnable specs: [`ecr-lambda-inference-api.yaml`](https://github.com/rkoren/kitchen-platform/blob/main/recipes/examples/ecr-lambda-inference-api.yaml)
(public) and [`kaggle-serving-stack.yaml`](https://github.com/rkoren/kitchen-platform/blob/main/recipes/examples/kaggle-serving-stack.yaml)
(IAM-authed).

### API Gateway (when you need more)

Reach for [API Gateway](https://docs.aws.amazon.com/apigateway/) instead of a function
URL when you need any of: a **custom domain**, **request throttling / usage plans / API
keys**, **WAF** integration, request/response transformation, or fan-out to multiple
backends. It's more moving parts (a REST or HTTP API, stages, routes, a Lambda
integration, and `lambda:InvokeFunction` permission for the API).

`recipes` doesn't generate API Gateway resources today — provision it with hand-written
Terraform alongside the generated `tf/`, or open a request for an `api_gateway` resource
type. For most single-model serving, the function URL above is sufficient.
