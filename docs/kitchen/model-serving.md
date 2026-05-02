# Model Serving

The trained model is served via a [FastAPI](https://fastapi.tiangolo.com) application, packaged as a Docker container and deployed to AWS Lambda via ECR.

## Architecture

```
ECR image → Lambda function URL
                │
         FastAPI + Mangum
                │
         /predict endpoint
```

[Mangum](https://mangum.io) adapts the FastAPI ASGI app to the Lambda event/response format.

## API

### `GET /health`

Health check.

```json
{"status": "ok"}
```

### `POST /predict`

Run inference on a single observation.

```json
// Request
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  ...
}

// Response
{
  "prediction": 1,
  "probability": 0.87
}
```

<!-- TODO: finalize request/response schema once feature engineering is complete -->

## Local development

```bash
cd src/serve
uvicorn app:app --reload
```

## Docker build

```bash
docker build -t my-project-serve ./src/serve
docker run -p 9000:8080 my-project-serve
```

## Deploy to Lambda

<!-- TODO: document ECR push + Lambda container update steps -->

!!! tip
    Use `recipes` to provision the Lambda function and ECR repo — see the [recipes quickstart](../recipes/quickstart.md).
