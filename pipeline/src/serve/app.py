"""FastAPI serving endpoint — deployed to Lambda via ECR/Mangum."""
from fastapi import FastAPI
from mangum import Mangum

app = FastAPI(title="pipeline-serve", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    raise NotImplementedError


# Lambda handler
handler = Mangum(app)
