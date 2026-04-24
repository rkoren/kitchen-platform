"""Generic FastAPI serving app — deployed to Lambda via ECR/Mangum.

Projects implement predictions by placing a ``predictor.py`` module alongside
this file (i.e. in the same Docker image working directory) that exposes::

    def predict(payload: dict) -> dict: ...

If no predictor is found the /predict endpoint returns 501 Not Implemented.
"""
from fastapi import FastAPI, HTTPException
from mangum import Mangum

app = FastAPI(title="kitchen-serve", version="0.1.0")

try:
    from predictor import predict as _predict_fn
except ImportError:
    _predict_fn = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict) -> dict:
    if _predict_fn is None:
        raise HTTPException(status_code=501, detail="No predictor implemented")
    return _predict_fn(payload)


handler = Mangum(app)
