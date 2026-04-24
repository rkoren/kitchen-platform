from fastapi.testclient import TestClient
from kitchen.serve.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_501_without_predictor():
    response = client.post("/predict", json={"feature": 1})
    assert response.status_code == 501
    assert "predictor" in response.json()["detail"].lower()
