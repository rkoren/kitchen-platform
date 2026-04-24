from fastapi.testclient import TestClient
from src.serve.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_not_implemented():
    with TestClient(app, raise_server_exceptions=False) as c:
        response = c.post("/predict", json={"feature": 1})
    assert response.status_code == 500
