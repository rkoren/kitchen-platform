"""Tests for kitchen.serve.app.

The app is a generic FastAPI serving layer. Projects plug in a ``predictor.py``
module that exposes ``predict(payload: dict) -> dict``.  Tests cover:

- The health endpoint (always available)
- The 501 fallback when no predictor is configured
- Project-provided predictor integration (S-007): correct dispatch, payload
  forwarding, response pass-through, and error propagation
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from kitchen.serve.app import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# No-predictor fallback
# ---------------------------------------------------------------------------


def test_predict_returns_501_without_predictor():
    response = client.post("/predict", json={"feature": 1})
    assert response.status_code == 501
    assert "predictor" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Project-provided predictor (S-007)
# ---------------------------------------------------------------------------


def _make_client(predict_fn):
    """Return a TestClient with *predict_fn* wired in as the active predictor."""
    with patch("kitchen.serve.app._predict_fn", predict_fn):
        return TestClient(app)


def test_predictor_returns_200():
    def predict(payload):
        return {"label": 1}

    with patch("kitchen.serve.app._predict_fn", predict):
        resp = client.post("/predict", json={"x": 0.5})
    assert resp.status_code == 200


def test_predictor_result_passed_through():
    """Whatever the predictor returns is the response body."""
    expected = {"label": 1, "score": 0.92}

    def predict(payload):
        return expected

    with patch("kitchen.serve.app._predict_fn", predict):
        resp = client.post("/predict", json={"x": 1.0})
    assert resp.json() == expected


def test_predictor_receives_exact_payload():
    """Predictor must be called with the exact dict sent by the client."""
    received: dict = {}

    def predict(payload):
        received.update(payload)
        return {"ok": True}

    payload = {"feature_a": 1.5, "feature_b": "red", "feature_c": True}
    with patch("kitchen.serve.app._predict_fn", predict):
        client.post("/predict", json=payload)
    assert received == payload


def test_predictor_nested_response():
    """Nested dicts and lists in the predictor response are serialised correctly."""
    def predict(payload):
        return {"scores": [0.1, 0.7, 0.2], "meta": {"version": "1.0"}}

    with patch("kitchen.serve.app._predict_fn", predict):
        resp = client.post("/predict", json={})
    assert resp.json()["scores"] == [0.1, 0.7, 0.2]
    assert resp.json()["meta"] == {"version": "1.0"}


def test_predictor_empty_payload():
    """Empty dict payload is valid — predictor may handle it however it likes."""
    def predict(payload):
        return {"default": True}

    with patch("kitchen.serve.app._predict_fn", predict):
        resp = client.post("/predict", json={})
    assert resp.status_code == 200
    assert resp.json() == {"default": True}


def test_predictor_unhandled_exception_returns_500():
    """An unhandled exception in the predictor surfaces as HTTP 500."""
    def predict(payload):
        raise RuntimeError("model file not found")

    with patch("kitchen.serve.app._predict_fn", predict):
        safe_client = TestClient(app, raise_server_exceptions=False)
        resp = safe_client.post("/predict", json={"x": 1.0})
    assert resp.status_code == 500


def test_predictor_http_exception_propagates():
    """A predictor that raises HTTPException controls its own status code."""
    from fastapi import HTTPException

    def predict(payload):
        raise HTTPException(status_code=422, detail="invalid feature value")

    with patch("kitchen.serve.app._predict_fn", predict):
        resp = client.post("/predict", json={"x": -999})
    assert resp.status_code == 422
    assert "invalid feature value" in resp.json()["detail"]


def test_predictor_multiple_requests_isolated():
    """Each request calls the predictor independently — no shared state leaks."""
    call_count = {"n": 0}

    def predict(payload):
        call_count["n"] += 1
        return {"call": call_count["n"]}

    with patch("kitchen.serve.app._predict_fn", predict):
        r1 = client.post("/predict", json={})
        r2 = client.post("/predict", json={})
        r3 = client.post("/predict", json={})

    assert r1.json()["call"] == 1
    assert r2.json()["call"] == 2
    assert r3.json()["call"] == 3


# ---------------------------------------------------------------------------
# CBB-023: structured /predict error instead of a bare 500
# ---------------------------------------------------------------------------


def test_predictor_exception_returns_structured_500():
    """An unhandled predictor exception returns a structured error body, not a bare 500."""
    def predict(payload):
        raise KeyError("['d_kp_APL_Def'] not in index")

    with patch("kitchen.serve.app._predict_fn", predict):
        safe_client = TestClient(app, raise_server_exceptions=False)
        resp = safe_client.post("/predict", json={"x": 1.0})
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["type"] == "KeyError"
    assert "d_kp_APL_Def" in detail["message"]
    assert "hint" in detail and "traceback" not in detail  # logs hint, no traceback by default


def test_predictor_exception_includes_traceback_under_debug(monkeypatch):
    monkeypatch.setenv("KITCHEN_DEBUG", "1")

    def predict(payload):
        raise RuntimeError("boom")

    with patch("kitchen.serve.app._predict_fn", predict):
        safe_client = TestClient(app, raise_server_exceptions=False)
        resp = safe_client.post("/predict", json={"x": 1.0})
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "traceback" in detail and "RuntimeError" in detail["traceback"]
    assert "hint" not in detail
