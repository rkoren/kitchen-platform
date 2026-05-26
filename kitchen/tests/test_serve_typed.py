"""Tests for the typed /predict route (S-002).

When a predictor exports *both* ``RequestModel`` and ``ResponseModel``
(Pydantic BaseModel subclasses) the ``build_app`` factory creates a typed
FastAPI route whose OpenAPI schema reflects the models.  When either is absent
the route falls back to the plain ``dict`` → ``dict`` contract.

All tests here use ``build_app`` to create an isolated app — they never touch
the module-level ``app`` so they don't disturb the existing ``test_serve.py``
tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kitchen.serve.app import build_app
from kitchen.serve.loader import load_predictor_bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_typed_predictor(directory: Path) -> None:
    """Write a predictor.py that exports RequestModel + ResponseModel."""
    (directory / "predictor.py").write_text(
        "from pydantic import BaseModel\n"
        "\n"
        "class RequestModel(BaseModel):\n"
        "    x: float\n"
        "    label: str = 'default'\n"
        "\n"
        "class ResponseModel(BaseModel):\n"
        "    prediction: int\n"
        "    confidence: float\n"
        "\n"
        "def predict(payload: dict) -> dict:\n"
        "    return {'prediction': int(payload['x'] > 0.5), 'confidence': payload['x']}\n"
    )


def _write_untyped_predictor(directory: Path) -> None:
    """Write a predictor.py with no Pydantic models."""
    (directory / "predictor.py").write_text(
        "def predict(payload: dict) -> dict:\n"
        "    return {'ok': True}\n"
    )


def _write_request_only_predictor(directory: Path) -> None:
    """Predictor with RequestModel only (no ResponseModel)."""
    (directory / "predictor.py").write_text(
        "from pydantic import BaseModel\n"
        "\n"
        "class RequestModel(BaseModel):\n"
        "    x: float\n"
        "\n"
        "def predict(payload: dict) -> dict:\n"
        "    return {'x': payload['x']}\n"
    )


def _write_response_only_predictor(directory: Path) -> None:
    """Predictor with ResponseModel only (no RequestModel)."""
    (directory / "predictor.py").write_text(
        "from pydantic import BaseModel\n"
        "\n"
        "class ResponseModel(BaseModel):\n"
        "    label: int\n"
        "\n"
        "def predict(payload: dict) -> dict:\n"
        "    return {'label': 1}\n"
    )


# ---------------------------------------------------------------------------
# build_app factory — basic wiring
# ---------------------------------------------------------------------------


def test_build_app_returns_fastapi_app():
    from fastapi import FastAPI

    fresh = build_app()
    assert isinstance(fresh, FastAPI)


def test_build_app_health_endpoint():
    fresh = build_app()
    client = TestClient(fresh)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_build_app_no_bundle_returns_501():
    fresh = build_app(bundle=None)
    client = TestClient(fresh)
    resp = client.post("/predict", json={"x": 1.0})
    assert resp.status_code == 501


# ---------------------------------------------------------------------------
# Typed route — both models present
# ---------------------------------------------------------------------------


def test_typed_route_returns_200(tmp_path):
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    resp = client.post("/predict", json={"x": 0.8, "label": "test"})
    assert resp.status_code == 200


def test_typed_route_response_body(tmp_path):
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    resp = client.post("/predict", json={"x": 0.8})
    body = resp.json()
    assert body["prediction"] == 1
    assert abs(body["confidence"] - 0.8) < 1e-6


def test_typed_route_payload_forwarded_as_dict(tmp_path):
    """predict() must receive the request body as a plain dict (not a Pydantic model)."""
    captured = tmp_path / "captured.json"

    (tmp_path / "predictor.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from pydantic import BaseModel\n"
        "\n"
        "class RequestModel(BaseModel):\n"
        "    a: float\n"
        "    b: str\n"
        "\n"
        "class ResponseModel(BaseModel):\n"
        "    ok: bool\n"
        "\n"
        "def predict(payload: dict) -> dict:\n"
        f"    Path({str(captured)!r}).write_text(json.dumps(payload))\n"
        "    return {'ok': True}\n"
    )
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    fresh = build_app(bundle)
    client = TestClient(fresh)
    client.post("/predict", json={"a": 1.5, "b": "hello"})

    import json

    saved = json.loads(captured.read_text())
    assert saved == {"a": 1.5, "b": "hello"}


def test_typed_route_request_validation_rejects_bad_input(tmp_path):
    """FastAPI validates the request against RequestModel; bad input → 422."""
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    resp = client.post("/predict", json={"x": "not-a-float"})
    assert resp.status_code == 422


def test_typed_route_missing_required_field_returns_422(tmp_path):
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    # 'x' is required; omitting it should fail schema validation
    resp = client.post("/predict", json={"label": "only-label"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# OpenAPI schema reflects the Pydantic models
# ---------------------------------------------------------------------------


def test_openapi_schema_has_predict_endpoint(tmp_path):
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]


def test_openapi_schema_request_model_name(tmp_path):
    """OpenAPI requestBody schema references RequestModel."""
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    schema = client.get("/openapi.json").json()
    # The $ref or component should contain "RequestModel"
    schema_str = str(schema)
    assert "RequestModel" in schema_str


def test_openapi_schema_response_model_name(tmp_path):
    """OpenAPI response schema references ResponseModel."""
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    schema = client.get("/openapi.json").json()
    schema_str = str(schema)
    assert "ResponseModel" in schema_str


def test_openapi_schema_request_fields_present(tmp_path):
    """RequestModel fields (x, label) appear in the OpenAPI schema."""
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    schema = client.get("/openapi.json").json()
    components = schema.get("components", {}).get("schemas", {})
    req_schema = components.get("RequestModel", {})
    props = req_schema.get("properties", {})
    assert "x" in props
    assert "label" in props


def test_openapi_schema_response_fields_present(tmp_path):
    """ResponseModel fields appear in the OpenAPI schema."""
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    schema = client.get("/openapi.json").json()
    components = schema.get("components", {}).get("schemas", {})
    resp_schema = components.get("ResponseModel", {})
    props = resp_schema.get("properties", {})
    assert "prediction" in props
    assert "confidence" in props


# ---------------------------------------------------------------------------
# Fallback to dict route when models absent or partial
# ---------------------------------------------------------------------------


def test_dict_route_when_no_models(tmp_path):
    """No models → plain dict route, accepts any JSON."""
    _write_untyped_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    resp = client.post("/predict", json={"anything": "goes"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_dict_route_when_only_request_model(tmp_path):
    """RequestModel only (no ResponseModel) → falls back to dict route."""
    _write_request_only_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    assert bundle is not None
    assert bundle.request_model is not None
    assert bundle.response_model is None
    client = TestClient(build_app(bundle))
    resp = client.post("/predict", json={"x": 1.0})
    assert resp.status_code == 200


def test_dict_route_when_only_response_model(tmp_path):
    """ResponseModel only (no RequestModel) → falls back to dict route."""
    _write_response_only_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    assert bundle is not None
    assert bundle.request_model is None
    assert bundle.response_model is not None
    client = TestClient(build_app(bundle))
    resp = client.post("/predict", json={"free": "form"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Multiple concurrent apps — isolation
# ---------------------------------------------------------------------------


def test_two_apps_independent(tmp_path):
    """Two build_app() calls create independent apps with separate route registrations."""
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    app_a = build_app(bundle)
    app_b = build_app()  # no predictor

    client_a = TestClient(app_a)
    client_b = TestClient(app_b)

    assert client_a.post("/predict", json={"x": 0.9}).status_code == 200
    assert client_b.post("/predict", json={"x": 0.9}).status_code == 501
