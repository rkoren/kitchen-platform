"""Tests for the POST /predict/batch endpoint (S-003).

The endpoint accepts ``{"items": [...]}`` and returns ``{"results": [...]}``.
When the predictor exports both ``RequestModel`` and ``ResponseModel`` the
route is typed (each item validated against ``RequestModel``; each result
against ``ResponseModel``).  Otherwise it falls back to plain ``dict`` mode.

All tests use ``build_app(bundle)`` to create isolated apps.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from kitchen.serve.app import _batch_max_items, build_app
from kitchen.serve.loader import PredictorBundle, load_predictor_bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bundle(predict_fn=None, features=None, request_model=None, response_model=None):
    if predict_fn is None:
        def predict_fn(p):
            return {"echo": p}
    return PredictorBundle(
        predict_fn=predict_fn,
        request_model=request_model,
        response_model=response_model,
        features=features,
    )


def _write_typed_predictor(directory: Path) -> None:
    (directory / "predictor.py").write_text(
        "from pydantic import BaseModel\n"
        "\n"
        "class RequestModel(BaseModel):\n"
        "    x: float\n"
        "\n"
        "class ResponseModel(BaseModel):\n"
        "    prediction: int\n"
        "\n"
        "def predict(payload: dict) -> dict:\n"
        "    return {'prediction': int(payload['x'] > 0.5)}\n"
    )


def _write_untyped_predictor(directory: Path) -> None:
    (directory / "predictor.py").write_text(
        "def predict(payload: dict) -> dict:\n"
        "    return {'value': payload.get('v', 0) * 2}\n"
    )


# ---------------------------------------------------------------------------
# Dict-mode batch route — basic contract
# ---------------------------------------------------------------------------


def test_batch_returns_200():
    client = TestClient(build_app(_make_bundle()))
    resp = client.post("/predict/batch", json={"items": [{"v": 1}, {"v": 2}]})
    assert resp.status_code == 200


def test_batch_results_key_present():
    client = TestClient(build_app(_make_bundle()))
    body = client.post("/predict/batch", json={"items": [{"a": 1}]}).json()
    assert "results" in body


def test_batch_results_length_matches_items():
    client = TestClient(build_app(_make_bundle()))
    body = client.post("/predict/batch", json={"items": [{"v": 1}, {"v": 2}, {"v": 3}]}).json()
    assert len(body["results"]) == 3


def test_batch_order_preserved():
    """results[i] must correspond to items[i]."""
    def predict_fn(p):
        return {"idx": p["idx"]}

    client = TestClient(build_app(_make_bundle(predict_fn=predict_fn)))
    body = client.post(
        "/predict/batch",
        json={"items": [{"idx": 0}, {"idx": 1}, {"idx": 2}]},
    ).json()
    assert [r["idx"] for r in body["results"]] == [0, 1, 2]


def test_batch_empty_items_returns_200():
    """Empty items list is valid — returns empty results, not 422."""
    client = TestClient(build_app(_make_bundle()))
    resp = client.post("/predict/batch", json={"items": []})
    assert resp.status_code == 200
    assert resp.json() == {"results": []}


def test_batch_predict_fn_called_once_per_item():
    """predict_fn must be called N times for N items, not once with a list."""
    mock_fn = MagicMock(return_value={"ok": True})
    bundle = _make_bundle(predict_fn=mock_fn)
    client = TestClient(build_app(bundle))
    client.post("/predict/batch", json={"items": [{"a": 1}, {"b": 2}, {"c": 3}]})
    assert mock_fn.call_count == 3


def test_batch_predict_fn_receives_individual_dicts():
    """Each call to predict_fn gets a single dict from items, not the whole list."""
    received = []

    def predict_fn(p):
        received.append(p)
        return {}

    client = TestClient(build_app(_make_bundle(predict_fn=predict_fn)))
    client.post("/predict/batch", json={"items": [{"x": 1}, {"x": 2}]})
    assert received == [{"x": 1}, {"x": 2}]


def test_batch_no_predictor_returns_501():
    client = TestClient(build_app(bundle=None))
    resp = client.post("/predict/batch", json={"items": [{"a": 1}]})
    assert resp.status_code == 501


def test_batch_missing_items_key_returns_422():
    """Body without 'items' key → 422."""
    client = TestClient(build_app(_make_bundle()))
    resp = client.post("/predict/batch", json={"data": [{"a": 1}]})
    assert resp.status_code == 422


def test_batch_items_not_a_list_returns_422():
    """items is a string, not a list → 422."""
    client = TestClient(build_app(_make_bundle()))
    resp = client.post("/predict/batch", json={"items": "not-a-list"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Dict-mode — end-to-end with real predictor file
# ---------------------------------------------------------------------------


def test_batch_dict_mode_end_to_end(tmp_path):
    _write_untyped_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    body = client.post("/predict/batch", json={"items": [{"v": 3}, {"v": 5}]}).json()
    assert body["results"] == [{"value": 6}, {"value": 10}]


# ---------------------------------------------------------------------------
# Typed batch route — both models present
# ---------------------------------------------------------------------------


def test_typed_batch_returns_200(tmp_path):
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    resp = client.post("/predict/batch", json={"items": [{"x": 0.8}, {"x": 0.2}]})
    assert resp.status_code == 200


def test_typed_batch_results_correct(tmp_path):
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    body = client.post("/predict/batch", json={"items": [{"x": 0.9}, {"x": 0.1}]}).json()
    assert body["results"] == [{"prediction": 1}, {"prediction": 0}]


def test_typed_batch_validates_each_item(tmp_path):
    """One invalid item in a typed batch → 422 for the whole request."""
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    resp = client.post(
        "/predict/batch",
        json={"items": [{"x": 0.5}, {"x": "not-a-float"}]},
    )
    assert resp.status_code == 422


def test_typed_batch_empty_items_returns_200(tmp_path):
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    resp = client.post("/predict/batch", json={"items": []})
    assert resp.status_code == 200
    assert resp.json() == {"results": []}


def test_typed_batch_predict_fn_called_with_dicts(tmp_path):
    """Typed batch: predict_fn must receive plain dicts (model_dump), not Pydantic objects."""
    captured = tmp_path / "captured.json"
    (tmp_path / "predictor.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from pydantic import BaseModel\n"
        "\n"
        "class RequestModel(BaseModel):\n"
        "    a: float\n"
        "\n"
        "class ResponseModel(BaseModel):\n"
        "    ok: bool\n"
        "\n"
        "_calls = []\n"
        "\n"
        "def predict(payload: dict) -> dict:\n"
        f"    data = json.loads(Path({str(captured)!r}).read_text()) if Path({str(captured)!r}).exists() else []\n"
        f"    data.append(payload)\n"
        f"    Path({str(captured)!r}).write_text(json.dumps(data))\n"
        "    return {'ok': True}\n"
    )
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    client.post("/predict/batch", json={"items": [{"a": 1.0}, {"a": 2.0}]})

    import json
    calls = json.loads(captured.read_text())
    assert calls == [{"a": 1.0}, {"a": 2.0}]


def test_typed_batch_openapi_schema_has_batch_endpoint(tmp_path):
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    schema = client.get("/openapi.json").json()
    assert "/predict/batch" in schema["paths"]


def test_typed_batch_openapi_schema_has_batch_request_model(tmp_path):
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    schema = client.get("/openapi.json").json()
    schema_str = str(schema)
    assert "BatchRequest" in schema_str


def test_typed_batch_openapi_schema_has_batch_response_model(tmp_path):
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    schema = client.get("/openapi.json").json()
    schema_str = str(schema)
    assert "BatchResponse" in schema_str


# ---------------------------------------------------------------------------
# Batch size cap — KITCHEN_BATCH_MAX_ITEMS
# ---------------------------------------------------------------------------


def test_batch_returns_413_when_over_limit(monkeypatch):
    monkeypatch.setenv("KITCHEN_BATCH_MAX_ITEMS", "3")
    client = TestClient(build_app(_make_bundle()))
    resp = client.post("/predict/batch", json={"items": [{"x": i} for i in range(4)]})
    assert resp.status_code == 413


def test_batch_returns_200_at_exact_limit(monkeypatch):
    monkeypatch.setenv("KITCHEN_BATCH_MAX_ITEMS", "3")
    client = TestClient(build_app(_make_bundle()))
    resp = client.post("/predict/batch", json={"items": [{"x": i} for i in range(3)]})
    assert resp.status_code == 200


def test_batch_413_error_detail_mentions_limit(monkeypatch):
    monkeypatch.setenv("KITCHEN_BATCH_MAX_ITEMS", "2")
    client = TestClient(build_app(_make_bundle()))
    resp = client.post("/predict/batch", json={"items": [{"x": i} for i in range(5)]})
    assert "2" in resp.json()["detail"]


def test_batch_max_items_from_env(monkeypatch):
    monkeypatch.setenv("KITCHEN_BATCH_MAX_ITEMS", "42")
    assert _batch_max_items() == 42


def test_batch_max_items_default_when_env_absent(monkeypatch):
    monkeypatch.delenv("KITCHEN_BATCH_MAX_ITEMS", raising=False)
    assert _batch_max_items() == 1000


def test_batch_max_items_invalid_env_uses_default(monkeypatch):
    monkeypatch.setenv("KITCHEN_BATCH_MAX_ITEMS", "not-an-int")
    assert _batch_max_items() == 1000


def test_batch_max_items_zero_uses_default(monkeypatch):
    """Zero is not a valid cap — fall back to default."""
    monkeypatch.setenv("KITCHEN_BATCH_MAX_ITEMS", "0")
    assert _batch_max_items() == 1000


def test_typed_batch_413_when_over_limit(tmp_path, monkeypatch):
    """Typed batch endpoint also enforces the size cap."""
    monkeypatch.setenv("KITCHEN_BATCH_MAX_ITEMS", "2")
    _write_typed_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    resp = client.post(
        "/predict/batch",
        json={"items": [{"x": 0.5}, {"x": 0.6}, {"x": 0.7}]},
    )
    assert resp.status_code == 413


# ---------------------------------------------------------------------------
# Isolation — single-item /predict is unaffected by /predict/batch
# ---------------------------------------------------------------------------


def test_single_predict_still_works_alongside_batch():
    def predict_fn(p):
        return {"out": p.get("in", 0)}

    client = TestClient(build_app(_make_bundle(predict_fn=predict_fn)))
    single = client.post("/predict", json={"in": 5}).json()
    batch = client.post("/predict/batch", json={"items": [{"in": 5}, {"in": 10}]}).json()
    assert single == {"out": 5}
    assert batch["results"] == [{"out": 5}, {"out": 10}]


def test_two_apps_batch_independent():
    """Two build_app() calls have independent /predict/batch handlers."""
    def fn_a(p):
        return {"src": "a"}

    def fn_b(p):
        return {"src": "b"}

    client_a = TestClient(build_app(_make_bundle(predict_fn=fn_a)))
    client_b = TestClient(build_app(_make_bundle(predict_fn=fn_b)))
    assert client_a.post("/predict/batch", json={"items": [{}]}).json()["results"] == [{"src": "a"}]
    assert client_b.post("/predict/batch", json={"items": [{}]}).json()["results"] == [{"src": "b"}]
