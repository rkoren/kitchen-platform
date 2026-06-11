"""Tests for the GET /metadata endpoint (S-004).

The endpoint returns four fields:
  - model_name    — from KITCHEN_MODEL_NAME env var (or null)
  - model_version — from KITCHEN_MODEL_VERSION env var (or null)
  - git_sha       — from GITHUB_SHA → GIT_SHA → subprocess (or null)
  - features      — from predictor.FEATURES (or null)

All tests use ``build_app(bundle)`` to create isolated apps so they don't
touch module-level state from ``kitchen.serve.app``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from kitchen.serve.app import _collect_metadata, _resolve_git_sha, build_app
from kitchen.serve.loader import PredictorBundle, load_predictor_bundle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_simple_predictor(directory: Path) -> None:
    (directory / "predictor.py").write_text(
        "def predict(payload: dict) -> dict:\n"
        "    return {'ok': True}\n"
    )


def _write_predictor_with_features(directory: Path, features: list) -> None:
    features_repr = repr(features)
    (directory / "predictor.py").write_text(
        f"FEATURES = {features_repr}\n"
        "def predict(payload: dict) -> dict:\n"
        "    return {'ok': True}\n"
    )


def _make_bundle(predict_fn=None, features=None):
    """Create a minimal PredictorBundle for metadata tests."""
    if predict_fn is None:
        def predict_fn(p):
            return {}
    return PredictorBundle(predict_fn=predict_fn, features=features)


# ---------------------------------------------------------------------------
# /metadata endpoint — basic contract
# ---------------------------------------------------------------------------


def test_metadata_endpoint_returns_200():
    client = TestClient(build_app(_make_bundle()))
    resp = client.get("/metadata")
    assert resp.status_code == 200


def test_metadata_endpoint_returns_json():
    client = TestClient(build_app(_make_bundle()))
    resp = client.get("/metadata")
    body = resp.json()
    assert isinstance(body, dict)


def test_metadata_endpoint_has_required_keys():
    """Response must contain exactly model_name, model_version, git_sha, features."""
    client = TestClient(build_app(_make_bundle()))
    body = client.get("/metadata").json()
    assert "model_name" in body
    assert "model_version" in body
    assert "git_sha" in body
    assert "features" in body


def test_metadata_endpoint_no_predictor_still_200():
    """build_app(bundle=None) still serves /metadata (returns nulls for all fields)."""
    client = TestClient(build_app(bundle=None))
    resp = client.get("/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] is None
    assert body["model_version"] is None
    assert body["features"] is None


# ---------------------------------------------------------------------------
# model_name / model_version from env vars
# ---------------------------------------------------------------------------


def test_model_name_from_env(monkeypatch):
    monkeypatch.setenv("KITCHEN_MODEL_NAME", "cbb-predictor")
    client = TestClient(build_app(_make_bundle()))
    body = client.get("/metadata").json()
    assert body["model_name"] == "cbb-predictor"


def test_model_version_from_env(monkeypatch):
    monkeypatch.setenv("KITCHEN_MODEL_VERSION", "3")
    client = TestClient(build_app(_make_bundle()))
    body = client.get("/metadata").json()
    assert body["model_version"] == "3"


def test_model_name_null_when_env_absent(monkeypatch):
    monkeypatch.delenv("KITCHEN_MODEL_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_MODEL_NAME", raising=False)
    client = TestClient(build_app(_make_bundle()))
    body = client.get("/metadata").json()
    assert body["model_name"] is None


def test_model_version_null_when_env_absent(monkeypatch):
    monkeypatch.delenv("KITCHEN_MODEL_VERSION", raising=False)
    client = TestClient(build_app(_make_bundle()))
    body = client.get("/metadata").json()
    assert body["model_version"] is None


def test_model_name_and_version_both_set(monkeypatch):
    monkeypatch.setenv("KITCHEN_MODEL_NAME", "iris-clf")
    monkeypatch.setenv("KITCHEN_MODEL_VERSION", "7")
    client = TestClient(build_app(_make_bundle()))
    body = client.get("/metadata").json()
    assert body["model_name"] == "iris-clf"
    assert body["model_version"] == "7"


# ---------------------------------------------------------------------------
# CBB-007: MLFLOW_MODEL_NAME fallback + predictor MODEL_NAME/MODEL_VERSION exports
# ---------------------------------------------------------------------------


def test_model_name_falls_back_to_mlflow_model_name(monkeypatch):
    monkeypatch.delenv("KITCHEN_MODEL_NAME", raising=False)
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "cbb-tournament-model")
    body = client_body(_make_bundle())
    assert body["model_name"] == "cbb-tournament-model"


def test_kitchen_model_name_takes_priority_over_mlflow(monkeypatch):
    monkeypatch.setenv("KITCHEN_MODEL_NAME", "explicit")
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "fallback")
    body = client_body(_make_bundle())
    assert body["model_name"] == "explicit"


def test_bundle_model_identity_wins_over_env(monkeypatch):
    monkeypatch.setenv("KITCHEN_MODEL_NAME", "from-env")
    monkeypatch.setenv("KITCHEN_MODEL_VERSION", "9")
    bundle = PredictorBundle(
        predict_fn=lambda p: {}, model_name="from-predictor", model_version="champion"
    )
    body = client_body(bundle)
    assert body["model_name"] == "from-predictor"
    assert body["model_version"] == "champion"


def test_predictor_module_model_name_version_loaded(tmp_path):
    (tmp_path / "predictor.py").write_text(
        "MODEL_NAME = 'cbb-tournament-model'\n"
        "MODEL_VERSION = 'champion'\n"
        "def predict(payload: dict) -> dict:\n"
        "    return {'ok': True}\n"
    )
    bundle = load_predictor_bundle(tmp_path)
    assert bundle.model_name == "cbb-tournament-model"
    assert bundle.model_version == "champion"


def client_body(bundle):
    return TestClient(build_app(bundle)).get("/metadata").json()


# ---------------------------------------------------------------------------
# git_sha resolution
# ---------------------------------------------------------------------------


def test_git_sha_from_github_sha_env(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc1234efgh5678")
    monkeypatch.delenv("GIT_SHA", raising=False)
    sha = _resolve_git_sha()
    assert sha == "abc1234efgh5678"


def test_git_sha_from_git_sha_env(monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setenv("GIT_SHA", "deadbeef")
    sha = _resolve_git_sha()
    assert sha == "deadbeef"


def test_github_sha_takes_priority_over_git_sha(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "from-github")
    monkeypatch.setenv("GIT_SHA", "from-git-sha")
    sha = _resolve_git_sha()
    assert sha == "from-github"


def test_git_sha_null_when_no_env_and_no_git(monkeypatch):
    """When no env var and subprocess fails, git_sha is None."""
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GIT_SHA", raising=False)
    with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
        sha = _resolve_git_sha()
    assert sha is None


def test_git_sha_null_when_subprocess_nonzero(monkeypatch):
    """subprocess returns non-zero (e.g. not a git repo) → None."""
    import subprocess

    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GIT_SHA", raising=False)
    mock_result = subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="")
    with patch("subprocess.run", return_value=mock_result):
        sha = _resolve_git_sha()
    assert sha is None


def test_git_sha_in_metadata_response(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "cafebabe")
    monkeypatch.delenv("GIT_SHA", raising=False)
    client = TestClient(build_app(_make_bundle()))
    body = client.get("/metadata").json()
    assert body["git_sha"] == "cafebabe"


# ---------------------------------------------------------------------------
# features field
# ---------------------------------------------------------------------------


def test_features_null_when_bundle_has_no_features():
    bundle = _make_bundle(features=None)
    client = TestClient(build_app(bundle))
    body = client.get("/metadata").json()
    assert body["features"] is None


def test_features_returned_when_bundle_has_features():
    bundle = _make_bundle(features=["sepal_length", "sepal_width", "petal_length"])
    client = TestClient(build_app(bundle))
    body = client.get("/metadata").json()
    assert body["features"] == ["sepal_length", "sepal_width", "petal_length"]


def test_features_empty_list_returned_as_empty_list():
    bundle = _make_bundle(features=[])
    client = TestClient(build_app(bundle))
    body = client.get("/metadata").json()
    assert body["features"] == []


def test_features_loaded_from_predictor_file(tmp_path):
    """End-to-end: FEATURES in predictor.py is surfaced on /metadata."""
    _write_predictor_with_features(tmp_path, ["home_court", "elo_diff", "pace"])
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    client = TestClient(build_app(bundle))
    body = client.get("/metadata").json()
    assert body["features"] == ["home_court", "elo_diff", "pace"]


# ---------------------------------------------------------------------------
# _collect_metadata unit tests
# ---------------------------------------------------------------------------


def test_collect_metadata_reads_env_at_call_time(monkeypatch):
    """_collect_metadata must read os.environ at call time, not import time."""
    monkeypatch.setenv("KITCHEN_MODEL_NAME", "before")
    bundle = _make_bundle()
    first = _collect_metadata(bundle)
    monkeypatch.setenv("KITCHEN_MODEL_NAME", "after")
    second = _collect_metadata(bundle)
    assert first["model_name"] == "before"
    assert second["model_name"] == "after"


def test_collect_metadata_none_bundle_returns_null_features(monkeypatch):
    monkeypatch.delenv("KITCHEN_MODEL_NAME", raising=False)
    monkeypatch.delenv("KITCHEN_MODEL_VERSION", raising=False)
    result = _collect_metadata(None)
    assert result["features"] is None
    assert result["model_name"] is None
    assert result["model_version"] is None


# ---------------------------------------------------------------------------
# Isolation — two apps have independent metadata
# ---------------------------------------------------------------------------


def test_two_apps_have_independent_metadata(monkeypatch):
    """Two build_app() calls create independent /metadata handlers."""
    bundle_a = _make_bundle(features=["x", "y"])
    bundle_b = _make_bundle(features=["a", "b", "c"])

    monkeypatch.setenv("KITCHEN_MODEL_NAME", "model-a")
    client_a = TestClient(build_app(bundle_a))
    client_b = TestClient(build_app(bundle_b))

    body_a = client_a.get("/metadata").json()
    body_b = client_b.get("/metadata").json()

    assert body_a["features"] == ["x", "y"]
    assert body_b["features"] == ["a", "b", "c"]
