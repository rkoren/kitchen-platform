"""Tests for kitchen.serve.loader (S-001).

Covers:
- Resolution order (explicit arg > env var > sys.path scan)
- Successful load and callable validation
- Sibling import support
- All error paths (missing predict, not callable, import error)
- Public API re-export from kitchen.serve
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kitchen.serve.loader import ENV_KEY, PredictorLoadError, load_predictor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_predictor(directory: Path, body: str = "def predict(payload): return {'ok': True}") -> Path:
    path = directory / "predictor.py"
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# Resolution: no predictor found
# ---------------------------------------------------------------------------


def test_returns_none_when_no_predictor_found(tmp_path, monkeypatch):
    """No predictor anywhere in explicit arg, env, or sys.path → None."""
    monkeypatch.delenv(ENV_KEY, raising=False)
    # Temporarily replace sys.path with entries that contain no predictor.py
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    result = load_predictor()
    assert result is None


def test_returns_none_when_explicit_dir_has_no_predictor(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    # tmp_path exists but contains no predictor.py
    result = load_predictor(predictor_dir=tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# Resolution: explicit predictor_dir argument
# ---------------------------------------------------------------------------


def test_loads_from_explicit_dir(tmp_path, monkeypatch):
    _write_predictor(tmp_path)
    fn = load_predictor(predictor_dir=tmp_path)
    assert callable(fn)
    assert fn({}) == {"ok": True}


def test_explicit_dir_accepts_string(tmp_path, monkeypatch):
    _write_predictor(tmp_path)
    fn = load_predictor(predictor_dir=str(tmp_path))
    assert callable(fn)


# ---------------------------------------------------------------------------
# Resolution: KITCHEN_PREDICTOR_DIR env var
# ---------------------------------------------------------------------------


def test_loads_from_env_var(tmp_path, monkeypatch):
    _write_predictor(tmp_path)
    monkeypatch.setenv(ENV_KEY, str(tmp_path))
    fn = load_predictor()
    assert callable(fn)
    assert fn({}) == {"ok": True}


def test_env_var_returns_none_when_predictor_missing(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_KEY, str(tmp_path))  # dir exists, no predictor.py
    result = load_predictor()
    assert result is None


# ---------------------------------------------------------------------------
# Resolution: sys.path scan fallback
# ---------------------------------------------------------------------------


def test_loads_from_sys_path_entry(tmp_path, monkeypatch):
    _write_predictor(tmp_path)
    monkeypatch.delenv(ENV_KEY, raising=False)
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    fn = load_predictor()
    assert callable(fn)


# ---------------------------------------------------------------------------
# Resolution: precedence order
# ---------------------------------------------------------------------------


def test_explicit_arg_overrides_env_var(tmp_path, monkeypatch):
    """Explicit predictor_dir must win over KITCHEN_PREDICTOR_DIR."""
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    _write_predictor(good_dir, "def predict(p): return {'source': 'good'}")

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    _write_predictor(bad_dir, "def predict(p): return {'source': 'bad'}")

    monkeypatch.setenv(ENV_KEY, str(bad_dir))
    fn = load_predictor(predictor_dir=good_dir)
    assert fn({})["source"] == "good"


def test_env_var_overrides_sys_path(tmp_path, monkeypatch):
    """KITCHEN_PREDICTOR_DIR must win over sys.path scan."""
    env_dir = tmp_path / "env_dir"
    env_dir.mkdir()
    _write_predictor(env_dir, "def predict(p): return {'source': 'env'}")

    path_dir = tmp_path / "path_dir"
    path_dir.mkdir()
    _write_predictor(path_dir, "def predict(p): return {'source': 'path'}")

    monkeypatch.setenv(ENV_KEY, str(env_dir))
    monkeypatch.setattr(sys, "path", [str(path_dir)])
    fn = load_predictor()
    assert fn({})["source"] == "env"


# ---------------------------------------------------------------------------
# Successful load: callable works correctly
# ---------------------------------------------------------------------------


def test_loaded_predict_fn_receives_payload(tmp_path):
    _write_predictor(tmp_path, "def predict(payload): return {'echo': payload}")
    fn = load_predictor(predictor_dir=tmp_path)
    result = fn({"x": 1, "y": "hello"})
    assert result == {"echo": {"x": 1, "y": "hello"}}


def test_loaded_predict_fn_is_the_module_function(tmp_path):
    _write_predictor(tmp_path, "def predict(payload): return {'v': 42}")
    fn = load_predictor(predictor_dir=tmp_path)
    assert fn.__name__ == "predict"


# ---------------------------------------------------------------------------
# Sibling import support
# ---------------------------------------------------------------------------


def test_sibling_import_resolves(tmp_path):
    """predictor.py may import other modules in its directory."""
    (tmp_path / "utils.py").write_text("CONSTANT = 99\n")
    _write_predictor(
        tmp_path,
        "from utils import CONSTANT\ndef predict(payload): return {'val': CONSTANT}\n",
    )
    fn = load_predictor(predictor_dir=tmp_path)
    assert fn({}) == {"val": 99}


def test_sibling_import_multi_level(tmp_path):
    """Predictor can import a sibling that itself imports a stdlib module."""
    (tmp_path / "helpers.py").write_text(
        "import json\ndef encode(d): return json.dumps(d)\n"
    )
    _write_predictor(
        tmp_path,
        "from helpers import encode\ndef predict(payload): return {'encoded': encode(payload)}\n",
    )
    fn = load_predictor(predictor_dir=tmp_path)
    import json

    result = fn({"a": 1})
    assert result["encoded"] == json.dumps({"a": 1})


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_raises_when_predict_fn_missing(tmp_path):
    """predictor.py found but no 'predict' function → PredictorLoadError."""
    _write_predictor(tmp_path, "# no predict here\nANSWER = 42\n")
    with pytest.raises(PredictorLoadError, match="no 'predict' function"):
        load_predictor(predictor_dir=tmp_path)


def test_raises_when_predict_not_callable(tmp_path):
    """predict exists but is not callable → PredictorLoadError."""
    _write_predictor(tmp_path, "predict = 'not a function'\n")
    with pytest.raises(PredictorLoadError, match="not callable"):
        load_predictor(predictor_dir=tmp_path)


def test_raises_when_predict_is_integer(tmp_path):
    _write_predictor(tmp_path, "predict = 42\n")
    with pytest.raises(PredictorLoadError, match="not callable"):
        load_predictor(predictor_dir=tmp_path)


def test_raises_on_module_import_error(tmp_path):
    """predictor.py raises at module level → PredictorLoadError wrapping the cause."""
    _write_predictor(tmp_path, "raise RuntimeError('model weights not found')\n")
    with pytest.raises(PredictorLoadError, match="model weights not found"):
        load_predictor(predictor_dir=tmp_path)


def test_raises_on_syntax_error(tmp_path):
    """predictor.py has a syntax error → PredictorLoadError."""
    _write_predictor(tmp_path, "def predict(payload)\n    return {}\n")  # missing ':'
    with pytest.raises(PredictorLoadError):
        load_predictor(predictor_dir=tmp_path)


def test_load_error_chained_to_original_exception(tmp_path):
    """PredictorLoadError.__cause__ is set so the original traceback is preserved."""
    _write_predictor(tmp_path, "import this_module_does_not_exist_abc123\n")
    with pytest.raises(PredictorLoadError) as exc_info:
        load_predictor(predictor_dir=tmp_path)
    assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# Public re-export from kitchen.serve
# ---------------------------------------------------------------------------


def test_public_api_re_exported():
    """load_predictor and PredictorLoadError must be importable from kitchen.serve."""
    from kitchen.serve import PredictorLoadError as PLE
    from kitchen.serve import load_predictor as lp

    assert lp is load_predictor
    assert PLE is PredictorLoadError


def test_predict_fn_type_alias_exported():
    from kitchen.serve import PredictFn  # noqa: F401 (import is the assertion)


# ---------------------------------------------------------------------------
# PredictorBundle — load_predictor_bundle
# ---------------------------------------------------------------------------


def test_load_predictor_bundle_returns_none_when_no_predictor(tmp_path, monkeypatch):
    """No predictor.py anywhere → None."""
    monkeypatch.delenv(ENV_KEY, raising=False)
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    from kitchen.serve.loader import load_predictor_bundle

    result = load_predictor_bundle()
    assert result is None


def test_load_predictor_bundle_returns_bundle(tmp_path):
    """A valid predictor.py → PredictorBundle with predict_fn set."""
    from kitchen.serve.loader import PredictorBundle, load_predictor_bundle

    _write_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    assert bundle is not None
    assert isinstance(bundle, PredictorBundle)
    assert callable(bundle.predict_fn)


def test_bundle_has_no_models_when_not_exported(tmp_path):
    """Predictor without RequestModel/ResponseModel → bundle.{request,response}_model is None."""
    from kitchen.serve.loader import load_predictor_bundle

    _write_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    assert bundle is not None
    assert bundle.request_model is None
    assert bundle.response_model is None


def test_bundle_has_request_model_when_exported(tmp_path):
    """Predictor that exports RequestModel → bundle.request_model is the class."""
    from kitchen.serve.loader import load_predictor_bundle

    (tmp_path / "predictor.py").write_text(
        "from pydantic import BaseModel\n"
        "class RequestModel(BaseModel):\n"
        "    x: float\n"
        "def predict(payload): return {'y': payload['x']}\n"
    )
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    assert bundle is not None
    assert bundle.request_model is not None
    assert bundle.request_model.__name__ == "RequestModel"


def test_bundle_has_response_model_when_exported(tmp_path):
    """Predictor that exports ResponseModel → bundle.response_model is the class."""
    from kitchen.serve.loader import load_predictor_bundle

    (tmp_path / "predictor.py").write_text(
        "from pydantic import BaseModel\n"
        "class ResponseModel(BaseModel):\n"
        "    label: int\n"
        "def predict(payload): return {'label': 1}\n"
    )
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    assert bundle is not None
    assert bundle.response_model is not None
    assert bundle.response_model.__name__ == "ResponseModel"


def test_bundle_has_both_models_when_both_exported(tmp_path):
    """Both models present → bundle carries them."""
    from kitchen.serve.loader import load_predictor_bundle

    (tmp_path / "predictor.py").write_text(
        "from pydantic import BaseModel\n"
        "class RequestModel(BaseModel):\n"
        "    x: float\n"
        "class ResponseModel(BaseModel):\n"
        "    label: int\n"
        "def predict(payload): return {'label': int(payload['x'] > 0)}\n"
    )
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    assert bundle is not None
    assert bundle.request_model is not None
    assert bundle.response_model is not None


def test_bundle_request_model_not_basemodel_raises(tmp_path):
    """RequestModel present but not a Pydantic BaseModel subclass → PredictorLoadError."""
    from kitchen.serve.loader import PredictorLoadError, load_predictor_bundle

    (tmp_path / "predictor.py").write_text(
        "RequestModel = 'not a class'\n"
        "def predict(payload): return {}\n"
    )
    with pytest.raises(PredictorLoadError, match="RequestModel"):
        load_predictor_bundle(predictor_dir=tmp_path)


def test_bundle_response_model_not_basemodel_raises(tmp_path):
    """ResponseModel present but not a Pydantic BaseModel subclass → PredictorLoadError."""
    from kitchen.serve.loader import PredictorLoadError, load_predictor_bundle

    (tmp_path / "predictor.py").write_text(
        "ResponseModel = 42\n"
        "def predict(payload): return {}\n"
    )
    with pytest.raises(PredictorLoadError, match="ResponseModel"):
        load_predictor_bundle(predictor_dir=tmp_path)


def test_load_predictor_delegates_to_bundle(tmp_path):
    """load_predictor() still returns the callable (thin wrapper around bundle)."""
    _write_predictor(tmp_path, "def predict(payload): return {'v': 7}")
    fn = load_predictor(predictor_dir=tmp_path)
    assert callable(fn)
    assert fn({}) == {"v": 7}


# ---------------------------------------------------------------------------
# Public re-export — bundle additions
# ---------------------------------------------------------------------------


def test_predictor_bundle_exported_from_serve():
    """PredictorBundle must be importable from kitchen.serve."""
    from kitchen.serve import PredictorBundle  # noqa: F401 (import is the assertion)


def test_load_predictor_bundle_exported_from_serve():
    """load_predictor_bundle must be importable from kitchen.serve."""
    from kitchen.serve import load_predictor_bundle as lpb  # noqa: F401


# ---------------------------------------------------------------------------
# FEATURES extraction — _extract_features / PredictorBundle.features
# ---------------------------------------------------------------------------


def test_bundle_features_is_none_when_not_exported(tmp_path):
    """No FEATURES in predictor.py → bundle.features is None."""
    from kitchen.serve.loader import load_predictor_bundle

    _write_predictor(tmp_path)
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    assert bundle is not None
    assert bundle.features is None


def test_bundle_features_returned_when_valid_list(tmp_path):
    """FEATURES = ['a', 'b'] → bundle.features == ['a', 'b']."""
    from kitchen.serve.loader import load_predictor_bundle

    (tmp_path / "predictor.py").write_text(
        "FEATURES = ['sepal_length', 'sepal_width']\n"
        "def predict(payload): return {}\n"
    )
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    assert bundle is not None
    assert bundle.features == ["sepal_length", "sepal_width"]


def test_bundle_features_empty_list_is_valid(tmp_path):
    """FEATURES = [] is a valid (empty) list — no error, bundle.features == []."""
    from kitchen.serve.loader import load_predictor_bundle

    (tmp_path / "predictor.py").write_text(
        "FEATURES = []\n"
        "def predict(payload): return {}\n"
    )
    bundle = load_predictor_bundle(predictor_dir=tmp_path)
    assert bundle is not None
    assert bundle.features == []


def test_bundle_features_not_a_list_raises(tmp_path):
    """FEATURES = 'a,b' (a string, not a list) → PredictorLoadError."""
    from kitchen.serve.loader import PredictorLoadError, load_predictor_bundle

    (tmp_path / "predictor.py").write_text(
        "FEATURES = 'feature_a,feature_b'\n"
        "def predict(payload): return {}\n"
    )
    with pytest.raises(PredictorLoadError, match="FEATURES"):
        load_predictor_bundle(predictor_dir=tmp_path)


def test_bundle_features_list_with_non_strings_raises(tmp_path):
    """FEATURES = [1, 2] (integers, not strings) → PredictorLoadError."""
    from kitchen.serve.loader import PredictorLoadError, load_predictor_bundle

    (tmp_path / "predictor.py").write_text(
        "FEATURES = [1, 2, 3]\n"
        "def predict(payload): return {}\n"
    )
    with pytest.raises(PredictorLoadError, match="FEATURES"):
        load_predictor_bundle(predictor_dir=tmp_path)


def test_bundle_features_mixed_types_raises(tmp_path):
    """FEATURES = ['a', 1] (mixed str + int) → PredictorLoadError."""
    from kitchen.serve.loader import PredictorLoadError, load_predictor_bundle

    (tmp_path / "predictor.py").write_text(
        "FEATURES = ['valid', 42]\n"
        "def predict(payload): return {}\n"
    )
    with pytest.raises(PredictorLoadError, match="FEATURES"):
        load_predictor_bundle(predictor_dir=tmp_path)


# --- lazy_model (S-005) ---


def test_lazy_model_defers_load_until_first_use():
    from kitchen.serve import lazy_model

    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return object()

    m = lazy_model(loader)
    assert m.loaded is False
    assert calls["n"] == 0  # not loaded at construction


def test_lazy_model_proxies_attributes_and_loads_once():
    from kitchen.serve import lazy_model

    calls = {"n": 0}

    class Model:
        def predict(self, X):
            return [sum(X[0])]

    def loader():
        calls["n"] += 1
        return Model()

    m = lazy_model(loader)
    assert m.predict([[1, 2, 3]]) == [6]  # transparent proxy triggers load
    assert m.loaded is True
    assert calls["n"] == 1
    m.predict([[4, 5]])  # cached
    m.unwrap()
    assert calls["n"] == 1  # never reloads


def test_lazy_model_unwrap_returns_underlying():
    from kitchen.serve import lazy_model

    sentinel = object()
    m = lazy_model(lambda: sentinel)
    assert m.unwrap() is sentinel
    assert m.loaded is True


# --- load_champion (MNT-003) ---


def _fake_mlflow_loader(monkeypatch, load_model):
    """Make importlib.import_module('mlflow.<flavor>') return a loader stub."""
    import types

    fake = types.SimpleNamespace(load_model=load_model)
    monkeypatch.setattr("importlib.import_module", lambda name: fake)


def test_load_champion_returns_model_on_success(monkeypatch):
    from kitchen.serve import load_champion

    sentinel = object()
    _fake_mlflow_loader(monkeypatch, lambda uri: sentinel)
    assert load_champion("models:/proj@champion") is sentinel


def test_load_champion_translates_artifact_drift(monkeypatch):
    from kitchen.serve import load_champion
    from kitchen.tracking import ArtifactLocationError

    def boom(uri):
        raise OSError("no such file")

    _fake_mlflow_loader(monkeypatch, boom)
    monkeypatch.setattr(
        "kitchen.tracking.explain_model_load_error",
        lambda uri, exc: ArtifactLocationError("unreachable artifact"),
    )
    with pytest.raises(ArtifactLocationError, match="unreachable artifact"):
        load_champion("models:/proj@champion")


def test_load_champion_passes_through_unrelated_errors(monkeypatch):
    from kitchen.serve import load_champion

    def boom(uri):
        raise ValueError("genuinely broken")

    _fake_mlflow_loader(monkeypatch, boom)
    monkeypatch.setattr("kitchen.tracking.explain_model_load_error", lambda uri, exc: None)
    with pytest.raises(ValueError, match="genuinely broken"):
        load_champion("models:/proj@champion")
