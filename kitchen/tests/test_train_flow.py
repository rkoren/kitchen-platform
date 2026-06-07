"""Tests for kitchen.flows.train_flow — _build, _train, and train_pipeline.

Project modules ``src.features.run`` and ``src.train.run`` are injected into
``sys.modules`` so the dynamic imports inside each function resolve without a
real project directory present.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import yaml

from kitchen.flows.train_flow import (
    EXPERIMENT,
    _apply_overrides,
    _build,
    _train,
    train_pipeline,
)

# ---------------------------------------------------------------------------
# Shared params fixture
# ---------------------------------------------------------------------------

PARAMS = {"experiment": "test-exp", "features": {"raw_file": "train.csv"}, "model": {}}


# ---------------------------------------------------------------------------
# Module injection helpers
# ---------------------------------------------------------------------------


def _inject_features(monkeypatch, raises=None):
    """Inject a fake src.features.run; return list that records build() calls."""
    calls: list[tuple] = []

    def fake_build(params, store):
        if raises is not None:
            raise raises
        calls.append((params, store))

    fake_mod = type(sys)("src.features.run")
    fake_mod.build = fake_build
    monkeypatch.setitem(sys.modules, "src.features.run", fake_mod)
    return calls


def _inject_train_mod(monkeypatch, raises=None):
    """Inject a fake src.train.run; return list that records train() calls."""
    calls: list[tuple] = []

    def fake_train(params, store, tracker):
        if raises is not None:
            raise raises
        calls.append((params, store, tracker))

    fake_mod = type(sys)("src.train.run")
    fake_mod.train = fake_train
    monkeypatch.setitem(sys.modules, "src.train.run", fake_mod)
    return calls


# ---------------------------------------------------------------------------
# Fixture: patch kitchen infra for _train
# ---------------------------------------------------------------------------


@pytest.fixture()
def train_env():
    """Patches that let _train() run without a live MLflow server."""
    mock_tracker_cls = MagicMock()
    with (
        patch("kitchen.flows.train_flow.configure_from_env") as mock_configure,
        patch("kitchen.flows.train_flow.init_experiment") as mock_init,
        patch("kitchen.flows.train_flow.Tracker", mock_tracker_cls),
    ):
        yield {
            "configure_from_env": mock_configure,
            "init_experiment": mock_init,
            "Tracker": mock_tracker_cls,
        }


# ---------------------------------------------------------------------------
# _build
# ---------------------------------------------------------------------------


def test_build_calls_project_build(monkeypatch):
    """_build() calls build(params, store) from src.features.run."""
    calls = _inject_features(monkeypatch)
    _build(PARAMS)
    assert len(calls) == 1


def test_build_passes_params_unchanged(monkeypatch):
    """_build forwards the params dict to build() without modification."""
    calls = _inject_features(monkeypatch)
    _build(PARAMS)
    received_params, _ = calls[0]
    assert received_params is PARAMS


def test_build_passes_datastore_instance(monkeypatch):
    """_build passes a DataStore instance as the second argument to build()."""
    from kitchen.store import DataStore

    calls = _inject_features(monkeypatch)
    _build(PARAMS)
    _, store = calls[0]
    assert isinstance(store, DataStore)


def test_build_missing_module_raises(monkeypatch):
    """_build raises ModuleNotFoundError when src.features.run is absent."""
    monkeypatch.delitem(sys.modules, "src.features.run", raising=False)
    with pytest.raises((ImportError, ModuleNotFoundError)):
        _build(PARAMS)


def test_build_propagates_build_exception(monkeypatch):
    """Exceptions raised by the project's build() are not swallowed."""
    _inject_features(monkeypatch, raises=RuntimeError("feature error"))
    with pytest.raises(RuntimeError, match="feature error"):
        _build(PARAMS)


# ---------------------------------------------------------------------------
# _train
# ---------------------------------------------------------------------------


def test_train_calls_project_train(monkeypatch, train_env):
    """_train() calls train(params, store, tracker) from src.train.run."""
    calls = _inject_train_mod(monkeypatch)
    _train(PARAMS)
    assert len(calls) == 1


def test_train_passes_params_unchanged(monkeypatch, train_env):
    """_train forwards params to the project's train() without modification."""
    calls = _inject_train_mod(monkeypatch)
    _train(PARAMS)
    received_params, _, _ = calls[0]
    assert received_params is PARAMS


def test_train_passes_datastore_instance(monkeypatch, train_env):
    """_train passes a DataStore instance as the second argument to train()."""
    from kitchen.store import DataStore

    calls = _inject_train_mod(monkeypatch)
    _train(PARAMS)
    _, store, _ = calls[0]
    assert isinstance(store, DataStore)


def test_train_passes_tracker_instance(monkeypatch, train_env):
    """_train passes the Tracker constructed from the experiment name to train()."""
    calls = _inject_train_mod(monkeypatch)
    _train(PARAMS)
    _, _, tracker = calls[0]
    # Tracker() was called — the mock instance was passed through
    train_env["Tracker"].assert_called_once()
    assert tracker is train_env["Tracker"].return_value


def test_train_experiment_name_from_params(monkeypatch, train_env):
    """_train uses params['experiment'] as the MLflow experiment name."""
    _inject_train_mod(monkeypatch)
    _train(PARAMS)
    train_env["Tracker"].assert_called_once_with("test-exp")
    train_env["init_experiment"].assert_called_once_with("test-exp")


def test_train_experiment_fallback_to_module_default(monkeypatch, train_env):
    """_train falls back to the EXPERIMENT constant when params has no 'experiment' key."""
    _inject_train_mod(monkeypatch)
    _train({"model": {}})  # no experiment key
    train_env["Tracker"].assert_called_once_with(EXPERIMENT)


def test_train_calls_configure_from_env(monkeypatch, train_env):
    """_train calls configure_from_env() to wire MLflow tracking URI from the environment."""
    _inject_train_mod(monkeypatch)
    _train(PARAMS)
    train_env["configure_from_env"].assert_called_once()


def test_train_calls_init_experiment(monkeypatch, train_env):
    """_train calls init_experiment() so the MLflow experiment exists before logging."""
    _inject_train_mod(monkeypatch)
    _train(PARAMS)
    train_env["init_experiment"].assert_called_once()


def test_train_missing_module_raises(monkeypatch, train_env):
    """_train raises ModuleNotFoundError when src.train.run is absent."""
    monkeypatch.delitem(sys.modules, "src.train.run", raising=False)
    with pytest.raises((ImportError, ModuleNotFoundError)):
        _train(PARAMS)


def test_train_propagates_train_exception(monkeypatch, train_env):
    """Exceptions raised by the project's train() are not swallowed."""
    _inject_train_mod(monkeypatch, raises=ValueError("bad params"))
    with pytest.raises(ValueError, match="bad params"):
        _train(PARAMS)


# ---------------------------------------------------------------------------
# train_pipeline (wiring)
# ---------------------------------------------------------------------------


def _write_params(tmp_path, params=None):
    path = tmp_path / "params.yaml"
    path.write_text(yaml.dump(params or PARAMS))
    return str(path)


def test_pipeline_calls_build_and_train(tmp_path, monkeypatch):
    """train_pipeline() calls _build then _train."""
    monkeypatch.chdir(tmp_path)
    params_file = _write_params(tmp_path)
    with (
        patch("kitchen.flows.train_flow._build") as mock_build,
        patch("kitchen.flows.train_flow._train") as mock_train,
    ):
        train_pipeline(params_file=params_file)
    mock_build.assert_called_once()
    mock_train.assert_called_once()


def test_pipeline_calls_build_before_train(tmp_path, monkeypatch):
    """_build is always invoked before _train."""
    monkeypatch.chdir(tmp_path)
    params_file = _write_params(tmp_path)
    order: list[str] = []
    with (
        patch("kitchen.flows.train_flow._build", side_effect=lambda p: order.append("build")),
        patch("kitchen.flows.train_flow._train", side_effect=lambda p, **kw: order.append("train")),
    ):
        train_pipeline(params_file=params_file)
    assert order == ["build", "train"]


def test_pipeline_passes_parsed_params_to_tasks(tmp_path, monkeypatch):
    """Both _build and _train receive the dict parsed from params.yaml."""
    monkeypatch.chdir(tmp_path)
    params_file = _write_params(tmp_path)
    received: dict[str, dict] = {}
    with (
        patch("kitchen.flows.train_flow._build", side_effect=lambda p: received.update({"build": p})),
        patch("kitchen.flows.train_flow._train", side_effect=lambda p, **kw: received.update({"train": p})),
    ):
        train_pipeline(params_file=params_file)
    assert received["build"]["experiment"] == "test-exp"
    assert received["build"] == received["train"]


def test_pipeline_default_reads_params_yaml_from_cwd(tmp_path, monkeypatch):
    """Without an explicit params_file, train_pipeline reads params.yaml from cwd."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(yaml.dump(PARAMS))
    with (
        patch("kitchen.flows.train_flow._build") as mock_build,
        patch("kitchen.flows.train_flow._train"),
    ):
        train_pipeline()  # default params_file="params.yaml"
    mock_build.assert_called_once()


def test_pipeline_missing_params_file_raises(tmp_path, monkeypatch):
    """train_pipeline() raises when the params file does not exist."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises((FileNotFoundError, OSError)):
        train_pipeline(params_file="nonexistent.yaml")


# ---------------------------------------------------------------------------
# run_id return value (SWEEP-005)
# ---------------------------------------------------------------------------


def test_train_returns_active_run_id(monkeypatch, train_env):
    """_train returns the active MLflow run's run_id (used by `kitchen sweep`)."""
    _inject_train_mod(monkeypatch)
    active_run = train_env["Tracker"].return_value.run.return_value.__enter__.return_value
    active_run.info.run_id = "run-xyz"
    assert _train(PARAMS) == "run-xyz"


def test_pipeline_propagates_run_id(tmp_path, monkeypatch):
    """train_pipeline returns whatever run_id _train produced."""
    monkeypatch.chdir(tmp_path)
    params_file = _write_params(tmp_path)
    with (
        patch("kitchen.flows.train_flow._build"),
        patch("kitchen.flows.train_flow._train", return_value="run-abc"),
    ):
        assert train_pipeline(params_file=params_file) == "run-abc"


# ---------------------------------------------------------------------------
# _apply_overrides (SWEEP-001)
# ---------------------------------------------------------------------------


def test_apply_overrides_flat_key():
    params = {"experiment": "test", "alpha": 0.1}
    _apply_overrides(params, {"alpha": 0.5})
    assert params["alpha"] == 0.5


def test_apply_overrides_nested_key():
    params = {"model": {"max_depth": 3, "eta": 0.1}}
    _apply_overrides(params, {"model.max_depth": 6})
    assert params["model"]["max_depth"] == 6
    assert params["model"]["eta"] == 0.1  # unchanged


def test_apply_overrides_creates_missing_intermediate_dicts():
    params: dict = {}
    _apply_overrides(params, {"new_section.key": 42})
    assert params["new_section"]["key"] == 42


def test_apply_overrides_multiple_keys():
    params = {"model": {"max_depth": 3, "eta": 0.1}}
    _apply_overrides(params, {"model.max_depth": 8, "model.eta": 0.05})
    assert params["model"]["max_depth"] == 8
    assert params["model"]["eta"] == 0.05


def test_apply_overrides_does_not_touch_other_keys():
    params = {"experiment": "proj", "model": {"max_depth": 3}}
    _apply_overrides(params, {"model.max_depth": 6})
    assert params["experiment"] == "proj"


# ---------------------------------------------------------------------------
# train_pipeline with overrides (SWEEP-001)
# ---------------------------------------------------------------------------


def test_pipeline_applies_overrides_to_params(tmp_path, monkeypatch):
    """train_pipeline mutates params before passing to tasks when overrides are given."""
    monkeypatch.chdir(tmp_path)
    params_file = _write_params(tmp_path, {"experiment": "test-exp", "model": {"max_depth": 3}})
    received: dict = {}
    with (
        patch("kitchen.flows.train_flow._build", side_effect=lambda p: received.update({"build": p})),
        patch("kitchen.flows.train_flow._train", side_effect=lambda p, **kw: None),
    ):
        train_pipeline(params_file=params_file, overrides={"model.max_depth": 6})
    assert received["build"]["model"]["max_depth"] == 6


def test_pipeline_passes_overrides_to_train_task(tmp_path, monkeypatch):
    """train_pipeline passes the overrides dict to _train as a keyword argument."""
    monkeypatch.chdir(tmp_path)
    params_file = _write_params(tmp_path)
    captured: dict = {}
    with (
        patch("kitchen.flows.train_flow._build"),
        patch(
            "kitchen.flows.train_flow._train",
            side_effect=lambda p, overrides=None: captured.update({"overrides": overrides}),
        ),
    ):
        train_pipeline(params_file=params_file, overrides={"model.max_depth": 6})
    assert captured["overrides"] == {"model.max_depth": 6}


def test_pipeline_no_overrides_passes_none_to_train(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params_file = _write_params(tmp_path)
    captured: dict = {}
    with (
        patch("kitchen.flows.train_flow._build"),
        patch(
            "kitchen.flows.train_flow._train",
            side_effect=lambda p, overrides=None: captured.update({"overrides": overrides}),
        ),
    ):
        train_pipeline(params_file=params_file)
    assert captured["overrides"] is None


# ---------------------------------------------------------------------------
# _train override tag logging (SWEEP-001)
# ---------------------------------------------------------------------------


def test_train_logs_override_tags(monkeypatch, train_env):
    """_train logs override keys as MLflow tags with the 'override.' prefix."""
    _inject_train_mod(monkeypatch)
    with patch("kitchen.flows.train_flow.mlflow") as mock_mlflow:
        _train(PARAMS, overrides={"model.max_depth": 6, "model.eta": 0.05})
    mock_mlflow.set_tags.assert_called_once_with(
        {"override.model.max_depth": "6", "override.model.eta": "0.05"}
    )


def test_train_no_override_tags_when_no_overrides(monkeypatch, train_env):
    """_train does not call mlflow.set_tags when overrides is None."""
    _inject_train_mod(monkeypatch)
    with patch("kitchen.flows.train_flow.mlflow") as mock_mlflow:
        _train(PARAMS)
    mock_mlflow.set_tags.assert_not_called()


def test_train_opens_tracker_run(monkeypatch, train_env):
    """_train always opens tracker.run() regardless of whether overrides are present."""
    _inject_train_mod(monkeypatch)
    _train(PARAMS)
    tracker = train_env["Tracker"].return_value
    tracker.run.assert_called_once()
