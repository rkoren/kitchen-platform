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
    _metric_lines,
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
# CBB-016: variant tagging + composition
# ---------------------------------------------------------------------------


def test_train_tags_model_variant(monkeypatch, train_env):
    """_train tags the run with model_variant=<name> so leaderboard/diff group it."""
    _inject_train_mod(monkeypatch)
    with patch("kitchen.flows.train_flow.mlflow") as mock_mlflow:
        _train(PARAMS, variant="kenpom_rich")
    mock_mlflow.set_tag.assert_any_call("model_variant", "kenpom_rich")


def test_train_no_variant_sets_no_variant_tag(monkeypatch, train_env):
    """Existing projects (no --variant) get no model_variant tag — behavior unchanged."""
    _inject_train_mod(monkeypatch)
    with patch("kitchen.flows.train_flow.mlflow") as mock_mlflow:
        _train(PARAMS)
    tagged = [c.args[0] for c in mock_mlflow.set_tag.call_args_list if c.args]
    assert "model_variant" not in tagged


def test_pipeline_applies_variant_before_overrides(tmp_path, monkeypatch):
    """base → variant → override: the variant adds a feature + bumps max_depth, then the
    override wins on max_depth; the `variants:` definition is dropped from the run params."""
    monkeypatch.chdir(tmp_path)
    menu = tmp_path / "menu.yaml"
    menu.write_text(
        "project: p\npipeline: [train]\n"
        "recipes:\n  train: {kind: stage, source: src/train/run.py}\n"
        "model: {max_depth: 4}\nfeature_candidates: [a, b]\n"
        "variants:\n  rich:\n    model: {max_depth: 5}\n    feature_candidates: {add: [c]}\n"
    )
    seen: dict = {}
    with (
        patch("kitchen.flows.train_flow._build", side_effect=lambda p: seen.update(p)),
        patch("kitchen.flows.train_flow._train", side_effect=lambda p, **kw: seen.update(p)),
    ):
        train_pipeline(params_file=str(menu), variant="rich", overrides={"model.max_depth": 9})
    assert seen["feature_candidates"] == ["a", "b", "c"]  # variant add applied
    assert seen["model"]["max_depth"] == 9  # override wins over the variant's 5
    assert "variants" not in seen  # the definition isn't logged as a run param


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
            side_effect=lambda p, overrides=None, variant=None: captured.update({"overrides": overrides}),
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
            side_effect=lambda p, overrides=None, variant=None: captured.update({"overrides": overrides}),
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


# ---------------------------------------------------------------------------
# _metric_lines — train success-path summary (CBB-011)
# ---------------------------------------------------------------------------


def _fake_run(metrics):
    run = MagicMock()
    run.data.metrics = metrics
    return run


def test_metric_lines_threshold_pass():
    """A lower-is-better metric under its max prints a PASS line with the bound."""
    params = {"thresholds": {"loto_brier": {"max": 0.175}}}
    with patch(
        "kitchen.flows.train_flow.mlflow.get_run", return_value=_fake_run({"loto_brier": 0.1653})
    ):
        lines = _metric_lines("rid", params)
    assert lines == ["  loto_brier = 0.1653 — PASS (<= 0.175)"]


def test_metric_lines_threshold_fail():
    """A value above the max prints FAIL."""
    params = {"thresholds": {"loto_brier": {"max": 0.175}}}
    with patch(
        "kitchen.flows.train_flow.mlflow.get_run", return_value=_fake_run({"loto_brier": 0.20})
    ):
        lines = _metric_lines("rid", params)
    assert "FAIL" in lines[0] and "<= 0.175" in lines[0]


def test_metric_lines_bare_float_is_lower_bound():
    """A bare-number threshold is a >= lower bound (higher-is-better convention)."""
    params = {"thresholds": {"acc": 0.8}}
    with patch("kitchen.flows.train_flow.mlflow.get_run", return_value=_fake_run({"acc": 0.9})):
        lines = _metric_lines("rid", params)
    assert lines == ["  acc = 0.9 — PASS (>= 0.8)"]


def test_metric_lines_headline_fallback_caps_and_counts():
    """With no thresholds, show up to 8 metrics + a (+N more) pointer."""
    metrics = {f"m{i}": float(i) for i in range(10)}
    with patch("kitchen.flows.train_flow.mlflow.get_run", return_value=_fake_run(metrics)):
        lines = _metric_lines("rid", {})
    assert len(lines) == 9
    assert "more — see kitchen leaderboard" in lines[-1]


def test_metric_lines_excludes_feature_importance():
    """fi.* keys never appear in the summary."""
    with patch(
        "kitchen.flows.train_flow.mlflow.get_run",
        return_value=_fake_run({"loto_brier": 0.16, "fi.team_a": 1.0}),
    ):
        lines = _metric_lines("rid", {})
    assert all("fi." not in line for line in lines)


def test_metric_lines_empty_when_run_unreadable():
    """A store/read failure degrades to [] (caller prints the legacy message)."""
    with patch("kitchen.flows.train_flow.mlflow.get_run", side_effect=RuntimeError("no store")):
        assert _metric_lines("rid", {}) == []


def test_train_prints_metric_summary(monkeypatch, train_env, capsys):
    """_train prints the threshold summary on success instead of the bare message."""
    _inject_train_mod(monkeypatch)
    active = train_env["Tracker"].return_value.run.return_value.__enter__.return_value
    active.info.run_id = "rid"
    with patch(
        "kitchen.flows.train_flow.mlflow.get_run", return_value=_fake_run({"loto_brier": 0.1653})
    ):
        _train({"experiment": "e", "thresholds": {"loto_brier": {"max": 0.175}}})
    out = capsys.readouterr().out
    assert "Training complete:" in out
    assert "loto_brier = 0.1653 — PASS" in out


# ---------------------------------------------------------------------------
# CBB-017: holdout scoring wired into the train success path
# ---------------------------------------------------------------------------


def test_train_scores_configured_holdout(monkeypatch, train_env, capsys):
    """_train scores the holdout after the run and surfaces it in the summary."""
    _inject_train_mod(monkeypatch)
    params = {
        "experiment": "e",
        "holdout": {"path": "data/holdout/h.parquet", "label": "Outcome"},
        "mlflow": {"model_artifact_path": "cbb_model"},
    }
    with patch(
        "kitchen.holdout.score_run_holdout",
        return_value={"holdout_brier": 0.0985, "holdout_n": 134.0},
    ) as mock_score:
        _train(params)
    # called once with the run id and the project's model_artifact_path threaded through
    assert mock_score.call_count == 1
    assert mock_score.call_args.kwargs["model_artifact_path"] == "cbb_model"
    out = capsys.readouterr().out
    assert "holdout_brier = 0.0985" in out and "never-trained-on" in out


def test_train_no_holdout_logs_no_lines(monkeypatch, train_env, capsys):
    """With no holdout configured, the scorer is a no-op ({}) → no holdout line in the summary."""
    _inject_train_mod(monkeypatch)
    with patch("kitchen.holdout.score_run_holdout", return_value={}):
        _train(PARAMS)
    assert "never-trained-on" not in capsys.readouterr().out


def test_train_holdout_failure_warns_not_crashes(monkeypatch, train_env, capsys):
    """A holdout-scoring failure is reported but never sinks an otherwise-good train run."""
    _inject_train_mod(monkeypatch)
    params = {"experiment": "e", "holdout": {"path": "h.parquet", "label": "Outcome"}}
    with patch(
        "kitchen.holdout.score_run_holdout", side_effect=RuntimeError("model load boom")
    ):
        run_id = _train(params)
    assert run_id is not None  # run still succeeds
    err = capsys.readouterr().err
    assert "holdout scoring failed" in err and "boom" in err
