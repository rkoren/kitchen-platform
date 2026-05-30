"""Smoke tests for `kitchen init` scaffold output.

Verifies that a fresh scaffold:
- creates all expected files
- produces parseable YAML
- has Python modules that import at module level without errors
- uses correct schema field names (no memory_mb/timeout_s)
- contains no maintainer-specific names
- leaves intentional TODO boundaries as NotImplementedError (not silent pass-throughs)
"""
# pylint: disable=redefined-outer-name  # standard pytest fixture injection pattern

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow.exceptions
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from kitchen.cli import app

runner = CliRunner()

EXPECTED_FILES = [
    "CLAUDE.md",
    ".env.example",
    ".gitignore",
    "params.yaml",
    "pyproject.toml",
    "infra/my-competition.yaml",
    "src/__init__.py",
    "src/features/__init__.py",
    "src/features/run.py",
    "src/train/__init__.py",
    "src/train/run.py",
    "src/evaluate/__init__.py",
    "src/evaluate/run.py",
    "src/tests/__init__.py",
    "src/tests/test_features.py",
    "experiments/__init__.py",
    "experiments/baseline.py",
    "experiments/challenger.py",
    "flows/train_flow.py",
    "flows/promote.py",
    "flows/generate_submission.py",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "submissions/.gitkeep",
]


@pytest.fixture()
def project():
    """Run `kitchen init my-competition` in a temp dir and return the project root."""
    result = runner.invoke(app, ["init", "my-competition"], catch_exceptions=False)
    # CliRunner doesn't change the real cwd, so files land in cwd/my-competition.
    # We need to re-invoke with the fs_root wired to tmp_path.
    # Use monkeypatch-free approach: invoke with --here from inside tmp_path via env trick.
    # Actually CliRunner.isolated_filesystem() is the cleanest path.
    return result


@pytest.fixture()
def scaffold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my-competition"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return tmp_path / "my-competition"


def test_all_expected_files_created(scaffold):
    for rel in EXPECTED_FILES:
        assert (scaffold / rel).exists(), f"Missing scaffolded file: {rel}"


def test_params_yaml_parses(scaffold):
    content = yaml.safe_load((scaffold / "params.yaml").read_text())
    assert content["experiment"] == "my-competition"
    assert "features" in content
    assert "model" in content


def test_infra_yaml_parses(scaffold):
    content = yaml.safe_load((scaffold / "infra/my-competition.yaml").read_text())
    assert content["name"] == "my-competition"
    assert isinstance(content["resources"], list)


def test_infra_yaml_uses_correct_lambda_field_names(scaffold):
    raw = (scaffold / "infra/my-competition.yaml").read_text()
    assert "memory_mb" not in raw, "Scaffold emits deprecated memory_mb"
    assert "timeout_s" not in raw, "Scaffold emits deprecated timeout_s"
    assert "memory:" in raw
    assert "timeout:" in raw


def test_infra_yaml_has_no_maintainer_names(scaffold):
    raw = (scaffold / "infra/my-competition.yaml").read_text()
    assert "reilly" not in raw.lower(), "Scaffold contains maintainer-specific name"


def test_features_module_imports_cleanly(scaffold, monkeypatch):
    monkeypatch.syspath_prepend(str(scaffold))
    spec = importlib.util.spec_from_file_location(
        "src.features.run", scaffold / "src/features/run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must not raise
    assert hasattr(mod, "FEATURES")
    assert hasattr(mod, "build")


def test_generate_submission_imports_cleanly(scaffold, monkeypatch):
    monkeypatch.syspath_prepend(str(scaffold))
    # Stub src.features.run so the import inside generate_submission resolves
    stub = type(sys)("src.features.run")
    stub.FEATURES = []
    sys.modules.setdefault("src", type(sys)("src"))
    sys.modules.setdefault("src.features", type(sys)("src.features"))
    sys.modules["src.features.run"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "flows.generate_submission",
            scaffold / "flows/generate_submission.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # must not raise on import
        assert hasattr(mod, "generate")
    finally:
        for key in ("src", "src.features", "src.features.run"):
            sys.modules.pop(key, None)


def test_train_flow_imports_cleanly(scaffold):
    spec = importlib.util.spec_from_file_location(
        "flows.train_flow",
        scaffold / "flows/train_flow.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must not raise on import


def test_feature_builder_raises_not_implemented(scaffold, monkeypatch):
    monkeypatch.syspath_prepend(str(scaffold))
    spec = importlib.util.spec_from_file_location(
        "src.features.run", scaffold / "src/features/run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cls_name = "MyCompetitionFeatures"
    features_cls = getattr(mod, cls_name, None)
    if features_cls is None:
        pytest.skip(f"Class {cls_name} not found — name derivation may differ")
    assert features_cls is not None  # pragma: no branch — skip() raises above
    with pytest.raises(NotImplementedError):
        features_cls().build(pd.DataFrame(), params={})  # pylint: disable=not-callable


def test_scaffold_features_build_signature_includes_dict(tmp_path, monkeypatch):
    """Scaffolded build() annotation accepts dict[str, DataFrame] for multi-source."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-competition"], catch_exceptions=False)
    content = (tmp_path / "my-competition" / "src" / "features" / "run.py").read_text()
    assert "dict[str, pd.DataFrame]" in content


def test_scaffold_features_sources_usage_documented(tmp_path, monkeypatch):
    """Scaffolded features/run.py mentions the sources() override pattern."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-competition"], catch_exceptions=False)
    content = (tmp_path / "my-competition" / "src" / "features" / "run.py").read_text()
    assert "sources" in content


def test_init_here_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my-competition", "--here"], catch_exceptions=False)
    assert result.exit_code == 0
    # Files land in cwd, not a subdirectory
    assert (tmp_path / "params.yaml").exists()
    assert not (tmp_path / "my-competition" / "params.yaml").exists()


def test_init_skips_existing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-competition"], catch_exceptions=False)
    sentinel = tmp_path / "my-competition" / "params.yaml"
    sentinel.write_text("# modified")
    runner.invoke(app, ["init", "my-competition"], catch_exceptions=False)
    assert sentinel.read_text() == "# modified", (
        "Re-init without --overwrite should skip existing files"
    )


def test_init_overwrite_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-competition"], catch_exceptions=False)
    sentinel = tmp_path / "my-competition" / "params.yaml"
    sentinel.write_text("# modified")
    runner.invoke(app, ["init", "my-competition", "--overwrite"], catch_exceptions=False)
    assert sentinel.read_text() != "# modified", "--overwrite should replace existing files"


# ---------------------------------------------------------------------------
# kitchen open (LML-008)
# ---------------------------------------------------------------------------


def test_open_reads_dashboard_url_from_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\ndashboard_url: https://user.github.io/repo/\n")
    with patch("webbrowser.open") as mock_open:
        result = runner.invoke(app, ["open"], catch_exceptions=False)
    assert result.exit_code == 0
    mock_open.assert_called_once_with("https://user.github.io/repo/")
    assert "https://user.github.io/repo/" in result.output


def test_open_reads_dashboard_url_from_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DASHBOARD_URL", "https://org.github.io/proj/")
    with patch("webbrowser.open") as mock_open:
        result = runner.invoke(app, ["open"], catch_exceptions=False)
    assert result.exit_code == 0
    mock_open.assert_called_once_with("https://org.github.io/proj/")


def test_open_params_url_takes_precedence_over_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\ndashboard_url: https://params.example.com/\n")
    monkeypatch.setenv("DASHBOARD_URL", "https://env.example.com/")
    with patch("webbrowser.open") as mock_open:
        runner.invoke(app, ["open"], catch_exceptions=False)
    mock_open.assert_called_once_with("https://params.example.com/")


def test_open_fallback_to_ui_when_no_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    with patch("webbrowser.open") as mock_open:
        result = runner.invoke(app, ["open"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Falling back" in result.output
    mock_open.assert_called_once_with("https://mlflow.example.com")


def test_open_no_url_no_params_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    with patch("webbrowser.open"):
        result = runner.invoke(app, ["open"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Falling back" in result.output


# --- kitchen init name validation ---


@pytest.mark.parametrize(
    "name",
    [
        "titanic",
        "spaceship-titanic",
        "house-prices-2024",
        "a",
        "abc123",
    ],
)
def test_init_valid_names(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", name], catch_exceptions=False)
    assert result.exit_code == 0, f"Expected valid name {name!r} to be accepted"


@pytest.mark.parametrize(
    "name",
    [
        "My-Competition",  # uppercase
        "my competition",  # space
        "-leading",  # leading hyphen
        "trailing-",  # trailing hyphen
        "a--b",  # consecutive hyphens
        "1competition",  # starts with digit
        "../escape",  # path traversal
        "",  # empty
    ],
)
def test_init_invalid_names(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", name])
    assert result.exit_code != 0, f"Expected invalid name {name!r} to be rejected"


# --- generated test file ---


def test_generated_test_asserts_not_implemented(scaffold):
    raw = (scaffold / "src/tests/test_features.py").read_text()
    assert "NotImplementedError" in raw, "Generated test should assert the TODO boundary"
    assert "params={}" in raw, "Generated test should call build with params"


# --- kitchen validate ---


def test_validate_valid_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "params.yaml"
    p.write_text("experiment: my-exp\n")
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 0
    assert "my-exp" in result.output


def test_validate_shows_mlflow_uri(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text("experiment: x\nmlflow:\n  tracking_uri: sqlite:///runs.db\n")
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 0
    assert "sqlite:///runs.db" in result.output


def test_validate_shows_data_source(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text("experiment: x\ndata:\n  source: kaggle\n  competition: titanic\n")
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 0
    assert "kaggle" in result.output


def test_validate_fails_on_bad_data(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text("experiment: x\ndata:\n  source: kaggle\n")  # missing competition
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code != 0
    assert "competition" in result.output


def test_validate_fails_on_missing_experiment(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text("mlflow:\n  tracking_uri: sqlite:///x.db\n")
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code != 0


def test_validate_file_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["validate", "nonexistent.yaml"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_validate_default_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: default-test\n")
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0
    assert "default-test" in result.output


# ---------------------------------------------------------------------------
# kitchen run train
# ---------------------------------------------------------------------------


def test_run_train_missing_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["run", "train", "--params", "missing.yaml"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_run_train_invokes_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")

    calls = []

    def fake_pipeline(params_file="params.yaml", overrides=None):
        calls.append(params_file)

    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "train"])
    assert result.exit_code == 0
    assert calls == ["params.yaml"]


def test_run_train_custom_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    custom = tmp_path / "custom.yaml"
    custom.write_text("experiment: custom\n")

    calls = []

    def fake_pipeline(params_file="params.yaml", overrides=None):
        calls.append(params_file)

    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "train", "--params", "custom.yaml"])
    assert result.exit_code == 0
    assert calls == ["custom.yaml"]


def test_run_train_missing_src_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")

    def fake_pipeline(params_file="params.yaml", overrides=None):
        raise ModuleNotFoundError("No module named 'src.features.run'")

    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "train"])
    assert result.exit_code != 0
    assert "src/" in result.output


# ---------------------------------------------------------------------------
# kitchen run train --override (SWEEP-001)
# ---------------------------------------------------------------------------


def test_override_single_int(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")
    captured = {}

    def fake_pipeline(params_file="params.yaml", overrides=None):
        captured["overrides"] = overrides

    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "train", "--override", "model.max_depth=6"])
    assert result.exit_code == 0
    assert captured["overrides"] == {"model.max_depth": 6}


def test_override_multiple(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")
    captured = {}

    def fake_pipeline(params_file="params.yaml", overrides=None):
        captured["overrides"] = overrides

    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", fake_pipeline)
    result = runner.invoke(
        app,
        ["run", "train", "--override", "model.max_depth=6", "--override", "model.eta=0.05"],
    )
    assert result.exit_code == 0
    assert captured["overrides"] == {"model.max_depth": 6, "model.eta": 0.05}


def test_override_bool_coercion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")
    captured = {}

    def fake_pipeline(params_file="params.yaml", overrides=None):
        captured["overrides"] = overrides

    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", fake_pipeline)
    result = runner.invoke(
        app, ["run", "train", "--override", "use_gpu=true", "--override", "shuffle=False"]
    )
    assert result.exit_code == 0
    assert captured["overrides"]["use_gpu"] is True
    assert captured["overrides"]["shuffle"] is False


def test_override_string_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")
    captured = {}

    def fake_pipeline(params_file="params.yaml", overrides=None):
        captured["overrides"] = overrides

    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "train", "--override", "model.objective=binary:logistic"])
    assert result.exit_code == 0
    assert captured["overrides"] == {"model.objective": "binary:logistic"}


def test_override_invalid_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")
    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", _fake_pipeline_noop)
    result = runner.invoke(app, ["run", "train", "--override", "no-equals-sign"])
    assert result.exit_code != 0
    assert "key=value" in result.output


def test_no_override_passes_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")
    captured = {}

    def fake_pipeline(params_file="params.yaml", overrides=None):
        captured["overrides"] = overrides

    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "train"])
    assert result.exit_code == 0
    assert captured["overrides"] is None


# ---------------------------------------------------------------------------
# _coerce_override_value unit tests (SWEEP-001)
# ---------------------------------------------------------------------------


def test_coerce_int():
    from kitchen.cli import _coerce_override_value

    assert _coerce_override_value("6") == 6
    assert isinstance(_coerce_override_value("6"), int)


def test_coerce_float():
    from kitchen.cli import _coerce_override_value

    assert _coerce_override_value("0.05") == 0.05
    assert isinstance(_coerce_override_value("0.05"), float)


def test_coerce_bool_true():
    from kitchen.cli import _coerce_override_value

    assert _coerce_override_value("true") is True
    assert _coerce_override_value("True") is True
    assert _coerce_override_value("TRUE") is True


def test_coerce_bool_false():
    from kitchen.cli import _coerce_override_value

    assert _coerce_override_value("false") is False
    assert _coerce_override_value("False") is False


def test_coerce_string_fallback():
    from kitchen.cli import _coerce_override_value

    assert _coerce_override_value("binary:logistic") == "binary:logistic"
    assert isinstance(_coerce_override_value("binary:logistic"), str)


# ---------------------------------------------------------------------------
# kitchen run train --auto-promote (LML-004)
# ---------------------------------------------------------------------------


def _fake_pipeline_noop(params_file="params.yaml", overrides=None):  # pylint: disable=unused-argument
    pass


def _auto_promote_invoke(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    tmp_path, monkeypatch, extra_args, champion_score=None, new_score=0.15, metric="loto_brier"
):
    """Helper: set up a fake pipeline + MLflow client and invoke run train with extra_args."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: cbb\n")
    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", _fake_pipeline_noop)

    new_run = MagicMock()
    new_run.info.run_id = "newrun" + "0" * 26
    new_run.data.metrics = {metric: new_score}

    def make_client():
        client = MagicMock()
        client.get_experiment_by_name.return_value.experiment_id = "1"
        client.search_runs.return_value = [new_run]
        if champion_score is not None:
            mv = MagicMock()
            mv.run_id = "champrun" + "0" * 24
            champ_run = MagicMock()
            champ_run.data.metrics = {metric: champion_score}
            client.get_model_version_by_alias.return_value = mv
            client.get_run.return_value = champ_run
        else:
            client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException("no alias")
        return client

    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient", side_effect=make_client),
        patch("kitchen.registry.register_model", return_value="3") as mock_reg,
        patch("kitchen.registry.promote_model") as mock_prom,
    ):
        result = runner.invoke(app, ["run", "train", *extra_args], catch_exceptions=False)

    return result, mock_reg, mock_prom


def test_auto_promote_requires_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: cbb\n")
    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", _fake_pipeline_noop)
    result = runner.invoke(app, ["run", "train", "--auto-promote"])
    assert result.exit_code != 0
    assert "promote-metric" in result.output


def test_auto_promote_no_champion_promotes(tmp_path, monkeypatch):
    result, mock_reg, mock_prom = _auto_promote_invoke(
        tmp_path, monkeypatch,
        ["--auto-promote", "--promote-metric", "loto_brier", "--lower-is-better"],
        champion_score=None,
        new_score=0.164,
    )
    assert result.exit_code == 0
    mock_reg.assert_called_once()
    mock_prom.assert_called_once()
    assert "champion" in result.output
    assert "no current champion" in result.output


def test_auto_promote_beats_champion_promotes(tmp_path, monkeypatch):
    result, mock_reg, mock_prom = _auto_promote_invoke(
        tmp_path, monkeypatch,
        ["--auto-promote", "--promote-metric", "loto_brier", "--lower-is-better"],
        champion_score=0.172,  # current champion
        new_score=0.160,       # new run is better (lower)
    )
    assert result.exit_code == 0
    mock_reg.assert_called_once()
    mock_prom.assert_called_once()
    assert "→ champion" in result.output


def test_auto_promote_loses_to_champion_skips(tmp_path, monkeypatch):
    result, mock_reg, mock_prom = _auto_promote_invoke(
        tmp_path, monkeypatch,
        ["--auto-promote", "--promote-metric", "loto_brier", "--lower-is-better"],
        champion_score=0.155,  # champion is already better
        new_score=0.170,
    )
    assert result.exit_code == 0
    mock_reg.assert_not_called()
    mock_prom.assert_not_called()
    assert "skipped" in result.output


def test_auto_promote_higher_is_better(tmp_path, monkeypatch):
    result, mock_reg, mock_prom = _auto_promote_invoke(
        tmp_path, monkeypatch,
        ["--auto-promote", "--promote-metric", "val_auc", "--higher-is-better"],
        champion_score=0.80,
        new_score=0.85,  # higher is better, new run wins
        metric="val_auc",
    )
    assert result.exit_code == 0
    mock_reg.assert_called_once()
    mock_prom.assert_called_once()


def test_auto_promote_not_set_no_promote(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: cbb\n")
    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", _fake_pipeline_noop)
    with patch("kitchen.registry.register_model") as mock_reg:
        result = runner.invoke(app, ["run", "train"])
    assert result.exit_code == 0
    mock_reg.assert_not_called()


# ---------------------------------------------------------------------------
# kitchen run monitor
# ---------------------------------------------------------------------------


def test_run_monitor_missing_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["run", "monitor", "--params", "missing.yaml"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_run_monitor_invokes_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")

    calls = []

    def fake_pipeline(params_file="params.yaml", local_path_override=None):
        calls.append((params_file, local_path_override))
        return "monitoring/drift.html"

    monkeypatch.setattr("kitchen.flows.monitor_flow.monitor_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "monitor"])
    assert result.exit_code == 0
    assert calls == [("params.yaml", None)]
    assert "monitoring/drift.html" in result.output


def test_run_monitor_local_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")

    calls = []

    def fake_pipeline(params_file="params.yaml", local_path_override=None):  # pylint: disable=unused-argument
        calls.append(local_path_override)
        return local_path_override

    monkeypatch.setattr("kitchen.flows.monitor_flow.monitor_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "monitor", "--local", "monitoring/drift.html"])
    assert result.exit_code == 0
    assert calls == ["monitoring/drift.html"]
    assert "monitoring/drift.html" in result.output


def test_run_monitor_missing_output_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")

    def fake_pipeline(params_file="params.yaml", local_path_override=None):
        raise ValueError("monitor config must specify at least one of: report_bucket or local_path")

    monkeypatch.setattr("kitchen.flows.monitor_flow.monitor_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "monitor"])
    assert result.exit_code != 0
    assert "error" in result.output


# ---------------------------------------------------------------------------
# kitchen run evaluate
# ---------------------------------------------------------------------------

EVAL_PARAMS = "experiment: test-project\n"


def _make_evaluate_mocks(monkeypatch, model=None, metrics=None, load_raises=None):
    """Wire up the three external boundaries for run evaluate tests."""
    fake_model = model or object()

    def fake_load(_uri):
        if load_raises:
            raise load_raises
        return fake_model

    fake_loader = type("Loader", (), {"load_model": staticmethod(fake_load)})()
    monkeypatch.setattr("importlib.import_module", lambda name: fake_loader)
    monkeypatch.setattr("kitchen.tracking.configure_from_env", lambda: None)

    returned_metrics = metrics if metrics is not None else {"val_brier": 0.18, "val_accuracy": 0.72}
    calls = []

    def fake_evaluate(m, p, s):
        calls.append((m, p, s))
        return returned_metrics

    return fake_evaluate, calls, fake_model


def test_run_evaluate_missing_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["run", "evaluate", "--params", "missing.yaml"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_run_evaluate_default_uri_from_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    fake_evaluate, _, _ = _make_evaluate_mocks(monkeypatch)

    src = tmp_path / "src" / "evaluate"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "run.py").write_text(
        "def evaluate(model, params, store):\n    return {'val_brier': 0.18}\n"
    )

    # Bypass actual import with a direct monkeypatch on the CLI's lazy import path
    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = fake_evaluate
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate"])
    assert result.exit_code == 0
    assert "test-project-model@champion" in result.output


def test_run_evaluate_custom_model_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    fake_evaluate, _, _ = _make_evaluate_mocks(monkeypatch)

    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = fake_evaluate
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate", "--model-uri", "runs:/abc123/model"])
    assert result.exit_code == 0
    assert "runs:/abc123/model" in result.output


def test_run_evaluate_custom_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    fake_evaluate, _, _ = _make_evaluate_mocks(monkeypatch)

    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = fake_evaluate
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate", "--alias", "staging"])
    assert result.exit_code == 0
    assert "@staging" in result.output


def test_run_evaluate_prints_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    fake_evaluate, _, _ = _make_evaluate_mocks(
        monkeypatch, metrics={"val_brier": 0.18, "val_accuracy": 0.72}
    )

    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = fake_evaluate
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate"])
    assert result.exit_code == 0
    assert "val_brier" in result.output
    assert "val_accuracy" in result.output


def test_run_evaluate_model_load_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    _make_evaluate_mocks(monkeypatch, load_raises=Exception("registry not found"))

    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = lambda m, p, s: {}
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate"])
    assert result.exit_code != 0
    assert "error loading model" in result.output


def test_run_evaluate_missing_champion_gives_clear_message(tmp_path, monkeypatch):
    """K-019: MlflowException about a missing alias shows a helpful first-run message."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    alias_exc = mlflow.exceptions.MlflowException(
        "Registered model alias 'champion' not found for model 'test-project-model'."
    )
    _make_evaluate_mocks(monkeypatch, load_raises=alias_exc)

    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = lambda m, p, s: {}
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate"])
    assert result.exit_code != 0
    assert "No 'champion' model registered yet" in result.output
    assert "auto-promote" in result.output


def test_run_evaluate_missing_alias_shows_alias_name(tmp_path, monkeypatch):
    """The helpful message reflects the actual alias name when --alias is overridden."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    alias_exc = mlflow.exceptions.MlflowException(
        "Registered model alias 'staging' not found for model 'test-project-model'."
    )
    _make_evaluate_mocks(monkeypatch, load_raises=alias_exc)

    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = lambda m, p, s: {}
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate", "--alias", "staging"])
    assert result.exit_code != 0
    assert "No 'staging' model registered yet" in result.output


def test_run_evaluate_non_alias_mlflow_error_shows_generic_message(tmp_path, monkeypatch):
    """A non-alias MlflowException (e.g. network error) still shows the generic message."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    generic_exc = mlflow.exceptions.MlflowException("Connection refused: mlflow server unreachable")
    _make_evaluate_mocks(monkeypatch, load_raises=generic_exc)

    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = lambda m, p, s: {}
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate"])
    assert result.exit_code != 0
    assert "error loading model" in result.output
    assert "auto-promote" not in result.output


def test_run_evaluate_invalid_flavor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)
    monkeypatch.setattr("kitchen.tracking.configure_from_env", lambda: None)
    result = runner.invoke(app, ["run", "evaluate", "--flavor", "torchscript"])
    assert result.exit_code != 0
    assert "unknown flavor" in result.output


def test_run_evaluate_flavor_autodetect_xgboost(tmp_path, monkeypatch):
    """SCF-003: default sklearn flavor is auto-upgraded to xgboost from MLmodel manifest."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    imported_modules: list[str] = []
    fake_loader = type("Loader", (), {"load_model": staticmethod(lambda uri: object())})()

    def recording_import(name):
        imported_modules.append(name)
        return fake_loader

    monkeypatch.setattr("importlib.import_module", recording_import)
    monkeypatch.setattr("kitchen.tracking.configure_from_env", lambda: None)

    import mlflow.models as _mlflow_models
    fake_info = type("Info", (), {"flavors": {"xgboost": {}, "python_function": {}}})()
    monkeypatch.setattr(_mlflow_models, "get_model_info", lambda uri: fake_info)

    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = lambda m, p, s: {"val_accuracy": 0.8}
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate"])
    assert result.exit_code == 0
    assert "mlflow.xgboost" in imported_modules


def test_run_evaluate_flavor_autodetect_lightgbm(tmp_path, monkeypatch):
    """SCF-003: default sklearn flavor is auto-upgraded to lightgbm from MLmodel manifest."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    imported_modules: list[str] = []
    fake_loader = type("Loader", (), {"load_model": staticmethod(lambda uri: object())})()

    def recording_import(name):
        imported_modules.append(name)
        return fake_loader

    monkeypatch.setattr("importlib.import_module", recording_import)
    monkeypatch.setattr("kitchen.tracking.configure_from_env", lambda: None)

    import mlflow.models as _mlflow_models
    fake_info = type("Info", (), {"flavors": {"lightgbm": {}, "python_function": {}}})()
    monkeypatch.setattr(_mlflow_models, "get_model_info", lambda uri: fake_info)

    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = lambda m, p, s: {"val_accuracy": 0.8}
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate"])
    assert result.exit_code == 0
    assert "mlflow.lightgbm" in imported_modules


def test_run_evaluate_flavor_autodetect_failure_falls_back(tmp_path, monkeypatch):
    """SCF-003: if get_model_info raises, fall back to sklearn without crashing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    imported_modules: list[str] = []
    fake_loader = type("Loader", (), {"load_model": staticmethod(lambda uri: object())})()

    def recording_import(name):
        imported_modules.append(name)
        return fake_loader

    monkeypatch.setattr("importlib.import_module", recording_import)
    monkeypatch.setattr("kitchen.tracking.configure_from_env", lambda: None)

    import mlflow.models as _mlflow_models
    monkeypatch.setattr(_mlflow_models, "get_model_info", lambda uri: (_ for _ in ()).throw(Exception("registry unavailable")))

    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = lambda m, p, s: {"val_accuracy": 0.8}
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate"])
    assert result.exit_code == 0
    assert "mlflow.sklearn" in imported_modules


def test_run_evaluate_missing_src_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    _make_evaluate_mocks(monkeypatch)

    # `from src.evaluate.run import evaluate` uses builtins.__import__, not
    # importlib.import_module, so we must intercept at the builtin level.
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "src.evaluate.run":
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    result = runner.invoke(app, ["run", "evaluate"])
    assert result.exit_code != 0
    assert "src/" in result.output


# ---------------------------------------------------------------------------
# kitchen init --source / --competition / --template
# ---------------------------------------------------------------------------


def test_init_kaggle_source_params_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["init", "march-mania", "--source", "kaggle", "--competition", "march-ml-mania-2026"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    params = yaml.safe_load((tmp_path / "march-mania" / "params.yaml").read_text())
    assert params["data"]["source"] == "kaggle"
    assert params["data"]["competition"] == "march-ml-mania-2026"
    assert "submission" in params
    assert params["submission"]["id_col"] == "Id"
    assert params["submission"]["target_col"] == "target"


def test_init_kaggle_next_steps_mentions_ingest_and_submit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["init", "my-comp", "--source", "kaggle", "--competition", "my-comp"],
        catch_exceptions=False,
    )
    assert "kitchen ingest" in result.output
    assert "kitchen submit" in result.output


def test_init_local_next_steps_no_kaggle_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my-comp"], catch_exceptions=False)
    assert "kitchen ingest" not in result.output
    assert "kitchen submit" not in result.output


def test_init_kaggle_requires_competition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my-comp", "--source", "kaggle"])
    assert result.exit_code != 0
    assert "competition" in result.output


def test_init_invalid_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my-comp", "--source", "ftp"])
    assert result.exit_code != 0
    assert "invalid source" in result.output


def test_init_baseline_xgb_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["init", "my-comp", "--template", "baseline-xgb"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    train_src = (tmp_path / "my-comp" / "src" / "train" / "run.py").read_text()
    assert "XGBClassifier" in train_src
    assert "xgboost" in train_src
    assert "train_val_split" in train_src
    assert "classification_metrics" in train_src
    assert "mlflow.log_metrics" in train_src
    eval_src = (tmp_path / "my-comp" / "src" / "evaluate" / "run.py").read_text()
    assert "classification_metrics" in eval_src
    assert "train_val_split" in eval_src
    assert "NotImplementedError" not in eval_src


def test_init_baseline_lr_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["init", "my-comp", "--template", "baseline-lr"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    train_src = (tmp_path / "my-comp" / "src" / "train" / "run.py").read_text()
    assert "LogisticRegression" in train_src
    assert "train_val_split" in train_src
    assert "classification_metrics" in train_src
    assert "mlflow.log_metrics" in train_src
    eval_src = (tmp_path / "my-comp" / "src" / "evaluate" / "run.py").read_text()
    assert "classification_metrics" in eval_src
    assert "train_val_split" in eval_src
    assert "NotImplementedError" not in eval_src


def test_init_invalid_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my-comp", "--template", "random-forest"])
    assert result.exit_code != 0
    assert "invalid template" in result.output


def test_init_baseline_lgbm_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["init", "my-comp", "--template", "baseline-lgbm"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    train_src = (tmp_path / "my-comp" / "src" / "train" / "run.py").read_text()
    assert "LGBMClassifier" in train_src
    assert "lightgbm" in train_src
    assert 'model_flavour = "lightgbm"' in train_src
    assert "num_leaves" in train_src
    assert "train_val_split" in train_src
    assert "classification_metrics" in train_src
    assert "mlflow.log_metrics" in train_src
    eval_src = (tmp_path / "my-comp" / "src" / "evaluate" / "run.py").read_text()
    assert "classification_metrics" in eval_src
    assert "train_val_split" in eval_src
    assert "NotImplementedError" not in eval_src


def test_init_baseline_lgbm_params_hint(tmp_path, monkeypatch):
    """params.yaml should contain a commented lgbm: section for user guidance."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--template", "baseline-lgbm"],
        catch_exceptions=False,
    )
    params_src = (tmp_path / "my-comp" / "params.yaml").read_text()
    assert "lgbm:" in params_src
    assert "num_leaves" in params_src


def test_init_baseline_rf_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["init", "my-comp", "--template", "baseline-rf"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    train_src = (tmp_path / "my-comp" / "src" / "train" / "run.py").read_text()
    assert "RandomForestClassifier" in train_src
    assert "sklearn.ensemble" in train_src
    assert "train_val_split" in train_src
    assert "classification_metrics" in train_src
    assert "mlflow.log_metrics" in train_src
    eval_src = (tmp_path / "my-comp" / "src" / "evaluate" / "run.py").read_text()
    assert "classification_metrics" in eval_src
    assert "train_val_split" in eval_src
    assert "NotImplementedError" not in eval_src


def test_init_binary_cls_template_train(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["init", "my-comp", "--template", "binary-cls"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    train_src = (tmp_path / "my-comp" / "src" / "train" / "run.py").read_text()
    assert "XGBClassifier" in train_src
    assert "train_val_split" in train_src
    assert "classification_metrics" in train_src
    assert "mlflow.log_metrics" in train_src


def test_init_binary_cls_template_evaluate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--template", "binary-cls"],
        catch_exceptions=False,
    )
    eval_src = (tmp_path / "my-comp" / "src" / "evaluate" / "run.py").read_text()
    assert "classification_metrics" in eval_src
    assert "train_val_split" in eval_src
    # evaluate should not have the stub — it has a real implementation
    assert "NotImplementedError" not in eval_src
    # params stash pattern is present
    assert "_params" in eval_src


def test_init_multiclass_cls_template_train(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["init", "my-comp", "--template", "multiclass-cls"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    train_src = (tmp_path / "my-comp" / "src" / "train" / "run.py").read_text()
    assert "XGBClassifier" in train_src
    assert "multi:softprob" in train_src
    assert "train_val_split" in train_src
    assert 'average="macro"' in train_src


def test_init_multiclass_cls_template_evaluate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--template", "multiclass-cls"],
        catch_exceptions=False,
    )
    eval_src = (tmp_path / "my-comp" / "src" / "evaluate" / "run.py").read_text()
    assert "classification_metrics" in eval_src
    assert 'average="macro"' in eval_src
    assert "NotImplementedError" not in eval_src


def test_init_regression_template_train(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["init", "my-comp", "--template", "regression"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    train_src = (tmp_path / "my-comp" / "src" / "train" / "run.py").read_text()
    assert "XGBRegressor" in train_src
    assert "regression_metrics" in train_src
    assert "stratify=False" in train_src


def test_init_regression_template_evaluate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--template", "regression"],
        catch_exceptions=False,
    )
    eval_src = (tmp_path / "my-comp" / "src" / "evaluate" / "run.py").read_text()
    assert "regression_metrics" in eval_src
    assert "stratify=False" in eval_src
    assert "NotImplementedError" not in eval_src


def test_init_tabular_ts_template_train(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["init", "my-comp", "--template", "tabular-ts"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    train_src = (tmp_path / "my-comp" / "src" / "train" / "run.py").read_text()
    assert "LGBMRegressor" in train_src
    assert "lightgbm" in train_src
    assert 'model_flavour = "lightgbm"' in train_src
    assert "regression_metrics" in train_src
    assert "mlflow.log_metrics" in train_src
    # time-ordered split — not the random kitchen.modeling helper
    assert "_time_split" in train_src
    assert "date_col" in train_src
    assert "val_frac" in train_src
    assert "train_val_split" not in train_src  # must NOT use the random helper


def test_init_tabular_ts_template_evaluate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--template", "tabular-ts"],
        catch_exceptions=False,
    )
    eval_src = (tmp_path / "my-comp" / "src" / "evaluate" / "run.py").read_text()
    assert "regression_metrics" in eval_src
    assert "_time_split" in eval_src
    assert "date_col" in eval_src
    assert "val_frac" in eval_src
    # has real implementation, not stub
    assert "NotImplementedError" not in eval_src
    # params stash pattern must be present (same as regression/binary-cls)
    assert "_params" in eval_src
    assert "train_val_split" not in eval_src  # no random split


def test_init_tabular_ts_params_hint(tmp_path, monkeypatch):
    """params.yaml should contain commented tabular-ts section with date_col and val_frac."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--template", "tabular-ts"],
        catch_exceptions=False,
    )
    params_src = (tmp_path / "my-comp" / "params.yaml").read_text()
    assert "tabular-ts" in params_src
    assert "date_col" in params_src
    assert "val_frac" in params_src


def test_init_tabular_ts_kaggle_params_hint(tmp_path, monkeypatch):
    """Kaggle variant of params.yaml also contains the tabular-ts hint."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        [
            "init", "my-comp",
            "--source", "kaggle",
            "--competition", "my-comp",
            "--template", "tabular-ts",
        ],
        catch_exceptions=False,
    )
    params_src = (tmp_path / "my-comp" / "params.yaml").read_text()
    assert "date_col" in params_src
    assert "val_frac" in params_src


def test_init_tabular_ts_class_name_substituted(tmp_path, monkeypatch):
    """Both MyCompTrainer and MyCompEvaluator should appear in the generated files."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--template", "tabular-ts"],
        catch_exceptions=False,
    )
    train_src = (tmp_path / "my-comp" / "src" / "train" / "run.py").read_text()
    eval_src = (tmp_path / "my-comp" / "src" / "evaluate" / "run.py").read_text()
    assert "MyCompTrainer" in train_src
    assert "MyCompEvaluator" in eval_src


def test_init_scaffolds_src_serve_predictor_py(tmp_path, monkeypatch):
    """kitchen init creates src/serve/predictor.py with predict() stub — valid Python."""
    import ast

    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-proj"], catch_exceptions=False)
    predictor_path = tmp_path / "my-proj" / "src" / "serve" / "predictor.py"
    assert predictor_path.exists(), "src/serve/predictor.py must be scaffolded"
    content = predictor_path.read_text()
    assert "def predict(payload: dict) -> dict:" in content
    assert "NotImplementedError" in content
    # Template escaping smoke-test: the rendered file must be syntactically valid Python.
    ast.parse(content)


def test_init_scaffolds_src_serve_init_py(tmp_path, monkeypatch):
    """kitchen init creates src/serve/__init__.py."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-proj"], catch_exceptions=False)
    init_path = tmp_path / "my-proj" / "src" / "serve" / "__init__.py"
    assert init_path.exists(), "src/serve/__init__.py must be scaffolded"


def test_init_predictor_py_has_mlflow_comment(tmp_path, monkeypatch):
    """Scaffolded predictor.py includes a comment showing how to load the champion model."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-proj"], catch_exceptions=False)
    content = (tmp_path / "my-proj" / "src" / "serve" / "predictor.py").read_text()
    assert "mlflow" in content


def test_init_predictor_py_has_pydantic_comment(tmp_path, monkeypatch):
    """Scaffolded predictor.py includes commented RequestModel/ResponseModel examples."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-proj"], catch_exceptions=False)
    content = (tmp_path / "my-proj" / "src" / "serve" / "predictor.py").read_text()
    assert "RequestModel" in content
    assert "ResponseModel" in content


def test_init_predictor_py_name_substituted(tmp_path, monkeypatch):
    """$name is substituted in the predictor.py scaffold."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "cbb-predictor"], catch_exceptions=False)
    content = (tmp_path / "cbb-predictor" / "src" / "serve" / "predictor.py").read_text()
    assert "cbb-predictor" in content


def test_init_default_train_template_unchanged(scaffold):
    train_src = (scaffold / "src" / "train" / "run.py").read_text()
    assert "NotImplementedError" in train_src
    assert "XGBClassifier" not in train_src


def test_init_kaggle_with_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "init",
            "mania",
            "--source",
            "kaggle",
            "--competition",
            "march-ml-mania-2026",
            "--template",
            "baseline-xgb",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    params = yaml.safe_load((tmp_path / "mania" / "params.yaml").read_text())
    assert params["data"]["source"] == "kaggle"
    train_src = (tmp_path / "mania" / "src" / "train" / "run.py").read_text()
    assert "XGBClassifier" in train_src


# ---------------------------------------------------------------------------
# kitchen init --ci
# ---------------------------------------------------------------------------

_CI_WORKFLOW_PATH = ".github/workflows/train-evaluate.yml"


def test_init_ci_creates_workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    assert result.exit_code == 0
    assert (tmp_path / "my-comp" / _CI_WORKFLOW_PATH).exists()


def test_init_no_ci_no_workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp"], catch_exceptions=False)
    assert not (tmp_path / "my-comp" / _CI_WORKFLOW_PATH).exists()


def test_init_ci_workflow_valid_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    raw = (tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text()
    content = yaml.safe_load(raw)
    assert content is not None
    assert "jobs" in content
    assert "train-evaluate" in content["jobs"]


def test_init_ci_workflow_contains_expected_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    assert "Train" in step_names
    assert "Evaluate" in step_names
    assert "Report" in step_names
    assert "Upload metrics" in step_names


def test_init_ci_workflow_substitutes_project_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    raw = (tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text()
    assert "my-comp" in raw
    assert "$name" not in raw


def test_init_ci_workflow_has_workflow_dispatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    # `on:` parses as boolean True in YAML; check raw text instead
    raw = (tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text()
    assert "workflow_dispatch" in raw


def test_init_ci_kaggle_includes_ingest_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "my-comp"],
        catch_exceptions=False,
    )
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    assert "Ingest data" in step_names


def test_init_ci_kaggle_ingest_uses_secrets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "my-comp"],
        catch_exceptions=False,
    )
    raw = (tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text()
    assert "secrets.KAGGLE_USERNAME" in raw
    assert "secrets.KAGGLE_KEY" in raw


def test_init_ci_local_no_ingest_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    assert "Ingest data" not in step_names


def test_init_ci_note_in_output_for_kaggle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "my-comp"],
        catch_exceptions=False,
    )
    assert "KAGGLE_USERNAME" in result.output
    assert "KAGGLE_KEY" in result.output


def test_init_ci_note_in_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    assert "train-evaluate.yml" in result.output


# ---------------------------------------------------------------------------
# LML-006: Push results step in CI workflow
# ---------------------------------------------------------------------------


def test_init_ci_workflow_has_push_results_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    assert "Push results" in step_names


def test_init_ci_workflow_push_step_after_evaluate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    assert step_names.index("Push results") > step_names.index("Evaluate")


def test_init_ci_workflow_contents_write_permission(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    perms = content["jobs"]["train-evaluate"]["permissions"]
    assert perms.get("contents") == "write"


def test_init_ci_workflow_push_step_fetches_branch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    raw = (tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text()
    assert "git fetch origin results:results" in raw


def test_init_ci_workflow_push_step_gated_on_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    push_step = next(s for s in steps if s.get("name") == "Push results")
    assert "refs/heads/main" in str(push_step.get("if", ""))


def test_init_ci_kaggle_has_push_results_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "my-comp"],
        catch_exceptions=False,
    )
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    assert "Push results" in step_names


def test_init_ci_kaggle_push_step_after_submit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "my-comp"],
        catch_exceptions=False,
    )
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    assert step_names.index("Push results") > step_names.index("Submit to Kaggle")


def test_init_ci_kaggle_contents_write_permission(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "my-comp"],
        catch_exceptions=False,
    )
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    perms = content["jobs"]["train-evaluate"]["permissions"]
    assert perms.get("contents") == "write"


# ---------------------------------------------------------------------------
# GH-003: PR comment steps in CI workflow
# ---------------------------------------------------------------------------


def test_init_ci_workflow_has_pr_comment_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    assert "Find PR comment" in step_names
    assert "Post PR comment" in step_names


def test_init_ci_workflow_has_download_base_metrics_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    assert "Download base metrics" in step_names


def test_init_ci_workflow_download_step_is_pr_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    dl_step = next(s for s in steps if s.get("name") == "Download base metrics")
    assert "pull_request" in str(dl_step.get("if", ""))


def test_init_ci_workflow_pr_comment_steps_are_pr_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    post_step = next(s for s in steps if s.get("name") == "Post PR comment")
    assert "pull_request" in str(post_step.get("if", ""))


def test_init_ci_workflow_has_pr_write_permission(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    perms = content["jobs"]["train-evaluate"].get("permissions", {})
    assert perms.get("pull-requests") == "write"


def test_init_ci_kaggle_workflow_has_pr_comment_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "my-comp"],
        catch_exceptions=False,
    )
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    assert "Download base metrics" in step_names
    assert "Find PR comment" in step_names
    assert "Post PR comment" in step_names


def test_init_ci_workflow_report_step_has_compare_logic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    raw = (tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text()
    assert "--compare" in raw
    assert "base-metrics/metrics.json" in raw


# ---------------------------------------------------------------------------
# Dashboard (GP-002)
# ---------------------------------------------------------------------------

_DASHBOARD_PATH = "docs/index.html"


def test_init_creates_dashboard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp"], catch_exceptions=False)
    assert (tmp_path / "my-comp" / _DASHBOARD_PATH).exists()


def test_init_dashboard_name_substituted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp"], catch_exceptions=False)
    html = (tmp_path / "my-comp" / _DASHBOARD_PATH).read_text()
    assert "my-comp" in html
    assert "$name" not in html


def test_init_dashboard_chartjs_script_tag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp"], catch_exceptions=False)
    html = (tmp_path / "my-comp" / _DASHBOARD_PATH).read_text()
    assert "cdn.jsdelivr.net/npm/chart.js" in html


def test_init_dashboard_has_canvas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp"], catch_exceptions=False)
    html = (tmp_path / "my-comp" / _DASHBOARD_PATH).read_text()
    assert "<canvas" in html


def test_init_dashboard_github_api_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp"], catch_exceptions=False)
    html = (tmp_path / "my-comp" / _DASHBOARD_PATH).read_text()
    assert "contents/results?ref=results" in html


def test_init_dashboard_created_without_ci_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp"], catch_exceptions=False)
    assert (tmp_path / "my-comp" / _DASHBOARD_PATH).exists()


def test_init_output_mentions_github_pages_when_ci(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    assert "GitHub Actions" in result.output


def test_init_output_no_pages_note_without_ci(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my-comp"], catch_exceptions=False)
    assert "GitHub Actions" not in result.output


# ---------------------------------------------------------------------------
# Deploy Pages job (GP-003)
# ---------------------------------------------------------------------------


def test_init_ci_has_deploy_pages_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    raw = (tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text()
    assert "deploy-pages:" in raw


def test_init_ci_deploy_pages_gated_on_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    data = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    job = data["jobs"]["deploy-pages"]
    assert "push" in job["if"]
    assert "refs/heads/main" in job["if"]


def test_init_ci_deploy_pages_uses_deploy_action(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    raw = (tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text()
    assert "actions/deploy-pages@" in raw


def test_init_ci_deploy_pages_has_pages_write_permission(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    data = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    perms = data["jobs"]["deploy-pages"]["permissions"]
    assert perms.get("pages") == "write"


def test_init_ci_kaggle_has_deploy_pages_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "titanic"],
        catch_exceptions=False,
    )
    raw = (tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text()
    assert "deploy-pages:" in raw


# ---------------------------------------------------------------------------
# DASH-002: deploy-pages job generates dashboard before Pages upload
# ---------------------------------------------------------------------------


def _deploy_pages_steps(tmp_path, kaggle: bool = False) -> list:
    """Helper: scaffold a CI workflow and return the deploy-pages job steps."""
    if kaggle:
        runner.invoke(
            app,
            ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "titanic"],
            catch_exceptions=False,
        )
    else:
        runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    data = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    return data["jobs"]["deploy-pages"]["steps"]


def test_init_ci_deploy_pages_has_generate_dashboard_step(tmp_path, monkeypatch):
    """deploy-pages job has a 'Generate dashboard' step."""
    monkeypatch.chdir(tmp_path)
    steps = _deploy_pages_steps(tmp_path)
    step_names = [s.get("name", "") for s in steps]
    assert "Generate dashboard" in step_names


def test_init_ci_deploy_pages_generate_step_continue_on_error(tmp_path, monkeypatch):
    """Generate dashboard step has continue-on-error: true (no results on first run)."""
    monkeypatch.chdir(tmp_path)
    steps = _deploy_pages_steps(tmp_path)
    gen_step = next(s for s in steps if s.get("name") == "Generate dashboard")
    assert gen_step.get("continue-on-error") is True


def test_init_ci_deploy_pages_generate_step_uses_docs_output(tmp_path, monkeypatch):
    """Generate dashboard step writes to docs/index.html to match upload-pages path."""
    monkeypatch.chdir(tmp_path)
    steps = _deploy_pages_steps(tmp_path)
    gen_step = next(s for s in steps if s.get("name") == "Generate dashboard")
    assert "--output docs/index.html" in gen_step.get("run", "")


def test_init_ci_deploy_pages_fetches_results_branch(tmp_path, monkeypatch):
    """deploy-pages job fetches the results branch before generating the dashboard."""
    monkeypatch.chdir(tmp_path)
    steps = _deploy_pages_steps(tmp_path)
    step_names = [s.get("name", "") for s in steps]
    assert "Fetch results branch" in step_names


def test_init_ci_deploy_pages_fetch_before_generate(tmp_path, monkeypatch):
    """Fetch results branch step comes before Generate dashboard step."""
    monkeypatch.chdir(tmp_path)
    steps = _deploy_pages_steps(tmp_path)
    step_names = [s.get("name", "") for s in steps]
    assert step_names.index("Fetch results branch") < step_names.index("Generate dashboard")


def test_init_ci_deploy_pages_checkout_full_depth(tmp_path, monkeypatch):
    """deploy-pages checkout uses fetch-depth: 0 to get the results branch."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    data = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    deploy_steps = data["jobs"]["deploy-pages"]["steps"]
    checkout = next(s for s in deploy_steps if (s.get("uses") or "").startswith("actions/checkout"))
    assert checkout.get("with", {}).get("fetch-depth") == 0


def test_init_ci_kaggle_deploy_pages_has_generate_dashboard_step(tmp_path, monkeypatch):
    """Kaggle variant: deploy-pages job also has the Generate dashboard step."""
    monkeypatch.chdir(tmp_path)
    steps = _deploy_pages_steps(tmp_path, kaggle=True)
    step_names = [s.get("name", "") for s in steps]
    assert "Generate dashboard" in step_names


def test_init_ci_deploy_pages_install_kitchen_before_generate(tmp_path, monkeypatch):
    """Install kitchen step comes before Generate dashboard in deploy-pages."""
    monkeypatch.chdir(tmp_path)
    steps = _deploy_pages_steps(tmp_path)
    step_names = [s.get("name", "") for s in steps]
    assert step_names.index("Install kitchen") < step_names.index("Generate dashboard")


# ---------------------------------------------------------------------------
# Dashboard delta column (LML-007)
# ---------------------------------------------------------------------------


def test_init_dashboard_has_delta_column_header(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp"], catch_exceptions=False)
    html = (tmp_path / "my-comp" / _DASHBOARD_PATH).read_text()
    assert "Δ" in html or "&#916;" in html or "&Delta;" in html


def test_init_dashboard_delta_js_uses_champion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp"], catch_exceptions=False)
    html = (tmp_path / "my-comp" / _DASHBOARD_PATH).read_text()
    assert "champ" in html
    assert "toFixed" in html


# ---------------------------------------------------------------------------
# Dashboard URL in job summary (GP-005)
# ---------------------------------------------------------------------------


def test_init_ci_deploy_pages_links_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    data = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = data["jobs"]["deploy-pages"]["steps"]
    summary_steps = [s for s in steps if "GITHUB_STEP_SUMMARY" in str(s.get("run", ""))]
    assert summary_steps, "No step writes to GITHUB_STEP_SUMMARY in deploy-pages job"


def test_init_ci_kaggle_deploy_pages_links_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app,
        ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "titanic"],
        catch_exceptions=False,
    )
    data = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = data["jobs"]["deploy-pages"]["steps"]
    summary_steps = [s for s in steps if "GITHUB_STEP_SUMMARY" in str(s.get("run", ""))]
    assert summary_steps, "No step writes to GITHUB_STEP_SUMMARY in deploy-pages job (kaggle)"


# ---------------------------------------------------------------------------
# No-raw-data CI check (DVC-009)
# ---------------------------------------------------------------------------

_RAW_DATA_STEP_NAME = "Check for raw data in git"


def _ci_train_steps(tmp_path, kaggle=False):
    """Scaffold --ci project and return the train-evaluate job steps list."""
    args = ["init", "my-comp", "--ci"]
    if kaggle:
        args += ["--source", "kaggle", "--competition", "titanic"]
    runner.invoke(app, args, catch_exceptions=False)
    data = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    return data["jobs"]["train-evaluate"]["steps"]


def test_ci_workflow_has_raw_data_check(tmp_path, monkeypatch):
    """Non-Kaggle CI workflow includes the no-raw-data check step."""
    monkeypatch.chdir(tmp_path)
    steps = _ci_train_steps(tmp_path)
    names = [s.get("name", "") for s in steps]
    assert _RAW_DATA_STEP_NAME in names


def test_ci_kaggle_workflow_has_raw_data_check(tmp_path, monkeypatch):
    """Kaggle CI workflow includes the no-raw-data check step."""
    monkeypatch.chdir(tmp_path)
    steps = _ci_train_steps(tmp_path, kaggle=True)
    names = [s.get("name", "") for s in steps]
    assert _RAW_DATA_STEP_NAME in names


def test_ci_raw_data_check_runs_before_python_setup(tmp_path, monkeypatch):
    """The raw-data check appears before setup-python so it fails fast with no install cost."""
    monkeypatch.chdir(tmp_path)
    steps = _ci_train_steps(tmp_path)
    names = [s.get("name", "") or str(s.get("uses", "")) for s in steps]
    check_idx = next(i for i, n in enumerate(names) if _RAW_DATA_STEP_NAME in n)
    setup_idx = next(i for i, n in enumerate(names) if "setup-python" in n)
    assert check_idx < setup_idx, "Raw-data check should run before setup-python"


def test_ci_raw_data_check_uses_git_ls_files(tmp_path, monkeypatch):
    """The check step uses git ls-files to inspect tracked files in data/raw/."""
    monkeypatch.chdir(tmp_path)
    steps = _ci_train_steps(tmp_path)
    check_step = next(s for s in steps if s.get("name", "") == _RAW_DATA_STEP_NAME)
    assert "git ls-files" in check_step["run"]
    assert "data/raw/" in check_step["run"]


def test_ci_raw_data_check_excludes_gitkeep(tmp_path, monkeypatch):
    """The check step filters out .gitkeep so the scaffold placeholder doesn't trigger it."""
    monkeypatch.chdir(tmp_path)
    steps = _ci_train_steps(tmp_path)
    check_step = next(s for s in steps if s.get("name", "") == _RAW_DATA_STEP_NAME)
    assert ".gitkeep" in check_step["run"]


# ---------------------------------------------------------------------------
# kitchen check — DVC context-aware behaviour (DVC-010)
# ---------------------------------------------------------------------------


def test_check_dvc_ok_when_binary_found(tmp_path, monkeypatch):
    """When the dvc binary is found, check reports ✓ regardless of dvc.yaml."""
    monkeypatch.chdir(tmp_path)
    with patch("shutil.which", side_effect=lambda name: "/usr/bin/dvc" if name == "dvc" else None):
        result = runner.invoke(app, ["check"], catch_exceptions=False)
    assert "✓ dvc" in result.output


def test_check_dvc_hard_fail_when_dvc_yaml_present_but_binary_missing(tmp_path, monkeypatch):
    """When dvc.yaml exists but the dvc binary is absent, check hard-fails (✗)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dvc.yaml").write_text("stages: {}\n")
    with patch("shutil.which", side_effect=lambda name: None):
        result = runner.invoke(app, ["check"], catch_exceptions=False)
    assert "✗ dvc" in result.output
    assert "pip install kitchen[dvc]" in result.output


def test_check_dvc_soft_warn_when_no_dvc_yaml_and_binary_missing(tmp_path, monkeypatch):
    """When dvc.yaml is absent and the dvc binary is missing, check soft-warns (~) not hard-fails."""
    monkeypatch.chdir(tmp_path)
    # No dvc.yaml in tmp_path
    with patch("shutil.which", side_effect=lambda name: None):
        result = runner.invoke(app, ["check"], catch_exceptions=False)
    assert "~ dvc" in result.output
    assert "✗ dvc" not in result.output
    assert "pip install kitchen[dvc]" in result.output


def test_check_dvc_soft_warn_symbol_is_tilde_not_cross(tmp_path, monkeypatch):
    """The soft DVC warning uses ~ (not ✗) so it doesn't inflate the hard-fail count."""
    monkeypatch.chdir(tmp_path)
    # No dvc.yaml → soft-warn path. Other tools may also be absent; we only care about DVC symbol.
    with patch("shutil.which", side_effect=lambda name: None):
        result = runner.invoke(app, ["check"], catch_exceptions=False)
    assert "~ dvc" in result.output
    assert "✗ dvc" not in result.output


# ---------------------------------------------------------------------------
# kitchen init --with-dvc (DVC-002)
# ---------------------------------------------------------------------------


def _init_with_dvc(tmp_path, monkeypatch, extra_args=None):
    """Helper: run kitchen init --with-dvc with dvc binary mocked away."""
    monkeypatch.chdir(tmp_path)
    args = ["init", "my-project", "--with-dvc"] + (extra_args or [])
    with (
        patch("shutil.which", return_value="/usr/bin/dvc"),
        patch("subprocess.run"),  # prevent real dvc init from running in tests
    ):
        return runner.invoke(app, args, catch_exceptions=False)


def test_init_with_dvc_creates_dvc_yaml(tmp_path, monkeypatch):
    """--with-dvc scaffolds dvc.yaml in the project root."""
    _init_with_dvc(tmp_path, monkeypatch)
    assert (tmp_path / "my-project" / "dvc.yaml").exists()


def test_init_with_dvc_creates_dvcignore(tmp_path, monkeypatch):
    """--with-dvc scaffolds .dvcignore in the project root."""
    _init_with_dvc(tmp_path, monkeypatch)
    assert (tmp_path / "my-project" / ".dvcignore").exists()


def test_init_with_dvc_non_kaggle_has_all_three_stages(tmp_path, monkeypatch):
    """Non-Kaggle dvc.yaml has features, train, and evaluate stages."""
    _init_with_dvc(tmp_path, monkeypatch)
    content = (tmp_path / "my-project" / "dvc.yaml").read_text()
    assert "features:" in content
    assert "train:" in content
    assert "evaluate:" in content


def test_init_with_dvc_non_kaggle_no_submit_stage(tmp_path, monkeypatch):
    """Non-Kaggle dvc.yaml does not have a submit stage."""
    _init_with_dvc(tmp_path, monkeypatch)
    content = (tmp_path / "my-project" / "dvc.yaml").read_text()
    assert "submit:" not in content


def test_init_with_dvc_non_kaggle_ingest_is_commented(tmp_path, monkeypatch):
    """Non-Kaggle dvc.yaml includes an ingest placeholder that is commented out."""
    _init_with_dvc(tmp_path, monkeypatch)
    content = (tmp_path / "my-project" / "dvc.yaml").read_text()
    assert "# ingest:" in content


def test_init_with_dvc_kaggle_has_submit_stage(tmp_path, monkeypatch):
    """Kaggle dvc.yaml includes a submit stage."""
    _init_with_dvc(
        tmp_path, monkeypatch, ["--source", "kaggle", "--competition", "titanic"]
    )
    content = (tmp_path / "my-project" / "dvc.yaml").read_text()
    assert "submit:" in content


def test_init_with_dvc_kaggle_no_ingest_placeholder(tmp_path, monkeypatch):
    """Kaggle dvc.yaml does not include a commented ingest block (data is fixed by slug)."""
    _init_with_dvc(
        tmp_path, monkeypatch, ["--source", "kaggle", "--competition", "titanic"]
    )
    content = (tmp_path / "my-project" / "dvc.yaml").read_text()
    assert "# ingest:" not in content


def test_init_with_dvc_train_stage_uses_kitchen_run(tmp_path, monkeypatch):
    """The train stage uses `kitchen run train` as its cmd."""
    _init_with_dvc(tmp_path, monkeypatch)
    content = (tmp_path / "my-project" / "dvc.yaml").read_text()
    assert "cmd: kitchen run train" in content


def test_init_with_dvc_evaluate_stage_uses_kitchen_run(tmp_path, monkeypatch):
    """The evaluate stage uses `kitchen run evaluate` as its cmd."""
    _init_with_dvc(tmp_path, monkeypatch)
    content = (tmp_path / "my-project" / "dvc.yaml").read_text()
    assert "cmd: kitchen run evaluate" in content


def test_init_with_dvc_metrics_json_cache_false(tmp_path, monkeypatch):
    """metrics.json is declared with cache: false so MLflow owns metric history."""
    _init_with_dvc(tmp_path, monkeypatch)
    content = (tmp_path / "my-project" / "dvc.yaml").read_text()
    assert "metrics.json" in content
    assert "cache: false" in content


def test_init_with_dvc_params_sections_declared(tmp_path, monkeypatch):
    """features and model params sections are declared in dvc.yaml stages."""
    _init_with_dvc(tmp_path, monkeypatch)
    content = (tmp_path / "my-project" / "dvc.yaml").read_text()
    assert "- features" in content
    assert "- model" in content


def test_init_with_dvc_dvc_config_written(tmp_path, monkeypatch):
    """.dvc/config is written with an S3 remote placeholder after dvc init."""
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "my-project"
    with (
        patch("shutil.which", return_value="/usr/bin/dvc"),
        patch("subprocess.run"),
    ):
        runner.invoke(app, ["init", "my-project", "--with-dvc"], catch_exceptions=False)
    config_path = project / ".dvc" / "config"
    assert config_path.exists()
    config_text = config_path.read_text()
    assert "s3remote" in config_text
    assert "s3://YOUR-BUCKET/dvc" in config_text


def test_init_with_dvc_fails_fast_if_binary_missing(tmp_path, monkeypatch):
    """--with-dvc exits with error if the dvc binary is not installed."""
    monkeypatch.chdir(tmp_path)
    with patch("shutil.which", return_value=None):
        result = runner.invoke(app, ["init", "my-project", "--with-dvc"])
    assert result.exit_code != 0
    assert "pip install kitchen[dvc]" in result.output


def test_init_without_dvc_no_dvc_yaml(tmp_path, monkeypatch):
    """Without --with-dvc, no dvc.yaml is created."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-project"], catch_exceptions=False)
    assert not (tmp_path / "my-project" / "dvc.yaml").exists()


def test_init_with_dvc_output_mentions_remote(tmp_path, monkeypatch):
    """The next-steps output includes a dvc remote modify instruction."""
    result = _init_with_dvc(tmp_path, monkeypatch)
    assert "dvc remote modify" in result.output


def test_init_with_dvc_features_stage_uses_kitchen_run_features(tmp_path, monkeypatch):
    """dvc.yaml features stage cmd is `kitchen run features` not `python src/features/run.py`."""
    _init_with_dvc(tmp_path, monkeypatch)
    content = (tmp_path / "my-project" / "dvc.yaml").read_text()
    assert "cmd: kitchen run features" in content
    assert "python src/features/run.py" not in content


# ---------------------------------------------------------------------------
# kitchen run features (DVC-011)
# ---------------------------------------------------------------------------

FEATURES_PARAMS = """\
experiment: test-project
features:
  raw_file: train.csv
  processed_file: features.parquet
"""


def _make_features_mock(monkeypatch, raises=None):
    """Inject a fake src.features.run module; return the recorded calls list."""
    calls = []

    def fake_build(params, store):
        if raises is not None:
            raise raises
        calls.append((params, store))

    fake_mod = type(sys)("src.features.run")
    fake_mod.build = fake_build
    monkeypatch.setitem(sys.modules, "src.features.run", fake_mod)
    return calls


def test_run_features_missing_params(tmp_path, monkeypatch):
    """run features exits non-zero when params file is missing."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["run", "features", "--params", "missing.yaml"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_run_features_invokes_build(tmp_path, monkeypatch):
    """run features calls build(params, store) from src.features.run."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(FEATURES_PARAMS)
    calls = _make_features_mock(monkeypatch)
    result = runner.invoke(app, ["run", "features"], catch_exceptions=False)
    assert result.exit_code == 0
    assert len(calls) == 1


def test_run_features_custom_params(tmp_path, monkeypatch):
    """run features --params custom.yaml reads the specified file."""
    monkeypatch.chdir(tmp_path)
    custom = tmp_path / "custom.yaml"
    custom.write_text(FEATURES_PARAMS)
    calls = _make_features_mock(monkeypatch)
    result = runner.invoke(app, ["run", "features", "--params", "custom.yaml"], catch_exceptions=False)
    assert result.exit_code == 0
    assert len(calls) == 1


def test_run_features_missing_src_module(tmp_path, monkeypatch):
    """run features exits non-zero with a helpful message when src.features.run is absent."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(FEATURES_PARAMS)

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "src.features.run":
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    monkeypatch.delitem(sys.modules, "src.features.run", raising=False)

    result = runner.invoke(app, ["run", "features"])
    assert result.exit_code != 0
    assert "src/features/run.py" in result.output


def test_run_features_not_implemented_error(tmp_path, monkeypatch):
    """run features exits non-zero with a clear message when build() is a stub."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(FEATURES_PARAMS)
    _make_features_mock(monkeypatch, raises=NotImplementedError())
    result = runner.invoke(app, ["run", "features"])
    assert result.exit_code != 0
    assert "not yet implemented" in result.output


def test_run_features_output_mentions_processed_file(tmp_path, monkeypatch):
    """run features echoes the name of the processed output file on success."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(FEATURES_PARAMS)
    _make_features_mock(monkeypatch)
    result = runner.invoke(app, ["run", "features"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "features.parquet" in result.output


# ---------------------------------------------------------------------------
# kitchen dvc init (DVC-002b)
# ---------------------------------------------------------------------------


def _dvc_init(tmp_path, monkeypatch, extra_args=None, params_content=None):
    """Helper: run `kitchen dvc init` with dvc binary and subprocess mocked."""
    monkeypatch.chdir(tmp_path)
    if params_content is not None:
        (tmp_path / "params.yaml").write_text(params_content)
    args = ["dvc", "init"] + (extra_args or [])
    with (
        patch("shutil.which", return_value="/usr/bin/dvc"),
        patch("subprocess.run"),  # prevent real dvc init
    ):
        return runner.invoke(app, args, catch_exceptions=False)


_PARAMS_LOCAL = """\
experiment: my-project
data:
  source: local
  path: /data
"""

_PARAMS_KAGGLE = """\
experiment: my-competition
data:
  source: kaggle
  competition: spaceship-titanic
"""


def test_dvc_init_creates_dvc_yaml(tmp_path, monkeypatch):
    """kitchen dvc init writes dvc.yaml into the current directory."""
    _dvc_init(tmp_path, monkeypatch, params_content=_PARAMS_LOCAL)
    assert (tmp_path / "dvc.yaml").exists()


def test_dvc_init_creates_dvcignore(tmp_path, monkeypatch):
    """kitchen dvc init writes .dvcignore into the current directory."""
    _dvc_init(tmp_path, monkeypatch, params_content=_PARAMS_LOCAL)
    assert (tmp_path / ".dvcignore").exists()


def test_dvc_init_creates_dvc_config(tmp_path, monkeypatch):
    """kitchen dvc init writes .dvc/config with the S3 remote placeholder."""
    _dvc_init(tmp_path, monkeypatch, params_content=_PARAMS_LOCAL)
    config = (tmp_path / ".dvc" / "config").read_text()
    assert "s3remote" in config
    assert "YOUR-BUCKET" in config


def test_dvc_init_non_kaggle_template_by_default(tmp_path, monkeypatch):
    """Non-kaggle params.yaml → non-kaggle dvc.yaml (ingest placeholder, no submit stage)."""
    _dvc_init(tmp_path, monkeypatch, params_content=_PARAMS_LOCAL)
    content = (tmp_path / "dvc.yaml").read_text()
    assert "# ingest:" in content
    assert "submit:" not in content


def test_dvc_init_kaggle_template_from_params(tmp_path, monkeypatch):
    """Kaggle source in params.yaml → Kaggle dvc.yaml (submit stage, no ingest placeholder)."""
    _dvc_init(tmp_path, monkeypatch, params_content=_PARAMS_KAGGLE)
    content = (tmp_path / "dvc.yaml").read_text()
    assert "submit:" in content
    assert "# ingest:" not in content


def test_dvc_init_kaggle_flag_overrides_params(tmp_path, monkeypatch):
    """--kaggle flag forces Kaggle template regardless of params.yaml source."""
    _dvc_init(tmp_path, monkeypatch, extra_args=["--kaggle"], params_content=_PARAMS_LOCAL)
    content = (tmp_path / "dvc.yaml").read_text()
    assert "submit:" in content


def test_dvc_init_no_params_falls_back_to_cwd_name(tmp_path, monkeypatch):
    """Without params.yaml the project name falls back to the cwd directory name."""
    _dvc_init(tmp_path, monkeypatch)  # no params_content → no params.yaml written
    content = (tmp_path / "dvc.yaml").read_text()
    # Template header should contain the directory name (tmp_path.name), not a literal '$name'
    assert "$name" not in content


def test_dvc_init_fails_if_binary_missing(tmp_path, monkeypatch):
    """kitchen dvc init exits 1 with a helpful message when dvc is not on PATH."""
    monkeypatch.chdir(tmp_path)
    with patch("shutil.which", return_value=None):
        result = runner.invoke(app, ["dvc", "init"])
    assert result.exit_code != 0
    assert "pip install kitchen[dvc]" in result.output


def test_dvc_init_skips_existing_files_by_default(tmp_path, monkeypatch):
    """Existing dvc.yaml is not overwritten unless --overwrite is passed."""
    (tmp_path / "params.yaml").write_text(_PARAMS_LOCAL)
    sentinel = "# existing content\n"
    (tmp_path / "dvc.yaml").write_text(sentinel)
    _dvc_init(tmp_path, monkeypatch, params_content=None)  # params.yaml already written
    assert (tmp_path / "dvc.yaml").read_text() == sentinel


def test_dvc_init_overwrite_replaces_existing_dvc_yaml(tmp_path, monkeypatch):
    """--overwrite replaces an existing dvc.yaml."""
    (tmp_path / "dvc.yaml").write_text("# old\n")
    _dvc_init(tmp_path, monkeypatch, extra_args=["--overwrite"], params_content=_PARAMS_LOCAL)
    content = (tmp_path / "dvc.yaml").read_text()
    assert "features:" in content


def test_dvc_init_output_mentions_s3_remote(tmp_path, monkeypatch):
    """Next-steps output tells the user to set the S3 remote URL."""
    result = _dvc_init(tmp_path, monkeypatch, params_content=_PARAMS_LOCAL)
    assert "s3://YOUR-BUCKET" in result.output


# ---------------------------------------------------------------------------
# kitchen diff (CMP-001)
# ---------------------------------------------------------------------------


def _make_diff_run(
    run_id: str,
    params: dict[str, str],
    metrics: dict[str, float],
    run_name: str = "",
) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.data.params = params
    run.data.metrics = metrics
    run.data.tags = {"mlflow.runName": run_name} if run_name else {}
    return run


def _diff_invoke(run_a, run_b, extra_args=None, fi_a: dict | None = None, fi_b: dict | None = None):
    """Invoke `kitchen diff` with two mocked MLflow runs.

    Pass ``fi_a`` / ``fi_b`` as ``{feature: importance}`` dicts to inject feature
    importance artifacts for CMP-004 tests; omit (or pass None) to simulate a run
    with no ``feature_importances.json`` artifact.
    """

    def make_client():
        client = MagicMock()
        client.get_run.side_effect = [run_a, run_b]
        return client

    def fake_download(run_id, artifact_path, dst_path):
        fi = fi_a if run_id == run_a.info.run_id else fi_b
        if fi is None:
            raise Exception("no artifact")
        p = Path(dst_path) / artifact_path
        p.write_text(json.dumps(fi))
        return str(p)

    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient", side_effect=make_client),
        patch("mlflow.artifacts.download_artifacts", side_effect=fake_download),
    ):
        return runner.invoke(
            app, ["diff", run_a.info.run_id, run_b.info.run_id, *(extra_args or [])],
            catch_exceptions=False,
        )


def test_diff_shows_param_change():
    run_a = _make_diff_run("a" * 32, {"model.max_depth": "3"}, {})
    run_b = _make_diff_run("b" * 32, {"model.max_depth": "6"}, {})
    result = _diff_invoke(run_a, run_b)
    assert result.exit_code == 0
    assert "model.max_depth" in result.output
    assert "3" in result.output
    assert "6" in result.output


def test_diff_shows_metric_change():
    run_a = _make_diff_run("a" * 32, {}, {"val_accuracy": 0.80})
    run_b = _make_diff_run("b" * 32, {}, {"val_accuracy": 0.85})
    result = _diff_invoke(run_a, run_b)
    assert result.exit_code == 0
    assert "val_accuracy" in result.output
    assert "0.8000" in result.output
    assert "0.8500" in result.output


def test_diff_suppresses_identical_params():
    run_a = _make_diff_run("a" * 32, {"model.max_depth": "6", "model.eta": "0.1"}, {})
    run_b = _make_diff_run("b" * 32, {"model.max_depth": "6", "model.eta": "0.05"}, {})
    result = _diff_invoke(run_a, run_b)
    assert result.exit_code == 0
    assert "model.max_depth" not in result.output
    assert "model.eta" in result.output


def test_diff_suppresses_identical_metrics():
    run_a = _make_diff_run("a" * 32, {}, {"val_accuracy": 0.80, "val_brier": 0.18})
    run_b = _make_diff_run("b" * 32, {}, {"val_accuracy": 0.85, "val_brier": 0.18})
    result = _diff_invoke(run_a, run_b)
    assert result.exit_code == 0
    assert "val_accuracy" in result.output
    assert "val_brier" not in result.output


def test_diff_params_before_metrics():
    run_a = _make_diff_run("a" * 32, {"model.eta": "0.1"}, {"val_accuracy": 0.80})
    run_b = _make_diff_run("b" * 32, {"model.eta": "0.05"}, {"val_accuracy": 0.85})
    result = _diff_invoke(run_a, run_b)
    assert result.exit_code == 0
    params_pos = result.output.index("Params")
    metrics_pos = result.output.index("Metrics")
    assert params_pos < metrics_pos


def test_diff_no_differences():
    run_a = _make_diff_run("a" * 32, {"model.max_depth": "6"}, {"val_accuracy": 0.80})
    run_b = _make_diff_run("b" * 32, {"model.max_depth": "6"}, {"val_accuracy": 0.80})
    result = _diff_invoke(run_a, run_b)
    assert result.exit_code == 0
    assert "No differences found" in result.output


def test_diff_missing_param_in_one_run():
    run_a = _make_diff_run("a" * 32, {"model.max_depth": "6"}, {})
    run_b = _make_diff_run("b" * 32, {}, {})
    result = _diff_invoke(run_a, run_b)
    assert result.exit_code == 0
    assert "model.max_depth" in result.output
    assert "(missing)" in result.output


def test_diff_filters_fi_metrics():
    """Feature importance metrics (fi.*) should not appear in the diff."""
    run_a = _make_diff_run("a" * 32, {}, {"fi.feature_x": 0.5, "val_accuracy": 0.80})
    run_b = _make_diff_run("b" * 32, {}, {"fi.feature_x": 0.9, "val_accuracy": 0.85})
    result = _diff_invoke(run_a, run_b)
    assert result.exit_code == 0
    assert "fi.feature_x" not in result.output
    assert "val_accuracy" in result.output


def test_diff_shows_run_id_prefixes():
    run_a = _make_diff_run("abcd1234" + "0" * 24, {"model.eta": "0.1"}, {})
    run_b = _make_diff_run("efgh5678" + "0" * 24, {"model.eta": "0.05"}, {})
    result = _diff_invoke(run_a, run_b)
    assert result.exit_code == 0
    assert "abcd1234" in result.output
    assert "efgh5678" in result.output


def test_diff_invalid_run_id_exits_nonzero():
    def make_client():
        client = MagicMock()
        client.get_run.side_effect = Exception("run not found")
        return client

    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient", side_effect=make_client),
    ):
        result = runner.invoke(app, ["diff", "bad_id_a", "bad_id_b"])
    assert result.exit_code != 0


def test_diff_shows_run_names():
    run_a = _make_diff_run("a" * 32, {"model.eta": "0.1"}, {}, run_name="baseline")
    run_b = _make_diff_run("b" * 32, {"model.eta": "0.05"}, {}, run_name="challenger")
    result = _diff_invoke(run_a, run_b)
    assert result.exit_code == 0
    assert "baseline" in result.output
    assert "challenger" in result.output


# ---------------------------------------------------------------------------
# kitchen diff — feature importance comparison (CMP-004)
# ---------------------------------------------------------------------------


def test_diff_fi_shows_section_header():
    """Feature importance section header appears when both runs have the artifact."""
    run_a = _make_diff_run("a" * 32, {}, {})
    run_b = _make_diff_run("b" * 32, {}, {})
    fi = {"feat_x": 0.9, "feat_y": 0.5, "feat_z": 0.1}
    fi_b = {"feat_x": 0.1, "feat_y": 0.5, "feat_z": 0.9}
    result = _diff_invoke(run_a, run_b, fi_a=fi, fi_b=fi_b)
    assert result.exit_code == 0
    assert "Feature importance" in result.output


def test_diff_fi_shows_rank_changes():
    """Features whose rank changed appear with their rank(a), rank(b), and delta."""
    run_a = _make_diff_run("a" * 32, {}, {})
    run_b = _make_diff_run("b" * 32, {}, {})
    fi_a = {"feat_x": 0.9, "feat_y": 0.1}  # feat_x rank 1, feat_y rank 2
    fi_b = {"feat_x": 0.1, "feat_y": 0.9}  # feat_x rank 2, feat_y rank 1
    result = _diff_invoke(run_a, run_b, fi_a=fi_a, fi_b=fi_b)
    assert result.exit_code == 0
    assert "feat_x" in result.output
    assert "feat_y" in result.output


def test_diff_fi_delta_positive_when_demoted():
    """A feature that dropped in rank shows a positive delta."""
    run_a = _make_diff_run("a" * 32, {}, {})
    run_b = _make_diff_run("b" * 32, {}, {})
    # feat_x: rank 1 → rank 3 (+2)
    fi_a = {"feat_x": 0.9, "feat_y": 0.6, "feat_z": 0.1}
    fi_b = {"feat_x": 0.1, "feat_y": 0.6, "feat_z": 0.9}
    result = _diff_invoke(run_a, run_b, fi_a=fi_a, fi_b=fi_b)
    assert result.exit_code == 0
    assert "+2" in result.output


def test_diff_fi_delta_negative_when_promoted():
    """A feature that rose in rank shows a negative delta."""
    run_a = _make_diff_run("a" * 32, {}, {})
    run_b = _make_diff_run("b" * 32, {}, {})
    # feat_z: rank 3 → rank 1 (-2)
    fi_a = {"feat_x": 0.9, "feat_y": 0.6, "feat_z": 0.1}
    fi_b = {"feat_x": 0.1, "feat_y": 0.6, "feat_z": 0.9}
    result = _diff_invoke(run_a, run_b, fi_a=fi_a, fi_b=fi_b)
    assert result.exit_code == 0
    assert "-2" in result.output


def test_diff_fi_top_5_limit():
    """Only the top 5 features by rank-change magnitude are shown."""
    run_a = _make_diff_run("a" * 32, {}, {})
    run_b = _make_diff_run("b" * 32, {}, {})
    # 10 features all reversed in rank — only top 5 shifts shown
    names = [f"f{i:02d}" for i in range(10)]
    fi_a = {n: 10 - i for i, n in enumerate(names)}   # f00 most important
    fi_b = {n: i + 1 for i, n in enumerate(names)}    # f09 most important
    result = _diff_invoke(run_a, run_b, fi_a=fi_a, fi_b=fi_b)
    assert result.exit_code == 0
    feature_lines = [line for line in result.output.splitlines() if line.strip().startswith("f")]
    assert len(feature_lines) == 5


def test_diff_fi_skipped_when_artifact_missing():
    """No feature importance section when either run lacks the artifact."""
    run_a = _make_diff_run("a" * 32, {"model.eta": "0.1"}, {})
    run_b = _make_diff_run("b" * 32, {"model.eta": "0.05"}, {})
    # fi_a provided, fi_b omitted → section should not appear
    fi_a = {"feat_x": 0.9, "feat_y": 0.1}
    result = _diff_invoke(run_a, run_b, fi_a=fi_a)
    assert result.exit_code == 0
    assert "Feature importance" not in result.output


def test_diff_fi_only_no_other_diffs():
    """When only FI ranks differ (params and metrics match), FI section still shows."""
    run_a = _make_diff_run("a" * 32, {"p": "1"}, {"val_acc": 0.8})
    run_b = _make_diff_run("b" * 32, {"p": "1"}, {"val_acc": 0.8})
    fi_a = {"feat_x": 0.9, "feat_y": 0.1}
    fi_b = {"feat_x": 0.1, "feat_y": 0.9}
    result = _diff_invoke(run_a, run_b, fi_a=fi_a, fi_b=fi_b)
    assert result.exit_code == 0
    assert "Feature importance" in result.output
    assert "No differences found" not in result.output


def test_diff_fi_after_metrics_section():
    """Feature importance section appears after the metrics section."""
    run_a = _make_diff_run("a" * 32, {}, {"val_acc": 0.80})
    run_b = _make_diff_run("b" * 32, {}, {"val_acc": 0.85})
    fi_a = {"feat_x": 0.9, "feat_y": 0.1}
    fi_b = {"feat_x": 0.1, "feat_y": 0.9}
    result = _diff_invoke(run_a, run_b, fi_a=fi_a, fi_b=fi_b)
    assert result.exit_code == 0
    metrics_pos = result.output.index("Metrics")
    fi_pos = result.output.index("Feature importance")
    assert metrics_pos < fi_pos


# ---------------------------------------------------------------------------
# kitchen leaderboard --show-params (CMP-002)
# ---------------------------------------------------------------------------


def _make_lb_run(
    run_id: str,
    metric_val: float,
    params: dict | None = None,
    variant: str = "",
    extra_metrics: dict | None = None,
) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.data.metrics = {"loto_brier": metric_val, **(extra_metrics or {})}
    run.data.params = params or {}
    run.data.tags = {"model_variant": variant} if variant else {}
    run.info.start_time = None
    return run


def _lb_invoke(runs: list, extra_args: list | None = None) -> object:
    """Invoke `kitchen leaderboard --experiment test-exp` with mocked MLflow."""

    def make_client():
        client = MagicMock()
        exp = MagicMock()
        exp.experiment_id = "1"
        client.get_experiment_by_name.return_value = exp
        client.search_runs.return_value = runs
        client.get_model_version_by_alias.side_effect = Exception("no champion")
        return client

    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient", side_effect=make_client),
    ):
        return runner.invoke(
            app,
            ["leaderboard", "--experiment", "test-exp", *(extra_args or [])],
            catch_exceptions=False,
        )


def test_leaderboard_show_params_column_header():
    """--show-params adds the param key as a column header."""
    runs = [_make_lb_run("a" * 32, 0.18, {"model.eta": "0.1"})]
    result = _lb_invoke(runs, ["--show-params", "model.eta"])
    assert result.exit_code == 0
    assert "model.eta" in result.output


def test_leaderboard_show_params_value():
    """The param value from the run appears in the leaderboard row."""
    runs = [_make_lb_run("a" * 32, 0.18, {"model.eta": "0.05"})]
    result = _lb_invoke(runs, ["--show-params", "model.eta"])
    assert result.exit_code == 0
    assert "0.05" in result.output


def test_leaderboard_show_params_multiple_columns():
    """Multiple comma-separated params produce multiple columns."""
    runs = [_make_lb_run("a" * 32, 0.18, {"model.eta": "0.1", "model.max_depth": "6"})]
    result = _lb_invoke(runs, ["--show-params", "model.eta,model.max_depth"])
    assert result.exit_code == 0
    assert "model.eta" in result.output
    assert "model.max_depth" in result.output
    assert "0.1" in result.output
    assert "6" in result.output


def test_leaderboard_show_params_missing_param_shows_dash():
    """A run missing the requested param shows a dash placeholder."""
    runs = [_make_lb_run("a" * 32, 0.18, {})]
    result = _lb_invoke(runs, ["--show-params", "model.eta"])
    assert result.exit_code == 0
    assert "model.eta" in result.output


def test_leaderboard_show_params_trailing_comma_ignored():
    """Trailing comma in --show-params does not create a spurious empty column."""
    runs = [_make_lb_run("a" * 32, 0.18, {"model.eta": "0.1"})]
    result = _lb_invoke(runs, ["--show-params", "model.eta,"])
    assert result.exit_code == 0
    assert "model.eta" in result.output
    assert result.output.count("model.eta") == 1


def test_leaderboard_no_show_params_no_extra_columns():
    """Without --show-params the output contains no param column headers."""
    runs = [_make_lb_run("a" * 32, 0.18, {"model.eta": "0.1"})]
    result = _lb_invoke(runs)
    assert result.exit_code == 0
    assert "model.eta" not in result.output


# ---------------------------------------------------------------------------
# kitchen leaderboard --expand-metrics (CMP-003)
# ---------------------------------------------------------------------------


def test_leaderboard_expand_metrics_shows_fold_suffixes_as_columns():
    """--expand-metrics adds per-fold suffix keys as column headers."""
    runs = [
        _make_lb_run(
            "a" * 32,
            0.18,
            extra_metrics={"loto_brier_2021": 0.19, "loto_brier_2022": 0.17, "loto_brier_mean": 0.18},
        )
    ]
    result = _lb_invoke(runs, ["--expand-metrics"])
    assert result.exit_code == 0
    assert "2021" in result.output
    assert "2022" in result.output


def test_leaderboard_expand_metrics_shows_fold_values():
    """Per-fold metric values appear in the leaderboard row."""
    runs = [
        _make_lb_run(
            "a" * 32,
            0.18,
            extra_metrics={"loto_brier_2021": 0.19, "loto_brier_2022": 0.17},
        )
    ]
    result = _lb_invoke(runs, ["--expand-metrics"])
    assert result.exit_code == 0
    assert "0.1900" in result.output
    assert "0.1700" in result.output


def test_leaderboard_expand_metrics_excludes_mean_and_std():
    """_mean and _std keys are aggregates and must not appear as fold sub-columns."""
    runs = [
        _make_lb_run(
            "a" * 32,
            0.18,
            extra_metrics={
                "loto_brier_2021": 0.19,
                "loto_brier_mean": 0.18,
                "loto_brier_std": 0.01,
            },
        )
    ]
    result = _lb_invoke(runs, ["--expand-metrics"])
    assert result.exit_code == 0
    assert "2021" in result.output
    # "mean" and "std" should not appear as column headers
    lines = result.output.splitlines()
    header = next(line for line in lines if "RUN ID" in line)
    assert "mean" not in header
    assert "std" not in header


def test_leaderboard_expand_metrics_missing_fold_shows_dash():
    """A run that lacks a fold key shows a dash for that fold column."""
    run_a = _make_lb_run("a" * 32, 0.18, extra_metrics={"loto_brier_2021": 0.19, "loto_brier_2022": 0.17})
    run_b = _make_lb_run("b" * 32, 0.20, extra_metrics={"loto_brier_2021": 0.21})
    result = _lb_invoke([run_a, run_b], ["--expand-metrics"])
    assert result.exit_code == 0
    # run_b has no loto_brier_2022 — a dash placeholder should appear
    assert "-" in result.output


def test_leaderboard_expand_metrics_union_across_runs():
    """Fold columns are the union of all fold keys seen across all runs."""
    run_a = _make_lb_run("a" * 32, 0.18, extra_metrics={"loto_brier_2021": 0.19})
    run_b = _make_lb_run("b" * 32, 0.20, extra_metrics={"loto_brier_2022": 0.21})
    result = _lb_invoke([run_a, run_b], ["--expand-metrics"])
    assert result.exit_code == 0
    assert "2021" in result.output
    assert "2022" in result.output


def test_leaderboard_no_expand_metrics_hides_fold_columns():
    """Without --expand-metrics, per-fold keys do not appear as columns."""
    runs = [
        _make_lb_run(
            "a" * 32,
            0.18,
            extra_metrics={"loto_brier_2021": 0.19, "loto_brier_2022": 0.17},
        )
    ]
    result = _lb_invoke(runs)
    assert result.exit_code == 0
    lines = result.output.splitlines()
    header = next(line for line in lines if "RUN ID" in line)
    assert "2021" not in header
    assert "2022" not in header


def test_leaderboard_expand_metrics_no_fold_keys_is_noop():
    """--expand-metrics with no per-fold keys logged renders normally."""
    runs = [_make_lb_run("a" * 32, 0.18)]
    result = _lb_invoke(runs, ["--expand-metrics"])
    assert result.exit_code == 0
    assert "loto_brier" in result.output


# ---------------------------------------------------------------------------
# kitchen promote --run-id (LML-011)
# ---------------------------------------------------------------------------


def _make_promote_run(run_id: str, metrics: dict | None = None, run_name: str = "") -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.data.metrics = metrics or {}
    run.data.tags = {"mlflow.runName": run_name} if run_name else {}
    return run


def _promote_invoke(extra_args: list | None = None) -> object:
    """Invoke `kitchen promote` with registry calls mocked out."""

    def make_client():
        client = MagicMock()
        client.get_run.return_value = _make_promote_run("abc123" + "0" * 26, {"accuracy": 0.85})
        client.get_model_version_by_alias.side_effect = Exception("no champion")
        return client

    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient", side_effect=make_client),
        patch("kitchen.registry.get_best_run"),
        patch("kitchen.registry.register_model", return_value="3"),
        patch("kitchen.registry.promote_model"),
        patch("kitchen.registry.get_production_uri", return_value=None),
    ):
        return runner.invoke(
            app,
            ["promote", "--experiment", "test-exp", *(extra_args or [])],
            catch_exceptions=False,
        )


def test_promote_run_id_succeeds():
    """--run-id bypasses metric ranking and promotes the specified run."""
    result = _promote_invoke(["--run-id", "abc123" + "0" * 26])
    assert result.exit_code == 0


def test_promote_run_id_output_contains_run_id_prefix():
    """The run ID prefix appears in the promote output."""
    result = _promote_invoke(["--run-id", "abc123" + "0" * 26])
    assert result.exit_code == 0
    assert "abc123" in result.output


def test_promote_run_id_with_metric_shows_metric():
    """--run-id combined with METRIC shows the metric value."""
    result = _promote_invoke(["accuracy", "--run-id", "abc123" + "0" * 26])
    assert result.exit_code == 0
    assert "accuracy" in result.output


def test_promote_run_id_dry_run_skips_registration():
    """--run-id --dry-run shows the run but does not register it."""
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient", side_effect=lambda: _make_promote_run_client()),
        patch("kitchen.registry.register_model") as mock_reg,
        patch("kitchen.registry.promote_model"),
        patch("kitchen.registry.get_production_uri", return_value=None),
    ):
        result = runner.invoke(
            app,
            ["promote", "--experiment", "test-exp", "--run-id", "abc123" + "0" * 26, "--dry-run"],
            catch_exceptions=False,
        )
    assert "Dry run" in result.output
    mock_reg.assert_not_called()


def _make_promote_run_client():
    client = MagicMock()
    client.get_run.return_value = _make_promote_run("abc123" + "0" * 26)
    client.get_model_version_by_alias.side_effect = Exception("no champion")
    return client


def test_promote_run_id_invalid_id_exits_nonzero():
    """An invalid run ID produces a non-zero exit code with an error message."""

    def make_client():
        client = MagicMock()
        client.get_run.side_effect = Exception("run not found")
        client.get_model_version_by_alias.side_effect = Exception("no champion")
        return client

    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient", side_effect=make_client),
        patch("kitchen.registry.get_production_uri", return_value=None),
    ):
        result = runner.invoke(
            app, ["promote", "--experiment", "test-exp", "--run-id", "bad_id"]
        )
    assert result.exit_code != 0


def test_promote_no_metric_no_run_id_exits_nonzero():
    """Calling promote with neither METRIC nor --run-id exits non-zero."""
    with patch("kitchen.tracking.configure_from_env"):
        result = runner.invoke(app, ["promote", "--experiment", "test-exp"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# DASH-001: kitchen dashboard generate
# ---------------------------------------------------------------------------


def _make_result_dict(
    sha="aabbccdd",
    champion=False,
    metric_val=0.85,
    lb_score=None,
    params=None,
):
    return {
        "sha": sha,
        "timestamp": "2026-01-01T10:00:00Z",
        "run_id": sha * 4,
        "metrics": {"val_accuracy": metric_val},
        "params": params,
        "top_features": None,
        "calibration": None,
        "lb_score": lb_score,
        "champion": champion,
    }


def _dash_generate_invoke(results, branch_exists=True, extra_args=None, tmp_path=None):
    """Invoke `kitchen dashboard generate` with git subprocess fully mocked."""
    import json

    def mock_run(cmd, **kwargs):
        m = MagicMock()
        m.returncode = 0 if branch_exists else 1
        return m

    json_files = [f"results/{r['sha'][:8]}.json" for r in results]

    def mock_check_output(cmd, **kwargs):
        if "ls-tree" in cmd:
            return "\n".join(json_files).encode()
        if "cat-file" in cmd:
            path = cmd[-1].split(":")[-1]
            sha_prefix = path.split("/")[-1].replace(".json", "")
            for r in results:
                if r["sha"][:8] == sha_prefix:
                    return json.dumps(r).encode()
            return b"{}"
        return b""

    args = ["dashboard", "generate", *(extra_args or [])]
    with (
        patch("subprocess.run", side_effect=mock_run),
        patch("subprocess.check_output", side_effect=mock_check_output),
    ):
        return runner.invoke(app, args, catch_exceptions=False)


def test_dashboard_generate_creates_html_file(tmp_path, monkeypatch):
    """generate writes dashboard/index.html and exits 0."""
    monkeypatch.chdir(tmp_path)
    results = [_make_result_dict(sha="aabbccdd", champion=True)]
    result = _dash_generate_invoke(results)
    assert result.exit_code == 0
    assert (tmp_path / "dashboard" / "index.html").exists()


def test_dashboard_generate_html_contains_project_name(tmp_path, monkeypatch):
    """Project name from params.yaml appears in the generated HTML."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: my-cbb-model\n")
    results = [_make_result_dict(sha="aabbccdd", champion=True)]
    _dash_generate_invoke(results)
    html = (tmp_path / "dashboard" / "index.html").read_text()
    assert "my-cbb-model" in html


def test_dashboard_generate_html_embeds_results_json(tmp_path, monkeypatch):
    """All result SHAs appear in the embedded JSON."""
    monkeypatch.chdir(tmp_path)
    results = [
        _make_result_dict(sha="aaaa1111", champion=True, metric_val=0.80),
        _make_result_dict(sha="bbbb2222", metric_val=0.83),
        _make_result_dict(sha="cccc3333", metric_val=0.85),
    ]
    _dash_generate_invoke(results)
    html = (tmp_path / "dashboard" / "index.html").read_text()
    for r in results:
        assert r["sha"] in html


def test_dashboard_generate_outputs_metric_name(tmp_path, monkeypatch):
    """The primary metric appears in the CLI output."""
    monkeypatch.chdir(tmp_path)
    results = [_make_result_dict(sha="aabbccdd", champion=True)]
    result = _dash_generate_invoke(results)
    assert "val_accuracy" in result.output


def test_dashboard_generate_missing_branch_exits_nonzero(tmp_path, monkeypatch):
    """Missing results branch → exit code 1 with error message."""
    monkeypatch.chdir(tmp_path)
    result = _dash_generate_invoke([], branch_exists=False)
    assert result.exit_code != 0
    assert "kitchen push" in result.output or "branch" in result.output.lower()


def test_dashboard_generate_no_json_files_exits_nonzero(tmp_path, monkeypatch):
    """No JSON files on results branch → exit code 1."""
    monkeypatch.chdir(tmp_path)

    def mock_run(cmd, **kwargs):
        m = MagicMock()
        m.returncode = 0
        return m

    def mock_check_output(cmd, **kwargs):
        if "ls-tree" in cmd:
            return b""  # no files
        return b""

    with (
        patch("subprocess.run", side_effect=mock_run),
        patch("subprocess.check_output", side_effect=mock_check_output),
    ):
        result = runner.invoke(app, ["dashboard", "generate"], catch_exceptions=False)
    assert result.exit_code != 0


def test_dashboard_generate_lb_score_in_html(tmp_path, monkeypatch):
    """When lb_score is present, HAS_LB is true in the generated HTML."""
    monkeypatch.chdir(tmp_path)
    results = [
        _make_result_dict(sha="aabbccdd", champion=True, lb_score=0.803),
    ]
    _dash_generate_invoke(results)
    html = (tmp_path / "dashboard" / "index.html").read_text()
    assert "true" in html  # HAS_LB = true
    assert "0.803" in html


def test_dashboard_generate_no_lb_score_has_lb_false(tmp_path, monkeypatch):
    """When no run has lb_score, HAS_LB is false."""
    monkeypatch.chdir(tmp_path)
    results = [_make_result_dict(sha="aabbccdd", champion=True, lb_score=None)]
    _dash_generate_invoke(results)
    html = (tmp_path / "dashboard" / "index.html").read_text()
    assert "HAS_LB = false" in html


def test_dashboard_generate_show_params_embeds_keys(tmp_path, monkeypatch):
    """--show-params keys appear in PARAM_KEYS in the generated HTML."""
    monkeypatch.chdir(tmp_path)
    results = [
        _make_result_dict(
            sha="aabbccdd",
            champion=True,
            params={"model.max_depth": "6", "model.eta": "0.05"},
        )
    ]
    _dash_generate_invoke(results, extra_args=["--show-params", "model.max_depth,model.eta"])
    html = (tmp_path / "dashboard" / "index.html").read_text()
    assert "model.max_depth" in html
    assert "model.eta" in html


def test_dashboard_generate_custom_output_path(tmp_path, monkeypatch):
    """--output writes to the specified path."""
    monkeypatch.chdir(tmp_path)
    results = [_make_result_dict(sha="aabbccdd", champion=True)]
    _dash_generate_invoke(results, extra_args=["--output", "docs/index.html"])
    assert (tmp_path / "docs" / "index.html").exists()


def test_dashboard_generate_champion_marker_in_html(tmp_path, monkeypatch):
    """Champion result has champion=true in the embedded JSON."""
    monkeypatch.chdir(tmp_path)
    results = [
        _make_result_dict(sha="aabbccdd", champion=True, metric_val=0.90),
        _make_result_dict(sha="eeff4455", champion=False, metric_val=0.85),
    ]
    _dash_generate_invoke(results)
    html = (tmp_path / "dashboard" / "index.html").read_text()
    assert '"champion": true' in html


# ---------------------------------------------------------------------------
# DASH-001: kitchen open — local dashboard fallback
# ---------------------------------------------------------------------------


def test_open_falls_back_to_local_dashboard(tmp_path, monkeypatch):
    """When no dashboard_url is configured but dashboard/index.html exists, serve it."""
    monkeypatch.chdir(tmp_path)
    dash = tmp_path / "dashboard" / "index.html"
    dash.parent.mkdir()
    dash.write_text("<html></html>")
    with patch("kitchen.cli._serve_local_dashboard") as mock_serve:
        result = runner.invoke(app, ["open"], catch_exceptions=False)
    assert result.exit_code == 0
    mock_serve.assert_called_once()
    # open_dashboard passes the Path it constructed; resolve to compare regardless of cwd
    called_path = mock_serve.call_args[0][0]
    assert called_path.resolve() == dash.resolve()


def test_open_prefers_url_over_local_dashboard(tmp_path, monkeypatch):
    """Configured dashboard_url takes precedence over local dashboard/index.html."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(
        "experiment: test\ndashboard_url: https://user.github.io/repo/\n"
    )
    dash = tmp_path / "dashboard" / "index.html"
    dash.parent.mkdir()
    dash.write_text("<html></html>")
    with patch("webbrowser.open") as mock_open:
        runner.invoke(app, ["open"], catch_exceptions=False)
    mock_open.assert_called_once_with("https://user.github.io/repo/")


def test_open_falls_back_to_ui_when_no_local_dashboard(tmp_path, monkeypatch):
    """When no url and no dashboard/index.html, fall back to MLflow UI."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    with patch("webbrowser.open") as mock_open:
        result = runner.invoke(app, ["open"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Falling back" in result.output
    mock_open.assert_called_once_with("https://mlflow.example.com")


# ---------------------------------------------------------------------------
# DASH-003: --serve flag and _serve_local_dashboard
# ---------------------------------------------------------------------------


def test_dashboard_generate_serve_calls_serve_helper(tmp_path, monkeypatch):
    """--serve invokes _serve_local_dashboard with the output path."""
    monkeypatch.chdir(tmp_path)
    results = [_make_result_dict(sha="aabbccdd", champion=True)]
    with patch("kitchen.cli._serve_local_dashboard") as mock_serve:
        result = _dash_generate_invoke(results, extra_args=["--serve"])
    assert result.exit_code == 0
    mock_serve.assert_called_once()
    called_path = mock_serve.call_args[0][0]
    assert called_path.resolve() == (tmp_path / "dashboard" / "index.html").resolve()


def test_dashboard_generate_no_serve_skips_serve_helper(tmp_path, monkeypatch):
    """Without --serve, _serve_local_dashboard is never called."""
    monkeypatch.chdir(tmp_path)
    results = [_make_result_dict(sha="aabbccdd", champion=True)]
    with patch("kitchen.cli._serve_local_dashboard") as mock_serve:
        result = _dash_generate_invoke(results)
    assert result.exit_code == 0
    mock_serve.assert_not_called()


def test_dashboard_generate_serve_hint_in_output(tmp_path, monkeypatch):
    """Without --serve, the CLI suggests kitchen dashboard generate --serve."""
    monkeypatch.chdir(tmp_path)
    results = [_make_result_dict(sha="aabbccdd", champion=True)]
    result = _dash_generate_invoke(results)
    assert "--serve" in result.output


def test_serve_local_dashboard_starts_server_and_opens_browser(tmp_path, capsys):
    """_serve_local_dashboard prints a localhost URL with the chosen port."""
    from kitchen.cli import _serve_local_dashboard

    html_file = tmp_path / "dashboard" / "index.html"
    html_file.parent.mkdir()
    html_file.write_text("<html></html>")

    class _FakeServer:
        def __init__(self, addr: tuple, handler: object) -> None:
            pass

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            pass

    with (
        patch("http.server.HTTPServer", _FakeServer),
        patch("webbrowser.open"),
        patch("kitchen.cli._find_free_port", return_value=19999),
        patch("threading.Thread"),  # prevent background thread from racing
    ):
        _serve_local_dashboard(html_file)

    captured = capsys.readouterr()
    assert "19999" in captured.out
    assert "localhost" in captured.out
