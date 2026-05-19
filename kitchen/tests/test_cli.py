"""Smoke tests for `kitchen init` scaffold output.

Verifies that a fresh scaffold:
- creates all expected files
- produces parseable YAML
- has Python modules that import at module level without errors
- uses correct schema field names (no memory_mb/timeout_s)
- contains no maintainer-specific names
- leaves intentional TODO boundaries as NotImplementedError (not silent pass-throughs)
"""
from __future__ import annotations

import importlib.util
import sys

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
def project(tmp_path):
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
    import pandas as pd
    cls_name = "MyCompetitionFeatures"
    features_cls = getattr(mod, cls_name, None)
    if features_cls is None:
        pytest.skip(f"Class {cls_name} not found — name derivation may differ")
    with pytest.raises(NotImplementedError):
        features_cls().build(pd.DataFrame(), params={})


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
    assert sentinel.read_text() == "# modified", "Re-init without --overwrite should skip existing files"


def test_init_overwrite_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "my-competition"], catch_exceptions=False)
    sentinel = tmp_path / "my-competition" / "params.yaml"
    sentinel.write_text("# modified")
    runner.invoke(app, ["init", "my-competition", "--overwrite"], catch_exceptions=False)
    assert sentinel.read_text() != "# modified", "--overwrite should replace existing files"


# --- kitchen init name validation ---

@pytest.mark.parametrize("name", [
    "titanic",
    "spaceship-titanic",
    "house-prices-2024",
    "a",
    "abc123",
])
def test_init_valid_names(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", name], catch_exceptions=False)
    assert result.exit_code == 0, f"Expected valid name {name!r} to be accepted"


@pytest.mark.parametrize("name", [
    "My-Competition",   # uppercase
    "my competition",   # space
    "-leading",         # leading hyphen
    "trailing-",        # trailing hyphen
    "a--b",             # consecutive hyphens
    "1competition",     # starts with digit
    "../escape",        # path traversal
    "",                 # empty
])
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

    def fake_pipeline(params_file="params.yaml"):
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

    def fake_pipeline(params_file="params.yaml"):
        calls.append(params_file)

    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "train", "--params", "custom.yaml"])
    assert result.exit_code == 0
    assert calls == ["custom.yaml"]


def test_run_train_missing_src_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")

    def fake_pipeline(params_file="params.yaml"):
        raise ModuleNotFoundError("No module named 'src.features.run'")

    monkeypatch.setattr("kitchen.flows.train_flow.train_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "train"])
    assert result.exit_code != 0
    assert "src/" in result.output


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

    def fake_pipeline(params_file="params.yaml"):
        calls.append(params_file)
        return "monitoring/drift.html"

    monkeypatch.setattr("kitchen.flows.monitor_flow.monitor_pipeline", fake_pipeline)
    result = runner.invoke(app, ["run", "monitor"])
    assert result.exit_code == 0
    assert calls == ["params.yaml"]
    assert "monitoring/drift.html" in result.output


def test_run_monitor_missing_output_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: test\n")

    def fake_pipeline(params_file="params.yaml"):
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

    def fake_load(uri):
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

    fake_evaluate, calls, fake_model = _make_evaluate_mocks(monkeypatch)

    src = tmp_path / "src" / "evaluate"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "run.py").write_text(
        "def evaluate(model, params, store):\n    return {'val_brier': 0.18}\n"
    )

    # Bypass actual import with a direct monkeypatch on the CLI's lazy import path
    import sys
    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = fake_evaluate
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate"])
    assert result.exit_code == 0
    assert "test-project-model@champion" in result.output


def test_run_evaluate_custom_model_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    fake_evaluate, calls, _ = _make_evaluate_mocks(monkeypatch)

    import sys
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

    import sys
    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = fake_evaluate
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate", "--alias", "staging"])
    assert result.exit_code == 0
    assert "@staging" in result.output


def test_run_evaluate_prints_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    fake_evaluate, _, _ = _make_evaluate_mocks(monkeypatch, metrics={"val_brier": 0.18, "val_accuracy": 0.72})

    import sys
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

    import sys
    fake_mod = type(sys)("src.evaluate.run")
    fake_mod.evaluate = lambda m, p, s: {}
    monkeypatch.setitem(sys.modules, "src.evaluate.run", fake_mod)

    result = runner.invoke(app, ["run", "evaluate"])
    assert result.exit_code != 0
    assert "error loading model" in result.output


def test_run_evaluate_invalid_flavor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)
    monkeypatch.setattr("kitchen.tracking.configure_from_env", lambda: None)
    result = runner.invoke(app, ["run", "evaluate", "--flavor", "torchscript"])
    assert result.exit_code != 0
    assert "unknown flavor" in result.output


def test_run_evaluate_missing_src_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(EVAL_PARAMS)

    _make_evaluate_mocks(monkeypatch)

    # `from src.evaluate.run import evaluate` uses builtins.__import__, not
    # importlib.import_module, so we must intercept at the builtin level.
    import builtins
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
        app, ["init", "march-mania", "--source", "kaggle", "--competition", "march-ml-mania-2026"],
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
        app, ["init", "my-comp", "--source", "kaggle", "--competition", "my-comp"],
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
        app, ["init", "my-comp", "--template", "baseline-xgb"], catch_exceptions=False,
    )
    assert result.exit_code == 0
    train_src = (tmp_path / "my-comp" / "src" / "train" / "run.py").read_text()
    assert "XGBClassifier" in train_src
    assert "xgboost" in train_src


def test_init_baseline_lr_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["init", "my-comp", "--template", "baseline-lr"], catch_exceptions=False,
    )
    assert result.exit_code == 0
    train_src = (tmp_path / "my-comp" / "src" / "train" / "run.py").read_text()
    assert "LogisticRegression" in train_src


def test_init_invalid_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my-comp", "--template", "random-forest"])
    assert result.exit_code != 0
    assert "invalid template" in result.output


def test_init_default_train_template_unchanged(scaffold):
    train_src = (scaffold / "src" / "train" / "run.py").read_text()
    assert "NotImplementedError" in train_src
    assert "XGBClassifier" not in train_src


def test_init_kaggle_with_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, [
            "init", "mania", "--source", "kaggle",
            "--competition", "march-ml-mania-2026", "--template", "baseline-xgb",
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
        app, ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "my-comp"],
        catch_exceptions=False,
    )
    content = yaml.safe_load((tmp_path / "my-comp" / _CI_WORKFLOW_PATH).read_text())
    steps = content["jobs"]["train-evaluate"]["steps"]
    step_names = [s.get("name", "") for s in steps]
    assert "Ingest data" in step_names


def test_init_ci_kaggle_ingest_uses_secrets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(
        app, ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "my-comp"],
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
        app, ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "my-comp"],
        catch_exceptions=False,
    )
    assert "KAGGLE_USERNAME" in result.output
    assert "KAGGLE_KEY" in result.output


def test_init_ci_note_in_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my-comp", "--ci"], catch_exceptions=False)
    assert "train-evaluate.yml" in result.output


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
        app, ["init", "my-comp", "--ci", "--source", "kaggle", "--competition", "my-comp"],
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
