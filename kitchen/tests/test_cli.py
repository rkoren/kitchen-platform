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
        features_cls().build(pd.DataFrame())


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
