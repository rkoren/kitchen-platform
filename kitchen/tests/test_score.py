"""Tests for `kitchen score` — register a project scoring callable as the metric source (GEN-006)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import mlflow
import pytest
from typer.testing import CliRunner

from kitchen.cli import app
from kitchen.config import KitchenConfig, ScorerConfig
from kitchen.menu import Menu, load_scorer_callable

runner = CliRunner()


def _menu(scorer_module: str, *, function: str = "score") -> str:
    return textwrap.dedent(f"""\
        project: demo
        experiment: demo
        pipeline: []
        recipes: {{}}
        mlflow:
          tracking_uri: sqlite:///mlruns.db
        scorer:
          source: {scorer_module}.py
          function: {function}
        thresholds:
          track_score: 0.5
        run_name: score-run
    """)


def _project(tmp_path: Path, scorer_module: str, body: str, *, function: str = "score") -> Path:
    """Write a menu + a top-level scorer module. A unique `scorer_module` per test avoids
    sys.modules caching collisions between tests that import a same-named module."""
    (tmp_path / "menu.yaml").write_text(_menu(scorer_module, function=function))
    (tmp_path / f"{scorer_module}.py").write_text(textwrap.dedent(body))
    return tmp_path


# ── config + loader units ────────────────────────────────────────────────────


def test_scorer_config_on_kitchen_config_via_menu_bridge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "menu.yaml").write_text(_menu("sc_cfg"))
    cfg = KitchenConfig.from_yaml("menu.yaml")
    assert isinstance(cfg.scorer, ScorerConfig)
    assert cfg.scorer.source == "sc_cfg.py"
    assert cfg.scorer.function == "score"
    # Menu carries it too (so it appears in the schema)
    assert Menu.from_yaml("menu.yaml").scorer.source == "sc_cfg.py"


def test_load_scorer_callable_imports_function(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import sys

    sys.path.insert(0, str(tmp_path))
    (tmp_path / "sc_loader.py").write_text("def my_score(params, store):\n    return {'m': 1.0}\n")
    fn = load_scorer_callable(ScorerConfig(source="sc_loader.py", function="my_score"))
    assert fn(None, None) == {"m": 1.0}


def test_load_scorer_callable_missing_attr_raises_importerror(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import sys

    sys.path.insert(0, str(tmp_path))
    (tmp_path / "sc_noattr.py").write_text("def other():\n    return {}\n")
    with pytest.raises(ImportError, match="cannot import name 'score'"):
        load_scorer_callable(ScorerConfig(source="sc_noattr.py", function="score"))


# ── kitchen score command ─────────────────────────────────────────────────────


def test_score_logs_metrics_to_run_and_writes_metrics_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Pin an absolute tracking URI so the command's write and the assertion's read hit one DB.
    uri = f"sqlite:///{tmp_path}/mlruns.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    _project(
        tmp_path, "sc_happy",
        "def score(params, store):\n    return {'track_score': 0.83, 'n_tracks': 1200}\n",
    )
    result = runner.invoke(app, ["score"], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    # metrics.json carries the scalar metrics + the run_id (for thresholds / report / push)
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["track_score"] == pytest.approx(0.83)
    assert payload["n_tracks"] == pytest.approx(1200.0)
    run_id = payload["run_id"]

    # and the same metrics are on an MLflow run (so leaderboard / promote rank on them)
    mlflow.set_tracking_uri(uri)
    logged = mlflow.tracking.MlflowClient().get_run(run_id).data.metrics
    assert logged["track_score"] == pytest.approx(0.83)
    assert "track_score" in result.output


def test_score_accepts_zero_arg_callable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project(tmp_path, "sc_zero", "def score():\n    return {'track_score': 0.9}\n")
    result = runner.invoke(app, ["score"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "metrics.json").read_text())["track_score"] == pytest.approx(0.9)


def test_score_no_scorer_configured_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "menu.yaml").write_text("project: demo\nexperiment: demo\npipeline: []\nrecipes: {}\n")
    result = runner.invoke(app, ["score"])
    assert result.exit_code != 0
    assert "no 'scorer:' configured" in result.output


def test_score_unimportable_scorer_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "menu.yaml").write_text(_menu("sc_absent"))  # module file never written
    result = runner.invoke(app, ["score"])
    assert result.exit_code != 0
    assert "could not load scorer" in result.output


def test_score_rejects_non_scalar_metric_without_logging_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project(
        tmp_path, "sc_bad",
        "def score(params, store):\n    return {'track_score': 0.8, 'notes': 'oops'}\n",
    )
    result = runner.invoke(app, ["score"])
    assert result.exit_code != 0
    assert "notes" in result.output and "str" in result.output
    # rejected before opening the run → no metrics.json, no run left behind
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "mlruns.db").exists()


def test_score_is_a_menu_run_pipeline_verb(tmp_path, monkeypatch):
    # `score` is a platform verb, so `pipeline: [score]` is valid and `menu run` dispatches it.
    monkeypatch.chdir(tmp_path)
    _project(tmp_path, "sc_pipe", "def score(params, store):\n    return {'track_score': 0.7}\n")
    menu = (tmp_path / "menu.yaml").read_text().replace("pipeline: []", "pipeline: [score]")
    (tmp_path / "menu.yaml").write_text(menu)
    result = runner.invoke(app, ["menu", "run", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "score: kitchen score" in result.output


def test_score_creates_a_distinct_run_each_call(tmp_path, monkeypatch):
    # score opens its own run (a separate leaderboard row from any train run — the GEN-001
    # "attach to an existing run" story is out of scope here). Two calls → two distinct runs.
    monkeypatch.chdir(tmp_path)
    _project(tmp_path, "sc_twice", "def score(params, store):\n    return {'track_score': 0.7}\n")
    r1 = runner.invoke(app, ["score"], catch_exceptions=False)
    id1 = json.loads((tmp_path / "metrics.json").read_text())["run_id"]
    r2 = runner.invoke(app, ["score"], catch_exceptions=False)
    id2 = json.loads((tmp_path / "metrics.json").read_text())["run_id"]
    assert r1.exit_code == 0 and r2.exit_code == 0
    assert id1 != id2
