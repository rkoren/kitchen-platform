"""Tests for the generic command sweep — `kitchen sweep --run` (GEN-004)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from typer.testing import CliRunner

from kitchen.cli import app
from kitchen.runstore import read_runs

runner = CliRunner()

# A trial command: score = 10 * a, written to the sweep's per-combo metrics file.
TRIAL = textwrap.dedent("""\
    import os, json, sys
    a = float(sys.argv[1])
    json.dump({"score": 10.0 * a}, open(os.environ["KITCHEN_METRICS_FILE"], "w"))
    print(f"ran a={a}")
""")

# A trial that writes the wrong metric name (no "score").
TRIAL_NO_METRIC = textwrap.dedent("""\
    import os, json, sys
    json.dump({"other": 1.0}, open(os.environ["KITCHEN_METRICS_FILE"], "w"))
""")

# A trial that always fails.
TRIAL_FAIL = "import sys; sys.exit(3)"


def _write(tmp_path: Path, body: str, name: str = "trial.py") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def _run_arg(trial: Path) -> str:
    return f"{sys.executable} {trial} {{a}}"


def test_generic_sweep_runs_ranks_and_writes_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trial = _write(tmp_path, TRIAL)
    result = runner.invoke(
        app,
        ["sweep", "--run", _run_arg(trial), "--param", "a=1,3,2",
         "--metric", "score", "--higher-is-better"],
    )
    assert result.exit_code == 0, result.output
    # best is a=3 (score 30)
    star = next(ln for ln in result.output.splitlines() if ln.strip().startswith("★"))
    assert "a=3" in star and "30.000000" in star
    # every combo recorded in the store, params coerced+stringified
    runs = read_runs(tmp_path / "sweep.jsonl")
    assert {r.params["a"] for r in runs} == {"1", "2", "3"}
    assert all(r.metrics.get("score") == 10.0 * float(r.params["a"]) for r in runs)


def test_generic_sweep_run_and_override_mutually_exclusive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["sweep", "--run", "echo {a}", "--param", "a=1,2", "--override", "model.x=1,2"]
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_generic_sweep_requires_param(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["sweep", "--run", "echo hi", "--metric", "score"])
    assert result.exit_code != 0
    assert "--run needs at least one --param" in result.output


def test_generic_sweep_requires_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["sweep", "--run", "echo {a}", "--param", "a=1,2"])
    assert result.exit_code != 0
    assert "--metric is required" in result.output


def test_generic_sweep_lints_unknown_placeholder_before_running(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["sweep", "--run", "echo {a} {typo}", "--param", "a=1,2", "--metric", "score"]
    )
    assert result.exit_code != 0
    assert "typo" in result.output and "not in --param" in result.output
    assert not (tmp_path / "sweep.jsonl").exists()  # nothing ran


def test_generic_sweep_metric_less_combo_recorded_but_unranked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trial = _write(tmp_path, TRIAL_NO_METRIC)
    result = runner.invoke(
        app, ["sweep", "--run", _run_arg(trial), "--param", "a=1,2", "--metric", "score"]
    )
    # no combo produced `score` → nothing to rank (exit 1), but combos are in the store
    assert result.exit_code != 0
    assert "nothing to rank" in result.output
    runs = read_runs(tmp_path / "sweep.jsonl")
    assert len(runs) == 2 and all("score" not in r.metrics for r in runs)


def test_generic_sweep_first_combo_failure_aborts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trial = _write(tmp_path, TRIAL_FAIL)
    result = runner.invoke(
        app, ["sweep", "--run", _run_arg(trial), "--param", "a=1,2", "--metric", "score"]
    )
    assert result.exit_code != 0
    assert "first sweep run failed" in result.output


def test_generic_sweep_dry_run_prints_argvs_without_running(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trial = _write(tmp_path, TRIAL_FAIL)  # would exit 3 if actually run
    result = runner.invoke(
        app,
        ["sweep", "--run", _run_arg(trial), "--param", "a=1,2", "--metric", "score", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert "[1/2]" in result.output and "[2/2]" in result.output
    assert not (tmp_path / "sweep.jsonl").exists()  # nothing ran


def test_generic_sweep_respects_max_combos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["sweep", "--run", "echo {a} {b}", "--param", "a=1,2,3", "--param", "b=1,2,3",
         "--metric", "score", "--max-combos", "4"],
    )
    assert result.exit_code != 0
    assert "would launch 9 runs (limit 4)" in result.output
