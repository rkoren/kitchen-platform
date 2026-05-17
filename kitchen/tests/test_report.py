"""Tests for `kitchen report`."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from kitchen.cli import app

runner = CliRunner()

MINIMAL_PARAMS = "experiment: test-project\n"


def _write_metrics(path, metrics: dict) -> None:
    path.write_text(json.dumps(metrics))


def _invoke(tmp_path, monkeypatch, extra_args=None):
    monkeypatch.chdir(tmp_path)
    return runner.invoke(app, ["report"] + (extra_args or []))


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_report_missing_metrics_file(tmp_path, monkeypatch):
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code != 0
    assert "not found" in result.output


def test_report_invalid_json(tmp_path, monkeypatch):
    (tmp_path / "metrics.json").write_text("not json{")
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code != 0
    assert "could not parse" in result.output


def test_report_custom_metrics_path(tmp_path, monkeypatch):
    custom = tmp_path / "subdir" / "my_metrics.json"
    custom.parent.mkdir()
    _write_metrics(custom, {"accuracy": 0.95})
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["report", "--metrics", str(custom)])
    assert result.exit_code == 0
    assert "accuracy" in result.output


# ---------------------------------------------------------------------------
# GitHub format (default)
# ---------------------------------------------------------------------------

def test_report_github_format_headers(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.876543})
    (tmp_path / "params.yaml").write_text(MINIMAL_PARAMS)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0
    assert "## Kitchen Report" in result.output
    assert "`test-project`" in result.output


def test_report_github_format_table(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.876543, "val_logloss": 0.312})
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0
    assert "| Metric | Value |" in result.output
    assert "`val_accuracy`" in result.output
    assert "0.876543" in result.output
    assert "`val_logloss`" in result.output


def test_report_github_shows_run_name(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {
        "_run": {"run_name": "baseline-run-42"},
        "accuracy": 0.9,
    })
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0
    assert "baseline-run-42" in result.output
    # _run metadata should not appear as a metric row
    assert "`_run`" not in result.output


def test_report_github_no_run_name(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"f1": 0.75})
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0
    assert "## Kitchen Report" in result.output


def test_report_integer_metric(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"n_samples": 1000})
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0
    assert "1000" in result.output


def test_report_metrics_sorted(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"z_metric": 0.1, "a_metric": 0.9})
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0
    lines = result.output.splitlines()
    metric_lines = [l for l in lines if "`a_metric`" in l or "`z_metric`" in l]
    assert len(metric_lines) == 2
    assert "a_metric" in metric_lines[0]
    assert "z_metric" in metric_lines[1]


# ---------------------------------------------------------------------------
# Plain format
# ---------------------------------------------------------------------------

def test_report_plain_format(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.876543})
    (tmp_path / "params.yaml").write_text(MINIMAL_PARAMS)
    result = _invoke(tmp_path, monkeypatch, extra_args=["--format", "plain"])
    assert result.exit_code == 0
    assert "Experiment: test-project" in result.output
    assert "val_accuracy: 0.876543" in result.output
    assert "| Metric |" not in result.output


def test_report_plain_no_markdown(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"accuracy": 0.9})
    result = _invoke(tmp_path, monkeypatch, extra_args=["--format", "plain"])
    assert "##" not in result.output
    assert "|" not in result.output


# ---------------------------------------------------------------------------
# Experiment name fallback
# ---------------------------------------------------------------------------

def test_report_experiment_from_params(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"auc": 0.88})
    (tmp_path / "params.yaml").write_text("experiment: my-competition\n")
    result = _invoke(tmp_path, monkeypatch)
    assert "my-competition" in result.output


def test_report_experiment_unknown_when_no_params(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"auc": 0.88})
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0
    assert "unknown" in result.output
