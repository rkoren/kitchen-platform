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
    metric_lines = [ln for ln in lines if "`a_metric`" in ln or "`z_metric`" in ln]
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


# ---------------------------------------------------------------------------
# --compare flag (GH-006 / GH-003)
# ---------------------------------------------------------------------------

def test_report_compare_missing_file(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"accuracy": 0.9})
    result = _invoke(tmp_path, monkeypatch, extra_args=["--compare", str(tmp_path / "base.json")])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_report_compare_invalid_json(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"accuracy": 0.9})
    (tmp_path / "base.json").write_text("not json{")
    result = _invoke(tmp_path, monkeypatch, extra_args=["--compare", str(tmp_path / "base.json")])
    assert result.exit_code != 0
    assert "could not parse" in result.output


def test_report_compare_shows_four_column_table(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"accuracy": 0.91})
    _write_metrics(tmp_path / "base.json", {"accuracy": 0.88})
    result = _invoke(tmp_path, monkeypatch, extra_args=["--compare", str(tmp_path / "base.json")])
    assert result.exit_code == 0
    assert "| Metric | Base | PR | Delta |" in result.output
    assert "0.910000" in result.output
    assert "0.880000" in result.output
    assert "+0.030000" in result.output


def test_report_compare_new_metric_shows_new_label(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"accuracy": 0.91, "f1": 0.85})
    _write_metrics(tmp_path / "base.json", {"accuracy": 0.88})
    result = _invoke(tmp_path, monkeypatch, extra_args=["--compare", str(tmp_path / "base.json")])
    assert result.exit_code == 0
    assert "(new)" in result.output


def test_report_compare_run_metadata_excluded(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"accuracy": 0.91})
    _write_metrics(tmp_path / "base.json", {"accuracy": 0.88, "_run": {"run_name": "old-run"}})
    result = _invoke(tmp_path, monkeypatch, extra_args=["--compare", str(tmp_path / "base.json")])
    assert result.exit_code == 0
    assert "`_run`" not in result.output


def test_report_compare_negative_delta(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"logloss": 0.45})
    _write_metrics(tmp_path / "base.json", {"logloss": 0.38})
    result = _invoke(tmp_path, monkeypatch, extra_args=["--compare", str(tmp_path / "base.json")])
    assert result.exit_code == 0
    assert "+0.070000" in result.output or "-" not in result.output.split("logloss")[0]
    assert "0.450000" in result.output


def test_report_compare_plain_format(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"accuracy": 0.91})
    _write_metrics(tmp_path / "base.json", {"accuracy": 0.88})
    result = _invoke(tmp_path, monkeypatch, extra_args=[
        "--compare", str(tmp_path / "base.json"), "--format", "plain"
    ])
    assert result.exit_code == 0
    assert "base:" in result.output
    assert "delta:" in result.output
    assert "| Metric |" not in result.output


def test_report_compare_integer_delta(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"n_samples": 1200})
    _write_metrics(tmp_path / "base.json", {"n_samples": 1000})
    result = _invoke(tmp_path, monkeypatch, extra_args=["--compare", str(tmp_path / "base.json")])
    assert result.exit_code == 0
    assert "+200" in result.output
