"""Tests for `kitchen report`."""

from __future__ import annotations

import json
from unittest.mock import patch

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
    _write_metrics(
        tmp_path / "metrics.json",
        {
            "_run": {"run_name": "baseline-run-42"},
            "accuracy": 0.9,
        },
    )
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
    result = _invoke(
        tmp_path,
        monkeypatch,
        extra_args=["--compare", str(tmp_path / "base.json"), "--format", "plain"],
    )
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


# ---------------------------------------------------------------------------
# Thresholds (GH-008)
# ---------------------------------------------------------------------------

PARAMS_WITH_THRESHOLD = """\
experiment: test-exp
thresholds:
  val_accuracy: 0.85
"""


def test_report_no_thresholds_exits_zero(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.5})
    (tmp_path / "params.yaml").write_text(MINIMAL_PARAMS)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0


def test_report_threshold_pass_exits_zero(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.90})
    (tmp_path / "params.yaml").write_text(PARAMS_WITH_THRESHOLD)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0


def test_report_threshold_fail_exits_one(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.75})
    (tmp_path / "params.yaml").write_text(PARAMS_WITH_THRESHOLD)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 1


def test_report_threshold_fail_shows_violation_table(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.75})
    (tmp_path / "params.yaml").write_text(PARAMS_WITH_THRESHOLD)
    result = _invoke(tmp_path, monkeypatch)
    assert "Threshold Violations" in result.output
    assert "0.850000" in result.output  # threshold
    assert "0.750000" in result.output  # actual


def test_report_threshold_fail_still_shows_metrics(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.75})
    (tmp_path / "params.yaml").write_text(PARAMS_WITH_THRESHOLD)
    result = _invoke(tmp_path, monkeypatch)
    assert "| Metric | Value |" in result.output
    assert "`val_accuracy`" in result.output


def test_report_threshold_missing_metric_skipped(tmp_path, monkeypatch):
    # threshold defined for a metric that isn't in metrics.json — should not fail
    _write_metrics(tmp_path / "metrics.json", {"other_metric": 0.5})
    (tmp_path / "params.yaml").write_text(PARAMS_WITH_THRESHOLD)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0


def test_report_threshold_fail_plain_format(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.75})
    (tmp_path / "params.yaml").write_text(PARAMS_WITH_THRESHOLD)
    result = _invoke(tmp_path, monkeypatch, extra_args=["--format", "plain"])
    assert result.exit_code == 1
    assert "Threshold violations" in result.output
    assert "FAIL" in result.output
    assert "0.75" in result.output


def test_report_threshold_multiple_only_fails_report_failing_ones(tmp_path, monkeypatch):
    params = "experiment: test-exp\nthresholds:\n  val_accuracy: 0.80\n  f1: 0.70\n"
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.85, "f1": 0.60})
    (tmp_path / "params.yaml").write_text(params)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 1
    assert "`f1`" in result.output
    # passing metric should not appear in violations table
    assert result.output.count("Threshold Violations") == 1
    violations_section = result.output.split("Threshold Violations")[1]
    assert "`val_accuracy`" not in violations_section


# ---------------------------------------------------------------------------
# Lower-is-better thresholds (K-012)
# ---------------------------------------------------------------------------

PARAMS_MAX_THRESHOLD = """\
experiment: test-exp
thresholds:
  val_logloss:
    max: 0.40
"""

PARAMS_MIXED_THRESHOLDS = """\
experiment: test-exp
thresholds:
  val_accuracy: 0.80
  val_logloss:
    max: 0.40
"""


def test_report_max_threshold_pass_exits_zero(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_logloss": 0.35})
    (tmp_path / "params.yaml").write_text(PARAMS_MAX_THRESHOLD)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0


def test_report_max_threshold_fail_exits_one(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_logloss": 0.52})
    (tmp_path / "params.yaml").write_text(PARAMS_MAX_THRESHOLD)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 1


def test_report_max_threshold_fail_shows_constraint(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_logloss": 0.52})
    (tmp_path / "params.yaml").write_text(PARAMS_MAX_THRESHOLD)
    result = _invoke(tmp_path, monkeypatch)
    assert "Threshold Violations" in result.output
    assert "<= 0.400000" in result.output
    assert "0.520000" in result.output


def test_report_max_threshold_fail_plain_format(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_logloss": 0.52})
    (tmp_path / "params.yaml").write_text(PARAMS_MAX_THRESHOLD)
    result = _invoke(tmp_path, monkeypatch, extra_args=["--format", "plain"])
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "<= 0.400000" in result.output


def test_report_min_threshold_via_spec(tmp_path, monkeypatch):
    params = "experiment: test-exp\nthresholds:\n  val_accuracy:\n    min: 0.85\n"
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.75})
    (tmp_path / "params.yaml").write_text(params)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 1
    assert ">= 0.850000" in result.output


def test_report_range_threshold_both_pass(tmp_path, monkeypatch):
    params = "experiment: test-exp\nthresholds:\n  score:\n    min: 0.60\n    max: 0.95\n"
    _write_metrics(tmp_path / "metrics.json", {"score": 0.80})
    (tmp_path / "params.yaml").write_text(params)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0


def test_report_range_threshold_upper_violated(tmp_path, monkeypatch):
    params = "experiment: test-exp\nthresholds:\n  score:\n    min: 0.60\n    max: 0.95\n"
    _write_metrics(tmp_path / "metrics.json", {"score": 0.97})
    (tmp_path / "params.yaml").write_text(params)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 1
    assert "<= 0.950000" in result.output


def test_report_mixed_min_max_thresholds(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.85, "val_logloss": 0.52})
    (tmp_path / "params.yaml").write_text(PARAMS_MIXED_THRESHOLDS)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 1
    assert "`val_logloss`" in result.output
    violations_section = result.output.split("Threshold Violations")[1]
    assert "`val_accuracy`" not in violations_section


# ---------------------------------------------------------------------------
# Kaggle leaderboard score (GH-007)
# ---------------------------------------------------------------------------


def test_report_kaggle_score_appears_in_dedicated_section(tmp_path, monkeypatch):
    _write_metrics(
        tmp_path / "metrics.json", {"val_accuracy": 0.88, "kaggle_public_score": 0.77123}
    )
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0
    assert "Kaggle Public Leaderboard" in result.output
    assert "0.771230" in result.output


def test_report_kaggle_score_not_in_metrics_table(tmp_path, monkeypatch):
    _write_metrics(
        tmp_path / "metrics.json", {"val_accuracy": 0.88, "kaggle_public_score": 0.77123}
    )
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0
    # Score should not appear as a row in the main metrics table
    assert "`kaggle_public_score`" not in result.output


def test_report_no_kaggle_score_no_leaderboard_section(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.88})
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 0
    assert "Kaggle" not in result.output


def test_report_kaggle_score_compare_shows_delta(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.91, "kaggle_public_score": 0.78})
    _write_metrics(tmp_path / "base.json", {"val_accuracy": 0.88, "kaggle_public_score": 0.75})
    result = _invoke(tmp_path, monkeypatch, extra_args=["--compare", str(tmp_path / "base.json")])
    assert result.exit_code == 0
    assert "Kaggle Public Leaderboard" in result.output
    assert "0.780000" in result.output
    assert "0.750000" in result.output
    assert "+0.030000" in result.output
    # Score should not appear in compare table
    assert "`kaggle_public_score`" not in result.output


def test_report_kaggle_score_compare_no_base_score(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.91, "kaggle_public_score": 0.78})
    _write_metrics(tmp_path / "base.json", {"val_accuracy": 0.88})
    result = _invoke(tmp_path, monkeypatch, extra_args=["--compare", str(tmp_path / "base.json")])
    assert result.exit_code == 0
    assert "Kaggle Public Leaderboard" in result.output
    assert "0.780000" in result.output
    assert "base:" not in result.output.split("Kaggle")[1].splitlines()[0]


def test_report_kaggle_score_plain_format(tmp_path, monkeypatch):
    _write_metrics(
        tmp_path / "metrics.json", {"val_accuracy": 0.88, "kaggle_public_score": 0.77123}
    )
    result = _invoke(tmp_path, monkeypatch, extra_args=["--format", "plain"])
    assert result.exit_code == 0
    assert "Kaggle Public Leaderboard: 0.771230" in result.output
    assert "kaggle_public_score" not in result.output.replace("Kaggle Public Leaderboard", "")


def test_report_kaggle_score_plain_format_compare_delta(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"kaggle_public_score": 0.78})
    _write_metrics(tmp_path / "base.json", {"kaggle_public_score": 0.75})
    result = _invoke(
        tmp_path,
        monkeypatch,
        extra_args=["--compare", str(tmp_path / "base.json"), "--format", "plain"],
    )
    assert result.exit_code == 0
    assert "Kaggle Public Leaderboard" in result.output
    assert "delta: +0.030000" in result.output


def test_report_kaggle_score_with_threshold_violation(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"val_accuracy": 0.70, "kaggle_public_score": 0.77})
    (tmp_path / "params.yaml").write_text(PARAMS_WITH_THRESHOLD)
    result = _invoke(tmp_path, monkeypatch)
    assert result.exit_code == 1
    assert "Kaggle Public Leaderboard" in result.output
    assert "Threshold Violations" in result.output


# ---------------------------------------------------------------------------
# --compare champion (GH-011): auto-fetch champion baseline from the registry
# ---------------------------------------------------------------------------


def test_report_compare_champion_fetches_baseline(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"accuracy": 0.91})
    (tmp_path / "params.yaml").write_text(MINIMAL_PARAMS)
    monkeypatch.chdir(tmp_path)
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("kitchen.registry.get_champion_metrics", return_value={"accuracy": 0.88}),
    ):
        result = runner.invoke(app, ["report", "--compare", "champion"])
    assert result.exit_code == 0, result.output
    assert "| Metric | Base | PR | Delta |" in result.output
    assert "0.880000" in result.output
    assert "+0.030000" in result.output


def test_report_compare_champion_resolves_model_name_from_experiment(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"accuracy": 0.91})
    (tmp_path / "params.yaml").write_text(MINIMAL_PARAMS)
    monkeypatch.delenv("MLFLOW_MODEL_NAME", raising=False)
    monkeypatch.chdir(tmp_path)
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("kitchen.registry.get_champion_metrics", return_value={}) as mock_fetch,
    ):
        runner.invoke(app, ["report", "--compare", "champion"])
    mock_fetch.assert_called_once_with("test-project-model")


def test_report_compare_champion_model_name_override(tmp_path, monkeypatch):
    _write_metrics(tmp_path / "metrics.json", {"accuracy": 0.91})
    (tmp_path / "params.yaml").write_text(MINIMAL_PARAMS)
    monkeypatch.chdir(tmp_path)
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("kitchen.registry.get_champion_metrics", return_value={}) as mock_fetch,
    ):
        runner.invoke(app, ["report", "--compare", "champion", "--model-name", "custom-model"])
    mock_fetch.assert_called_once_with("custom-model")


def test_report_compare_champion_no_champion_is_graceful(tmp_path, monkeypatch):
    """The realistic SQLite-CI path: no champion → warn, exit 0, plain report."""
    _write_metrics(tmp_path / "metrics.json", {"accuracy": 0.91})
    (tmp_path / "params.yaml").write_text(MINIMAL_PARAMS)
    monkeypatch.chdir(tmp_path)
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("kitchen.registry.get_champion_metrics", return_value=None),
    ):
        result = runner.invoke(app, ["report", "--compare", "champion"])
    assert result.exit_code == 0
    assert "no champion registered" in result.output
    # Falls back to the single-column report — no delta table.
    assert "| Metric | Base | PR | Delta |" not in result.output
    assert "| Metric | Value |" in result.output
