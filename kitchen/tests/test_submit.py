"""Tests for `kitchen submit`."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import kaggle as _kaggle
import pandas as pd
import pytest
from typer.testing import CliRunner

from kitchen.cli import app
from kitchen.submit import check_feature_parity, fetch_score, log_submission, validate_submission

runner = CliRunner()

# ---------------------------------------------------------------------------
# params.yaml fixtures
# ---------------------------------------------------------------------------

KAGGLE_PARAMS = """\
experiment: test
data:
  source: kaggle
  competition: march-machine-learning-mania-2026
submission:
  id_col: Id
  target_col: Pred
  message: ci-test
"""

KAGGLE_PARAMS_NO_SUB_SECTION = """\
experiment: test
data:
  source: kaggle
  competition: march-machine-learning-mania-2026
"""

KAGGLE_PARAMS_NO_COMPETITION = """\
experiment: test
submission:
  id_col: Id
  target_col: Pred
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _setup(tmp_path: Path, params: str, sub_rows: list[dict], sample_rows: list[dict]):
    """Write params.yaml, submission CSV, and sample_submission.csv to tmp_path."""
    (tmp_path / "params.yaml").write_text(params)

    sub_path = tmp_path / "submissions" / "submission.csv"
    sub_path.parent.mkdir()
    _write_csv(sub_path, sub_rows)

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    _write_csv(raw_dir / "sample_submission.csv", sample_rows)


SAMPLE = [{"Id": 1, "Pred": 0.5}, {"Id": 2, "Pred": 0.6}]
VALID_SUB = [{"Id": 1, "Pred": 0.9}, {"Id": 2, "Pred": 0.8}]


# ---------------------------------------------------------------------------
# Unit tests for check_feature_parity (KG-013)
# ---------------------------------------------------------------------------


def test_parity_ok():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    assert check_feature_parity(["a", "b"], df) == []


def test_parity_missing_one():
    df = pd.DataFrame({"a": [1]})
    errors = check_feature_parity(["a", "b"], df)
    assert len(errors) == 1
    assert "missing feature" in errors[0]
    assert "'b'" in errors[0]


def test_parity_missing_multiple():
    df = pd.DataFrame({"a": [1]})
    errors = check_feature_parity(["a", "b", "c"], df)
    assert len(errors) == 2
    missing = {e for e in errors}
    assert any("'b'" in e for e in missing)
    assert any("'c'" in e for e in missing)


def test_parity_empty_expected():
    df = pd.DataFrame({"a": [1]})
    assert check_feature_parity([], df) == []


def test_parity_extra_columns_ignored():
    # Columns in df that weren't in training are not reported as errors
    df = pd.DataFrame({"a": [1], "b": [2], "extra": [3]})
    assert check_feature_parity(["a", "b"], df) == []


def test_parity_all_missing():
    df = pd.DataFrame({"x": [1]})
    errors = check_feature_parity(["a", "b", "c"], df)
    assert len(errors) == 3


# ---------------------------------------------------------------------------
# Unit tests for validate_submission
# ---------------------------------------------------------------------------


def _df(rows):
    return pd.DataFrame(rows)


def test_validate_ok():
    errors = validate_submission(_df(VALID_SUB), _df(SAMPLE), "Id", "Pred")
    assert errors == []


def test_validate_missing_id_col():
    sub = _df([{"X": 1, "Pred": 0.5}, {"X": 2, "Pred": 0.6}])
    errors = validate_submission(sub, _df(SAMPLE), "Id", "Pred")
    assert any("missing column" in e and "Id" in e for e in errors)


def test_validate_missing_target_col():
    sub = _df([{"Id": 1, "Y": 0.5}, {"Id": 2, "Y": 0.6}])
    errors = validate_submission(sub, _df(SAMPLE), "Id", "Pred")
    assert any("missing column" in e and "Pred" in e for e in errors)


def test_validate_row_count_mismatch():
    sub = _df([{"Id": 1, "Pred": 0.5}])
    errors = validate_submission(sub, _df(SAMPLE), "Id", "Pred")
    assert any("row count" in e for e in errors)


def test_validate_null_predictions():
    sub = _df([{"Id": 1, "Pred": None}, {"Id": 2, "Pred": 0.8}])
    errors = validate_submission(sub, _df(SAMPLE), "Id", "Pred")
    assert any("null" in e for e in errors)


def test_validate_duplicate_ids():
    sub = _df([{"Id": 1, "Pred": 0.5}, {"Id": 1, "Pred": 0.6}])
    errors = validate_submission(sub, _df(SAMPLE), "Id", "Pred")
    assert any("duplicate" in e for e in errors)


def test_validate_collects_all_errors():
    # row count off AND nulls — both should be reported
    sub = _df([{"Id": 1, "Pred": None}])
    errors = validate_submission(sub, _df(SAMPLE), "Id", "Pred")
    assert len(errors) >= 2


# ---------------------------------------------------------------------------
# log_submission unit tests
# ---------------------------------------------------------------------------


def test_log_submission_validates_and_raises(tmp_path):
    bad = _df([{"ID": 1, "Pred": None}, {"ID": 1, "Pred": 0.5}])
    sample = _df(SAMPLE)
    path = tmp_path / "sub.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="validation failed"):
        log_submission(bad, sample, path)


def test_log_submission_logs_artifact_to_active_run(tmp_path):
    sub = _df(VALID_SUB)
    sample = _df(SAMPLE)
    path = tmp_path / "sub.csv"
    sub.to_csv(path, index=False)

    import mlflow
    with mlflow.start_run():
        result = log_submission(sub, sample, path, id_col="Id")
        run = mlflow.active_run()
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id, "submission")

    assert result == {}
    assert any(a.path == "submission/sub.csv" for a in artifacts)


def test_log_submission_no_active_run_skips_artifact(tmp_path):
    sub = _df(VALID_SUB)
    sample = _df(SAMPLE)
    path = tmp_path / "sub.csv"
    sub.to_csv(path, index=False)
    # No mlflow.start_run() — should not raise, just skips artifact logging
    result = log_submission(sub, sample, path, id_col="Id")
    assert result == {}


def test_log_submission_uploads_when_competition_set(tmp_path):
    sub = _df(VALID_SUB)
    sample = _df(SAMPLE)
    path = tmp_path / "sub.csv"
    sub.to_csv(path, index=False)

    with patch("kitchen.submit.upload") as mock_upload:
        result = log_submission(sub, sample, path, id_col="Id", competition="test-comp", message="v1")

    mock_upload.assert_called_once_with(path, "v1", "test-comp")
    assert result == {}


def test_log_submission_fetches_and_logs_lb_score(tmp_path):
    sub = _df(VALID_SUB)
    sample = _df(SAMPLE)
    path = tmp_path / "sub.csv"
    sub.to_csv(path, index=False)

    import mlflow
    with mlflow.start_run():
        with (
            patch("kitchen.submit.upload"),
            patch("kitchen.submit.fetch_score", return_value=0.1765),
        ):
            result = log_submission(
                sub, sample, path,
                id_col="Id",
                competition="test-comp",
                fetch_lb_score=True,
            )
        run_id = mlflow.active_run().info.run_id

    assert result == {"lb_score": pytest.approx(0.1765)}
    client = mlflow.tracking.MlflowClient()
    metrics = client.get_run(run_id).data.metrics
    assert metrics["lb_score"] == pytest.approx(0.1765)


def test_log_submission_no_fetch_when_flag_false(tmp_path):
    sub = _df(VALID_SUB)
    sample = _df(SAMPLE)
    path = tmp_path / "sub.csv"
    sub.to_csv(path, index=False)

    with (
        patch("kitchen.submit.upload"),
        patch("kitchen.submit.fetch_score") as mock_fetch,
    ):
        result = log_submission(sub, sample, path, id_col="Id", competition="test-comp", fetch_lb_score=False)

    mock_fetch.assert_not_called()
    assert result == {}


def test_log_submission_score_none_excluded_from_result(tmp_path):
    sub = _df(VALID_SUB)
    sample = _df(SAMPLE)
    path = tmp_path / "sub.csv"
    sub.to_csv(path, index=False)

    with (
        patch("kitchen.submit.upload"),
        patch("kitchen.submit.fetch_score", return_value=None),
    ):
        result = log_submission(sub, sample, path, id_col="Id", competition="test-comp", fetch_lb_score=True)

    assert result == {}


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def test_submit_missing_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["submit", "--params", "missing.yaml"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_submit_no_competition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(KAGGLE_PARAMS_NO_COMPETITION)
    with patch("pathlib.Path.home", return_value=tmp_path):
        result = runner.invoke(app, ["submit"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert result.exit_code != 0
    assert "competition" in result.output


def test_submit_missing_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(KAGGLE_PARAMS)
    with patch("pathlib.Path.home", return_value=tmp_path):
        result = runner.invoke(app, ["submit"], env={})
    assert result.exit_code != 0
    assert "Kaggle credentials" in result.output


def test_submit_missing_submission_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(KAGGLE_PARAMS)
    with patch("pathlib.Path.home", return_value=tmp_path):
        result = runner.invoke(app, ["submit"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert result.exit_code != 0
    assert "not found" in result.output


def test_submit_missing_sample_submission(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(KAGGLE_PARAMS)
    sub_path = tmp_path / "submissions" / "submission.csv"
    sub_path.parent.mkdir()
    _write_csv(sub_path, VALID_SUB)
    with patch("pathlib.Path.home", return_value=tmp_path):
        result = runner.invoke(app, ["submit"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert result.exit_code != 0
    assert "sample submission" in result.output


def test_submit_validation_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad_sub = [{"Id": 1, "Pred": None}, {"Id": 1, "Pred": 0.5}]
    _setup(tmp_path, KAGGLE_PARAMS, bad_sub, SAMPLE)
    with patch("pathlib.Path.home", return_value=tmp_path):
        result = runner.invoke(app, ["submit"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert result.exit_code != 0
    assert "validation failed" in result.output


def test_submit_happy_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, KAGGLE_PARAMS, VALID_SUB, SAMPLE)
    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("kitchen.submit.upload") as mock_upload,
    ):
        result = runner.invoke(app, ["submit"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert result.exit_code == 0
    assert "Submitted" in result.output
    mock_upload.assert_called_once()
    _, call_msg, call_comp = mock_upload.call_args.args
    assert call_comp == "march-machine-learning-mania-2026"
    assert call_msg == "ci-test"


def test_submit_competition_from_data_section(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, KAGGLE_PARAMS_NO_SUB_SECTION, VALID_SUB, SAMPLE)
    # KAGGLE_PARAMS_NO_SUB_SECTION has no submission section — id_col/target_col
    # default to "Id"/"target", but our sample + sub use "Pred" not "target"
    # so swap to defaults
    raw_dir = tmp_path / "data" / "raw"
    sample_default = [{"Id": 1, "target": 0.5}, {"Id": 2, "target": 0.6}]
    _write_csv(raw_dir / "sample_submission.csv", sample_default)
    sub_path = tmp_path / "submissions" / "submission.csv"
    _write_csv(sub_path, [{"Id": 1, "target": 0.9}, {"Id": 2, "target": 0.8}])
    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("kitchen.submit.upload") as mock_upload,
    ):
        result = runner.invoke(app, ["submit"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert result.exit_code == 0
    _, _, call_comp = mock_upload.call_args.args
    assert call_comp == "march-machine-learning-mania-2026"


def test_submit_custom_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, KAGGLE_PARAMS, VALID_SUB, SAMPLE)
    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("kitchen.submit.upload") as mock_upload,
    ):
        result = runner.invoke(
            app,
            ["submit", "--message", "my custom msg"],
            env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"},
        )
    assert result.exit_code == 0
    _, call_msg, _ = mock_upload.call_args.args
    assert call_msg == "my custom msg"


def test_submit_upload_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, KAGGLE_PARAMS, VALID_SUB, SAMPLE)
    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("kitchen.submit.upload", side_effect=RuntimeError("rate limit exceeded")),
    ):
        result = runner.invoke(app, ["submit"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert result.exit_code != 0
    assert "rate limit exceeded" in result.output


def test_submit_credentials_via_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, KAGGLE_PARAMS, VALID_SUB, SAMPLE)
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").write_text('{"username":"u","key":"k"}')
    with patch("pathlib.Path.home", return_value=tmp_path), patch("kitchen.submit.upload"):
        result = runner.invoke(app, ["submit"], env={})
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# fetch_score unit tests
# ---------------------------------------------------------------------------


def _make_submission(status: str, public_score=None):
    return SimpleNamespace(status=status, publicScore=public_score)


def test_fetch_score_complete():
    sub = _make_submission("complete", "0.85432")
    with (
        patch.object(_kaggle.api, "authenticate"),
        patch.object(_kaggle.api, "competition_submissions", return_value=[sub]),
    ):
        score = fetch_score("test-comp", timeout=5, interval=0)
    assert score == pytest.approx(0.85432)


def test_fetch_score_complete_float_value():
    sub = _make_submission("complete", 0.72)
    with (
        patch.object(_kaggle.api, "authenticate"),
        patch.object(_kaggle.api, "competition_submissions", return_value=[sub]),
    ):
        score = fetch_score("test-comp", timeout=5, interval=0)
    assert score == pytest.approx(0.72)


def test_fetch_score_error_status():
    sub = _make_submission("error")
    with (
        patch.object(_kaggle.api, "authenticate"),
        patch.object(_kaggle.api, "competition_submissions", return_value=[sub]),
    ):
        score = fetch_score("test-comp", timeout=5, interval=0)
    assert score is None


def test_fetch_score_timeout_on_pending():
    sub = _make_submission("pending")
    with (
        patch.object(_kaggle.api, "authenticate"),
        patch.object(_kaggle.api, "competition_submissions", return_value=[sub]),
        patch("time.sleep"),
    ):
        score = fetch_score("test-comp", timeout=0, interval=0)
    assert score is None


def test_fetch_score_pending_then_complete():
    pending = _make_submission("pending")
    complete = _make_submission("complete", "0.91")
    mock_submissions = MagicMock(side_effect=[[pending], [complete]])
    with (
        patch.object(_kaggle.api, "authenticate"),
        patch.object(_kaggle.api, "competition_submissions", mock_submissions),
        patch("time.sleep"),
    ):
        score = fetch_score("test-comp", timeout=60, interval=0)
    assert score == pytest.approx(0.91)


def test_fetch_score_empty_submissions():
    with (
        patch.object(_kaggle.api, "authenticate"),
        patch.object(_kaggle.api, "competition_submissions", return_value=[]),
        patch("time.sleep"),
    ):
        score = fetch_score("test-comp", timeout=0, interval=0)
    assert score is None


def test_fetch_score_api_exception():
    with (
        patch.object(_kaggle.api, "authenticate"),
        patch.object(_kaggle.api, "competition_submissions", side_effect=RuntimeError("api error")),
    ):
        score = fetch_score("test-comp", timeout=5, interval=0)
    assert score is None


def test_fetch_score_unparseable_score():
    sub = _make_submission("complete", "not-a-number")
    with (
        patch.object(_kaggle.api, "authenticate"),
        patch.object(_kaggle.api, "competition_submissions", return_value=[sub]),
    ):
        score = fetch_score("test-comp", timeout=5, interval=0)
    assert score is None


def test_fetch_score_none_public_score():
    sub = _make_submission("complete", None)
    with (
        patch.object(_kaggle.api, "authenticate"),
        patch.object(_kaggle.api, "competition_submissions", return_value=[sub]),
    ):
        score = fetch_score("test-comp", timeout=5, interval=0)
    assert score is None


# ---------------------------------------------------------------------------
# CLI submit + score integration tests
# ---------------------------------------------------------------------------


def test_submit_wait_writes_score_to_metrics_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, KAGGLE_PARAMS, VALID_SUB, SAMPLE)
    (tmp_path / "metrics.json").write_text('{"val_accuracy": 0.9}\n')

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("kitchen.submit.upload"),
        patch("kitchen.submit.fetch_score", return_value=0.78) as mock_fetch,
    ):
        result = runner.invoke(
            app, ["submit", "--wait"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}
        )

    assert result.exit_code == 0, result.output
    assert "Leaderboard score: 0.780000" in result.output
    assert "Score written to metrics.json" in result.output
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["kaggle_public_score"] == pytest.approx(0.78)
    assert metrics["val_accuracy"] == pytest.approx(0.9)
    mock_fetch.assert_called_once_with("march-machine-learning-mania-2026")


def test_submit_default_skips_fetch(tmp_path, monkeypatch):
    """By default, submit does not poll for a leaderboard score."""
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, KAGGLE_PARAMS, VALID_SUB, SAMPLE)

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("kitchen.submit.upload"),
        patch("kitchen.submit.fetch_score") as mock_fetch,
    ):
        result = runner.invoke(
            app,
            ["submit"],
            env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"},
        )

    assert result.exit_code == 0
    mock_fetch.assert_not_called()
    assert "Leaderboard score" not in result.output


def test_submit_wait_score_not_available_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, KAGGLE_PARAMS, VALID_SUB, SAMPLE)

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("kitchen.submit.upload"),
        patch("kitchen.submit.fetch_score", return_value=None),
    ):
        result = runner.invoke(
            app, ["submit", "--wait"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}
        )

    assert result.exit_code == 0
    assert "not yet available" in result.output
    assert not (tmp_path / "metrics.json").exists()


def test_submit_wait_creates_metrics_json_if_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, KAGGLE_PARAMS, VALID_SUB, SAMPLE)

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("kitchen.submit.upload"),
        patch("kitchen.submit.fetch_score", return_value=0.82),
    ):
        result = runner.invoke(
            app, ["submit", "--wait"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}
        )

    assert result.exit_code == 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["kaggle_public_score"] == pytest.approx(0.82)
