"""Tests for `kitchen submit`."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from typer.testing import CliRunner

from kitchen.cli import app
from kitchen.submit import validate_submission

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
    with patch("pathlib.Path.home", return_value=tmp_path), \
         patch("kitchen.submit.upload") as mock_upload:
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
    with patch("pathlib.Path.home", return_value=tmp_path), \
         patch("kitchen.submit.upload") as mock_upload:
        result = runner.invoke(app, ["submit"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert result.exit_code == 0
    _, _, call_comp = mock_upload.call_args.args
    assert call_comp == "march-machine-learning-mania-2026"


def test_submit_custom_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, KAGGLE_PARAMS, VALID_SUB, SAMPLE)
    with patch("pathlib.Path.home", return_value=tmp_path), \
         patch("kitchen.submit.upload") as mock_upload:
        result = runner.invoke(
            app, ["submit", "--message", "my custom msg"],
            env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"},
        )
    assert result.exit_code == 0
    _, call_msg, _ = mock_upload.call_args.args
    assert call_msg == "my custom msg"


def test_submit_upload_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, KAGGLE_PARAMS, VALID_SUB, SAMPLE)
    with patch("pathlib.Path.home", return_value=tmp_path), \
         patch("kitchen.submit.upload", side_effect=RuntimeError("rate limit exceeded")):
        result = runner.invoke(app, ["submit"], env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert result.exit_code != 0
    assert "rate limit exceeded" in result.output


def test_submit_credentials_via_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, KAGGLE_PARAMS, VALID_SUB, SAMPLE)
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").write_text('{"username":"u","key":"k"}')
    with patch("pathlib.Path.home", return_value=tmp_path), \
         patch("kitchen.submit.upload"):
        result = runner.invoke(app, ["submit"], env={})
    assert result.exit_code == 0
