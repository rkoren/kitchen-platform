"""Tests for kitchen.flows.monitor_flow."""
# pylint: disable=redefined-outer-name  # standard pytest fixture injection pattern

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from kitchen.flows.monitor_flow import (
    _run_drift_report,
    _save_report,
    _validate_output,
    monitor_pipeline,
)


@pytest.fixture()
def frames():
    ref = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    cur = pd.DataFrame({"a": [1.1, 2.1, 3.1], "b": [4.1, 5.1, 6.1]})
    return ref, cur


@pytest.fixture()
def report(frames):
    ref, cur = frames
    return _run_drift_report.fn(ref, cur)


def _write_params(tmp_path, monitor_cfg: dict) -> str:
    path = tmp_path / "params.yaml"
    path.write_text(yaml.dump({"monitor": monitor_cfg}))
    return str(path)


def test_pipeline_saves_local_report(report, tmp_path):
    report_path = tmp_path / "monitoring" / "drift.html"
    cfg = {"local_path": str(report_path)}
    result = _save_report.fn(report, cfg)
    assert result == str(report_path)
    assert report_path.exists()
    assert len(report_path.read_text()) > 0


def test_pipeline_uploads_to_s3(report):
    mock_s3 = MagicMock()
    cfg = {"report_bucket": "my-bucket", "report_key": "monitoring/report.html"}
    with patch("boto3.client", return_value=mock_s3):
        result = _save_report.fn(report, cfg)
    assert result == "s3://my-bucket/monitoring/report.html"
    mock_s3.put_object.assert_called_once()


def test_pipeline_fails_without_output_config(report):
    with pytest.raises(ValueError, match="report_bucket.*local_path|local_path.*report_bucket"):
        _save_report.fn(report, {})


# --- _validate_output (MON-001 / MON-002: fail fast before data loading) ---


def test_validate_output_passes_with_local_path():
    _validate_output({"local_path": "/tmp/drift.html"})


def test_validate_output_passes_with_bucket():
    _validate_output({"report_bucket": "my-bucket"})


def test_validate_output_passes_with_both():
    _validate_output({"local_path": "/tmp/drift.html", "report_bucket": "my-bucket"})


def test_validate_output_raises_with_neither():
    with pytest.raises(ValueError, match="No output configured"):
        _validate_output({})


def test_pipeline_fails_fast_before_data_loading(tmp_path):
    """_validate_output fires before any I/O — data loading must not be called."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("experiment: test\n")  # no monitor section

    with (
        patch("kitchen.flows.monitor_flow._load_reference") as mock_ref,
        patch("kitchen.flows.monitor_flow._load_current") as mock_cur,
    ):
        with pytest.raises(ValueError, match="No output configured"):
            monitor_pipeline.fn(params_file=str(params_file))

    mock_ref.assert_not_called()
    mock_cur.assert_not_called()


def test_pipeline_local_and_s3_both_run(report, tmp_path):
    report_path = tmp_path / "drift.html"
    cfg = {
        "local_path": str(report_path),
        "report_bucket": "my-bucket",
        "report_key": "monitoring/report.html",
    }
    mock_s3 = MagicMock()
    with patch("boto3.client", return_value=mock_s3):
        _save_report.fn(report, cfg)
    assert report_path.exists()
    mock_s3.put_object.assert_called_once()


def test_pipeline_wiring(frames, tmp_path):
    """monitor_pipeline.fn() reads params and calls stages in order."""
    ref, cur = frames
    report_path = tmp_path / "drift.html"
    params_file = _write_params(
        tmp_path,
        {
            "reference_file": "reference.parquet",
            "current_file": "current.parquet",
            "local_path": str(report_path),
        },
    )
    fake_report = _run_drift_report.fn(ref, cur)
    with (
        patch("kitchen.flows.monitor_flow.DataStore"),
        patch("kitchen.flows.monitor_flow._load_reference", return_value=ref),
        patch("kitchen.flows.monitor_flow._load_current", return_value=cur),
        patch("kitchen.flows.monitor_flow._run_drift_report", return_value=fake_report),
        patch(
            "kitchen.flows.monitor_flow._save_report",
            side_effect=_save_report.fn,
        ),
    ):
        result = monitor_pipeline.fn(params_file=params_file)
    assert result == str(report_path)
    assert report_path.exists()


def test_local_path_override_bypasses_params_config(frames, tmp_path):
    """local_path_override takes precedence over missing monitor config in params.yaml."""
    ref, cur = frames
    report_path = tmp_path / "out" / "drift.html"
    params_file = tmp_path / "params.yaml"
    params_file.write_text("experiment: test\n")  # no monitor section

    fake_report = _run_drift_report.fn(ref, cur)
    with (
        patch("kitchen.flows.monitor_flow.DataStore"),
        patch("kitchen.flows.monitor_flow._load_reference", return_value=ref),
        patch("kitchen.flows.monitor_flow._load_current", return_value=cur),
        patch("kitchen.flows.monitor_flow._run_drift_report", return_value=fake_report),
        patch(
            "kitchen.flows.monitor_flow._save_report",
            side_effect=_save_report.fn,
        ),
    ):
        result = monitor_pipeline.fn(
            params_file=str(params_file), local_path_override=str(report_path)
        )
    assert result == str(report_path)
    assert report_path.exists()


def test_local_path_override_wins_over_params_local_path(frames, tmp_path):
    """local_path_override replaces local_path from params.yaml."""
    ref, cur = frames
    params_local = tmp_path / "params_report.html"
    override_path = tmp_path / "override_report.html"
    params_file = _write_params(
        tmp_path,
        {
            "reference_file": "reference.parquet",
            "current_file": "current.parquet",
            "local_path": str(params_local),
        },
    )
    fake_report = _run_drift_report.fn(ref, cur)
    with (
        patch("kitchen.flows.monitor_flow.DataStore"),
        patch("kitchen.flows.monitor_flow._load_reference", return_value=ref),
        patch("kitchen.flows.monitor_flow._load_current", return_value=cur),
        patch("kitchen.flows.monitor_flow._run_drift_report", return_value=fake_report),
        patch(
            "kitchen.flows.monitor_flow._save_report",
            side_effect=_save_report.fn,
        ),
    ):
        result = monitor_pipeline.fn(
            params_file=params_file, local_path_override=str(override_path)
        )
    assert result == str(override_path)
    assert override_path.exists()
    assert not params_local.exists()
