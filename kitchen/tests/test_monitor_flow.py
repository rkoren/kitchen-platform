"""Tests for kitchen.flows.monitor_flow."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from kitchen.flows.monitor_flow import monitor_pipeline


@pytest.fixture()
def frames():
    ref = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    cur = pd.DataFrame({"a": [1.1, 2.1, 3.1], "b": [4.1, 5.1, 6.1]})
    return ref, cur


def _write_params(tmp_path, monitor_cfg: dict) -> str:
    path = tmp_path / "params.yaml"
    path.write_text(yaml.dump({"monitor": monitor_cfg}))
    return str(path)


def test_pipeline_saves_local_report(frames, tmp_path):
    ref, cur = frames
    report_path = tmp_path / "monitoring" / "drift.html"
    params_file = _write_params(tmp_path, {
        "reference_file": "reference.parquet",
        "current_file": "current.parquet",
        "local_path": str(report_path),
    })
    with patch("kitchen.flows.monitor_flow.DataStore") as MockStore:
        MockStore.return_value.load_parquet.side_effect = [ref, cur]
        result = monitor_pipeline(params_file=params_file)

    assert result == str(report_path)
    assert report_path.exists()
    assert len(report_path.read_text()) > 0


def test_pipeline_uploads_to_s3(frames, tmp_path):
    ref, cur = frames
    params_file = _write_params(tmp_path, {
        "reference_file": "reference.parquet",
        "current_file": "current.parquet",
        "report_bucket": "my-bucket",
        "report_key": "monitoring/report.html",
    })
    mock_s3 = MagicMock()
    with patch("kitchen.flows.monitor_flow.DataStore") as MockStore, \
         patch("boto3.client", return_value=mock_s3):
        MockStore.return_value.load_parquet.side_effect = [ref, cur]
        result = monitor_pipeline(params_file=params_file)

    assert result == "s3://my-bucket/monitoring/report.html"
    mock_s3.put_object.assert_called_once()


def test_pipeline_fails_without_output_config(frames, tmp_path):
    ref, cur = frames
    params_file = _write_params(tmp_path, {
        "reference_file": "reference.parquet",
        "current_file": "current.parquet",
    })
    with patch("kitchen.flows.monitor_flow.DataStore") as MockStore:
        MockStore.return_value.load_parquet.side_effect = [ref, cur]
        with pytest.raises(Exception, match="report_bucket.*local_path|local_path.*report_bucket"):
            monitor_pipeline(params_file=params_file)


def test_pipeline_local_and_s3_both_run(frames, tmp_path):
    ref, cur = frames
    report_path = tmp_path / "drift.html"
    params_file = _write_params(tmp_path, {
        "reference_file": "reference.parquet",
        "current_file": "current.parquet",
        "local_path": str(report_path),
        "report_bucket": "my-bucket",
        "report_key": "monitoring/report.html",
    })
    mock_s3 = MagicMock()
    with patch("kitchen.flows.monitor_flow.DataStore") as MockStore, \
         patch("boto3.client", return_value=mock_s3):
        MockStore.return_value.load_parquet.side_effect = [ref, cur]
        monitor_pipeline(params_file=params_file)

    assert report_path.exists()
    mock_s3.put_object.assert_called_once()
