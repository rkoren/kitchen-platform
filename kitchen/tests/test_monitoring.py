"""Tests for DriftReport."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kitchen.monitoring import DriftReport


@pytest.fixture()
def frames():
    ref = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    cur = pd.DataFrame({"a": [1.1, 2.1, 3.1], "b": [4.1, 5.1, 6.1]})
    return ref, cur


def test_run_returns_self(frames):
    ref, cur = frames
    report = DriftReport(ref, cur)
    assert report.run() is report


def test_as_html_requires_run(frames):
    ref, cur = frames
    with pytest.raises(RuntimeError, match="run()"):
        DriftReport(ref, cur).as_html()


def test_as_html_returns_string(frames):
    ref, cur = frames
    html = DriftReport(ref, cur).run().as_html()
    assert isinstance(html, str)
    assert len(html) > 0


def test_column_mapping_applied(frames):
    ref, cur = frames
    report = DriftReport(ref, cur, target="a", numerical=["b"])
    assert report._column_mapping is not None
    assert report._column_mapping.target == "a"


def test_no_column_mapping_when_not_configured(frames):
    ref, cur = frames
    assert DriftReport(ref, cur)._column_mapping is None


def test_upload_calls_s3(frames):
    ref, cur = frames
    report = DriftReport(ref, cur).run()

    mock_s3 = MagicMock()
    with patch("boto3.client", return_value=mock_s3):
        url = report.upload(bucket="my-bucket", key="monitoring/report.html")

    mock_s3.put_object.assert_called_once()
    call_kwargs = mock_s3.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "my-bucket"
    assert call_kwargs["Key"] == "monitoring/report.html"
    assert call_kwargs["ContentType"] == "text/html"
    assert url == "s3://my-bucket/monitoring/report.html"


def test_upload_requires_run(frames):
    ref, cur = frames
    mock_s3 = MagicMock()
    with patch("boto3.client", return_value=mock_s3):
        with pytest.raises(RuntimeError, match="run()"):
            DriftReport(ref, cur).upload(bucket="b", key="k")


def test_save_html_writes_file(frames, tmp_path):
    ref, cur = frames
    out = tmp_path / "report.html"
    DriftReport(ref, cur).run().save_html(str(out))
    assert out.exists()
    assert len(out.read_text()) > 0


def test_save_html_creates_parent_dirs(frames, tmp_path):
    ref, cur = frames
    out = tmp_path / "nested" / "dir" / "report.html"
    DriftReport(ref, cur).run().save_html(str(out))
    assert out.exists()


def test_save_html_requires_run(frames, tmp_path):
    ref, cur = frames
    with pytest.raises(RuntimeError, match="run()"):
        DriftReport(ref, cur).save_html(str(tmp_path / "report.html"))
