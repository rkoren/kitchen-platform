"""Tests for `kitchen ui` command (LML-001)."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from kitchen.cli import app

runner = CliRunner()


def _invoke_ui(env: dict, *extra_args: str):
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("subprocess.run") as mock_subprocess,
        patch("threading.Thread") as mock_thread,
        patch("webbrowser.open") as mock_browser,
    ):
        result = runner.invoke(app, ["ui", *extra_args], env=env)
    return result, mock_subprocess, mock_thread, mock_browser


# ---------------------------------------------------------------------------
# Remote tracking URI
# ---------------------------------------------------------------------------


def test_ui_remote_http_opens_browser():
    result, mock_subprocess, _, mock_browser = _invoke_ui(
        {"MLFLOW_TRACKING_URI": "http://mlflow.example.com:5000"}
    )
    assert result.exit_code == 0
    mock_browser.assert_called_once_with("http://mlflow.example.com:5000")
    mock_subprocess.assert_not_called()


def test_ui_remote_https_opens_browser():
    result, mock_subprocess, _, mock_browser = _invoke_ui(
        {"MLFLOW_TRACKING_URI": "https://mlflow.example.com"}
    )
    assert result.exit_code == 0
    mock_browser.assert_called_once_with("https://mlflow.example.com")
    mock_subprocess.assert_not_called()


# ---------------------------------------------------------------------------
# Local SQLite tracking URI
# ---------------------------------------------------------------------------


def test_ui_local_sqlite_starts_mlflow_server():
    result, mock_subprocess, mock_thread, _ = _invoke_ui(
        {"MLFLOW_TRACKING_URI": "sqlite:///mlruns.db"}
    )
    assert result.exit_code == 0
    mock_subprocess.assert_called_once()
    cmd = mock_subprocess.call_args[0][0]
    assert "mlflow" in cmd
    assert "ui" in cmd
    assert "sqlite:///mlruns.db" in cmd


def test_ui_local_sqlite_uses_correct_port():
    result, mock_subprocess, _, _ = _invoke_ui(
        {"MLFLOW_TRACKING_URI": "sqlite:///mlruns.db"}
    )
    cmd = mock_subprocess.call_args[0][0]
    port_idx = cmd.index("--port")
    assert cmd[port_idx + 1] == "5000"


def test_ui_custom_port():
    result, mock_subprocess, _, _ = _invoke_ui(
        {"MLFLOW_TRACKING_URI": "sqlite:///mlruns.db"}, "--port", "6789"
    )
    assert result.exit_code == 0
    cmd = mock_subprocess.call_args[0][0]
    assert "6789" in cmd


def test_ui_local_spawns_browser_thread():
    """Browser open is deferred to a background thread so the server starts first."""
    result, _, mock_thread, _ = _invoke_ui(
        {"MLFLOW_TRACKING_URI": "sqlite:///mlruns.db"}
    )
    assert result.exit_code == 0
    mock_thread.assert_called_once()


def test_ui_default_uri_when_env_unset(monkeypatch):
    """Falls back to sqlite:///mlruns.db when MLFLOW_TRACKING_URI is not set."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    result, mock_subprocess, _, _ = _invoke_ui({})
    assert result.exit_code == 0
    cmd = mock_subprocess.call_args[0][0]
    assert "sqlite:///mlruns.db" in cmd


def test_ui_prints_url_and_tracking_uri():
    result, _, _, _ = _invoke_ui({"MLFLOW_TRACKING_URI": "sqlite:///mlruns.db"})
    assert "localhost:5000" in result.output
    assert "sqlite:///mlruns.db" in result.output
