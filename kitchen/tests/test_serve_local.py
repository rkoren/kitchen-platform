"""Tests for `kitchen serve local` (S-006)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from kitchen.cli import app

runner = CliRunner()


def _invoke(args: list[str] | None = None):
    return runner.invoke(app, ["serve", "local"] + (args or []), catch_exceptions=False)


# ---------------------------------------------------------------------------
# Subprocess command construction
# ---------------------------------------------------------------------------


def test_default_invocation_calls_uvicorn(tmp_path, monkeypatch):
    """With no options, subprocess.run receives a uvicorn command on port 8080."""
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        result = _invoke()
    assert result.exit_code == 0
    cmd = mock_run.call_args[0][0]
    assert "-m" in cmd
    assert "uvicorn" in cmd
    assert "kitchen.serve.app:app" in cmd
    assert "8080" in cmd


def test_custom_port(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        result = _invoke(["--port", "9000"])
    assert result.exit_code == 0
    cmd = mock_run.call_args[0][0]
    port_idx = cmd.index("--port")
    assert cmd[port_idx + 1] == "9000"


def test_reload_flag_included_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        _invoke()
    cmd = mock_run.call_args[0][0]
    assert "--reload" in cmd


def test_no_reload_omits_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        _invoke(["--no-reload"])
    cmd = mock_run.call_args[0][0]
    assert "--reload" not in cmd


def test_host_is_0_0_0_0(tmp_path, monkeypatch):
    """uvicorn must bind to 0.0.0.0 so port-forwards work in containers."""
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        _invoke()
    cmd = mock_run.call_args[0][0]
    host_idx = cmd.index("--host")
    assert cmd[host_idx + 1] == "0.0.0.0"


# ---------------------------------------------------------------------------
# Browser / thread behaviour
# ---------------------------------------------------------------------------


def test_browser_thread_spawned_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run"),
        patch("threading.Thread") as mock_thread,
        patch("webbrowser.open"),
    ):
        result = _invoke()
    assert result.exit_code == 0
    mock_thread.assert_called_once()


def test_no_open_skips_browser_thread(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run"),
        patch("threading.Thread") as mock_thread,
        patch("webbrowser.open"),
    ):
        result = _invoke(["--no-open"])
    assert result.exit_code == 0
    mock_thread.assert_not_called()


# ---------------------------------------------------------------------------
# Predictor directory resolution
# ---------------------------------------------------------------------------


def test_predictor_dir_explicit(tmp_path, monkeypatch):
    """--predictor-dir sets PYTHONPATH to the supplied path."""
    monkeypatch.chdir(tmp_path)
    pred_dir = tmp_path / "custom_preds"
    pred_dir.mkdir()
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        result = _invoke(["--predictor-dir", str(pred_dir)])
    assert result.exit_code == 0
    env = mock_run.call_args[1]["env"]
    assert str(pred_dir.resolve()) in env["PYTHONPATH"]


def test_predictor_dir_resolved_from_src_serve(tmp_path, monkeypatch):
    """src/serve/predictor.py present → PYTHONPATH uses src/serve/."""
    monkeypatch.chdir(tmp_path)
    serve_dir = tmp_path / "src" / "serve"
    serve_dir.mkdir(parents=True)
    (serve_dir / "predictor.py").write_text("def predict(x): return x\n")
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        _invoke()
    env = mock_run.call_args[1]["env"]
    assert str(serve_dir.resolve()) in env["PYTHONPATH"]


def test_predictor_dir_falls_back_to_cwd(tmp_path, monkeypatch):
    """./predictor.py present → PYTHONPATH uses cwd."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "predictor.py").write_text("def predict(x): return x\n")
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        _invoke()
    env = mock_run.call_args[1]["env"]
    assert str(tmp_path.resolve()) in env["PYTHONPATH"]


def test_default_predictor_dir_when_none_found(tmp_path, monkeypatch):
    """No predictor.py anywhere → defaults to src/serve/ (app returns 501 gracefully)."""
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        _invoke()
    env = mock_run.call_args[1]["env"]
    expected = str((tmp_path / "src" / "serve").resolve())
    assert expected in env["PYTHONPATH"]


def test_pythonpath_prepended_to_existing(tmp_path, monkeypatch):
    """Existing PYTHONPATH entries are preserved; pred_dir is prepended."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        _invoke()
    env = mock_run.call_args[1]["env"]
    assert "/existing/path" in env["PYTHONPATH"]
    # pred_dir comes before the existing path
    assert env["PYTHONPATH"].index("/existing/path") > 0


def test_pythonpath_set_when_not_in_env(tmp_path, monkeypatch):
    """When PYTHONPATH is not set in the environment it is still populated."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        _invoke()
    env = mock_run.call_args[1]["env"]
    assert "PYTHONPATH" in env
    assert env["PYTHONPATH"]  # non-empty


def test_kitchen_predictor_dir_set_in_subprocess_env(tmp_path, monkeypatch):
    """KITCHEN_PREDICTOR_DIR must be set alongside PYTHONPATH for deterministic loader resolution."""
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        _invoke()
    env = mock_run.call_args[1]["env"]
    assert "KITCHEN_PREDICTOR_DIR" in env
    assert env["KITCHEN_PREDICTOR_DIR"]  # non-empty


def test_kitchen_predictor_dir_matches_pred_dir(tmp_path, monkeypatch):
    """KITCHEN_PREDICTOR_DIR must point to the same directory as PYTHONPATH entry."""
    monkeypatch.chdir(tmp_path)
    serve_dir = tmp_path / "src" / "serve"
    serve_dir.mkdir(parents=True)
    (serve_dir / "predictor.py").write_text("def predict(x): return x")
    with (
        patch("subprocess.run") as mock_run,
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        _invoke()
    env = mock_run.call_args[1]["env"]
    assert str(serve_dir.resolve()) in env["KITCHEN_PREDICTOR_DIR"]


# ---------------------------------------------------------------------------
# Output / UX
# ---------------------------------------------------------------------------


def test_output_shows_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run"),
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        result = _invoke()
    assert "localhost:8080" in result.output


def test_output_shows_predictor_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run"),
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        result = _invoke()
    assert "Predictor" in result.output


def test_output_mentions_ctrl_c(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run"),
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        result = _invoke()
    assert "Ctrl+C" in result.output


def test_keyboard_interrupt_prints_stopped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch("subprocess.run", side_effect=KeyboardInterrupt),
        patch("threading.Thread"),
        patch("webbrowser.open"),
    ):
        result = runner.invoke(app, ["serve", "local"], catch_exceptions=True)
    assert "Stopped" in result.output
