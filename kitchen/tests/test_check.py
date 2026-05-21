"""Tests for `kitchen check`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from kitchen.cli import app

runner = CliRunner()

MINIMAL_PARAMS = "experiment: test-project\n"


def _invoke(tmp_path, monkeypatch, params_content=MINIMAL_PARAMS, extra_args=None, env=None):
    monkeypatch.chdir(tmp_path)
    if params_content is not None:
        (tmp_path / "params.yaml").write_text(params_content)
    args = ["check"] + (extra_args or [])
    return runner.invoke(app, args, env=env or {})


# ---------------------------------------------------------------------------
# Pantry section
# ---------------------------------------------------------------------------


def test_check_python_ok(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path,
            monkeypatch,
            env={"MLFLOW_TRACKING_URI": "sqlite:///x.db", "KAGGLE_USERNAME": "user"},
        )
    assert "✓ python" in result.output


def test_check_terraform_missing(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path,
            monkeypatch,
            env={"MLFLOW_TRACKING_URI": "sqlite:///x.db", "KAGGLE_USERNAME": "u"},
        )
    assert "✗ terraform" in result.output
    assert result.exit_code != 0


def test_check_tool_present_shows_version(tmp_path, monkeypatch):
    def fake_which(name):
        return f"/usr/bin/{name}" if name == "terraform" else None

    with (
        patch("shutil.which", side_effect=fake_which),
        patch("subprocess.check_output", return_value="Terraform v1.7.4\n"),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "✓ terraform" in result.output
    assert "Terraform v1.7.4" in result.output


# ---------------------------------------------------------------------------
# Burners section
# ---------------------------------------------------------------------------


def test_check_mlflow_uri_present(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path,
            monkeypatch,
            env={"MLFLOW_TRACKING_URI": "sqlite:///mlflow.db", "KAGGLE_USERNAME": "u"},
        )
    assert "✓ MLFLOW_TRACKING_URI" in result.output
    assert "sqlite:///mlflow.db" in result.output


def test_check_mlflow_uri_missing(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(tmp_path, monkeypatch, env={"KAGGLE_USERNAME": "u"})
    assert "✗ MLFLOW_TRACKING_URI" in result.output
    assert result.exit_code != 0


def test_check_aws_creds_present(tmp_path, monkeypatch):
    mock_creds = MagicMock()
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = mock_creds
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "✓ AWS credentials" in result.output


def test_check_aws_creds_missing(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "✗ AWS credentials" in result.output
    assert result.exit_code != 0


def test_check_kaggle_env_var(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "user"}
        )
    assert "✓ Kaggle credentials" in result.output


def test_check_kaggle_json_file(tmp_path, monkeypatch):
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").write_text('{"username":"u","key":"k"}')

    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x"})
    assert "✓ Kaggle credentials" in result.output


def test_check_kaggle_missing(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x"})
    assert "✗ Kaggle credentials" in result.output
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Recipe section
# ---------------------------------------------------------------------------


def test_check_valid_params(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "✓ params.yaml" in result.output
    assert "test-project" in result.output


def test_check_invalid_params(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content="not_valid: true\n",
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "✗ params.yaml" in result.output
    assert result.exit_code != 0


def test_check_no_params_file(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=None,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "not found" in result.output


# ---------------------------------------------------------------------------
# Prep section
# ---------------------------------------------------------------------------


def test_check_shows_prep_when_src_exists(tmp_path, monkeypatch):
    src = tmp_path / "src" / "features"
    src.mkdir(parents=True)
    (src / "run.py").write_text("")

    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "✓ src/features/run.py" in result.output


def test_check_fails_missing_src_file(tmp_path, monkeypatch):
    src = tmp_path / "src" / "features"
    src.mkdir(parents=True)
    (src / "run.py").write_text("")
    # src/train/run.py intentionally absent

    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "✗ src/train/run.py" in result.output
    assert result.exit_code != 0


def test_check_no_prep_section_outside_project(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "src/features/run.py" not in result.output


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_check_all_pass_summary(tmp_path, monkeypatch):
    with (
        patch("shutil.which", side_effect=lambda n: f"/usr/bin/{n}"),
        patch("subprocess.check_output", return_value="v1.0\n"),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "All checks passed" in result.output
    assert result.exit_code == 0


def test_check_issues_summary(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(tmp_path, monkeypatch, env={})
    assert "found" in result.output
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Monitor config section (MON-001 / MON-002)
# ---------------------------------------------------------------------------


def test_check_shows_monitor_config_when_present(tmp_path, monkeypatch):
    params = (
        "experiment: test-project\n"
        "monitor:\n"
        "  local_path: monitoring/drift.html\n"
    )
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=params,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "monitor config" in result.output
    assert "monitoring/drift.html" in result.output


def test_check_no_monitor_section_shows_no_monitor_line(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path,
            monkeypatch,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "monitor config" not in result.output
