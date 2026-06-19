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


def test_check_terraform_missing_warns_not_fails(tmp_path, monkeypatch):
    """Missing terraform is a soft warning (only gates `recipes generate`), not a hard fail.

    Everything else is green (docker/dvc present, AWS + Kaggle resolved) so the only finding is
    terraform — which must NOT drive a non-zero exit.
    """
    def fake_which(name):
        return None if name == "terraform" else f"/usr/bin/{name}"

    with (
        patch("shutil.which", side_effect=fake_which),
        patch("subprocess.check_output", return_value="v1.0\n"),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "~ terraform" in result.output
    assert "✗ terraform" not in result.output
    assert result.exit_code == 0


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
# Postgres driver (Gap 1: postgresql backend needs psycopg2)
# ---------------------------------------------------------------------------


def _fake_find_spec(*, psycopg2_present: bool):
    """find_spec stub: control psycopg2 presence, delegate everything else to the real one."""
    import importlib.util as _ilu

    real = _ilu.find_spec

    def f(name, *args, **kwargs):
        if name == "psycopg2":
            return MagicMock() if psycopg2_present else None
        return real(name, *args, **kwargs)

    return f


def test_check_postgres_uri_missing_driver_fails(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value="/usr/bin/x"),
        patch("subprocess.check_output", return_value="v1\n"),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("importlib.util.find_spec", side_effect=_fake_find_spec(psycopg2_present=False)),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path,
            monkeypatch,
            env={"MLFLOW_TRACKING_URI": "postgresql://u:p@host/mlflow", "KAGGLE_USERNAME": "u"},
        )
    assert "✗ postgres driver" in result.output
    assert "kitchen[postgres]" in result.output
    assert result.exit_code != 0


def test_check_postgres_uri_with_driver_ok(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value="/usr/bin/x"),
        patch("subprocess.check_output", return_value="v1\n"),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("importlib.util.find_spec", side_effect=_fake_find_spec(psycopg2_present=True)),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path,
            monkeypatch,
            env={"MLFLOW_TRACKING_URI": "postgresql://u:p@host/mlflow", "KAGGLE_USERNAME": "u"},
        )
    assert "✓ postgres driver" in result.output
    assert "✗ postgres driver" not in result.output


def test_check_sqlite_uri_skips_driver_check(tmp_path, monkeypatch):
    with (
        patch("shutil.which", return_value="/usr/bin/x"),
        patch("subprocess.check_output", return_value="v1\n"),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "sqlite:///x.db", "KAGGLE_USERNAME": "u"}
        )
    assert "postgres driver" not in result.output


# ---------------------------------------------------------------------------
# DVC remote placeholder section (DVC-012)
# ---------------------------------------------------------------------------


def test_check_dvc_remote_placeholder_warns(tmp_path, monkeypatch):
    """Scaffolded .dvc/config with YOUR-BUCKET → warn with actionable fix hint."""
    dvc_dir = tmp_path / ".dvc"
    dvc_dir.mkdir()
    (dvc_dir / "config").write_text(
        "[core]\n    remote = s3remote\n[remote \"s3remote\"]\n    url = s3://YOUR-BUCKET/dvc\n"
    )
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "~ DVC remote" in result.output
    assert "YOUR-BUCKET" not in result.output  # hint should not reproduce the placeholder
    assert "dvc remote modify" in result.output


def test_check_dvc_remote_configured_no_warning(tmp_path, monkeypatch):
    """.dvc/config with a real bucket URL should produce no DVC remote warning."""
    dvc_dir = tmp_path / ".dvc"
    dvc_dir.mkdir()
    (dvc_dir / "config").write_text(
        "[core]\n    remote = s3remote\n[remote \"s3remote\"]\n    url = s3://my-real-bucket/dvc\n"
    )
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "DVC remote" not in result.output


def test_check_no_dvc_config_no_remote_check(tmp_path, monkeypatch):
    """No .dvc/config (DVC not initialised) → no DVC remote line at all."""
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "DVC remote" not in result.output


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
    """Unset MLFLOW_TRACKING_URI should warn (not fail) — the platform defaults to SQLite."""
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(tmp_path, monkeypatch, env={"KAGGLE_USERNAME": "u"})
    assert "~ MLFLOW_TRACKING_URI" in result.output
    assert "sqlite:///mlruns.db" in result.output
    # Should NOT count as a hard failure
    assert "✗ MLFLOW_TRACKING_URI" not in result.output


S3_PARAMS = (
    "experiment: test-project\n"
    "data:\n"
    "  source: s3\n"
    "  bucket: my-bucket\n"
)


def test_check_aws_creds_present(tmp_path, monkeypatch):
    """AWS credentials show ✓ when present and project uses S3."""
    mock_creds = MagicMock()
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = mock_creds
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=S3_PARAMS,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "✓ AWS credentials" in result.output


def test_check_aws_creds_missing(tmp_path, monkeypatch):
    """AWS credentials show ✗ (hard fail) when missing and project uses S3."""
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=S3_PARAMS,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "✗ AWS credentials" in result.output
    assert result.exit_code != 0


def test_check_aws_skipped_for_kaggle_project(tmp_path, monkeypatch):
    """Kaggle+SQLite projects have no AWS dependency — the check should not appear."""
    kaggle_params = (
        "experiment: test-project\n"
        "data:\n"
        "  source: kaggle\n"
        "  competition: spaceship-titanic\n"
    )
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=kaggle_params,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "AWS credentials" not in result.output


def test_check_aws_soft_warn_when_no_params_file(tmp_path, monkeypatch):
    """Without params.yaml the project type is unknown — warn but don't hard-fail."""
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=None,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "~ AWS credentials" in result.output
    assert "✗ AWS credentials" not in result.output


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


_STUB_EVALUATE = """\
from kitchen.steps import Evaluator

class MyEvaluator(Evaluator):
    def evaluate(self, model, df):
        raise NotImplementedError("fill in your evaluator")
"""

_IMPLEMENTED_EVALUATE = """\
from kitchen.steps import Evaluator

class MyEvaluator(Evaluator):
    def evaluate(self, model, df):
        return {"val_accuracy": 0.9}
"""

_STUB_TRAIN = """\
from kitchen.steps import Trainer

class MyTrainer(Trainer):
    model_flavour = "sklearn"

    def fit(self, df, params):
        raise NotImplementedError("fill in your trainer")
"""


def _write_stub_src(tmp_path, evaluate_content, train_content=None):
    """Write src layout with controllable content for evaluate (and optionally train)."""
    for subdir in ("features", "train", "evaluate"):
        (tmp_path / "src" / subdir).mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "features" / "run.py").write_text("")
    (tmp_path / "src" / "train" / "run.py").write_text(train_content or "")
    (tmp_path / "src" / "evaluate" / "run.py").write_text(evaluate_content)


def test_check_stub_evaluate_warns(tmp_path, monkeypatch):
    """A scaffolded evaluate() with raise NotImplementedError → warn (not fail)."""
    _write_stub_src(tmp_path, _STUB_EVALUATE)
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "~ src/evaluate/run.py" in result.output
    assert "evaluate() is a stub" in result.output
    assert "✗ src/evaluate/run.py" not in result.output


def test_check_implemented_evaluate_ok(tmp_path, monkeypatch):
    """A fully implemented evaluate() → ✓ ok, no stub warning."""
    _write_stub_src(tmp_path, _IMPLEMENTED_EVALUATE)
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "✓ src/evaluate/run.py" in result.output
    assert "stub" not in result.output


def test_check_stub_train_warns(tmp_path, monkeypatch):
    """Stub detection works for src/train/run.py (fit) too."""
    _write_stub_src(tmp_path, _IMPLEMENTED_EVALUATE, train_content=_STUB_TRAIN)
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path, monkeypatch, env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"}
        )
    assert "~ src/train/run.py" in result.output
    assert "fit() is a stub" in result.output
    assert "✓ src/evaluate/run.py" in result.output


def test_check_stub_warns_but_exit_zero_when_no_other_failures(tmp_path, monkeypatch):
    """Stub warnings do not cause a non-zero exit — they are informational only."""
    _write_stub_src(tmp_path, _STUB_EVALUATE)
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
    assert result.exit_code == 0
    assert "~ src/evaluate/run.py" in result.output


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


# ---------------------------------------------------------------------------
# secrets manifest + legacy check.required_env alias (SECR-001 / CBB-012)
# ---------------------------------------------------------------------------

_REQ_ENV_PARAMS = "experiment: test-project\ncheck:\n  required_env:\n    - KENPOM_API_KEY\n"
_SECRETS_ENV_PARAMS = (
    "experiment: test-project\nsecrets:\n  KENPOM_API_KEY:\n    required: true\n"
)
_SECRETS_CLOUD_PARAMS = (
    "experiment: test-project\nsecrets:\n"
    "  KENPOM_API_KEY:\n    aws_secret: cbb-model/prod\n    key: KENPOM_API_KEY\n    required: true\n"
)


def test_check_required_env_present(tmp_path, monkeypatch):
    """A legacy required_env secret present in env shows a ✓ secret: line, not a failure."""
    with (
        patch("shutil.which", return_value=None),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=_REQ_ENV_PARAMS,
            env={
                "MLFLOW_TRACKING_URI": "x",
                "KAGGLE_USERNAME": "u",
                "KENPOM_API_KEY": "secret",
            },
        )
    assert "secret: KENPOM_API_KEY" in result.output
    assert "✗ secret: KENPOM_API_KEY" not in result.output


def test_check_required_env_missing_fails(tmp_path, monkeypatch):
    """A required env-only secret absent from env and .env hard-fails check (exit 1)."""
    # which truthy + creds present so the ONLY failure is the missing secret.
    with (
        patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=_REQ_ENV_PARAMS,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "✗ secret: KENPOM_API_KEY" in result.output
    assert "environment variable" in result.output  # resolver remediation
    assert result.exit_code == 1


def test_check_required_env_warns_deprecated(tmp_path, monkeypatch):
    """Using the legacy check.required_env emits a deprecation warning toward secrets:."""
    with (
        patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=_REQ_ENV_PARAMS,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u", "KENPOM_API_KEY": "s"},
        )
    assert "deprecated" in result.output
    assert "secrets:" in result.output


def test_check_secrets_manifest_env_only(tmp_path, monkeypatch):
    """An env-only secret in the secrets: manifest is presence-checked (no deprecation warning)."""
    with (
        patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=_SECRETS_ENV_PARAMS,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "✗ secret: KENPOM_API_KEY" in result.output
    assert "deprecated" not in result.output
    assert result.exit_code == 1


def test_check_secret_cloud_resolves(tmp_path, monkeypatch):
    """SECR-003: a cloud secret that resolves shows ✓ resolved from its source (real boto call)."""
    sm_client = MagicMock()
    sm_client.get_secret_value.return_value = {"SecretString": '{"KENPOM_API_KEY": "v"}'}
    with (
        patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch("boto3.Session") as mock_session,
        patch("boto3.client", return_value=sm_client),
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()  # identity present
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=_SECRETS_CLOUD_PARAMS,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "✓ secret: KENPOM_API_KEY" in result.output
    assert "resolved from SM cbb-model/prod" in result.output
    assert result.exit_code == 0


def test_check_secret_cloud_unresolvable_hard_fails(tmp_path, monkeypatch):
    """SECR-003: a required cloud secret that can't be resolved hard-fails pre-flight."""
    with (
        patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None  # no AWS identity
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=_SECRETS_CLOUD_PARAMS,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "✗ secret: KENPOM_API_KEY" in result.output
    assert "no AWS identity" in result.output
    assert result.exit_code == 1


def test_check_secret_cloud_optional_not_hard_failed(tmp_path, monkeypatch):
    """An optional cloud secret that can't be resolved warns, but does not fail check."""
    params = (
        "experiment: test-project\nsecrets:\n"
        "  OPT:\n    aws_secret: proj/prod\n    key: OPT\n    required: false\n"
    )
    with (
        patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch("boto3.Session") as mock_session,
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        mock_session.return_value.get_credentials.return_value = None
        result = _invoke(
            tmp_path,
            monkeypatch,
            params_content=params,
            env={"MLFLOW_TRACKING_URI": "x", "KAGGLE_USERNAME": "u"},
        )
    assert "~ secret: OPT" in result.output
    assert "✗ secret: OPT" not in result.output
    assert result.exit_code == 0


def test_check_no_secrets_section_no_secret_lines(tmp_path, monkeypatch):
    """Projects with no secrets/required_env never emit secret: lines."""
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
    assert "secret:" not in result.output
