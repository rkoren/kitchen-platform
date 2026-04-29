"""Tests for the recipes CLI."""
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from recipes.cli import app, _workspace, _refresh_tf_files
from recipes.schema import RecipeSpec

runner = CliRunner()

VALID_SPEC = """\
name: test-infra
region: us-east-1
resources:
  - type: s3
    name: test-bucket
    versioning: false
  - type: iam_role
    name: test-role
    service: lambda.amazonaws.com
"""

LAMBDA_SPEC = """\
name: lambda-infra
region: us-east-1
resources:
  - type: s3
    name: my-data
    versioning: true
  - type: iam_role
    name: my-role
    service: lambda.amazonaws.com
    policies:
      - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  - type: ecr
    name: my-repo
    lambda_access: true
  - type: lambda
    name: my-fn
    role: my-role
    ecr_repo: my-repo
    memory: 1024
    timeout: 30
"""


# ── generate ───────────────────────────────────────────────────────────────────

def test_generate_exits_zero(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)
    result = runner.invoke(app, ["generate", str(spec), "--out", str(tmp_path / "tf")])
    assert result.exit_code == 0


def test_generate_creates_provider_tf(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)
    out = tmp_path / "tf"
    runner.invoke(app, ["generate", str(spec), "--out", str(out)])
    assert (out / "provider.tf").exists()


def test_generate_provider_contains_region(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text("name: x\nregion: eu-west-1\nresources: []\n")
    out = tmp_path / "tf"
    runner.invoke(app, ["generate", str(spec), "--out", str(out)])
    assert 'region = "eu-west-1"' in (out / "provider.tf").read_text()


def test_generate_creates_resource_tf_files(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)
    out = tmp_path / "tf"
    runner.invoke(app, ["generate", str(spec), "--out", str(out)])
    assert (out / "s3-test-bucket.tf").exists()
    assert (out / "iam-role-test-role.tf").exists()


def test_generate_creates_output_dir(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)
    out = tmp_path / "nested" / "tf"
    runner.invoke(app, ["generate", str(spec), "--out", str(out)])
    assert out.is_dir()


def test_generate_missing_spec_exits_nonzero():
    result = runner.invoke(app, ["generate", "does-not-exist.yaml"])
    assert result.exit_code != 0


def test_generate_lambda_uses_memory_and_timeout(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(LAMBDA_SPEC)
    out = tmp_path / "tf"
    runner.invoke(app, ["generate", str(spec), "--out", str(out)])
    lambda_tf = (out / "lambda-my-fn.tf").read_text()
    assert "memory_size = 1024" in lambda_tf
    assert "timeout     = 30" in lambda_tf


# ── validate ──────────────────────────────────────────────────────────────────

def test_validate_valid_spec(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)
    result = runner.invoke(app, ["validate", str(spec)])
    assert result.exit_code == 0
    assert "valid" in result.output


def test_validate_invalid_spec_exits_nonzero(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text("name: x\nresources:\n  - type: ec2\n    name: bad\n")
    result = runner.invoke(app, ["validate", str(spec)])
    assert result.exit_code != 0


# ── workspace helpers ─────────────────────────────────────────────────────────

def test_workspace_creates_directory(tmp_path):
    with patch("recipes.cli._WORKSPACE_ROOT", tmp_path):
        ws = _workspace("my-project")
    assert ws.exists()
    assert ws.name == "my-project"


def test_refresh_tf_files_removes_stale_tf(tmp_path):
    stale = tmp_path / "old-resource.tf"
    stale.write_text("# stale")
    spec = RecipeSpec.model_validate({"name": "x", "resources": []})
    _refresh_tf_files(spec, tmp_path)
    assert not stale.exists()


def test_refresh_tf_files_preserves_terraform_cache(tmp_path):
    cache = tmp_path / ".terraform"
    cache.mkdir()
    (cache / "providers").write_text("cached")
    spec = RecipeSpec.model_validate({"name": "x", "resources": []})
    _refresh_tf_files(spec, tmp_path)
    assert (cache / "providers").exists()


def test_refresh_tf_files_writes_provider(tmp_path):
    spec = RecipeSpec.model_validate({"name": "x", "region": "us-west-2", "resources": []})
    _refresh_tf_files(spec, tmp_path)
    assert (tmp_path / "provider.tf").exists()
    assert 'region = "us-west-2"' in (tmp_path / "provider.tf").read_text()


# ── apply ─────────────────────────────────────────────────────────────────────

def test_apply_missing_spec_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["apply", "no-such-file.yaml", "--state-bucket", "my-bucket", "--yes"])
    assert result.exit_code != 0


def test_apply_missing_state_bucket_exits_nonzero(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)
    # No --state-bucket and no env var set
    result = runner.invoke(app, ["apply", str(spec), "--yes"], env={"RECIPES_STATE_BUCKET": ""})
    assert result.exit_code != 0


def test_apply_streams_terraform_output(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)

    mock_proc = MagicMock()
    mock_proc.stdout = iter(["Terraform initialized\n", "Apply complete!\n"])
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0

    with (
        patch("recipes.cli._WORKSPACE_ROOT", tmp_path),
        patch("shutil.which", return_value="/usr/bin/terraform"),
        patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
    ):
        runner.invoke(
            app,
            ["apply", str(spec), "--state-bucket", "my-bucket", "--yes"],
        )

    assert mock_popen.called
    calls = [c.args[0] for c in mock_popen.call_args_list]
    # init then apply
    assert any("init" in c for c in calls)
    assert any("apply" in c for c in calls)


def test_apply_aborts_on_terraform_failure(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)

    mock_proc = MagicMock()
    mock_proc.stdout = iter(["Error: something went wrong\n"])
    mock_proc.wait.return_value = None
    mock_proc.returncode = 1

    with (
        patch("recipes.cli._WORKSPACE_ROOT", tmp_path),
        patch("shutil.which", return_value="/usr/bin/terraform"),
        patch("subprocess.Popen", return_value=mock_proc),
    ):
        result = runner.invoke(
            app,
            ["apply", str(spec), "--state-bucket", "my-bucket", "--yes"],
        )

    assert result.exit_code != 0


# ── destroy ───────────────────────────────────────────────────────────────────

def test_destroy_missing_spec_exits_nonzero():
    result = runner.invoke(app, ["destroy", "no-such-file.yaml", "--state-bucket", "my-bucket", "--yes"])
    assert result.exit_code != 0


def test_destroy_calls_terraform_destroy(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)

    mock_proc = MagicMock()
    mock_proc.stdout = iter(["Destroy complete!\n"])
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0

    with (
        patch("recipes.cli._WORKSPACE_ROOT", tmp_path),
        patch("shutil.which", return_value="/usr/bin/terraform"),
        patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
    ):
        runner.invoke(
            app,
            ["destroy", str(spec), "--state-bucket", "my-bucket", "--yes"],
        )

    calls = [c.args[0] for c in mock_popen.call_args_list]
    assert any("destroy" in c for c in calls)
