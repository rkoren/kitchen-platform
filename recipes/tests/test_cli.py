"""Tests for the recipes CLI."""

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from recipes.cli import _refresh_tf_files, _workspace, app
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


def test_generate_provider_hardens_backend(tmp_path):
    """SEC-005/SEC-006: generated backend encrypts state and locks via S3 lockfile."""
    spec = tmp_path / "infra.yaml"
    spec.write_text("name: x\nregion: us-east-1\nresources: []\n")
    out = tmp_path / "tf"
    runner.invoke(app, ["generate", str(spec), "--out", str(out)])
    provider = (out / "provider.tf").read_text()
    assert "encrypt      = true" in provider, "state must be encrypted at rest (SEC-005)"
    assert "use_lockfile = true" in provider, "state must use S3-native locking (SEC-006)"
    # use_lockfile needs Terraform >= 1.10 — the gate must be explicit.
    assert 'required_version = ">= 1.10"' in provider


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


# ── plan ──────────────────────────────────────────────────────────────────────


def test_plan_missing_spec_exits_nonzero():
    result = runner.invoke(app, ["plan", "no-such-file.yaml", "--state-bucket", "my-bucket"])
    assert result.exit_code != 0


def test_plan_missing_state_bucket_exits_nonzero(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)
    result = runner.invoke(app, ["plan", str(spec)], env={"RECIPES_STATE_BUCKET": ""})
    assert result.exit_code != 0


def test_plan_runs_init_then_plan(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)

    mock_proc = MagicMock()
    mock_proc.stdout = iter(["Terraform initialized\n", "No changes.\n"])
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0

    with (
        patch("recipes.cli._WORKSPACE_ROOT", tmp_path),
        patch("shutil.which", return_value="/usr/bin/terraform"),
        patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
    ):
        result = runner.invoke(app, ["plan", str(spec), "--state-bucket", "my-bucket"])

    assert result.exit_code == 0
    calls = [c.args[0] for c in mock_popen.call_args_list]
    assert any("init" in c for c in calls)
    assert any("plan" in c for c in calls)
    # plan is read-only: it must never apply.
    assert not any("apply" in c for c in calls)


def test_plan_does_not_prompt_for_confirmation(tmp_path):
    """plan is read-only — it should run without --yes and never prompt."""
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)

    mock_proc = MagicMock()
    mock_proc.stdout = iter(["Plan: 2 to add.\n"])
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0

    with (
        patch("recipes.cli._WORKSPACE_ROOT", tmp_path),
        patch("shutil.which", return_value="/usr/bin/terraform"),
        patch("subprocess.Popen", return_value=mock_proc),
    ):
        # No stdin provided — a confirmation prompt would abort with a non-zero code.
        result = runner.invoke(app, ["plan", str(spec), "--state-bucket", "my-bucket"])

    assert result.exit_code == 0


def test_plan_aborts_on_terraform_failure(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)

    mock_proc = MagicMock()
    mock_proc.stdout = iter(["Error: backend init failed\n"])
    mock_proc.wait.return_value = None
    mock_proc.returncode = 1

    with (
        patch("recipes.cli._WORKSPACE_ROOT", tmp_path),
        patch("shutil.which", return_value="/usr/bin/terraform"),
        patch("subprocess.Popen", return_value=mock_proc),
    ):
        result = runner.invoke(app, ["plan", str(spec), "--state-bucket", "my-bucket"])

    assert result.exit_code != 0


def test_plan_refreshes_tf_files(tmp_path):
    """plan regenerates .tf files in the workspace before planning."""
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)

    mock_proc = MagicMock()
    mock_proc.stdout = iter(["No changes.\n"])
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0

    with (
        patch("recipes.cli._WORKSPACE_ROOT", tmp_path),
        patch("shutil.which", return_value="/usr/bin/terraform"),
        patch("subprocess.Popen", return_value=mock_proc),
    ):
        runner.invoke(app, ["plan", str(spec), "--state-bucket", "my-bucket"])

    workspace = tmp_path / "test-infra"
    assert (workspace / "provider.tf").exists()
    assert any(workspace.glob("s3-*.tf"))


# ── apply ─────────────────────────────────────────────────────────────────────


def test_apply_missing_spec_exits_nonzero():
    result = runner.invoke(
        app, ["apply", "no-such-file.yaml", "--state-bucket", "my-bucket", "--yes"]
    )
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
    result = runner.invoke(
        app, ["destroy", "no-such-file.yaml", "--state-bucket", "my-bucket", "--yes"]
    )
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


# ── R-004: terraform fmt / validate checks on generated HCL ──────────────────────


@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform not installed")
def test_generate_check_passes_on_generated_hcl(tmp_path):
    """The generator must emit canonically-formatted HCL (terraform fmt -check)."""
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)
    out = tmp_path / "tf"
    result = runner.invoke(app, ["generate", str(spec), "--out", str(out), "--check"])
    assert result.exit_code == 0, result.output
    assert "terraform fmt" in result.output


@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform not installed")
def test_generate_check_passes_on_rds_hcl(tmp_path):
    """R-015: the rds generator must emit canonical HCL, including the subnet/sg variant."""
    spec = tmp_path / "infra.yaml"
    spec.write_text(
        "name: rds-infra\n"
        "region: us-east-1\n"
        "resources:\n"
        "  - type: rds\n"
        "    name: mlflow-backend\n"
        "    db_subnet_group_name: mlflow-subnets\n"
        "    vpc_security_group_ids: [sg-0123, sg-0456]\n"
    )
    out = tmp_path / "tf"
    result = runner.invoke(app, ["generate", str(spec), "--out", str(out), "--check"])
    assert result.exit_code == 0, result.output
    assert "terraform fmt" in result.output


@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform not installed")
def test_generate_check_passes_on_security_group_and_rds_ref(tmp_path):
    """R-016: a security_group + an rds that references it must emit canonical HCL."""
    spec = tmp_path / "infra.yaml"
    spec.write_text(
        "name: sg-rds\n"
        "region: us-east-1\n"
        "resources:\n"
        "  - type: security_group\n"
        "    name: mlflow-db-sg\n"
        "  - type: rds\n"
        "    name: mlflow-backend\n"
        "    publicly_accessible: true\n"
        "    security_groups: [mlflow-db-sg]\n"
    )
    out = tmp_path / "tf"
    result = runner.invoke(app, ["generate", str(spec), "--out", str(out), "--check"])
    assert result.exit_code == 0, result.output
    assert "terraform fmt" in result.output


def test_generate_check_fails_when_terraform_missing(tmp_path):
    spec = tmp_path / "infra.yaml"
    spec.write_text(VALID_SPEC)
    out = tmp_path / "tf"
    with patch("recipes.cli.shutil.which", return_value=None):
        result = runner.invoke(app, ["generate", str(spec), "--out", str(out), "--check"])
    assert result.exit_code == 1
    assert "terraform not found" in result.output
    # Files are still generated before the check runs.
    assert (out / "provider.tf").exists()


# ── R-002: recipes doctor ────────────────────────────────────────────────────────


def _fake_which(mapping):
    return lambda name: mapping.get(name)


def _fake_run(responses):
    """Return a subprocess.run stub keyed on a substring of the command."""

    def run(cmd, *args, **kwargs):
        key = next((k for k in responses if k in cmd), None)
        rc, out = responses.get(key, (0, ""))
        return SimpleNamespace(returncode=rc, stdout=out, stderr="")

    return run


def test_doctor_all_present():
    which = _fake_which({"terraform": "/usr/bin/terraform", "aws": "/usr/bin/aws"})
    runs = _fake_run({
        "version": (0, '{"terraform_version": "1.10.5"}'),
        "sts": (0, "123456789012\n"),
    })
    with patch("recipes.cli.shutil.which", side_effect=which), \
         patch("recipes.cli.subprocess.run", side_effect=runs):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "All checks passed" in result.output
    assert "123456789012" in result.output


def test_doctor_fails_without_terraform():
    which = _fake_which({"aws": "/usr/bin/aws"})  # no terraform
    runs = _fake_run({"sts": (0, "123456789012\n")})
    with patch("recipes.cli.shutil.which", side_effect=which), \
         patch("recipes.cli.subprocess.run", side_effect=runs):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "terraform not found" in result.output


def test_doctor_fails_on_bad_aws_credentials():
    which = _fake_which({"terraform": "/usr/bin/terraform", "aws": "/usr/bin/aws"})
    runs = _fake_run({
        "version": (0, '{"terraform_version": "1.10.5"}'),
        "sts": (255, ""),  # get-caller-identity fails
    })
    with patch("recipes.cli.shutil.which", side_effect=which), \
         patch("recipes.cli.subprocess.run", side_effect=runs):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "AWS credentials not found" in result.output


def test_doctor_warns_on_old_terraform():
    which = _fake_which({"terraform": "/usr/bin/terraform", "aws": "/usr/bin/aws"})
    runs = _fake_run({
        "version": (0, '{"terraform_version": "1.9.8"}'),
        "sts": (0, "123456789012\n"),
    })
    with patch("recipes.cli.shutil.which", side_effect=which), \
         patch("recipes.cli.subprocess.run", side_effect=runs):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "< 1.10" in result.output


def test_doctor_checks_state_bucket_access():
    which = _fake_which({"terraform": "/usr/bin/terraform", "aws": "/usr/bin/aws"})
    runs = _fake_run({
        "version": (0, '{"terraform_version": "1.10.5"}'),
        "sts": (0, "123456789012\n"),
        "head-bucket": (255, ""),  # bucket inaccessible
    })
    with patch("recipes.cli.shutil.which", side_effect=which), \
         patch("recipes.cli.subprocess.run", side_effect=runs):
        result = runner.invoke(app, ["doctor", "--state-bucket", "my-bucket"])
    assert result.exit_code == 1
    assert "cannot access state bucket" in result.output


# ── R-006: ECR + Lambda inference API example ────────────────────────────────────

_INFERENCE_EXAMPLE = Path(__file__).parent.parent / "examples" / "ecr-lambda-inference-api.yaml"


def test_generate_inference_api_example(tmp_path):
    out = tmp_path / "tf"
    result = runner.invoke(app, ["generate", str(_INFERENCE_EXAMPLE), "--out", str(out)])
    assert result.exit_code == 0, result.output
    lambda_tf = (out / "lambda-inference-api.tf").read_text()
    assert "aws_lambda_function_url" in lambda_tf
    assert 'output "inference_api_url"' in lambda_tf


@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform not installed")
def test_inference_api_example_is_canonical_hcl(tmp_path):
    out = tmp_path / "tf"
    result = runner.invoke(
        app, ["generate", str(_INFERENCE_EXAMPLE), "--out", str(out), "--check"]
    )
    assert result.exit_code == 0, result.output


# ── R-008: full Kaggle serving stack example ─────────────────────────────────────

_SERVING_STACK_EXAMPLE = Path(__file__).parent.parent / "examples" / "kaggle-serving-stack.yaml"


def test_generate_serving_stack_example(tmp_path):
    out = tmp_path / "tf"
    result = runner.invoke(app, ["generate", str(_SERVING_STACK_EXAMPLE), "--out", str(out)])
    assert result.exit_code == 0, result.output
    # Exercises the full feature set across files.
    assert (out / "s3-titanic-mlflow-artifacts.tf").exists()
    assert "aws_iam_role_policy" in (out / "iam-role-titanic-serve-exec.tf").read_text()
    lambda_tf = (out / "lambda-titanic-serve.tf").read_text()
    assert "aws_cloudwatch_log_group" in lambda_tf
    assert "aws_lambda_function_url" in lambda_tf


@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform not installed")
def test_serving_stack_example_is_canonical_hcl(tmp_path):
    out = tmp_path / "tf"
    result = runner.invoke(
        app, ["generate", str(_SERVING_STACK_EXAMPLE), "--out", str(out), "--check"]
    )
    assert result.exit_code == 0, result.output
