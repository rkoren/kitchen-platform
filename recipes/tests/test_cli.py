"""Tests for the recipes CLI."""
from typer.testing import CliRunner

from recipes.cli import app

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
