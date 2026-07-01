"""Tests for `recipes schema` — JSON Schema export of the recipe YAML spec."""

import json

import pytest
import yaml
from typer.testing import CliRunner

from kitchen.recipes.cli import app

# jsonschema lives only in the recipes `dev` extra, so a bare `uv run python -m
# pytest` (the documented convention) would otherwise error at collection. Guard
# the import and skip just the round-trip tests that need it — the two tests that
# exercise the `recipes schema` command output keep running without the extra.
try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised only when the dev extra is absent
    jsonschema = None

requires_jsonschema = pytest.mark.skipif(
    jsonschema is None, reason="jsonschema not installed (recipes 'dev' extra)"
)

runner = CliRunner()

VALID_SPEC = """\
name: my-api
region: us-east-1
resources:
  - type: s3
    name: my-api-artifacts
    versioning: true
  - type: iam_role
    name: my-api-exec
    service: lambda.amazonaws.com
  - type: lambda
    name: my-api
    role: my-api-exec
    image_uri: "123456789.dkr.ecr.us-east-1.amazonaws.com/my-api:latest"
    memory: 512
    timeout: 30
"""


def _export() -> dict:
    result = runner.invoke(app, ["schema"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_schema_stdout_is_valid_json_document():
    doc = _export()
    assert doc["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert doc["title"] == "RecipeSpec"
    assert set(doc["properties"]) == {"name", "region", "resources"}
    # The four resource types (plus the InlinePolicy sub-model) are described.
    assert {"S3Spec", "IAMRoleSpec", "ECRSpec", "LambdaSpec"} <= set(doc["$defs"])


@requires_jsonschema
def test_exported_schema_is_itself_valid():
    """The emitted document is a well-formed draft 2020-12 schema."""
    jsonschema.Draft202012Validator.check_schema(_export())


@requires_jsonschema
def test_exported_schema_accepts_a_valid_spec():
    """Round-trip: a spec that RecipeSpec accepts also validates against the export."""
    jsonschema.validate(instance=yaml.safe_load(VALID_SPEC), schema=_export())


@requires_jsonschema
def test_exported_schema_rejects_unknown_field():
    """extra='forbid' on the model surfaces as additionalProperties: false."""
    bad = yaml.safe_load(VALID_SPEC)
    bad["nonsense"] = True
    try:
        jsonschema.validate(instance=bad, schema=_export())
    except jsonschema.ValidationError:
        return
    raise AssertionError("schema should reject an unknown top-level field")


def test_schema_out_writes_file(tmp_path):
    dest = tmp_path / "recipe.schema.json"
    result = runner.invoke(app, ["schema", "--out", str(dest)], catch_exceptions=False)
    assert result.exit_code == 0
    written = json.loads(dest.read_text())
    assert written["title"] == "RecipeSpec"
