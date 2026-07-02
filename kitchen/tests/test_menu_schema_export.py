"""Tests for `kitchen menu schema` — JSON Schema export of the menu.yaml manifest (REL-001)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from kitchen.cli import app

# jsonschema lives only in the `dev` extra, so a bare `uv run python -m pytest` would otherwise
# error at collection. Guard the import and skip just the round-trip tests that need it — the
# tests that only exercise the command output keep running without the extra.
try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised only when the dev extra is absent
    jsonschema = None

requires_jsonschema = pytest.mark.skipif(
    jsonschema is None, reason="jsonschema not installed (the 'dev' extra)"
)

runner = CliRunner()

# parents[2] == repo root (this file is kitchen/tests/test_menu_schema_export.py)
_COMMITTED = Path(__file__).resolve().parents[2] / "docs" / "kitchen" / "menu.schema.json"

VALID_MENU = """\
project: demo
region: us-east-1
pipeline: [provision, train]
recipes:
  artifacts:
    kind: s3
    role: mlflow-artifacts
  train:
    kind: stage
    source: src/train/run.py
mlflow:
  artifact_bucket: {from_role: mlflow-artifacts}
thresholds:
  val_accuracy: 0.8
"""


def _export() -> dict:
    result = runner.invoke(app, ["menu", "schema"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_schema_stdout_is_valid_json_document():
    doc = _export()
    assert doc["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert doc["title"] == "Menu"
    # the manifest's core fields are described
    assert {"project", "pipeline", "recipes", "mlflow"} <= set(doc["properties"])
    # the sub-models are described in $defs
    assert {"RecipeEntry", "NetworkSpec", "MlflowSettings"} <= set(doc["$defs"])


def test_committed_schema_is_in_sync_with_the_model():
    """`docs/kitchen/menu.schema.json` must match the current Menu model. If this fails,
    regenerate it: `kitchen menu schema -o docs/kitchen/menu.schema.json`."""
    assert _COMMITTED.exists(), f"missing committed schema: {_COMMITTED}"
    committed = json.loads(_COMMITTED.read_text(encoding="utf-8"))
    assert committed == _export(), "committed menu.schema.json is stale — regenerate it"


@requires_jsonschema
def test_exported_schema_is_itself_valid():
    """The emitted document is a well-formed draft 2020-12 schema."""
    jsonschema.Draft202012Validator.check_schema(_export())


@requires_jsonschema
def test_exported_schema_accepts_a_valid_menu():
    """Round-trip: a menu that the Menu model accepts also validates against the export.

    (The schema is structural — the cross-field validators like infra-field checking, S-2,
    are runtime-only and not expressible in JSON Schema.)"""
    jsonschema.validate(instance=yaml.safe_load(VALID_MENU), schema=_export())


def test_schema_out_writes_file(tmp_path):
    dest = tmp_path / "menu.schema.json"
    result = runner.invoke(app, ["menu", "schema", "--out", str(dest)], catch_exceptions=False)
    assert result.exit_code == 0
    written = json.loads(dest.read_text())
    assert written["title"] == "Menu"
