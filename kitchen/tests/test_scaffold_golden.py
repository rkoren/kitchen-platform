"""Golden-fixture tests for `kitchen init` generated projects.

`test_cli.py` checks scaffold *behaviour* (files exist, YAML parses, modules import).
This locks the scaffold *content*: each config below is captured as a JSON snapshot
(relative path → file text) under ``tests/fixtures/scaffold/``. Regenerating a project
and comparing against the snapshot catches unintended drift — an added/removed file or
any edit to a template — that the behavioural tests miss.

The snapshot is a single inert JSON per config (not a committed project tree) so pytest
never collects the scaffolded ``src/tests/test_features.py`` and ruff never lints the
generated modules.

When a template change is intentional, regenerate the fixtures:

    UPDATE_SCAFFOLD_GOLDEN=1 pytest tests/test_scaffold_golden.py

and review the resulting JSON diff before committing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kitchen.cli import app

runner = CliRunner()

FIXTURES = Path(__file__).parent / "fixtures" / "scaffold"

# Two configs span the axes of scaffold variation: source (local vs kaggle) and
# template (generic binary-cls vs the canonical XGBoost baseline with CI).
CONFIGS: dict[str, list[str]] = {
    "binary-cls-local": [
        "init", "demo", "--template", "binary-cls", "--source", "local", "--here",
    ],
    "baseline-xgb-kaggle-ci": [
        "init", "demo", "--template", "baseline-xgb",
        "--source", "kaggle", "--competition", "demo-comp", "--ci", "--here",
    ],
}


def _snapshot(root: Path) -> dict[str, str]:
    """Map every generated file to its text content, keyed by POSIX relative path."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rel = path.relative_to(root).as_posix()
        out[rel] = path.read_text(encoding="utf-8")
    return out


def _generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]) -> dict[str, str]:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return _snapshot(tmp_path)


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_scaffold_matches_golden(config: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generated = _generate(tmp_path, monkeypatch, CONFIGS[config])
    golden_file = FIXTURES / f"{config}.json"

    if os.environ.get("UPDATE_SCAFFOLD_GOLDEN"):
        FIXTURES.mkdir(parents=True, exist_ok=True)
        golden_file.write_text(json.dumps(generated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"updated golden fixture {golden_file.name}")

    assert golden_file.exists(), (
        f"missing golden fixture {golden_file} — create it with "
        f"UPDATE_SCAFFOLD_GOLDEN=1 pytest tests/test_scaffold_golden.py"
    )
    golden = json.loads(golden_file.read_text(encoding="utf-8"))

    # File set first — clearest signal when a template adds or drops a file.
    missing = sorted(set(golden) - set(generated))
    extra = sorted(set(generated) - set(golden))
    assert not missing and not extra, (
        f"scaffold file set drifted for {config}.\n"
        f"  missing (in golden, not generated): {missing}\n"
        f"  extra (generated, not in golden): {extra}\n"
        f"Regenerate with UPDATE_SCAFFOLD_GOLDEN=1 if intended."
    )

    # Then per-file content.
    changed = [rel for rel in golden if generated[rel] != golden[rel]]
    assert not changed, (
        f"scaffold content drifted for {config} in: {changed}\n"
        f"Regenerate with UPDATE_SCAFFOLD_GOLDEN=1 if intended."
    )
