"""Source-aware stage loading — `stage_module_name` + a divergent-source run (S-8, INT-019).

A stage's code module is derived from its menu-declared `source` (INT-006) if the project
overrides the convention, else `src.<stage>.run`. This lets stage code live anywhere
(`src/training/main.py`), not just at `src/<stage>/run.py`.
"""

from __future__ import annotations

import sys

from typer.testing import CliRunner

from kitchen.cli import app
from kitchen.menu import stage_module_name

runner = CliRunner()


# --- derivation ---------------------------------------------------------------


def test_no_source_falls_back_to_convention():
    assert stage_module_name("features", {}) == "src.features.run"
    assert stage_module_name("train", {"experiment": "x"}) == "src.train.run"  # params.yaml project


def test_divergent_source_is_honored():
    params = {"recipes": {"train": {"kind": "stage", "source": "src/training/main.py"}}}
    assert stage_module_name("train", params) == "src.training.main"


def test_source_matching_the_convention_round_trips():
    """A menu that declares the conventional path resolves to the same module as the fallback."""
    params = {"recipes": {"features": {"kind": "stage", "source": "src/features/run.py"}}}
    assert stage_module_name("features", params) == "src.features.run"


def test_stage_without_a_recipe_uses_convention():
    params = {"recipes": {"other": {"kind": "stage", "source": "src/other/main.py"}}}
    assert stage_module_name("features", params) == "src.features.run"


# --- end-to-end: a divergent source actually gets loaded ----------------------


def test_run_features_loads_a_divergent_source(tmp_path, monkeypatch):
    """`kitchen run features` imports the stage's code from the menu-declared `source`, not the
    hard-coded `src/features/run.py` — the S-8 acceptance."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "menu.yaml").write_text(
        "project: demo\n"
        "pipeline: [features]\n"
        "recipes:\n"
        "  features:\n"
        "    kind: stage\n"
        "    source: src/featureeng/build_step.py\n",
        encoding="utf-8",
    )
    stage_dir = tmp_path / "src" / "featureeng"
    stage_dir.mkdir(parents=True)
    # A divergent stage module (not src/features/run.py) — writes a marker so we can prove it ran.
    (stage_dir / "build_step.py").write_text(
        "FEATURES = []\n"
        "def build(params, store):\n"
        "    from pathlib import Path\n"
        "    Path('BUILT_BY_DIVERGENT_SOURCE').write_text('ok', encoding='utf-8')\n",
        encoding="utf-8",
    )

    try:
        result = runner.invoke(app, ["run", "features", "--params", "menu.yaml"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "BUILT_BY_DIVERGENT_SOURCE").exists(), "divergent build() did not run"
    finally:
        for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
            del sys.modules[name]
