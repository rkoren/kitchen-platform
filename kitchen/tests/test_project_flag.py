"""`-C/--project DIR` on the project-scoped read/compare commands (CBB-030)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kitchen._cli import enter_project
from kitchen.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "argv",
    [
        ["status"],
        ["leaderboard"],
        ["diff"],
        ["check"],
        ["report"],
        ["promote"],
        ["log"],
        ["dashboard"],
        ["run"],
        ["ingest"],
        ["submit"],
        ["stage"],
    ],
)
def test_project_scoped_command_accepts_C(argv):
    # Inspect the declared options, not `--help` text: Rich colors help output in CI, and the
    # ANSI codes split `--project` so a substring check passes locally and fails there.
    import typer.main

    cmd = typer.main.get_command(app)
    for name in argv:
        cmd = cmd.commands[name]
    opts = {opt for param in cmd.params for opt in param.opts}
    assert {"--project", "-C"} <= opts


def test_enter_project_chdirs_and_loads_project_env(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".env").write_text("KITCHEN_TEST_FROM_ENV=proj\nKITCHEN_TEST_PRESET=proj\n")
    monkeypatch.chdir(tmp_path)  # restores cwd at teardown
    monkeypatch.setenv("KITCHEN_TEST_FROM_ENV", "x")
    monkeypatch.delenv("KITCHEN_TEST_FROM_ENV")  # registered → removed again at teardown
    monkeypatch.setenv("KITCHEN_TEST_PRESET", "caller")

    enter_project(str(proj))

    assert Path.cwd() == proj.resolve()
    assert os.environ["KITCHEN_TEST_FROM_ENV"] == "proj"
    assert os.environ["KITCHEN_TEST_PRESET"] == "caller"  # existing env wins


def test_enter_project_none_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enter_project(None)
    assert Path.cwd() == tmp_path.resolve()


def test_log_C_writes_store_under_project(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(tmp_path)  # deliberately NOT in the project dir
    result = runner.invoke(app, ["log", "-C", str(proj), "--metric", "acc=0.9"])
    assert result.exit_code == 0, result.output
    record = json.loads((proj / "runs.jsonl").read_text().splitlines()[0])
    assert record["metrics"] == {"acc": 0.9}
    assert not (tmp_path / "runs.jsonl").exists()


def test_check_C_reads_menu_from_project(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "menu.yaml").write_text("project: from-proj-dir\n")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["check", "-C", str(proj)])
    assert "not found (run from a project directory)" not in result.output
    assert "menu.yaml" in result.output  # found (and validated) in the project dir


# ── CBB-046: top-level `kitchen -C DIR <cmd>` + models.X.variant validation ──


def test_top_level_C_runs_any_command_in_project(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["-C", str(proj), "log", "--metric", "acc=0.5"])
    assert result.exit_code == 0, result.output
    assert (proj / "runs.jsonl").exists()
    assert not (tmp_path / "runs.jsonl").exists()


def test_top_level_C_composes_with_subcommand_C_like_git(tmp_path, monkeypatch):
    (tmp_path / "outer" / "inner").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["-C", "outer", "log", "-C", "inner", "--metric", "acc=0.5"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "outer" / "inner" / "runs.jsonl").exists()


_BAD_VARIANT_MENU = """\
project: p
pipeline: []
models:
  reg: {model_name: r, variant: reg_seasn}
variants:
  reg_season: {model: {a: 1}}
"""


def test_menu_rejects_model_with_undeclared_variant(tmp_path):
    from kitchen.menu import Menu

    path = tmp_path / "menu.yaml"
    path.write_text(_BAD_VARIANT_MENU)
    with pytest.raises(ValueError, match=r"models\.reg\.variant: 'reg_seasn' is not a declared"):
        Menu.from_yaml(str(path))


def test_check_flags_model_with_undeclared_variant(tmp_path, monkeypatch):
    (tmp_path / "menu.yaml").write_text(_BAD_VARIANT_MENU)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["check"])
    assert result.exit_code != 0
    assert "reg_seasn" in result.output and "reg_season" in result.output


def test_check_lists_models(tmp_path, monkeypatch):
    (tmp_path / "menu.yaml").write_text(_BAD_VARIANT_MENU.replace("reg_seasn", "reg_season"))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["check"])
    assert "reg (variant reg_season)" in result.output
