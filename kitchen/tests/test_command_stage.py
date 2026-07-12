"""Tests for command/subprocess stages + per-stage environments (GEN-002/003)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kitchen.cli import app
from kitchen.menu import Menu, RecipeEntry
from kitchen.menu_run import PipelineError, run_command_stage, run_pipeline, stage_argv

runner = CliRunner()


# ── schema + validation ───────────────────────────────────────────────────────


def test_recipe_entry_parses_command_fields():
    e = RecipeEntry(
        kind="stage", cmd=["python", "-m", "x"], python=".venv/bin/python",
        inputs=["data/"], outputs=["out.parquet"],
    )
    assert e.cmd == ["python", "-m", "x"] and e.python == ".venv/bin/python"
    assert e.inputs == ["data/"] and e.outputs == ["out.parquet"]


def test_stage_requires_source_xor_cmd():
    # neither source nor cmd
    with pytest.raises(Exception, match="must declare a `source`.*or a `cmd`"):
        Menu.model_validate({"project": "p", "pipeline": [], "recipes": {"s": {"kind": "stage"}}})
    # both
    with pytest.raises(Exception, match="declares both `source` and `cmd`"):
        Menu.model_validate(
            {"project": "p", "pipeline": [],
             "recipes": {"s": {"kind": "stage", "source": "a.py", "cmd": "python x"}}}
        )
    # cmd only → valid
    m = Menu.model_validate(
        {"project": "p", "pipeline": ["s"], "recipes": {"s": {"kind": "stage", "cmd": "python x"}}}
    )
    assert m.recipes["s"].cmd == "python x"


# ── stage_argv ────────────────────────────────────────────────────────────────


def test_stage_argv_string_is_shlex_split():
    e = RecipeEntry(kind="stage", cmd='python -m pipeline.run --thresh 0.5')
    assert stage_argv(e) == ["python", "-m", "pipeline.run", "--thresh", "0.5"]


def test_stage_argv_preserves_quoted_args():
    e = RecipeEntry(kind="stage", cmd='python -c "print(1); print(2)"')
    assert stage_argv(e) == ["python", "-c", "print(1); print(2)"]


def test_stage_argv_list_used_verbatim():
    e = RecipeEntry(kind="stage", cmd=["./run.sh", "--fast", "a b"])
    assert stage_argv(e) == ["./run.sh", "--fast", "a b"]


def test_stage_argv_python_is_interpreter_and_cmd_is_args():
    # per-stage env: python is the interpreter, cmd is the args passed to it (GEN-003)
    e = RecipeEntry(kind="stage", python=".venv-track/bin/python", cmd="-m pipeline.run")
    assert stage_argv(e) == [".venv-track/bin/python", "-m", "pipeline.run"]


# ── run_command_stage ─────────────────────────────────────────────────────────


def test_run_command_stage_invokes_argv():
    e = RecipeEntry(kind="stage", cmd=["echo", "hi"])
    calls: list[list[str]] = []
    run_command_stage("s", e, run=calls.append)
    assert calls == [["echo", "hi"]]


def test_run_command_stage_dry_run_prints_without_running():
    e = RecipeEntry(kind="stage", cmd="python -m x", inputs=["nope"])
    calls: list[list[str]] = []
    echoed: list[str] = []
    run_command_stage("s", e, run=calls.append, echo=echoed.append, dry_run=True)
    assert not calls  # nothing ran
    assert any("python -m x" in line for line in echoed)  # argv previewed
    # dry run must not fail on the missing declared input either


def test_run_command_stage_missing_input_fails_fast(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    e = RecipeEntry(kind="stage", cmd=["echo", "hi"], inputs=["data/raw"])
    with pytest.raises(PipelineError, match="declared input.*data/raw"):
        run_command_stage("s", e, run=lambda _argv: None)


def test_run_command_stage_missing_output_warns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    e = RecipeEntry(kind="stage", cmd=["echo", "hi"], outputs=["out.parquet"])
    echoed: list[str] = []
    run_command_stage("s", e, run=lambda _argv: None, echo=echoed.append)
    assert any("output(s) not found" in line and "out.parquet" in line for line in echoed)


# ── menu run dispatch ─────────────────────────────────────────────────────────


def test_pipeline_routes_command_stage_to_subprocess():
    menu = Menu.model_validate(
        {"project": "p", "pipeline": ["track"],
         "recipes": {"track": {"kind": "stage", "cmd": ["python", "track.py"]}}}
    )
    calls: list[list[str]] = []
    run_pipeline(menu, menu_path="menu.yaml", run=calls.append)
    assert calls == [["python", "track.py"]]  # ran the argv, not `kitchen run track`


def test_pipeline_fails_fast_on_missing_input_before_later_stages(tmp_path, monkeypatch):
    # A missing declared input in a pipeline stage must stop `menu run` before later stages run.
    monkeypatch.chdir(tmp_path)
    menu = Menu.model_validate(
        {"project": "p", "pipeline": ["a", "b"],
         "recipes": {
             "a": {"kind": "stage", "cmd": ["echo", "a"], "inputs": ["missing_x"]},
             "b": {"kind": "stage", "cmd": ["echo", "b"]},
         }}
    )
    calls: list[list[str]] = []
    with pytest.raises(PipelineError, match="missing_x"):
        run_pipeline(menu, menu_path="menu.yaml", run=calls.append)
    assert calls == []  # neither a nor b ran — fail-fast, not fail-open


def test_pipeline_source_stage_still_shells_kitchen_run():
    menu = Menu.model_validate(
        {"project": "p", "pipeline": ["train"],
         "recipes": {"train": {"kind": "stage", "source": "src/train/run.py"}}}
    )
    calls: list[list[str]] = []
    run_pipeline(menu, menu_path="menu.yaml", run=calls.append)
    assert calls == [["kitchen", "run", "train", "--params", "menu.yaml"]]


# ── kitchen stage <name> ──────────────────────────────────────────────────────


def _menu_with_writer_stage(tmp_path: Path) -> None:
    (tmp_path / "menu.yaml").write_text(textwrap.dedent(f"""\
        project: p
        pipeline: [track]
        recipes:
          track:
            kind: stage
            python: {sys.executable}
            cmd: -c "open('out.txt','w').write('done')"
            outputs: [out.txt]
    """))


def test_stage_command_runs_a_command_stage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _menu_with_writer_stage(tmp_path)
    result = runner.invoke(app, ["stage", "track"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out.txt").read_text() == "done"


def test_stage_command_dry_run_does_not_execute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _menu_with_writer_stage(tmp_path)
    result = runner.invoke(app, ["stage", "track", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "out.txt").exists()
    assert "track:" in result.output  # argv previewed


def test_stage_command_propagates_child_exit_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "menu.yaml").write_text(textwrap.dedent(f"""\
        project: p
        pipeline: [boom]
        recipes:
          boom:
            kind: stage
            python: {sys.executable}
            cmd: -c "import sys; sys.exit(2)"
    """))
    result = runner.invoke(app, ["stage", "boom"])
    assert result.exit_code == 2  # the child's exit code, not a generic 1


def test_stage_command_unknown_stage_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _menu_with_writer_stage(tmp_path)
    result = runner.invoke(app, ["stage", "ghost"])
    assert result.exit_code != 0
    assert "no command stage named 'ghost'" in result.output
    assert "track" in result.output  # lists the available command stages


def test_stage_command_project_flag_runs_from_dir(tmp_path, monkeypatch):
    # -C runs from the project dir even when cwd is elsewhere (inputs/outputs resolve there).
    proj = tmp_path / "proj"
    proj.mkdir()
    _menu_with_writer_stage(proj)
    monkeypatch.chdir(tmp_path)  # NOT the project dir
    result = runner.invoke(app, ["stage", "track", "-C", str(proj)])
    assert result.exit_code == 0, result.output
    assert (proj / "out.txt").exists()
