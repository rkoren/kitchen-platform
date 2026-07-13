"""Tests for `kitchen submit --notebook` — submission notebook assembly (GEN-005a)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kitchen.cli import app
from kitchen.notebook import (
    NotebookSafetyError,
    build_submission_notebook,
    notebook_code,
)

runner = CliRunner()

SAFE_SCRIPT = textwrap.dedent('''\
    """A notebook-safe submission producer."""
    from pathlib import Path


    def generate():
        import pandas as pd
        Path("submissions").mkdir(exist_ok=True)
        pd.DataFrame({"Id": [1, 2], "Pred": [1, 0]}).to_csv("submissions/submission.csv", index=False)


    if __name__ == "__main__":
        import os, sys
        os.chdir(Path(__file__).resolve().parent)   # excluded from the notebook
        generate()
''')


def _build(source: str = SAFE_SCRIPT, call: str = "generate") -> dict:
    nb_json = build_submission_notebook(
        source=source, call=call, competition="demo-comp", id_col="Id", target_col="Pred",
        sample_path="data/raw/sample_submission.csv", submission_file="submissions/submission.csv",
    )
    return json.loads(nb_json)


# ── builder ───────────────────────────────────────────────────────────────────


def test_notebook_structure_and_metadata():
    nb = _build()
    assert [c["cell_type"] for c in nb["cells"]] == ["markdown", "code", "code", "code"]
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert nb["nbformat"] == 4
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    # __main__ block (with its os.chdir) is excluded; an explicit call replaces it
    assert "__main__" not in codes[0] and "os.chdir" not in codes[0]
    assert codes[1].strip() == "generate()"
    assert "validate_submission" in codes[2]


def test_builder_rejects_module_level_bootstrap():
    unsafe = textwrap.dedent('''\
        import sys, os
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        os.chdir("x")
        def generate():
            pass
    ''')
    with pytest.raises(NotebookSafetyError) as exc:
        _build(unsafe)
    msg = str(exc.value)
    assert "__file__" in msg and "os.chdir" in msg and "sys.path.insert" in msg


def test_builder_ignores_bootstrap_words_in_docstrings():
    # The false-positive an AST-based check must avoid: `__file__`/`os.chdir` mentioned in prose.
    doc = textwrap.dedent('''\
        """This is notebook-safe: no module-level __file__, os.chdir, or sys.path.insert."""
        def generate():
            pass
    ''')
    nb = _build(doc)  # must not raise
    assert [c["cell_type"] for c in nb["cells"]] == ["markdown", "code", "code", "code"]


def test_builder_requires_the_entry_function():
    with pytest.raises(NotebookSafetyError, match="no `def make_it"):
        _build("def generate():\n    pass\n", call="make_it")


def test_builder_rejects_unparseable_source():
    with pytest.raises(NotebookSafetyError, match="does not parse"):
        _build("def generate(:\n  pass\n")


def test_notebook_code_concatenates_code_cells():
    nb_json = build_submission_notebook(
        source=SAFE_SCRIPT, call="generate", competition="c", id_col="Id", target_col="Pred",
        sample_path="s.csv", submission_file="sub.csv",
    )
    code = notebook_code(nb_json)
    assert "def generate" in code and "\ngenerate()" in code and "validate_submission" in code
    assert "markdown" not in code  # only code cells


# ── CLI: kitchen submit --notebook ────────────────────────────────────────────


def _project(tmp_path: Path, script: str = SAFE_SCRIPT) -> None:
    (tmp_path / "menu.yaml").write_text(textwrap.dedent("""\
        project: demo
        pipeline: []
        recipes: {}
        submission:
          id_col: Id
          target_col: Pred
          competition: demo-comp
          sample_submission: sample_submission.csv
    """))
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "sample_submission.csv").write_text("Id,Pred\n1,0\n2,0\n")
    (tmp_path / "flows").mkdir()
    (tmp_path / "flows" / "generate_submission.py").write_text(script)


def test_submit_notebook_generates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project(tmp_path)
    result = runner.invoke(app, ["submit", "--notebook"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "submission.ipynb").exists()
    assert "generated, not uploaded" in result.output


def test_submit_notebook_execute_runs_and_validates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project(tmp_path)
    result = runner.invoke(app, ["submit", "--notebook", "--execute"])
    assert result.exit_code == 0, result.output
    # the notebook ran (wrote the CSV) and its validation cell passed
    assert (tmp_path / "submissions" / "submission.csv").exists()
    assert "ran clean and validated" in result.output


def test_submit_notebook_execute_surfaces_a_failing_notebook(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # a producer that writes a submission with the WRONG row count → validation cell fails
    bad = textwrap.dedent('''\
        from pathlib import Path
        def generate():
            import pandas as pd
            Path("submissions").mkdir(exist_ok=True)
            pd.DataFrame({"Id": [1], "Pred": [1]}).to_csv("submissions/submission.csv", index=False)
    ''')
    _project(tmp_path, bad)
    result = runner.invoke(app, ["submit", "--notebook", "--execute"])
    assert result.exit_code != 0
    assert "execution failed" in result.output


def test_submit_notebook_missing_script_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _project(tmp_path)
    result = runner.invoke(app, ["submit", "--notebook", "--from", "flows/nope.py"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_submit_notebook_rejects_unsafe_script(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    unsafe = textwrap.dedent('''\
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        def generate():
            pass
    ''')
    _project(tmp_path, unsafe)
    result = runner.invoke(app, ["submit", "--notebook"])
    assert result.exit_code != 0
    assert "not notebook-safe" in result.output
    assert not (tmp_path / "submission.ipynb").exists()


def test_generated_notebook_is_valid_nbformat():
    # The JSON must be a real notebook nbformat can read — the concatenation test doesn't prove that.
    import nbformat

    nb_json = build_submission_notebook(
        source=SAFE_SCRIPT, call="generate", competition="c", id_col="Id", target_col="Pred",
        sample_path="s.csv", submission_file="sub.csv",
    )
    nb = nbformat.reads(nb_json, as_version=4)
    nbformat.validate(nb)  # raises if the notebook structure is invalid
    assert nb.metadata["kernelspec"]["name"] == "python3"
