"""Assemble a Kaggle submission notebook from a project's submission code (GEN-005a).

`kitchen submit --notebook` wraps a **notebook-safe** submission script into a runnable
``submission.ipynb``: a header, the script's source, an explicit call to its entry function, and a
final validation cell that checks the produced CSV against the sample with the platform's own
``validate_submission``. This is the notebook half of code-competition support; bundling offline
dependencies as a Kaggle dataset (for internet-off competitions) is separate (GEN-005b).

**The notebook-safe contract.** A notebook runs top-to-bottom with no ``__main__`` and an
unpredictable working directory (Kaggle's is ``/kaggle/working``), so a wrapped script must:

* do its real work in a function the notebook calls explicitly — a ``if __name__ == "__main__":``
  block is *excluded* from the notebook (an explicit call cell replaces it), so relying on it to
  run the submission silently produces nothing on Kaggle;
* have **no module-level** ``__file__``, ``sys.path.insert(...)``, or ``os.chdir(...)`` — those
  break in a notebook. Put that bootstrap inside ``if __name__ == "__main__":`` (kept for the
  standalone-script path) and import project/heavy modules *inside* the function.

The builder rejects a script that violates this rather than silently stripping code it doesn't
understand.
"""

from __future__ import annotations

import ast
import json
import re


class NotebookSafetyError(ValueError):
    """A submission script isn't notebook-safe (module-level bootstrap or no callable entry)."""


_MAIN_RE = re.compile(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", re.M)
# Calls that break in a notebook if they run at module level (Kaggle cwd / no __file__).
_UNSAFE_CALLS = frozenset({"os.chdir", "sys.path.insert", "sys.path.append"})


def _strip_main_block(source: str) -> str:
    """Drop the trailing ``if __name__ == "__main__":`` block (the notebook calls the entry itself)."""
    m = _MAIN_RE.search(source)
    return (source[: m.start()].rstrip() + "\n") if m else source


def _unsafe_constructs(body: str) -> list[str]:
    """Real (AST-level) notebook-unsafe constructs in ``body`` — a ``__file__`` reference or an
    ``os.chdir`` / ``sys.path.insert`` call. Ignores mentions in docstrings/comments (which is why
    this is AST-based, not a substring scan)."""
    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        raise NotebookSafetyError(f"submission script does not parse: {exc}") from exc
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "__file__":
            found.add("__file__")
        elif isinstance(node, ast.Call):
            try:
                fn = ast.unparse(node.func)
            except Exception:  # noqa: BLE001 — unparse is best-effort; skip exotic call targets
                continue
            if fn in _UNSAFE_CALLS:
                found.add(f"{fn}(...)")
    return sorted(found)


def _md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.strip("\n").splitlines(keepends=True),
    }


def _validation_code(
    submission_file: str, sample_path: str, id_col: str, target_col: str
) -> str:
    return (
        "# Validate the produced submission against the sample (the same check `kitchen submit` runs).\n"
        "import pandas as pd\n"
        "from kitchen.submit import validate_submission\n"
        f"_sub = pd.read_csv({submission_file!r})\n"
        f"_sample = pd.read_csv({sample_path!r})\n"
        f"_errors = validate_submission(_sub, _sample, {id_col!r}, {target_col!r})\n"
        'assert not _errors, "Submission validation failed:\\n  " + "\\n  ".join(_errors)\n'
        'print(f"Validated {len(_sub)} rows against sample_submission — submission is well-formed.")\n'
    )


def build_submission_notebook(
    *,
    source: str,
    call: str,
    competition: str | None,
    id_col: str,
    target_col: str,
    sample_path: str,
    submission_file: str,
) -> str:
    """Return a submission notebook (ipynb JSON text) wrapping ``source``.

    Raises :class:`NotebookSafetyError` if the script has a module-level bootstrap that would break
    in a notebook, or no ``call`` entry function to invoke.
    """
    body = _strip_main_block(source)

    unsafe = _unsafe_constructs(body)
    if unsafe:
        raise NotebookSafetyError(
            f"submission script is not notebook-safe — {', '.join(unsafe)} breaks in a "
            "notebook (no __file__, Kaggle cwd). Move that bootstrap inside "
            '`if __name__ == "__main__":` and import project/heavy modules inside the function.'
        )
    if re.search(rf"^\s*def\s+{re.escape(call)}\s*\(", body, re.M) is None:
        raise NotebookSafetyError(
            f"submission script has no `def {call}(...)` to call from the notebook "
            "(pass --call <name> to select the entry function)."
        )

    title = f"# {competition or 'Submission'} — submission notebook"
    header = (
        f"{title}\n\n"
        "Generated by `kitchen submit --notebook`. Runs top-to-bottom: the submission code, an "
        f"explicit `{call}()` call, then a validation check against `sample_submission`.\n"
    )
    cells = [
        _md_cell(header),
        _code_cell(body),
        _code_cell(f"{call}()"),
        _code_cell(_validation_code(submission_file, sample_path, id_col, target_col)),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1) + "\n"


def notebook_code(notebook_json: str) -> str:
    """Concatenate a notebook's code cells into a single script (for a linear ``--execute`` run)."""
    nb = json.loads(notebook_json)
    blocks = [
        "".join(cell["source"]) for cell in nb["cells"] if cell.get("cell_type") == "code"
    ]
    return "\n\n".join(blocks) + "\n"
