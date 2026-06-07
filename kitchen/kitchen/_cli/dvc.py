from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Annotated

import typer

from kitchen._cli._templates import _DASHBOARD_HTML, _DVC_CONFIG, _DVC_YAML, _DVC_YAML_KAGGLE, _DVCIGNORE


def _to_class_name(name: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[-_\s]+", name))


def _render(tmpl: str, name: str, class_name: str, **extra) -> str:
    return string.Template(tmpl).substitute(name=name, class_name=class_name, **extra)


def _write(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        typer.echo(f"  skip   {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    typer.echo(f"  create {path}")


def _run_dvc_init(root: Path) -> None:
    """Run `dvc init` in root if .dvc/ doesn't exist yet; always write .dvc/config."""
    import subprocess as _subprocess

    dvc_dir = root / ".dvc"
    if not dvc_dir.exists():
        try:
            _subprocess.run(
                ["dvc", "init"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            typer.echo(f"  dvc    initialized DVC repository in {root}")
        except _subprocess.CalledProcessError as exc:
            typer.echo(
                f"warning: dvc init failed: {exc.stderr.strip() or exc.stdout.strip()}",
                err=True,
            )
    else:
        typer.echo("  dvc    DVC already initialized — skipping dvc init")
    _write(dvc_dir / "config", _DVC_CONFIG, overwrite=True)




# ---------------------------------------------------------------------------
# kitchen dvc — add DVC scaffolding to an existing project
# ---------------------------------------------------------------------------

dvc_app = typer.Typer(help="DVC scaffolding helpers.", no_args_is_help=True)



@dvc_app.command("init")
def dvc_init(
    params_file: Annotated[
        str,
        typer.Option("--params", help="Path to params.yaml (used to detect project name and source)"),
    ] = "params.yaml",
    kaggle: Annotated[
        bool,
        typer.Option(
            "--kaggle/--no-kaggle",
            help="Use the Kaggle DVC template (submit stage, no ingest placeholder)",
        ),
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing dvc.yaml and .dvcignore")
    ] = False,
) -> None:
    """Add DVC scaffolding (dvc.yaml, .dvcignore, .dvc/config) to an existing project.

    Reads params.yaml for the project name and data source. If params.yaml is
    not found, falls back to the current directory name and the non-Kaggle template
    (override with --kaggle).

    Requires the dvc binary: pip install kitchen[dvc]
    """
    import shutil as _shutil
    import subprocess as _subprocess

    if not _shutil.which("dvc"):
        typer.echo(
            "error: dvc binary not found — run `pip install kitchen[dvc]` first",
            err=True,
        )
        raise typer.Exit(1)

    # Resolve project name and source from params.yaml, or fall back to cwd name.
    project_name = Path.cwd().name
    is_kaggle = kaggle
    p = Path(params_file)
    if p.exists():
        import yaml as _yaml

        try:
            raw = _yaml.safe_load(p.read_text(encoding="utf-8"))
            project_name = raw.get("experiment", project_name)
            if not kaggle:
                is_kaggle = raw.get("data", {}).get("source") == "kaggle"
        except Exception:
            pass  # unparseable params.yaml — use defaults
    else:
        typer.echo(
            f"note: {params_file!r} not found — using directory name {project_name!r} and "
            f"{'kaggle' if is_kaggle else 'non-kaggle'} template. "
            "Pass --params or --kaggle to override.",
        )

    root = Path.cwd()
    class_name = _to_class_name(project_name)

    typer.echo(f"\nAdding DVC scaffolding to {root}\n")

    dvc_tmpl = _DVC_YAML_KAGGLE if is_kaggle else _DVC_YAML
    files = [
        (root / "dvc.yaml", _render(dvc_tmpl, project_name, class_name)),
        (root / ".dvcignore", _DVCIGNORE),
    ]
    for path, content in files:
        _write(path, content, overwrite)

    _run_dvc_init(root)

    typer.echo("""
Done. Next steps:

  dvc remote modify s3remote url s3://YOUR-BUCKET/dvc  # set your S3 remote
  dvc push                  # upload data/processed/ + models/ to S3
  # dvc pull               # restore on a new machine
  # dvc repro              # run the full pipeline (skips unchanged stages)
""")

