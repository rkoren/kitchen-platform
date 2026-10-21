"""Shared CLI helpers for the ``kitchen`` sub-command modules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

#: ``-C/--project DIR`` for every project-scoped command (DX-012/013, CBB-030) — like ``git -C``.
ProjectOption = Annotated[
    str | None,
    typer.Option(
        "--project",
        "-C",
        metavar="DIR",
        help="Run from this project directory (like `git -C`). Default: the current directory.",
    ),
]


def enter_project(project: str | None) -> None:
    """``chdir`` into ``project`` and load its ``.env`` (CBB-030).

    The menu, ``data/``, ``src/<stage>`` and a relative ``sqlite:///mlruns.db`` tracking URI all
    resolve against the working directory, so this must run before any of them is read. The
    project's ``.env`` is loaded after the move (existing environment variables win), so
    ``MLFLOW_TRACKING_URI`` & co. come from the project being driven, not the caller's cwd.
    """
    if not project:
        return
    os.chdir(project)
    env_file = Path(".env")
    if env_file.is_file():
        try:
            from dotenv import load_dotenv
        except ImportError:
            return  # python-dotenv not installed — env vars must be set externally
        load_dotenv(env_file)
