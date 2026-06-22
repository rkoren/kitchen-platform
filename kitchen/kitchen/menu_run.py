"""Execute a ``menu.yaml`` pipeline (INT-005, minimal).

Sequences existing commands, fail-fast, in the manifest's ``pipeline`` order:

* ``provision`` → ``recipes apply <menu>`` (the infra recipes), then materialize the resolved
  MLflow env (INT-003) into ``os.environ`` so the stages that follow inherit it;
* a ``stage`` recipe (``kind: stage``) → ``kitchen run <step>``;
* ``monitor`` → ``kitchen run monitor``;
* ``serve`` (a ``lambda`` recipe) is recognized but **not yet wired** (INT-006 — see the
  ``role`` collision, simplification S-4): it is reported and skipped.

Skip-unchanged, ``--from`` resume, and retries are deliberately out of scope (additive). The
runner shells out to the ``recipes`` and ``kitchen`` CLIs (two entry points — simplification
S-7); a merged package would call in-process.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable

from kitchen.menu import Menu
from kitchen.menu_resolve import resolve_mlflow_env


class PipelineError(RuntimeError):
    """A pipeline could not run (e.g. a required input is missing)."""


def _run(cmd: list[str]) -> None:
    """Run a subcommand, inheriting the (possibly materialized) environment; raise on failure."""
    subprocess.run(cmd, check=True)


def run_pipeline(
    menu: Menu,
    *,
    menu_path: str,
    state_bucket: str | None = None,
    dry_run: bool = False,
    run: Callable[[list[str]], None] = _run,
    echo: Callable[[str], None] = lambda _msg: None,
) -> None:
    """Execute ``menu.pipeline`` in order. ``run`` is injectable for testing; ``--dry-run``
    prints the plan without executing. Raises :class:`PipelineError` / propagates the failing
    step's error (fail-fast)."""
    for step in menu.pipeline:
        if step == "provision":
            if not state_bucket:
                raise PipelineError(
                    "`provision` needs a Terraform state bucket — pass --state-bucket or set "
                    "RECIPES_STATE_BUCKET."
                )
            echo(f"→ provision: recipes apply {menu_path}")
            if not dry_run:
                run(["recipes", "apply", menu_path, "--state-bucket", state_bucket, "--yes"])
                env = resolve_mlflow_env(menu)
                os.environ.update(env)
                echo(f"  materialized: {', '.join(env) or '(nothing)'}")
        elif step == "monitor":
            echo("→ monitor: kitchen run monitor")
            if not dry_run:
                run(["kitchen", "run", "monitor"])
        elif step in menu.recipes:
            entry = menu.recipes[step]
            if entry.kind == "stage":
                echo(f"→ {step}: kitchen run {step}")
                if not dry_run:
                    run(["kitchen", "run", step])
            elif entry.kind == "lambda":
                echo(f"→ {step}: serve deploy not yet wired (INT-006) — skipped")
            else:
                echo(f"→ {step}: infra recipe (deployed by `provision`) — skipped")
        # Menu validation guarantees every step is a platform verb or a recipe key.
