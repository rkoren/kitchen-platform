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
                run(["kitchen", "run", "monitor", "--params", menu_path])
        elif step in menu.recipes:
            entry = menu.recipes[step]
            if entry.kind == "stage":
                # A stage may carry CLI flags via `args:` (e.g. `--auto-promote` on train).
                extra_args = list(entry.fields.get("args") or [])
                echo(f"→ {step}: kitchen run {step} {' '.join(extra_args)}".rstrip())
                if not dry_run:
                    # Pass the manifest explicitly so the stage loads the menu's project
                    # sections (the raw-YAML `params["model"]` contract, INT-007/008).
                    run(["kitchen", "run", step, "--params", menu_path, *extra_args])
            elif entry.kind == "lambda":
                # The serve lambda + its KITCHEN_PREDICTOR_DIR provision via `provision`
                # (INT-006); the image build/push that points it at the latest code is the
                # remaining deploy step (S-008/S-010 territory), not yet wired here.
                echo(f"→ {step}: serve — provisioned by `provision`; image build/deploy not yet wired")
            else:
                echo(f"→ {step}: infra recipe (deployed by `provision`) — skipped")
        # Menu validation guarantees every step is a platform verb or a recipe key.
