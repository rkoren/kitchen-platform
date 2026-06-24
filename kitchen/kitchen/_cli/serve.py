from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from kitchen._cli._templates import _DASHBOARD_GENERATED_HTML


def _render_dashboard_html(
    project: str,
    status: str,
    results: "list[dict]",
    metric: str,
    param_keys: "list[str]",
    has_lb: bool,
) -> str:
    import html as _html
    import json as _json

    html = _DASHBOARD_GENERATED_HTML
    html = html.replace("__PROJECT_ESCAPED__", _html.escape(project))
    html = html.replace("__STATUS_ESCAPED__", _html.escape(status))
    html = html.replace("__RESULTS_JSON__", _json.dumps(results))
    html = html.replace("__METRIC_JS__", _json.dumps(metric))
    html = html.replace("__PARAM_KEYS_JSON__", _json.dumps(param_keys))
    html = html.replace("__HAS_LB__", "true" if has_lb else "false")
    return html


def _find_free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _serve_local_dashboard(html_path: Path) -> None:
    """Start a local HTTP server in CWD and open html_path in the browser.

    Blocks until the user presses Ctrl+C.
    """
    import http.server
    import threading
    import time
    import webbrowser

    port = _find_free_port()

    try:
        rel = html_path.resolve().relative_to(Path.cwd())
    except ValueError:
        rel = html_path

    url = f"http://localhost:{port}/{rel}"

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # suppress request log lines
            pass

    httpd = http.server.HTTPServer(("", port), _QuietHandler)

    typer.echo(f"Dashboard → {url}")
    typer.echo("Press Ctrl+C to stop.\n")

    def _open() -> None:
        time.sleep(0.5)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nStopped.")
    finally:
        httpd.server_close()



# ---------------------------------------------------------------------------
# kitchen serve — start the FastAPI serving app locally
# ---------------------------------------------------------------------------

serve_app = typer.Typer(help="Serving helpers.", no_args_is_help=True)



@serve_app.command("local")
def serve_local(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind (default 8080)")] = 8080,
    predictor_dir: Annotated[
        str | None,
        typer.Option(
            "--predictor-dir",
            help=(
                "Directory containing predictor.py. "
                "Defaults to src/serve/ if it contains predictor.py, else ./"
            ),
        ),
    ] = None,
    reload: Annotated[
        bool,
        typer.Option(
            "--reload/--no-reload",
            help="Enable uvicorn auto-reload on code changes (requires watchfiles)",
        ),
    ] = True,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open /docs in the browser after startup"),
    ] = True,
) -> None:
    """Start the kitchen FastAPI serving app locally with uvicorn.

    Resolves predictor.py in this order:
      1. --predictor-dir <dir>/predictor.py
      2. src/serve/predictor.py       (project default location)
      3. ./predictor.py               (current directory)

    The resolved directory is prepended to PYTHONPATH so the app can
    import the predictor module at startup. If none of the above
    locations contain predictor.py the app still starts and returns
    HTTP 501 until predictor.py is created.

    Press Ctrl+C to stop.
    """
    import os
    import subprocess
    import sys
    import threading
    import webbrowser

    # ── Resolve the directory that contains predictor.py ─────────────────────
    cwd = Path.cwd()
    if predictor_dir is not None:
        pred_dir = Path(predictor_dir).resolve()
    elif (cwd / "src" / "serve" / "predictor.py").exists():
        pred_dir = (cwd / "src" / "serve").resolve()
    elif (cwd / "predictor.py").exists():
        pred_dir = cwd.resolve()
    else:
        # Default: src/serve/ — app returns 501 if predictor.py is absent.
        pred_dir = (cwd / "src" / "serve").resolve()

    url = f"http://localhost:{port}"
    typer.echo(f"Serving   → {url}")
    typer.echo(f"Predictor → {pred_dir}  (exported as $KITCHEN_PREDICTOR_DIR — reserved)")
    if reload:
        typer.echo("Reload    → enabled (watchfiles)")
    typer.echo("Press Ctrl+C to stop.\n")

    # ── Open /docs after a short delay ───────────────────────────────────────
    if open_browser:
        def _open_after_delay() -> None:
            import time

            time.sleep(1.5)
            webbrowser.open(f"{url}/docs")

        threading.Thread(target=_open_after_delay, daemon=True).start()

    # ── Prepend pred_dir to PYTHONPATH and set KITCHEN_PREDICTOR_DIR ─────────
    # PYTHONPATH: backwards-compatible for any code that scans sys.path.
    # KITCHEN_PREDICTOR_DIR: used by kitchen.serve.loader for deterministic
    # resolution without a full sys.path scan.
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    new_pythonpath = (
        f"{pred_dir}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(pred_dir)
    )
    env = {
        **os.environ,
        "PYTHONPATH": new_pythonpath,
        "KITCHEN_PREDICTOR_DIR": str(pred_dir),
    }

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "kitchen.serve.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    if reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd, env=env, check=False)
    except KeyboardInterrupt:
        typer.echo("\nStopped.")


# ---------------------------------------------------------------------------
# kitchen dashboard — generate and view the static results dashboard
# ---------------------------------------------------------------------------


dashboard_app = typer.Typer(help="Dashboard helpers.", no_args_is_help=True)



@dashboard_app.command("generate")
def dashboard_generate(
    output: Annotated[
        str, typer.Option("--output", help="Path to write the HTML file")
    ] = "dashboard/index.html",
    branch: Annotated[
        str, typer.Option("--branch", help="Git branch holding results/*.json files")
    ] = "results",
    metric: Annotated[
        str | None, typer.Option("--metric", help="Primary metric to chart (auto-detected if omitted)")
    ] = None,
    show_params: Annotated[
        str | None,
        typer.Option(
            "--show-params",
            help="Comma-separated param keys to show as extra columns (e.g. model.eta,model.max_depth)",
        ),
    ] = None,
    params_file: Annotated[
        str, typer.Option("--params", help="Path to params.yaml (used for project name)")
    ] = "params.yaml",
    serve: Annotated[
        bool,
        typer.Option(
            "--serve/--no-serve",
            help="After generating, start a local HTTP server and open the dashboard in the browser",
        ),
    ] = False,
) -> None:
    """Re-render dashboard/index.html from results/*.json on the results branch.

    Reads result snapshots written by `kitchen push` from the local results branch
    and produces a self-contained HTML file with all data embedded — no server or
    GitHub Pages needed to view it locally.

    Pass --serve to start a local HTTP server and open the dashboard immediately.
    The generated file is NOT committed to git; it is a local build artifact.
    """
    import json
    import subprocess
    from datetime import datetime, timezone

    branch_ref = f"refs/heads/{branch}"
    check = subprocess.run(
        ["git", "rev-parse", "--verify", branch_ref],
        capture_output=True,
        check=False,
    )
    if check.returncode != 0:
        typer.echo(
            f"error: branch '{branch}' not found locally — run `kitchen push` first.",
            err=True,
        )
        raise typer.Exit(1)

    ls_raw = subprocess.check_output(
        ["git", "ls-tree", "--name-only", branch_ref, "results/"]
    ).decode().strip()
    json_files = [f for f in ls_raw.splitlines() if f.endswith(".json")]
    if not json_files:
        typer.echo(
            f"error: no .json files found under results/ on branch '{branch}'.",
            err=True,
        )
        raise typer.Exit(1)

    results: list[dict] = []
    for file_path in json_files:
        raw = subprocess.check_output(
            ["git", "cat-file", "-p", f"{branch_ref}:{file_path}"]
        ).decode()
        try:
            results.append(json.loads(raw))
        except json.JSONDecodeError:
            typer.echo(f"warning: skipping malformed result file {file_path}", err=True)

    if not results:
        typer.echo("error: no valid result files found.", err=True)
        raise typer.Exit(1)

    results.sort(key=lambda r: r.get("timestamp", ""))

    project = "project"
    from kitchen.config import resolve_params_path

    params_file = resolve_params_path(params_file)
    if Path(params_file).exists():
        try:
            from kitchen.config import KitchenConfig

            cfg = KitchenConfig.from_yaml(params_file)
            project = cfg.experiment
        except Exception:
            pass

    resolved_metric = metric
    if resolved_metric is None:
        priority = [
            "val_accuracy", "val_brier", "val_log_loss", "val_roc_auc",
            "val_rmse", "val_mae", "loto_brier",
        ]
        for candidate in priority:
            if any(candidate in (r.get("metrics") or {}) for r in results):
                resolved_metric = candidate
                break
        if resolved_metric is None:
            for r in results:
                for k in (r.get("metrics") or {}):
                    if not k.startswith("fi."):
                        resolved_metric = k
                        break
                if resolved_metric:
                    break

    param_keys: list[str] = []
    if show_params:
        requested = [p.strip() for p in show_params.split(",") if p.strip()]
        if any(r.get("params") for r in results):
            param_keys = requested

    has_lb = any(r.get("lb_score") is not None for r in results)

    status = (
        f"{len(results)} run(s) loaded. "
        f"Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    html = _render_dashboard_html(
        project=project,
        status=status,
        results=results,
        metric=resolved_metric or "",
        param_keys=param_keys,
        has_lb=has_lb,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    typer.echo(f"Dashboard written → {out_path}  ({len(results)} run(s))")
    typer.echo(f"  metric  : {resolved_metric or '—'}")
    if param_keys:
        typer.echo(f"  params  : {', '.join(param_keys)}")

    if serve:
        _serve_local_dashboard(out_path)
    else:
        typer.echo("\nOpen with: kitchen open  (or kitchen dashboard generate --serve)")


