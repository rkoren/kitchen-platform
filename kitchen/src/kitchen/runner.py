"""Run lifecycle: provenance, environment, foreground or detached execution, finalize."""

import itertools
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import yaml

from kitchen import __version__
from kitchen.config import Config, KitchenError, NeedsHuman, git
from kitchen.ledger import Ledger

OUTPUTS = ("oof.parquet", "oof.csv", "scores.csv", "metrics.json", "submission.csv", "test.parquet")
GATED = (
    {"kaggle", "competitions", "submit"},
    {"kaggle", "kernels", "push"},
    {"kaggle", "datasets", "create"},
    {"kaggle", "datasets", "version"},
)
UNTRACKED_MAX = 5_000_000  # bytes of untracked files snapshotted per run
SPAWN = {"stderr": subprocess.STDOUT, "start_new_session": True}  # own process group, merged output
DEFAULTS = {
    "job": None,
    "cmd": None,
    "shell": False,
    "params": {},
    "track": None,
    "parent": None,
    "op": None,
    "hypothesis": None,
    "host": "local",
    "scratch": False,
    "detached": False,
    "timeout": None,
}


def grid(items: list[str]) -> list[dict]:
    """['depth=4,5', 'lr=0.1'] -> [{'depth': 4, 'lr': 0.1}, {'depth': 5, 'lr': 0.1}]."""
    axes = []
    for item in items:
        key, sep, values = item.partition("=")
        if not sep or not key.strip():
            raise KitchenError(f"-p expects key=value[,value2], got {item!r}")
        axes.append([(key.strip(), yaml.safe_load(v)) for v in values.split(",")])
    return [dict(combo) for combo in itertools.product(*axes)]


def read_stdin() -> bytes | None:
    """Heredoc or piped input for a command; never blocks on a terminal or an idle pipe."""
    if sys.stdin is None or sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0.2)
    return (sys.stdin.buffer.read() if ready else b"") or None


def git_state(root: Path) -> tuple[dict, str, list[str]]:
    sha = git(root, "rev-parse", "HEAD")
    if sha is None:
        return {"sha": None}, "", []
    diff = git(root, "diff", "HEAD") or ""
    others = (git(root, "ls-files", "--others", "--exclude-standard") or "").splitlines()
    untracked = [f for f in others if f and not f.startswith(".kitchen/")]
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
    return {"sha": sha, "branch": branch, "dirty": bool(diff or untracked)}, diff, untracked


def record(cfg: Config, ledger: Ledger, **fields) -> str:
    """Validate a run, write its `run` event and snapshot its code; returns the run id."""
    run = {**DEFAULTS, **fields}
    if run["parent"]:
        ledger.state().get(run["parent"])
    if run["track"]:
        from kitchen import splits

        run["split"] = splits.ensure(cfg, ledger, cfg.track(run["track"])[1]["split"])[1]["id"]
    run["git"], diff, untracked = git_state(cfg.root)
    rid = ledger.new_run({**run, "pid": os.getpid(), "kitchen": __version__})
    run_dir = ledger.run_dir(rid)
    if diff:
        (run_dir / "diff.patch").write_text(diff + "\n")
    files = [cfg.root / f for f in untracked if (cfg.root / f).is_file()]
    if files and sum(f.stat().st_size for f in files) <= UNTRACKED_MAX:
        with tarfile.open(run_dir / "untracked.tar.gz", "w:gz") as tar:
            for f in files:
                tar.add(f, arcname=str(f.relative_to(cfg.root)))
    elif files:
        (run_dir / "untracked.txt").write_text("\n".join(untracked) + "\n")
    return rid


def environment(cfg: Config, ledger: Ledger, rid: str, run: dict) -> dict:
    split = run.get("split")
    split_path = ledger.dir / "splits" / f"{split.split('@')[0]}.parquet" if split else ""
    env = cfg.env()
    env |= {
        "KITCHEN_RUN_ID": rid,
        "KITCHEN_RUN_DIR": str(ledger.run_dir(rid)),
        "KITCHEN_SPLIT": str(split_path),
        "KITCHEN_PROJECT": str(cfg.root),
        "KITCHEN_PARAMS": json.dumps(run.get("params") or {}),
        "KITCHEN_TRACK": run.get("track") or "",
    }
    return env


def start(
    cfg: Config, ledger: Ledger, *, detach=False, stdin: bytes | None = None, echo=None, **fields
) -> str:
    """Record a run and execute it (foreground: returns when finalized; detached: at once)."""
    cmd = fields["cmd"]
    if any(g <= set(cmd.split() if isinstance(cmd, str) else cmd) for g in GATED):
        raise NeedsHuman(
            "Kaggle submissions and pushes go through kitchen's gated commands, not `kitchen run`"
        )
    rid = record(cfg, ledger, detached=detach, **fields)
    run_dir = ledger.run_dir(rid)
    if stdin:
        (run_dir / "stdin.txt").write_bytes(stdin)
    if detach:
        argv = [sys.executable, "-m", "kitchen.runner", str(cfg.root), rid]
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            argv = ["caffeinate", "-i", *argv]  # keep the Mac awake until the run ends
        with open(run_dir / "log.txt", "ab") as log:
            proc = subprocess.Popen(
                argv, cwd=cfg.root, stdin=subprocess.DEVNULL, stdout=log, **SPAWN
            )
        (run_dir / "pid").write_text(str(proc.pid))
        return rid
    finalize(cfg, ledger, rid, *execute(cfg, ledger, rid, echo))
    return rid


def _kill(proc: subprocess.Popen, flag: threading.Event | None) -> None:
    if flag is not None:
        flag.set()
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(grace)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def execute(cfg: Config, ledger: Ledger, rid: str, echo=None) -> tuple[int, float, bool]:
    """Run the recorded command in its own process group; tee output to `echo` if given."""
    run, run_dir = ledger.state().get(rid), ledger.run_dir(rid)
    stdin_path = run_dir / "stdin.txt"
    timed_out, t0 = threading.Event(), time.monotonic()
    env = environment(cfg, ledger, rid, run)
    with (
        open(stdin_path if stdin_path.exists() else os.devnull, "rb") as stdin,
        open(run_dir / "log.txt", "ab") as log,
    ):
        out = subprocess.PIPE if echo else log
        proc = subprocess.Popen(
            run["cmd"], shell=run["shell"], cwd=cfg.root, env=env, stdin=stdin, stdout=out, **SPAWN
        )
        timer = (
            threading.Timer(run["timeout"], _kill, (proc, timed_out))
            if run.get("timeout")
            else None
        )
        if timer:
            timer.daemon = True
            timer.start()
        try:
            for line in iter(proc.stdout.readline, b"") if echo else ():
                log.write(line)
                log.flush()
                echo.write(line.decode(errors="replace"))
                echo.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            _kill(proc, None)
            code = 130
        finally:
            if timer:
                timer.cancel()
    return (124 if timed_out.is_set() else code), time.monotonic() - t0, timed_out.is_set()


def finalize(
    cfg: Config, ledger: Ledger, rid: str, code: int, seconds: float, timed_out=False
) -> dict:
    """Score the run on its track and compare it with the champion (or parent), then record the end.

    `end` is written last: it is the commit marker that `status --wait` waits for.
    """
    status = "timeout" if timed_out else "ok" if code == 0 else "failed"
    run = ledger.state().get(rid)
    if status == "ok" and run.get("track"):
        from kitchen import scoring, stats

        try:
            sc = scoring.score(cfg, ledger, rid, run["track"])
            if sc:
                ledger.append({"ev": "score", "id": rid, **sc})
                base = ledger.state().champions.get(run["track"]) or run.get("parent")
                if base and base != rid:
                    ledger.append(
                        {"ev": "verdict", "id": rid, **stats.compare(cfg, ledger, rid, base)}
                    )
        except Exception as e:  # a scoring problem must never lose the run record
            ledger.append({"ev": "score_error", "id": rid, "error": f"{type(e).__name__}: {e}"})
    ledger.append(
        {"ev": "end", "id": rid, "exit": code, "seconds": round(seconds, 1), "status": status}
    )
    run = ledger.state().get(rid)
    (ledger.run_dir(rid) / "run.json").write_text(json.dumps(run, indent=2, default=str))
    return run


def import_dir(cfg: Config, ledger: Ledger, src: Path, **fields) -> str:
    """Record outputs produced elsewhere (a notebook, Colab, a legacy export) as a run."""
    src = Path(src).expanduser().resolve()
    if not src.is_dir():
        raise KitchenError(f"--import: {src} is not a directory")
    fields["op"] = fields.get("op") or "import"
    rid = record(cfg, ledger, host="import", source=str(src), **fields)
    for name in OUTPUTS:
        if (src / name).is_file():
            shutil.copy2(src / name, ledger.run_dir(rid) / name)
    finalize(cfg, ledger, rid, 0, 0.0)
    return rid


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def refresh(ledger: Ledger) -> list[str]:
    """Close runs whose process died without finalizing, as `lost`."""
    lost = []
    for rid, r in ledger.state().runs.items():
        pid_file = ledger.run_dir(rid) / "pid"
        pid = int(pid_file.read_text()) if r.get("detached") and pid_file.exists() else r.get("pid")
        if r["status"] == "running" and pid and not _alive(pid):
            time.sleep(0.2)  # the finalizer may be mid-write
            if ledger.state().runs[rid]["status"] == "running":
                ledger.append(
                    {"ev": "end", "id": rid, "exit": None, "seconds": None, "status": "lost"}
                )
                lost.append(rid)
    return lost


def wait(ledger: Ledger, rid: str, timeout: float | None = None, poll: float = 2.0) -> dict:
    end = time.monotonic() + timeout if timeout else None
    while True:
        refresh(ledger)
        run = ledger.state().get(rid)
        if run["status"] != "running" or (end and time.monotonic() >= end):
            return run
        time.sleep(poll)


def watch(ledger: Ledger, rid: str, out=sys.stdout) -> dict:
    """Stream a run's log until it ends (Monitor-friendly: one line per output line)."""
    log, pos = ledger.run_dir(rid) / "log.txt", 0
    while True:
        run = ledger.state().get(rid)  # status first, so the final lines are never missed
        if log.exists():
            with open(log, "rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
            pos += len(chunk)
            out.write(chunk.decode(errors="replace"))
            out.flush()
        if run["status"] != "running":
            return run
        refresh(ledger)
        time.sleep(1)


if __name__ == "__main__":  # detached child: python -m kitchen.runner <project-root> <run-id>
    _cfg = Config.load(Path(sys.argv[1]))
    _ledger = Ledger(_cfg.state)
    finalize(_cfg, _ledger, sys.argv[2], *execute(_cfg, _ledger, sys.argv[2]))
