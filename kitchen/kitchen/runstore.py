"""Framework-agnostic run tracking (GEN-001) — a dependency-light local run store.

Log ``{params → metrics}`` for a run from **any** process, env, or venv, then rank/compare it —
without MLflow, the ``Trainer`` ABC, or kitchen's interpreter. Records are appended as
**JSON Lines** (one JSON object per line) to a local file. The line format *is* the cross-env
contract, so a script in a separate venv (no mlflow installed) can append a record directly:

    {"id": "...", "params": {...}, "metrics": {...}, "tags": {...},
     "timestamp": "<iso8601>", "git_sha": "<sha|null>"}

``kitchen log`` is the convenience writer for envs that have kitchen installed; ``kitchen
leaderboard --store <path>`` reads the store. (The *format* needs nothing but ``json`` — importing
``kitchen`` itself still pulls the ML stack today; making ``kitchen.runstore`` importable in
isolation is a separate lazy-import concern.)

**Concurrency.** The store is single-writer: :func:`log_run` takes an exclusive ``flock`` around
the append, so concurrent ``kitchen log`` processes (a parallel sweep) *serialize* rather than
interleave bytes — a bare append past ``PIPE_BUF`` (~4 KB) would risk corrupting a line. A single
process appending with shell ``>>`` is fine; concurrent shell appends are the caller's problem.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#: Default store filename (relative to the working directory).
DEFAULT_STORE = "runs.jsonl"

#: On-disk record format version, written on every line so future readers can branch. A record
#: without the field is treated as version 1 (the original format).
SCHEMA_VERSION = 1


@dataclass
class RunRecord:
    """One tracked run: identity + ``{params}`` + ``{metrics}`` + ``{tags}`` + provenance."""

    id: str
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
    git_sha: str | None = None


def new_run_id() -> str:
    """A short, collision-resistant run id."""
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str | None:
    """Best-effort current commit sha; ``None`` outside a git repo (never raises)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except Exception:  # noqa: BLE001 — git absent / not a repo / timeout → no provenance
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def make_record(
    run_id: str | None = None,
    *,
    params: dict | None = None,
    metrics: dict | None = None,
    tags: dict | None = None,
) -> RunRecord:
    """Build a :class:`RunRecord` — id defaults to a fresh one, timestamp + git sha are stamped.

    Params/tags are stringified and metrics coerced to ``float`` so the record is JSON-clean.
    """
    return RunRecord(
        id=run_id or new_run_id(),
        params={k: str(v) for k, v in (params or {}).items()},
        metrics={k: float(v) for k, v in (metrics or {}).items()},
        tags={k: str(v) for k, v in (tags or {}).items()},
        timestamp=_now_iso(),
        git_sha=_git_sha(),
    )


def log_run(store_path: str | Path, record: RunRecord) -> None:
    """Append ``record`` to the JSONL store, single-writer-safe via an exclusive ``flock``.

    Concurrent writers serialize on the lock rather than interleaving bytes. Creates the store
    (and parent dirs) on first write.
    """
    import fcntl  # noqa: PLC0415 — POSIX-only; imported lazily so the module imports anywhere

    path = Path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"schema_version": SCHEMA_VERSION, **asdict(record)}, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _read_all(store_path: str | Path) -> Iterator[RunRecord]:
    path = Path(store_path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON in run store: {exc}") from exc
            yield RunRecord(
                id=d.get("id", ""),
                params=d.get("params", {}) or {},
                metrics=d.get("metrics", {}) or {},
                tags=d.get("tags", {}) or {},
                timestamp=d.get("timestamp", "") or "",
                git_sha=d.get("git_sha"),
            )


def read_runs(store_path: str | Path) -> list[RunRecord]:
    """All records in the store, in file order (newest last).

    Wraps the private :func:`_read_all` iterator so a future switch to a streaming
    ``Iterator[RunRecord]`` return is a one-line change, not a caller migration.
    """
    return list(_read_all(store_path))
