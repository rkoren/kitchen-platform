"""Append-only event ledger (.kitchen/ledger.jsonl), folded into run and champion state."""

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


class State:
    """Runs, champions and splits as of the last event."""

    def __init__(self, events: list[dict]):
        self.runs: dict[str, dict] = {}
        self.champions: dict[str, str] = {}
        self.splits: dict[str, dict] = {}
        for e in events:
            kind, rid = e.get("ev"), e.get("id")
            run = self.runs.get(rid)
            if kind == "run":
                self.runs[rid] = {**e, "status": "running", "notes": []}
            elif kind == "split":
                self.splits[e["name"]] = e
            elif kind == "promote":
                self.champions[e["track"]] = rid
            elif run is None:
                continue
            elif kind == "end":
                run.update(
                    status=e["status"], exit=e.get("exit"), seconds=e.get("seconds"), ended=e["t"]
                )
            elif kind == "score":
                run["score"] = _payload(e)
            elif kind == "score_error":
                run["score_error"] = e.get("error")
            elif kind == "verdict":
                run["verdict"] = _payload(e)
            elif kind == "mark":
                if e.get("note"):
                    run["notes"].append(e["note"])
                run.update({k: e[k] for k in ("quarantine", "retire", "baseline") if k in e})

    def track_runs(self, track: str) -> list[dict]:
        return [r for r in self.runs.values() if r.get("track") == track]

    def get(self, rid: str) -> dict:
        from kitchen.config import KitchenError

        if rid not in self.runs:
            raise KitchenError(f"unknown run {rid!r}")
        return self.runs[rid]


def _payload(e: dict) -> dict:
    return {k: v for k, v in e.items() if k not in ("ev", "id", "t")}


class Ledger:
    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir)
        self.path = self.dir / "ledger.jsonl"

    def run_dir(self, rid: str) -> Path:
        return self.dir / "runs" / rid

    @contextmanager
    def _locked(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        if not (self.dir / ".gitignore").exists():  # the state dir ignores itself: no project edits
            (self.dir / ".gitignore").write_text("*\n")
        with open(self.dir / ".lock", "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:  # a torn line from a crash mid-write
                    continue
        return out

    def state(self) -> State:
        return State(self.events())

    def _write(self, events) -> None:
        torn = False
        if self.path.exists() and self.path.stat().st_size:
            with open(self.path, "rb") as fh:
                fh.seek(-1, os.SEEK_END)
                torn = fh.read(1) != b"\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write("\n" * torn)
            for ev in events:
                fh.write(json.dumps({**ev, "t": round(time.time(), 3)}, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def append(self, *events: dict) -> None:
        with self._locked():
            self._write(events)

    def new_run(self, record: dict) -> str:
        """Allocate the next run id and write its `run` event, atomically."""
        with self._locked():
            n = max((int(e["id"][1:]) for e in self.events() if e.get("ev") == "run"), default=0)
            rid = f"r{n + 1:04d}"
            self.run_dir(rid).mkdir(parents=True, exist_ok=True)
            self._write([{"ev": "run", "id": rid, **record}])
        return rid

    def set_champion(self, track: str, rid: str) -> Path:
        """Point .kitchen/champion/<track> at the run dir (a stable path for predict/export)."""
        link = self.dir / "champion" / track
        link.parent.mkdir(parents=True, exist_ok=True)
        tmp = link.with_name(f".{track}.tmp")
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(Path("..") / "runs" / rid)
        os.replace(tmp, link)
        return link
