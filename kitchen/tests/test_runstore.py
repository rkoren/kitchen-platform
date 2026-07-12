"""Tests for kitchen.runstore + `kitchen log` / `kitchen leaderboard --store` (GEN-001)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kitchen.cli import app
from kitchen.runstore import log_run, make_record, read_runs

runner = CliRunner()


# ── runstore unit ─────────────────────────────────────────────────────────────


def test_make_record_stamps_and_coerces():
    rec = make_record(params={"linker": "ilp", "n": 3}, metrics={"score": 1}, tags={"env": "track"})
    assert rec.params == {"linker": "ilp", "n": "3"}     # params stringified
    assert rec.metrics == {"score": 1.0}                 # metrics floated
    assert rec.tags == {"env": "track"}
    assert rec.timestamp                                  # ISO timestamp stamped
    assert isinstance(rec.id, str) and len(rec.id) == 12


def test_log_and_read_round_trip(tmp_path):
    store = tmp_path / "runs.jsonl"
    r1 = make_record("aaa", metrics={"score": 0.8})
    r2 = make_record("bbb", metrics={"score": 0.9}, params={"k": "v"})
    log_run(store, r1)
    log_run(store, r2)
    runs = read_runs(store)
    assert [r.id for r in runs] == ["aaa", "bbb"]         # file order preserved
    assert runs[1].params == {"k": "v"}
    lines = store.read_text().splitlines()
    assert len(lines) == 2                                # one JSON line per record
    assert all(json.loads(ln)["schema_version"] == 1 for ln in lines)  # versioned format


def test_read_missing_store_is_empty(tmp_path):
    assert read_runs(tmp_path / "nope.jsonl") == []


def test_read_corrupt_line_raises_naming_line(tmp_path):
    store = tmp_path / "runs.jsonl"
    store.write_text('{"id": "ok", "metrics": {"s": 1.0}}\nnot json\n')
    with pytest.raises(ValueError, match=r"runs\.jsonl:2"):
        read_runs(store)


def test_concurrent_writers_keep_the_store_consistent(tmp_path):
    # Property we guarantee: concurrent writers leave a fully-parseable store with every record
    # present. log_run holds an exclusive flock for this; the flock is portable correctness
    # insurance (POSIX only guarantees atomic appends up to PIPE_BUF) — on some filesystems a
    # bare append happens to be atomic, so this asserts the guarantee, it doesn't prove the lock
    # is load-bearing on this platform.
    store = tmp_path / "runs.jsonl"
    n = 40
    blob = "x" * 8192  # large records → multiple write() syscalls per line without the lock

    def _write(i: int) -> None:
        log_run(store, make_record(f"run{i:03d}", tags={"blob": blob}))

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every line must be valid JSON and we must get exactly n distinct records back.
    runs = read_runs(store)  # raises if any line is corrupt
    assert len(runs) == n
    assert len({r.id for r in runs}) == n
    for line in store.read_text().splitlines():
        json.loads(line)  # each line independently parseable


# ── kitchen log command ───────────────────────────────────────────────────────


def test_log_command_writes_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["log", "--metric", "track_score=0.83", "--param", "linker=ilp", "--tag", "env=track"],
    )
    assert result.exit_code == 0, result.output
    runs = read_runs(tmp_path / "runs.jsonl")  # DEFAULT_STORE
    assert len(runs) == 1
    assert runs[0].metrics == {"track_score": 0.83}
    assert runs[0].params == {"linker": "ilp"}
    assert runs[0].tags == {"env": "track"}
    assert "logged run" in result.output


def test_log_command_custom_store_and_run_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["log", "--store", "sub/runs.jsonl", "--run", "myrun", "--metric", "s=1.0"]
    )
    assert result.exit_code == 0, result.output
    runs = read_runs(tmp_path / "sub" / "runs.jsonl")
    assert runs[0].id == "myrun"


def test_log_command_rejects_non_numeric_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["log", "--metric", "track_score=high"])
    assert result.exit_code != 0
    assert "not a number" in result.output


def test_log_command_requires_something_to_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["log", "--tag", "env=track"])  # tags alone aren't enough
    assert result.exit_code != 0
    assert "nothing to log" in result.output


# ── kitchen leaderboard --store ───────────────────────────────────────────────


def _seed(tmp_path: Path) -> Path:
    store = tmp_path / "runs.jsonl"
    log_run(store, make_record("a", metrics={"track_score": 0.81}, params={"linker": "greedy"}))
    log_run(store, make_record("b", metrics={"track_score": 0.87}, params={"linker": "ilp"}))
    log_run(store, make_record("c", metrics={"track_score": 0.79}, params={"linker": "greedy"}))
    return store


def test_leaderboard_store_ranks_higher_is_better(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    result = runner.invoke(
        app,
        ["leaderboard", "--store", "runs.jsonl", "--metric", "track_score", "--higher-is-better"],
    )
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    # the ★ (top) row must be run b (0.87)
    star_row = next(ln for ln in lines if ln.startswith("★"))
    assert "b" in star_row and "0.8700" in star_row


def test_leaderboard_store_ranks_lower_is_better(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    result = runner.invoke(app, ["leaderboard", "--store", "runs.jsonl", "--metric", "track_score"])
    assert result.exit_code == 0, result.output
    star_row = next(ln for ln in result.output.splitlines() if ln.startswith("★"))
    assert "c" in star_row and "0.7900" in star_row  # lowest first


def test_leaderboard_store_requires_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    result = runner.invoke(app, ["leaderboard", "--store", "runs.jsonl"])
    assert result.exit_code != 0
    assert "--metric is required" in result.output
    assert "track_score" in result.output  # names available metrics


def test_leaderboard_store_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["leaderboard", "--store", "missing.jsonl", "--metric", "s"])
    assert result.exit_code == 0
    assert "No runs in store" in result.output
