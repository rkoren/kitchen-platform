"""End-to-end smoke test for the offline-quickstart showcase example.

`examples/offline-quickstart/` is the v1.0.0 "one complete showcase example": a full
features → train → evaluate loop that runs with no credentials and no network. This
test copies it to a temp dir and drives the real CLI so the example can't silently rot
before a release — if the scaffold contract or these commands break, this fails.

Kept as a coarse smoke check (pipeline runs, champion promoted, metrics sane); golden
fixed-value snapshots are DX-007's job.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "offline-quickstart"


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "kitchen.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.skipif(not EXAMPLE_DIR.exists(), reason="offline-quickstart example missing")
def test_offline_quickstart_runs_end_to_end(tmp_path: Path) -> None:
    project = tmp_path / "offline-quickstart"
    shutil.copytree(EXAMPLE_DIR, project)

    train = _run(["run", "train", "--auto-promote"], project)
    assert train.returncode == 0, f"train failed:\n{train.stdout}\n{train.stderr}"
    assert "champion" in (train.stdout + train.stderr).lower()

    evaluate = _run(["run", "evaluate"], project)
    assert evaluate.returncode == 0, f"evaluate failed:\n{evaluate.stdout}\n{evaluate.stderr}"

    metrics = json.loads((project / "metrics.json").read_text())
    # Real signal in the synthetic data — accuracy clears the menu.yaml threshold (0.70).
    assert metrics["accuracy"] >= 0.70, metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0, metrics
    assert metrics["run_id"], "metrics.json should carry the run_id"
