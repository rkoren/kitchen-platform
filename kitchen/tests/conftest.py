import csv
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

TRAIN = '''"""Toy model: pred = sigmoid(1.5*x1 + noise*N(0,1)); writes $KITCHEN_RUN_DIR/oof.csv."""
import argparse, csv, math, os, random
ap = argparse.ArgumentParser(); ap.add_argument("--noise", type=float); ap.add_argument("--seed", type=int)
a = ap.parse_args(); rng = random.Random(a.seed)
rows = list(csv.DictReader(open("data/train.csv")))
with open(os.path.join(os.environ["KITCHEN_RUN_DIR"], "oof.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "pred"])
    for r in rows:
        w.writerow([r["id"], 1 / (1 + math.exp(-(1.5 * float(r["x1"]) + a.noise * rng.gauss(0, 1))))])
print("trained", len(rows))
'''


def _git(path: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=path,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A git repo with 2,000 rows over 10 seasons, a toy job and a logo split scored by AUC."""
    (tmp_path / "data").mkdir()
    rng = random.Random(7)
    with open(tmp_path / "data" / "train.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "season", "x1", "y"])
        for i in range(2000):
            x1 = rng.gauss(0, 1)
            w.writerow(
                [
                    f"g{i:04d}",
                    2010 + i % 10,
                    round(x1, 4),
                    int(rng.random() < 1 / (1 + math.exp(-1.5 * x1))),
                ]
            )
    (tmp_path / "train.py").write_text(TRAIN)
    (tmp_path / "kitchen.yaml").write_text(f"""project: toy
python: {sys.executable}
splits:
  cv: {{data: data/train.csv, id: id, target: y, scheme: logo, group: season}}
tracks:
  main: {{split: cv, metric: auc}}
jobs:
  base: {{cmd: "{{python}} train.py --noise {{noise}} --seed {{seed}}", params: {{noise: 1.0, seed: 0}}}}
run: {{timeout: 2m}}
""")
    (tmp_path / ".gitignore").write_text(".kitchen/\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def kitchen(
    project: Path, *args: str, stdin: str | None = None, env: dict | None = None
) -> subprocess.CompletedProcess:
    """Run the real CLI in a subprocess (as an agent would)."""
    return subprocess.run(
        [sys.executable, "-m", "kitchen", "-C", str(project), *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, **(env or {})},
    )


def kj(project: Path, *args: str, **kw):
    """Run with --json (placed before any `--`) and parse stdout."""
    out = kitchen(project, args[0], "--json", *args[1:], **kw)
    return out.returncode, json.loads(out.stdout) if out.stdout.strip() else None
