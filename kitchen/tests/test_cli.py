import json
import os
import subprocess
import sys
import time

from conftest import kitchen, kj


def test_runs_get_verdicts_and_promotion_is_gated(project):
    assert kitchen(project, "split", "cv").returncode == 0
    code, runs = kj(project, "run", "base", "-m", "baseline", "--op", "draft")
    assert code == 0 and runs[0]["score"]["value"] > 0.7 and "verdict" not in runs[0]
    assert (
        kitchen(project, "promote", "r0001").returncode == 0
    )  # the first champion needs no verdict
    _, better = kj(project, "run", "base", "-p", "noise=0.5", "--parent", "r0001")
    _, worse = kj(project, "run", "base", "-p", "noise=3", "--parent", "r0001")
    assert better[0]["verdict"]["verdict"] == "better" and worse[0]["verdict"]["verdict"] == "worse"
    refused = kitchen(project, "promote", "r0003")
    assert refused.returncode == 1 and "not better" in refused.stderr
    assert kitchen(project, "promote", "r0002", env={"KITCHEN_AUTONOMOUS": "1"}).returncode == 3
    assert kitchen(project, "promote", "r0002").returncode == 0
    _, champ = kj(project, "show", "--champion")
    assert champ["main"]["id"] == "r0002" and os.path.islink(champ["main"]["path"])
    assert kitchen(project, "run", "nope").returncode == 2  # config error


def test_grid_scratch_failure_timeout_and_gated_passthrough(project):
    code, runs = kj(project, "run", "base", "-p", "noise=0.4,0.6", "-p", "seed=1")
    assert code == 0 and [r["params"] for r in runs] == [
        {"noise": 0.4, "seed": 1},
        {"noise": 0.6, "seed": 1},
    ]
    script = (
        "import csv, os\nrows = list(csv.DictReader(open('data/train.csv')))\n"
        "out = os.path.join(os.environ['KITCHEN_RUN_DIR'], 'oof.csv')\n"
        "open(out, 'w').write('id,pred\\n' + ''.join(f\"{r['id']},{r['x1']}\\n\" for r in rows))\n"
    )
    code, runs = kj(
        project, "run", "--scratch", "-m", "heredoc", "--", sys.executable, "-", stdin=script
    )
    assert code == 0 and runs[0]["scratch"] and runs[0]["score"]["value"] > 0.7
    assert open(os.path.join(runs[0]["dir"], "stdin.txt")).read() == script  # the heredoc is kept
    assert kitchen(project, "promote", runs[0]["id"]).returncode == 1  # scratch needs --force
    code, runs = kj(
        project, "run", "--", sys.executable, "-c", "print('boom'); raise SystemExit(3)"
    )
    assert code == 10 and runs[0]["status"] == "failed" and "boom" in runs[0]["log_tail"]
    code, runs = kj(
        project, "run", "--timeout", "1s", "--", sys.executable, "-c", "import time; time.sleep(30)"
    )
    assert code == 124 and runs[0]["status"] == "timeout"
    assert (
        kitchen(project, "run", "--", "kaggle", "competitions", "submit", "-c", "x").returncode == 3
    )


def test_detached_run_is_scored_before_wait_returns(project):
    kitchen(project, "run", "base")
    kitchen(project, "promote", "r0001")
    code, runs = kj(project, "run", "base", "--detach", "-p", "noise=0.4")
    assert code == 0 and runs[0]["status"] == "running"
    code, run = kj(project, "status", "--wait", runs[0]["id"], "--timeout", "90s")
    assert code == 0 and run["status"] == "ok" and run["verdict"]["verdict"] == "better"


def test_brief_lineage_board_compare_and_quarantine(project):
    kitchen(project, "run", "base", "-m", "baseline")
    kitchen(project, "promote", "r0001")
    kitchen(project, "run", "base", "-p", "noise=0.5", "--parent", "r0001", "--op", "improve")
    kitchen(project, "run", "base", "-p", "noise=2", "--parent", "r0001", "--op", "improve")
    assert kitchen(project, "mark", "r0003", "--note", "too noisy").returncode == 0
    brief = kitchen(project, "show", "--brief").stdout
    assert (
        "champion r0001" in brief
        and "awaiting human promotion: r0002" in brief
        and "too noisy" in brief
    )
    _, lin = kj(project, "show", "r0002", "--lineage")
    assert lin["ancestors"][0].startswith("r0001") and lin["siblings"][0].startswith("r0003")
    _, rows = kj(project, "show")
    assert rows[0]["id"] == "r0002"  # best AUC first
    md = kitchen(project, "compare", "r0002", "--md").stdout
    assert "**better**" in md and "noise: 1.0 -> 0.5" in md and "kitchen promote r0002" in md
    kitchen(project, "mark", "r0002", "--quarantine", "leak")
    code, v = kj(project, "compare", "r0002")
    assert code == 5 and v["verdict"] == "incomparable"


def test_import_and_agent_install_merge(project):
    out = project / "nb_out"
    out.mkdir()
    rows = [r.split(",") for r in (project / "data" / "train.csv").read_text().splitlines()[1:]]
    (out / "oof.csv").write_text("id,pred\n" + "".join(f"{r[0]},{r[2]}\n" for r in rows))
    code, runs = kj(project, "run", "--import", str(out), "-m", "notebook result")
    assert code == 0 and runs[0]["host"] == "import" and runs[0]["score"]["value"] > 0.7
    (project / ".claude").mkdir()
    (project / ".claude" / "settings.json").write_text(
        '{"model": "opus", "permissions": {"allow": ["Bash(ls)"]}}'
    )
    for _ in range(2):  # idempotent
        assert kitchen(project, "init", "--agent").returncode == 0
    settings = json.loads((project / ".claude" / "settings.json").read_text())
    assert settings["model"] == "opus" and "Bash(ls)" in settings["permissions"]["allow"]
    assert (
        "Bash(kitchen promote *)" in settings["permissions"]["ask"]
        and len(settings["hooks"]["SessionStart"]) == 1
    )
    assert (project / "CLAUDE.md").read_text().count("<!-- kitchen -->") == 1
    assert (
        "kitchen-version" in (project / ".claude" / "skills" / "kitchen" / "SKILL.md").read_text()
    )


def test_hook_is_silent_outside_projects_and_the_cli_stays_light(project, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("not-a-project")
    out = subprocess.run(
        [sys.executable, "-m", "kitchen", "-C", str(elsewhere), "show", "--brief", "--hook"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0 and out.stdout == ""
    probe = "import sys, kitchen.cli; print([m for m in ('numpy', 'pandas', 'pyarrow') if m in sys.modules])"
    assert (
        subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout.strip()
        == "[]"
    )
    kitchen(project, "run", "base")
    t0 = time.perf_counter()
    assert kitchen(project, "show", "--brief", "--hook").returncode == 0
    assert time.perf_counter() - t0 < 1.5  # ~0.1 s locally; generous for CI
