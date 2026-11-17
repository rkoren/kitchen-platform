"""Compact views of the ledger: the session brief and lineage for agents, the board for humans.

No numpy/pandas here: the brief runs in a SessionStart hook and must stay fast.
"""

import re
import shutil
import sys
import time

from kitchen import __version__

BRIEF_CHARS = 6000  # ~1,500 tokens; Claude Code caps SessionStart context at 10,000 chars
RULES = (
    "rules: train/eval via `kitchen run`; quote its verdicts; mark every outcome; humans promote"
)
BETTER = {"min": "lower", "max": "higher"}


def _f(x, signed=False) -> str:
    return "n/a" if x is None else f"{x:+.5f}" if signed else f"{x:.5f}"


def _age(r: dict) -> str:
    m = int((time.time() - r["t"]) // 60)
    return f"{m}m" if m < 120 else f"{m // 60}h"


def label(r: dict) -> str:
    """One line per run: id, op<-parent, outcome, flags, latest note (or the hypothesis)."""
    head = f"{r['id']} {r.get('op') or 'run'}" + (f"<-{r['parent']}" if r.get("parent") else "")
    v, s, status = r.get("verdict") or {}, r.get("score") or {}, r["status"]
    if status == "running":
        out = f"running {_age(r)}" + " (detached)" * bool(r.get("detached"))
    elif status != "ok":
        out = status + (f" (exit {r['exit']})" if r.get("exit") not in (None, 0) else "")
    elif v.get("verdict") in ("better", "worse", "tie"):
        lo, hi = v.get("ci") or (None, None)
        out = f"{v['verdict']:<6} {_f(v['delta'], 1)} [{_f(lo, 1)},{_f(hi, 1)}] vs {v['vs']}"
    elif s:
        cov = f" coverage {s['coverage']:.0%}" if s.get("coverage", 1) < 1 else ""
        out = f"score {_f(s.get('value'))}{cov}" + (f" ({v['reason']})" if v.get("reason") else "")
    else:
        out = "ok, unscored" + (f" — {r['score_error']}" if r.get("score_error") else "")
    flags = "".join(f" [{k}]" for k in ("quarantine", "retire", "scratch") if r.get(k))
    text = (r["notes"][-1] if r.get("notes") else r.get("hypothesis")) or ""
    return f"{head:<17} {out}{flags}  {text}".rstrip()[:220]


def _track(cfg, st, name: str, track: dict) -> list[str]:
    split, champ = st.splits.get(track["split"]), st.champions.get(name)
    runs = sorted(st.track_runs(name), key=lambda r: r["t"])
    scored = [r["score"] for r in runs if r.get("score")]
    where = split and f"{split['id']} ({split['folds']} folds, {split['val_rows']} rows)"
    better = f" ({BETTER[scored[-1]['direction']]} is better)" if scored else ""
    head = f"{name}: {cfg.metric_spec(track).get('name')}{better} on {where or '(not built)'}"
    lines = ["", head]
    if champ:
        c = st.runs[champ]
        note = f"  — {c['notes'][-1]}" if c.get("notes") else ""
        lines.append(f"  champion {champ} {_f((c.get('score') or {}).get('value'))}{note}")
    else:
        lines.append("  champion: none yet (a human promotes the first one)")
    waiting = [
        r["id"]
        for r in runs
        if (r.get("verdict") or {}).get("verdict")
        == "better"  # checked first: unscored runs have none
        and r["verdict"].get("vs") == champ != r["id"]
        and not r.get("quarantine")
    ]
    lines += [f"  awaiting human promotion: {', '.join(waiting[-3:])}"] * bool(waiting)
    return lines + [f"  {label(r)}" for r in runs[::-1][:8]]


def brief(cfg, st) -> str:
    runs = list(st.runs.values())
    project = cfg.data.get("project") or cfg.root.name
    lines = [f"kitchen {__version__} | {project} | {shutil.which('kitchen') or sys.argv[0]}", RULES]
    for name, track in cfg.tracks.items():
        lines += _track(cfg, st, name, track)
    loose = [r for r in runs if not r.get("track")]
    live = [f"{r['id']} {_age(r)}" for r in runs if r["status"] == "running"]
    bad = [f"{r['id']} ({r['quarantine']})" for r in runs if r.get("quarantine")]
    lines += [f"\nuntracked runs: {len(loose)}; latest: {label(r)}" for r in loose[-1:]]
    lines += [f"in flight: {', '.join(live)}"] * bool(live)
    lines += [f"quarantined: {'; '.join(bad[-5:])}"] * bool(bad)
    skill = cfg.root / ".claude" / "skills" / "kitchen" / "SKILL.md"
    m = re.search(r"kitchen-version: (\S+)", skill.read_text()) if skill.exists() else None
    if m and m[1] != __version__:
        lines.append(f"skill files are from kitchen {m[1]}; run `kitchen init --agent` to update")
    text = "\n".join(lines)
    return text if len(text) <= BRIEF_CHARS else text[: BRIEF_CHARS - 16] + "\n…(truncated)"


def lineage(st, rid: str) -> dict:
    run = cur = st.get(rid)
    chain, parent = [], run.get("parent")
    while cur.get("parent") in st.runs and len(chain) < 25:
        cur = st.runs[cur["parent"]]
        chain.append(cur)
    kin = {
        "ancestors": chain,
        "children": [r for r in st.runs.values() if r.get("parent") == rid],
        "siblings": [
            r for r in st.runs.values() if parent and r.get("parent") == parent and r["id"] != rid
        ],
    }
    return {"run": label(run), **{k: [label(r) for r in v] for k, v in kin.items()}}


def render_lineage(lin: dict) -> str:
    groups = [
        f"{k}:\n" + "\n".join(f"  {x}" for x in lin[k])
        for k in ("ancestors", "siblings", "children")
        if lin[k]
    ]
    return "\n".join([lin["run"], *groups])


def board(cfg, st, track: str | None = None) -> tuple[list[dict], str]:
    rows, lines = [], []
    for name in [track] if track else list(cfg.tracks):
        runs = [r for r in st.track_runs(name) if (r.get("score") or {}).get("value") is not None]
        if not runs:
            lines.append(f"{name}: no scored runs yet")
            continue
        direction, champ = runs[-1]["score"]["direction"], st.champions.get(name)
        sign = -1 if direction == "max" else 1  # flagged runs sink below clean ones
        runs.sort(key=lambda r: sign * r["score"]["value"])
        runs.sort(key=lambda r: bool(r.get("quarantine") or r.get("retire")))  # stable
        lines.append(
            f"{name}: {runs[0]['score']['metric']} ({BETTER[direction]} is better; * champion)"
        )
        for r in runs[:20]:
            keep = {k: r.get(k) for k in ("id", "params", "score", "verdict")}
            rows.append(keep | {"track": name, "champion": r["id"] == champ})
            mark = "*" if r["id"] == champ else " "
            lines.append(f" {mark} {_f(r['score']['value'])}  {label(r)}")
    return rows, "\n".join(lines)
